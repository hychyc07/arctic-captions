# Set the random number generators' seeds for consistency
import copy
import os
import urllib
import random
import stat
import subprocess
import timeit
import numpy
import theano
import time

import newlstm, myelman, mylstm
import flickr30k, flickr8k, coco, videodata
import metrics

# Otherwise the deepcopy fails
import sys
sys.setrecursionlimit(10000)

# datasets: 'name', 'load_data: returns iterator', 'prepare_data: some preprocessing'
datasets = { 'flickr8k': (flickr8k.load_data, flickr8k.prepare_data),
            'flickr30k': (flickr30k.load_data, flickr30k.prepare_data),
            'coco': (coco.load_data, coco.prepare_data),
            'video' : (videodata.load_data, videodata.prepare_data)}

def get_dataset(name):
    return datasets[name][0], datasets[name][1]

# utils functions
def shuffle(lol):
    shuffled_idx = range(lol)
    random.seed(time.time()%1000)
    random.shuffle(shuffled_idx)
    return shuffled_idx

# metrics function using conlleval.pl
def conlleval(p, g, w, filename, script_path):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score

    OTHER:
    script_path :: path to the directory containing the
    conlleval.pl script
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    return get_perf(filename, script_path)


def download(origin, destination):
    '''
    download the corresponding atis file
    from http://www-etud.iro.umontreal.ca/~mesnilgr/atis/
    '''
    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, destination)


def get_perf(filename, folder):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''
    _conlleval = os.path.join(folder, 'conlleval.pl')
    if not os.path.isfile(_conlleval):
        url = 'http://www-etud.iro.umontreal.ca/~mesnilgr/atis/conlleval.pl'
        download(url, _conlleval)
        os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

    proc = subprocess.Popen(["perl",
                            _conlleval],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    stdout, _ = proc.communicate(''.join(open(filename).readlines()))
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])

    return {'p': precision, 'r': recall, 'f1': f1score}

def main(param=None):
    if not param:
        param = {
            'model' : 'lstm_att',
            # model, by default it is lstm
            'fold': 3,
            # 5 folds 0,1,2,3,4
            'data': 'video', #'flickr8k',
            'lr': 0.2970806646812754,
            'verbose': 1,
            'decay': True,
            # decay on the learning rate if improvement stops
            # number of words in the context window
            'nhidden': 200,
            # number of hidden units
            'seed': 345,
            'nepochs': 60,
            # 60 is recommended
            'savemodel': False,
            'minibatch': 8}

    print param
    minibatch = param['minibatch']

    folder_name = os.path.basename(__file__).split('.')[0]
    folder = os.path.join(os.path.dirname(__file__), folder_name)
    if not os.path.exists(folder):
        os.mkdir(folder)

    # load the dataset
    print 'Loading data'
    load_data, prepare_data = get_dataset(param['data'])
    train_set, valid_set, test_set, dictionary, vocab_restriction = load_data()

    # index 0 and 1 always code for the end of sentence and unknown token
    word_idict = dict()
    for kk, vv in dictionary.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'
    idx2word = word_idict
    idx2label = word_idict

    print "final vocab size" , len(idx2label), [idx2label[i] for i in range(10)]
 
    n_word_limit = 8000 #len(word_idict)
    print n_word_limit

    print "restrictions ", [len(a[1]) for a in vocab_restriction.items()[:10]]
    print "restrictions ", sum([len(a[1]) for a in vocab_restriction.items()[:n_word_limit]])
    print "restriction examples" 
    print '  a',  vocab_restriction['a']
    print '  <eos>',  vocab_restriction['<eos>']
    print '  .',  vocab_restriction['.']
    print '  is',  vocab_restriction['is']
    def construct_restrictionmask( idx2word, dictionary, vocab_restriction):
        restriction_mask = numpy.zeros((n_word_limit, n_word_limit))
        for i in range(n_word_limit):
            if i == 1:
                restriction_mask[i, : ] = numpy.ones((1, n_word_limit))
            else:
                for w in vocab_restriction[idx2word[i]]:
                    j = dictionary[w]
                    if j < n_word_limit:
                        restriction_mask[i, j] = 1
        return restriction_mask

    restriction_matrix = construct_restrictionmask(idx2word, dictionary, vocab_restriction)
    print "all restriction" , numpy.sum(restriction_matrix), numpy.average(restriction_matrix)
    print restriction_matrix[0]
    print restriction_matrix[2] , idx2label[2]

    def getdata_prep(datasets):
        x, x_mask, ctx = prepare_data(datasets[0], datasets[1], dictionary, n_words=n_word_limit)
        return ctx, x_mask, x

    train_lex, train_ne, train_y = getdata_prep(train_set)
    valid_lex, valid_ne, valid_y = getdata_prep(valid_set)
    test_lex, test_ne, test_y = getdata_prep(test_set)

    print "train y example" , train_y[0] , train_y.shape

    input_featsize = 4096
    output_nclasses = min(len(dictionary), n_word_limit)
    nsentences = len(train_lex)
    maxlenSentences = max([len(x) for x in train_y])
    print 'maxlenSentences', maxlenSentences
    print 'nclass', output_nclasses

    groundtruth_valid = [map(lambda x: idx2label[x], y[:sum(mask)-1]) for (y, mask) in zip(valid_y, valid_ne)]
    groundtruth_test = [map(lambda x: idx2label[x], y[:sum(mask)-1]) for (y, mask) in zip(test_y, test_ne)]

    print "test set length", len(groundtruth_test), groundtruth_test[:10]

    # instanciate the model
    numpy.random.seed(param['seed'])
    random.seed(param['seed'])
    pad_front_input = False
    pad_front_output = False
    bidirection = False

    maxlen_input = 64 # maxlenSentences
    maxlen_output = maxlenSentences

    def sparsepad_1hot(onebatch_xy, xysize, maxlen=maxlen_input, pad_front=False, inverse=False):
        allbatch = []
        for xeach in onebatch_xy:
            s = numpy.zeros((xysize, maxlen), dtype=theano.config.floatX)
            for i,idx in enumerate(xeach):
                if idx == 0:
                    continue
                loc = i
                if inverse:
                    loc = len(xeach) - 1 - i
                if pad_front:
                    s[idx][max(0, maxlen-len(xeach)+loc)] = 1.0 #to handle out of bound in test
                else:
                    s[idx][min(maxlen-1, loc)] = 1.0 #to handle out of bound in test
            allbatch+=[s]
        data_returned = numpy.swapaxes(numpy.array(allbatch, dtype=theano.config.floatX).transpose(), 1,2)
        return data_returned   # len * batch * sizeoffeature/output

    def get_batch(raw_x, raw_y, shuffled_idx=None, sparsex=False):
        alltraining= zip(raw_x, raw_y)
        batch_lex = []
        batch_y = []
        for b in range(len(raw_x)/minibatch):  #batch iteration
            onebatch_x = []
            onebatch_y = []
            for j in range(minibatch):
                idx = b * minibatch + j
                if shuffled_idx is not None:
                    idx = shuffled_idx[idx]
                onebatch_x += [alltraining[idx][0]]
                onebatch_y += [alltraining[idx][1]]
            if sparsex:
                batch_lex += [sparsepad_1hot(onebatch_x, input_featsize, maxlen_input, pad_front_input)]
            else:
                batch_lex += [numpy.swapaxes(numpy.array(onebatch_x, dtype=theano.config.floatX), 0, 1)]
            batch_y += [sparsepad_1hot(onebatch_y, output_nclasses, maxlen_output, pad_front_output)]
        return batch_lex, batch_y

    # train with early stopping on validation set
    best_f1 = -numpy.inf
    param['clr'] = param['lr']

    input_size = input_featsize

    if param['model'] == 'elman':
        rnn = myelman.ELMANSLU(nh=param['nhidden'],  nc=output_nclasses, nf=input_size, mb=minibatch)
    if param['model'] == 'lstm':
        rnn = mylstm.LSTMSLU(nh=param['nhidden'],  nc=output_nclasses, nf=input_size, mb=minibatch)

    if param['model'] == 'lstm_att':
        rnn = newlstm.LSTM_att(nh_enc=param['nhidden'], nh_dec=param['nhidden'], nh_att=50,
                           nx=input_size, ny=output_nclasses, mb=minibatch,
                               lt=maxlen_input, bidir=bidirection+1, nonlstm_encode=False,
                               restriction = restriction_matrix)

    for e in xrange(param['nepochs']):
        # shuffle
        shuffled_idx = shuffle(len(train_lex))
        param['ce'] = e
        tic = timeit.default_timer()
        trainbatch_x, trainbatch_y = get_batch(train_lex, train_y, shuffled_idx)
        print len(trainbatch_x)
        print trainbatch_x[0].shape
        print trainbatch_y[0].shape

        print shuffled_idx[:10]

        allcost = 0
        for i, (x, y) in enumerate(zip( trainbatch_x, trainbatch_y )):
            cost = rnn.train(x,  y, param['clr']) 
            if cost is not None:
                allcost += cost
            print '[learning] epoch %i >> %2.2f%%' % (
                e, (i + 1) * 100. / (nsentences/minibatch)),
            print 'completed in %.2f (sec) <<\r' % (timeit.default_timer() - tic),
            sys.stdout.flush()
        print "finish training, all cost = ", allcost

        # evaluation // back into the real world : idx -> words
        testbatch_x, testbatch_y = get_batch(test_lex, test_y)
        all_testbatchresult = []
        for (x, y) in zip( testbatch_x, testbatch_y):
	    if param['model'] == 'lstm_att':
                rawpred_y = rnn.classify(x, y) 
            else:
                rawpred_y = rnn.classify(x)
            all_testbatchresult += [rawpred_y.transpose()]
        all_testresult=numpy.concatenate(all_testbatchresult, axis=0)
        predictions_test = []
        for i in range(all_testresult.shape[0]):
            if pad_front_output:
                predictions_test += [[idx2label[z] for z in all_testresult[i][-len(groundtruth_test[i]):]] ]
            else:
                predictions_test += [[idx2label[z] for z in all_testresult[i][0:len(groundtruth_test[i])]] ]
        print predictions_test[2:5]

        validbatch_x, validbatch_y = get_batch(valid_lex, valid_y)
        all_validbatchresult = []
        for (x, y) in zip( validbatch_x, validbatch_y):
            if param['model'] == 'lstm_att':
                rawpred_y = rnn.classify(x, y) 
            else:
                rawpred_y = rnn.classify(x)
            all_validbatchresult += [rawpred_y.transpose()]
        all_validresult=numpy.concatenate(all_validbatchresult, axis=0)

        predictions_valid = []
        for i in range(all_validresult.shape[0]):
            if pad_front_output:
                predictions_valid += [[idx2label[z] for z in all_validresult[i][-len(groundtruth_valid[i]):]] ]
            else:
                predictions_valid += [[idx2label[z] for z in all_validresult[i][0:len(groundtruth_valid[i])]] ]
        print predictions_valid[2:5]


        max_ref = videodata.maxcap_persent 
        def dump4score(predictions, groundtruth, name):
            f_ref = open(name +'_ref.txt', 'w')
            f_hyp = open(name + '_hyp.txt', 'w')
            for (ref, hyp) in zip(groundtruth, predictions):
                f_ref.write(' '.join(ref)+'\n')
                f_hyp.write(' '.join(hyp)+'\n')
                f_ref.flush() ; f_hyp.flush()
            f_ref.close()
            f_hyp.close()
            try:
                refs, hypos = metrics.load_textfiles([open(name +'_ref'+'.txt', 'r'), open(name +'_ref'+'.txt', 'r')],
                                                open(name +'_hyp.txt', 'r'))
                final_scores = metrics.score(refs, hypos)
                print final_scores
                print " === finished dump and compute score ===="
            except e:
                print "error with compute scores ", e

        print "dump and compute the score now"
        try:
            dump4score(predictions_test, groundtruth_test, folder + '/current.test')
            dump4score(predictions_valid, groundtruth_valid, folder + '/current.valid')
        except e:
            print "Exceptino in dumping and bleu score computateion" , e
            continue

        continue
        if False:
            # evaluation // compute the accuracy using conlleval.pl
            res_test = conlleval(predictions_test,
                                 groundtruth_test,
                                 groundtruth_test,
                                 folder + '/current.test.txt',
                                 folder)

            res_valid = conlleval(predictions_valid,
                                  groundtruth_valid,
                                  groundtruth_valid,
                                  folder + '/current.valid.txt',
                                  folder)
            print res_valid
            print res_test

            if res_valid['f1'] > best_f1:

                if param['savemodel']:
                    rnn.save(folder)

                best_rnn = copy.deepcopy(rnn)
                best_f1 = res_valid['f1']

                if param['verbose']:
                    print('NEW BEST: epoch', e,
                          'valid F1', res_valid['f1'],
                          'best test F1', res_test['f1'])

                param['vf1'], param['tf1'] = res_valid['f1'], res_test['f1']
                param['vp'], param['tp'] = res_valid['p'], res_test['p']
                param['vr'], param['tr'] = res_valid['r'], res_test['r']
                param['be'] = e

                subprocess.call(['mv', folder + '/current.test.txt',
                                folder + '/best.test.txt'])
                subprocess.call(['mv', folder + '/current.valid.txt',
                                folder + '/best.valid.txt'])
            else:
                if param['verbose']:
                    print ''

        # finish f score computation ... no need here

        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.5
            rnn = best_rnn

        if param['clr'] < 1e-5:
            break

    print "FINSIH??" 
    exit(0)
    print('BEST RESULT: epoch', param['be'],
          'valid F1', param['vf1'],
          'best test F1', param['tf1'],
          'with the model', folder)


if __name__ == '__main__':
    main()
