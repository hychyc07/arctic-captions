import cPickle as pkl
import gzip
import os
import sys
import time
import operator
from sets import Set
import numpy as np

def prepare_data(caps, features, worddict, maxlen=64, n_words=10000, zero_pad=False):
    # x: a list of sentences
    seqs = []
    feat_list = []
    for cc in caps:
        seqs.append([worddict[w] if w in worddict and worddict[w] < n_words else 1 for w in cc[0].split()])
        feat_list.append(features[cc[1]])
    lengths = [len(s) for s in seqs]
    print "first 10 examples" , seqs[:10]

    if maxlen != None:
        new_seqs = []
        new_feat_list = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, feat_list):
            if l < maxlen:
                new_seqs.append(s)
                new_feat_list.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        feat_list = new_feat_list
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = np.max(lengths)+1

    feat_array = np.array(feat_list)
    print "data feat ", y.shape
   # y = np.zeros((len(feat_list), feat_list[0].shape[0], feat_list[0].shape[1])).astype('float32')
   # for idx, ff in enumerate(feat_list):
   #     y[idx,:] = np.array(ff.todense())
   # y = y.reshape([y.shape[0], maxlen, 4096 ])  #14*14, 512])
    if zero_pad:
        feat_array_pad = np.zeros((feat_array.shape[0], feat_array.shape[1]+1, feat_array.shape[2])).astype('float32')
        feat_array_pad[:,:-1,:] = feat_array
        feat_array = feat_array_pad

    capture = np.zeros((n_samples, maxlen)).astype('float32') #.astype('int64')
    capture_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, s in enumerate(seqs):
        capture[idx, :lengths[idx]] = s
        capture_mask[idx, :lengths[idx]+1] = 1.
    print "data cap ", capture.shape, capture[0, :]
    return capture, capture_mask, feat_array

maxcap_persent = 1
def load_data(load_train=True, load_dev=True, load_test=True, path='../../MANIAC/'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    max_videolen = 64

    training_size = 1200
    testing_size = 500

    allfile = open(path + "filelist.txt", 'r')
    allfeat = open(path + "Feats1d.txt" , 'r')

    capfile = open(path + "msrvdc_eng.csv", 'r')

    vectorlen = 0
    capfeat = {}
    for picstr, picfeat in zip(allfile, allfeat):
        picname_str = picstr[picstr.rindex('/')+1: picstr.rindex('.')]
        picname = picname_str[:picname_str.index('.')]
        picno = picname_str[picname_str.rindex('_')+1: ]
        picfeat = np.array( [float(x) for x in picfeat.replace('\n','').split(' ')])
        vectorlen = picfeat.shape[0]
        if picname in capfeat:
            capfeat[picname] += [(int(picno), picfeat)]
        else: 
            capfeat[picname] = [(int(picno), picfeat)]
    overalldata =  len(capfeat.keys())
    print "overall datasize" , overalldata
    capfeat = [(name, sorted(capfeat[name], key = operator.itemgetter(0))) for name in capfeat.keys()]


    def padding_feat(list_of_vect):
        a = np.zeros((max_videolen, vectorlen))
        if len(list_of_vect)< max_videolen:
            a[max_videolen-len(list_of_vect):, :] = np.array(list_of_vect)
        else:
            a = np.array(list_of_vect[:max_videolen])
        return a

    capfeat = [(s[0], padding_feat([x[1] for x in s[1]])) for s in capfeat]
    capfeat = dict(capfeat)
    print "double check shape align", [s.shape for s in capfeat.values()[:10]  ]

    cap_infolist = []
    cap_info = {}
    cap_vocab = {}
    cap_vocab2gram = {}
    cap_vocab2gram['<eos>'] = []
    for line in capfile:
        capline = line.replace('\n','').split(',')
        captitle = '_'.join(capline[0:3])  #'_'.join(capline[0:2])
        capture = capline[-1].replace('\r','').replace('"', '').replace('.',' .').lower()
        cap_words = capture.split(' ')
        for w_idx in range(len(cap_words)):
            words = cap_words[w_idx]
            if words in cap_vocab:
                cap_vocab[words] = 1 + cap_vocab[words]
            else:
                cap_vocab[words] = 1

            if w_idx < len(cap_words)-1:
                if w_idx == 0:
                    cap_vocab2gram['<eos>'] += [words]
                if words in cap_vocab2gram:
                    cap_vocab2gram[words] += [cap_words[w_idx+1]]
                else:
                    cap_vocab2gram[words] = [cap_words[w_idx+1]]
            else:
                if words not in cap_vocab2gram:
                    cap_vocab2gram[words] = []

        if captitle in cap_info: 
            cap_info[captitle] += [capture]
        else:
            cap_info[captitle] = [capture]

    sorted_cap_vocab = sorted(cap_vocab.items(), key=operator.itemgetter(1))
    sorted_cap_vocab.reverse()
    vocab_index = dict([(s[1][0], s[0]+2) for s in enumerate(sorted_cap_vocab)])
    print 'vocab size' , len(vocab_index), vocab_index.items()[:10]

    cap_vocab2gram = [(a, set(cap_vocab2gram[a]) ) for a in cap_vocab2gram.keys()]
    print "2gram example" , cap_vocab2gram[:10]
    vocab_restriction = dict(cap_vocab2gram)

    def filter_dataset(start_index, end_index):
        cap_infolist_set = []
        capfeat_set = []
        for captitle in capfeat.keys()[start_index: end_index]:
            cap_infolist_set += [(cap, captitle) for cap in cap_info[captitle][:maxcap_persent] ]
            capfeat_set += [(captitle, capfeat[captitle])]
        capfeat_set = dict(capfeat_set)
        print len(cap_infolist_set)
        print len(capfeat_set)
        return cap_infolist_set, capfeat_set

    alltrain = filter_dataset(0, training_size)
    alltest = filter_dataset(training_size, training_size+testing_size)
    allvalid = filter_dataset(training_size+testing_size, None)

    return alltrain, alltest, allvalid, vocab_index, vocab_restriction


if __name__ == '__main__':
    alltrain, allvalid, alltest, sorted_cap_vocab, vocab_restriction = load_data()
    prepare_data(alltrain[0], alltrain[1], sorted_cap_vocab)