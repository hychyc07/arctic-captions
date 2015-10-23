from collections import OrderedDict
import os
import numpy
import theano
from theano import tensor as T
import optimizers
import theano.sparse as Sparse


# define an elman rnn
class ELMANSLU(object):
    ''' elman neural net model '''

    def __init__(self, nh, nc, nf, mb):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        nf :: number of features
        mb :: batch size : mini batch
        '''
        # parameters of the model
        self.wx = theano.shared(name='wx',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                                                 (nf, nh))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                                                 (nh, nh))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                                                (nh, nc))
                               .astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',
                                value=numpy.zeros((nh, 1),
                                                  dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros((nc, 1),
                                                 dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros((mb, nh),
                                                  dtype=theano.config.floatX))
        self.I_mb = theano.shared(name='I',
                                value=numpy.ones((mb, 1),
                                                  dtype=theano.config.floatX))

        # bundle
        self.params = [self.wx, self.wh, self.w,
                       self.bh, self.b]

        lr = T.scalar('lr')

        idxs = T.tensor3()     # input, since batched, dim rise to 3
        x = idxs.astype(theano.config.floatX)
        yinput = T.tensor3()   # labels
        y_sentence = yinput.astype(theano.config.floatX)
        # no batch:
        #idxs = T.imatrix()
        #y_sentence = T.ivector('y_sentence')  # labels

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx)
                                 + T.dot(h_tm1, self.wh) + T.dot(self.I_mb, self.bh.T)) #
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + T.dot(self.I_mb, self.b.T))  #
            # trying for the sparse version? //TODO: suppose to be much faster since both cost and grad are sparse
            ## Sparse.structured_dot(Sparse.csc_from_dense(x_t), self.wx)
            return [h_t, s_t]   # output is dimension len x (nc x 1) but s_t is of len x 1 x nc

        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0, None],
                                n_steps=x.shape[0])

        p_y_given_x_sentence = s[:, :, :]      # here size len x nc x mb
        y_pred = T.argmax(p_y_given_x_sentence, axis=2)
        # no batch:
        #p_y_given_x_sentence = s[:, 0, :]
        #y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        # y is matrix (nlabel , batch) now instead of pure vector
        # TODO: NEED TO FIGURE OUT PROPER WAY TO COMPUTE COST, NOW MEAN DOES NOT MAKE SENSE ....
        #sentence_nll = -T.mean(T.log(p_y_given_x_sentence) * y_sentence) * mb * 5
        sentence_nll = -T.mean(T.log(T.nonzero_values(p_y_given_x_sentence * y_sentence))) * mb
                    # sparse version?
                    #   T.mean(T.log(  T.nonzero_values(p_y_given_x_sentence * y_sentence)))
        # non-batch version:
        # sentence_nll = -T.mean(T.log(p_y_given_x_sentence)[T.arange(x.shape[0]), y_sentence])

        sentence_gradients = T.grad(sentence_nll, self.params)

        sentence_updates = OrderedDict((p, p - lr * g)
                                       for p, g in
                                         zip(self.params, sentence_gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
        # this is not going to be used .....
        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates)

        # by default it is sgd
        self.optm = optimizers.rmsprop
        self.f_grad_shared, self.f_update = self.optm(lr, dict(zip([s.name for s in self.params], self.params)),
                                                      sentence_gradients,x, y_sentence, sentence_nll)
                                                       #lr, tparams, grads, x, y, cost):

    def train(self, x, y, learning_rate):
        self.sentence_train(x, y, learning_rate)
        # self.normalize()

    def train_optimizer(self, x, y, learning_rate):
        cost = self.f_grad_shared(x, y)
        self.f_update(learning_rate)

    def set_optimizer(self, optmname):
        if optmname == 'sgd':
            self.optm = optimizers.sgd
        if optmname == 'adadelta':
            self.optm = optimizers.adadelta
        if optmname == 'rmsprop':
            self.optm = optimizers.rmsprop
        else:
            print 'Warning: optimizer not recognized, use sgd by default'
            self.optm = optimizers.sgd

    def save(self, folder):
        for param in self.params:
            numpy.savetxt(os.path.join(folder,
                                   'elman_'+ param.name + '.npy'), param.get_value(), fmt='%10.15f')

    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.loadtxt(os.path.join(folder,
                               'elman_'+ param.name + '.npy'), param.get_value(), fmt='%10.15f'))