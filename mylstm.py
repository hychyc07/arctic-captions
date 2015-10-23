from collections import OrderedDict
import os
import numpy as np
import theano
from theano import tensor as T
import optimizers
from theano import config
from theano import ProfileMode
profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())

import theano.sparse as Sparse
# define a lstm rnn
def generate_weight(dim1, dim2, weight_name, weight_scaler=0.2):
    return theano.shared(name=weight_name,
                         value=weight_scaler * np.random.uniform(-1.0, 1.0,(dim1, dim2))
                         .astype(config.floatX))

# define a lstm rnn
class LSTMSLU(object):
    def __init__(self, nh, nc, nf, mb):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        nf :: input feature size
        mb :: mini batch size
        '''
        # parameters of the model
        # first level : input to hidden bias
        self.wx_z = generate_weight(nf, nh, 'wx_z')

        # first level: recurrent : hidden to hidden state
        self.wh_z = generate_weight(nh, nh, 'wh_z')

        # first level: input to hidden bias
        self.bh_z = generate_weight(1, nh, 'bh_z')

        # first level : input to hidden bias
        self.wx_i = theano.shared(name='wx_i',
                                  value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                   (nf, nh))
                                  .astype(theano.config.floatX))

        # first level: recurrent : hidden to hidden state
        self.wh_i = theano.shared(name='wh_i',
                                  value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                   (nh, nh))
                                  .astype(theano.config.floatX))

        # first level: input to hidden bias
        self.bh_i = theano.shared(name='bh_i',
                                  value=np.zeros((1, nh),
                                                    dtype=theano.config.floatX))

        # first level : input to hidden bias
        self.wx_f = theano.shared(name='wx_f',
                                  value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                   (nf, nh))
                                  .astype(theano.config.floatX))

        # first level: recurrent : hidden to hidden state
        self.wh_f = theano.shared(name='wh_f',
                                  value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                   (nh, nh))
                                  .astype(theano.config.floatX))

        # first level: input to hidden bias
        self.bh_f = theano.shared(name='bh_f',
                                  value=np.zeros((1, nh),
                                                    dtype=theano.config.floatX))

        # first level : input to hidden bias
        self.wx_o = theano.shared(name='wx_o',
                                  value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                   (nf, nh))
                                  .astype(theano.config.floatX))

        # first level: recurrent : hidden to hidden state
        self.wh_o = theano.shared(name='wh_o',
                                  value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                   (nh, nh))
                                  .astype(theano.config.floatX))

        # first level: input to hidden bias
        self.bh_o = theano.shared(name='bh_o',
                                  value=np.zeros((1, nh),
                                                    dtype=theano.config.floatX))

        ## the peephole weights
        self.ph_o = theano.shared(name='ph_o',
                                  value=np.zeros((1, nh),
                                                    dtype=theano.config.floatX))
        self.ph_i = theano.shared(name='ph_i',
                                  value=np.zeros((1, nh),
                                                    dtype=theano.config.floatX))
        self.ph_f = theano.shared(name='ph_f',
                                  value=np.zeros((1, nh),
                                                    dtype=theano.config.floatX))

        # hidden layer value :
        self.h0 = theano.shared(name='h0',
                                value=np.zeros((mb, nh),
                                                  dtype=theano.config.floatX))
        # hidden layer value :
        self.c0 = theano.shared(name='c0',
                                value=np.zeros((mb, nh),
                                                  dtype=theano.config.floatX))
        ## LAST Level
        # last level:  hidden to output
        self.w = theano.shared(name='w',
                               value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                (nh, nc))
                               .astype(theano.config.floatX))

        # last level:  hidden to output
        self.b = theano.shared(name='b',
                               value=np.zeros((1, nc),
                                                 dtype=theano.config.floatX))

        self.I_mb = theano.shared(name='I',
                                value=np.ones((mb, 1),
                                                  dtype=theano.config.floatX))

        # bundle
        self.params = [self.wx_z, self.wx_f, self.wx_i, self.wx_o,
                       self.wh_z, self.wh_f, self.wh_i, self.wh_o,
                       self.bh_z, self.bh_f, self.bh_i, self.bh_o,
                       self.ph_i, self.ph_o, self.ph_f,
                       self.w, self.b]

        lr = T.scalar('lr')

        idxs = T.tensor3()     # input, since batched, dim rise to 3
        x = idxs.astype(theano.config.floatX)
        yinput = T.tensor3()   # labels
        y_sentence = yinput.astype(theano.config.floatX)
        #idxs = T.imatrix()
        #y_sentence = T.ivector()  # no batch version

        def recurrence(x_t, h_tm1, c_tm1):
            z_t = T.tanh(T.dot(x_t, self.wx_z) + T.dot(h_tm1, self.wh_z)
                              + T.dot(self.I_mb, self.bh_z))
            i_t = T.nnet.sigmoid(T.dot(x_t, self.wx_i) + T.dot(h_tm1, self.wh_i)
                                 + T.dot(self.I_mb, self.ph_i) * c_tm1
                                 + T.dot(self.I_mb, self.bh_i))
            f_t = T.nnet.sigmoid(T.dot(x_t, self.wx_f) + T.dot(h_tm1, self.wh_f)
                                 + T.dot(self.I_mb, self.ph_f) * c_tm1
                                 + T.dot(self.I_mb, self.bh_f))
            c_t = z_t * i_t + c_tm1 * f_t
            o_t = T.nnet.sigmoid(T.dot(x_t, self.wx_o) + T.dot(h_tm1, self.wh_o)
                                 + T.dot(self.I_mb, self.ph_o) * c_t
                                 + T.dot(self.I_mb, self.bh_o))
            h_t = T.tanh(c_t) * o_t

            s_t = T.nnet.softmax(T.dot(h_t, self.w) + T.dot(self.I_mb, self.b))

            '''no batch, raw math equations'''
            '''
            z_t = T.tanh(T.dot(x_t, self.wx_z) + T.dot(h_tm1, self.wh_z) + self.bh_z)
            i_t = T.nnet.sigmoid(T.dot(x_t, self.wx_i) + T.dot(h_tm1, self.wh_i) + self.bh_i + self.ph_i * c_tm1)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.wx_f) + T.dot(h_tm1, self.wh_f) + self.bh_f + self.ph_f * c_tm1)
            c_t = z_t * i_t + c_tm1 * f_t
            o_t = T.nnet.sigmoid(T.dot(x_t, self.wx_o) + T.dot(h_tm1, self.wh_o) + self.bh_o + self.ph_o * c_t)
            h_t = T.tanh(c_t) * o_t
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            '''
            return [h_t, c_t, s_t]

        [h, c, s], _ = theano.scan(fn=recurrence,
                                   sequences=x,
                                   outputs_info=[self.h0, self.c0, None],
                                   n_steps=x.shape[0])

        p_y_given_x_sentence = s[:, :, :]      # here size len x nc x mb
        y_pred = T.argmax(p_y_given_x_sentence, axis=2)
        #no batch:
        #p_y_given_x_sentence = s[:, 0, :]
        #y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        sentence_nll = -T.mean(T.log(T.nonzero_values(p_y_given_x_sentence * y_sentence))) * mb
        #no batch:
        #sentence_nll = -T.mean(T.log(p_y_given_x_sentence)[T.arange(x.shape[0]), y_sentence])

        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p - lr * g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)#, mode=profmode)
        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates)

         # by default it is sgd
        self.optm = optimizers.sgd
        self.f_grad_shared, self.f_update = self.optm(lr, dict(zip([s.name for s in self.params], self.params)),
                                                      sentence_gradients,x, y_sentence, sentence_nll)

        #self.optm(lr, self.params, sentence_gradients, x, y_sentence, sentence_nll)

    def train(self, x, y, learning_rate):
        #for (x_batch, y_batch) in train_batches:
            # here x_batch and y_batch are elements of train_batches and
            # therefore numpy arrays; function MSGD also updates the params
        #    print('Current loss is ', self.sentence_train(x_batch, y_batch, learning_rate))
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
        print "save" , self.params
        for param in self.params:
            np.savetxt(os.path.join(folder,
                                    'lstm_' + param.name + '.npy'), param.get_value(), fmt='%10.15f')

    def load(self, folder):
        for param in self.params:
            param.set_value(np.loadtxt(os.path.join(folder,
                                   'lstm_' + param.name + '.npy'), param.get_value(), fmt='%10.15f'))

