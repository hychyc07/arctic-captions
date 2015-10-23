from collections import OrderedDict
import os
import numpy as np
import theano
from theano import config
from theano import tensor as T
import optimizers
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# define a lstm rnn
def generate_weight(dim1, dim2, weight_name, weight_scaler=0.2):
    return theano.shared(name=weight_name,
                         value=weight_scaler * np.random.uniform(-1.0, 1.0,(dim1, dim2))
                         .astype(config.floatX))

SEED=123
def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def dropout(state_before, use_noise, trng):
    result = T.switch(use_noise, (state_before * trng.binomial(state_before.shape, p=0.5, n=1,
            dtype=state_before.dtype)), state_before * 0.5)
    return result

class LSTM_att(object):
    def __init__(self, nh_enc, nh_dec, nh_att, nx, ny, mb, lt, bidir, nonlstm_encode=False, restriction=None):
        '''
        nh_enc :: dimension of the hidden layer of encoder
        nh_dec :: dimension of the hidden layer of decoder
        nh_att :: dimension of the hidden layer of attention
        ny :: number of classes
        nx :: input feature size
        mb :: mini batch size
        lt :: length of input, after padding .. for attention
        bidir:: bidirection or not ... 2 is bidirection, 1 is single ...
        '''
        self.nh_enc = nh_enc
        self.nh_dec = nh_dec
        self.nh_att = nh_att
        self.nx = nx
        self.ny = ny
        self.lt = lt
        self.bidir = bidir

        # parameters of the model
        xhdim = nx+nh_enc*bidir
        # encoder forward
        # 1 level : input to hidden bias, *4 below is since we compressed the W, H, b computation
        self.Wf_enc_z = generate_weight(nx, nh_enc, "Wf_enc_z")
        self.Wf_enc_i = generate_weight(nx, nh_enc, "Wf_enc_i")
        self.Wf_enc_f = generate_weight(nx, nh_enc, "Wf_enc_f")
        self.Wf_enc_o = generate_weight(nx, nh_enc, "Wf_enc_o")

        self.Hf_enc_z = generate_weight(nh_enc, nh_enc, "Hf_enc_z")
        self.Hf_enc_i = generate_weight(nh_enc, nh_enc, "Hf_enc_i")
        self.Hf_enc_f = generate_weight(nh_enc, nh_enc, "Hf_enc_f")
        self.Hf_enc_o = generate_weight(nh_enc, nh_enc, "Hf_enc_o")

        self.bf_enc_z = generate_weight(1, nh_enc, "bf_enc_z")
        self.bf_enc_i = generate_weight(1, nh_enc, "bf_enc_i")
        self.bf_enc_f = generate_weight(1, nh_enc, "bf_enc_f")
        self.bf_enc_o = generate_weight(1, nh_enc, "bf_enc_o")

        # encoder backward:
        self.Wb_enc_z = generate_weight(nx, nh_enc, "Wb_enc_z")
        self.Wb_enc_i = generate_weight(nx, nh_enc, "Wb_enc_i")
        self.Wb_enc_f = generate_weight(nx, nh_enc, "Wb_enc_f")
        self.Wb_enc_o = generate_weight(nx, nh_enc, "Wb_enc_o")

        self.Hb_enc_z = generate_weight(nh_enc, nh_enc, "Hb_enc_z")
        self.Hb_enc_i = generate_weight(nh_enc, nh_enc, "Hb_enc_i")
        self.Hb_enc_f = generate_weight(nh_enc, nh_enc, "Hb_enc_f")
        self.Hb_enc_o = generate_weight(nh_enc, nh_enc, "Hb_enc_o")

        self.bb_enc_z = generate_weight(1, nh_enc, "bb_enc_z")
        self.bb_enc_i = generate_weight(1, nh_enc, "bb_enc_i")
        self.bb_enc_f = generate_weight(1, nh_enc, "bb_enc_f")
        self.bb_enc_o = generate_weight(1, nh_enc, "bb_enc_o")

        ## attention level:
        self.UV_att = generate_weight(xhdim, nh_att, "UV_att")
        self.W_att = generate_weight(nh_dec, nh_att, "W_att")
        self.v_att = generate_weight(nh_att, 1, "v_att")

        # decoder level : input to hidden bias
        self.W_dec_z = generate_weight(xhdim, nh_dec, "W_dec_z")
        self.W_dec_i = generate_weight(xhdim, nh_dec, "W_dec_i")
        self.W_dec_f = generate_weight(xhdim, nh_dec, "W_dec_f")
        self.W_dec_o = generate_weight(xhdim, nh_dec, "W_dec_o")

        self.H_dec_z = generate_weight(nh_dec, nh_dec, "H_dec_z")
        self.H_dec_i = generate_weight(nh_dec, nh_dec, "H_dec_i")
        self.H_dec_f = generate_weight(nh_dec, nh_dec, "H_dec_f")
        self.H_dec_o = generate_weight(nh_dec, nh_dec, "H_dec_o")

        self.b_dec_z = generate_weight(1, nh_dec, "b_dec_z")
        self.b_dec_i = generate_weight(1, nh_dec, "b_dec_i")
        self.b_dec_f = generate_weight(1, nh_dec, "b_dec_f")
        self.b_dec_o = generate_weight(1, nh_dec, "b_dec_o")

            # e is extra in decoder, for previous outut
        self.E_dec_z = generate_weight(ny, nh_dec, "E_dec_z")
        self.E_dec_i = generate_weight(ny, nh_dec, "E_dec_i")
        self.E_dec_f = generate_weight(ny, nh_dec, "E_dec_f")
        self.E_dec_o = generate_weight(ny, nh_dec, "E_dec_o")

        ## LAST Level
        # last level:  hidden to output
        self.W_y = generate_weight(xhdim, ny, "W_y")
        self.H_y = generate_weight(nh_dec, ny, "H_y")
        self.E_y = generate_weight(ny, ny, "E_y")
        self.b_y = generate_weight(1, ny, "b_y", 0.0)

        ## INTERMEDIATE value
        hf0 = theano.shared(name='hf0', value=np.zeros((mb, nh_enc), dtype=config.floatX)) # forward
        cf0 = theano.shared(name='cf0', value=np.zeros((mb, nh_enc), dtype=config.floatX))
        hb0 = theano.shared(name='hb0', value=np.zeros((mb, nh_enc), dtype=config.floatX)) # backward
        cb0 = theano.shared(name='cb0', value=np.zeros((mb, nh_enc), dtype=config.floatX))

        sd0 = theano.shared(name='sd0', value=np.zeros((mb, nh_dec), dtype=config.floatX))
        cd0 = theano.shared(name='cd0', value=np.zeros((mb, nh_dec), dtype=config.floatX))
        a= np.zeros((1, mb, ny), dtype=config.floatX)
        a[:,:,0] =1
        y0 = theano.shared(name='y0', value=a)


        # all one vector for batch size ... , deprecated, should be matching automatically
        I_mb = theano.shared(name='I', value=np.ones((mb, 1), dtype=config.floatX))

        WHb_f_enc = [self.Wf_enc_z, self.Hf_enc_z, self.bf_enc_z,
                       self.Wf_enc_i, self.Hf_enc_i, self.bf_enc_i,
                       self.Wf_enc_f, self.Hf_enc_f, self.bf_enc_f,
                       self.Wf_enc_o, self.Hf_enc_o, self.bf_enc_o]
        WHb_b_enc = [self.Wb_enc_z, self.Hb_enc_z, self.bb_enc_z,
                       self.Wb_enc_i, self.Hb_enc_i, self.bb_enc_i,
                       self.Wb_enc_f, self.Hb_enc_f, self.bb_enc_f,
                       self.Wb_enc_o, self.Hb_enc_o, self.bb_enc_o]
        WHEb_dec =  [self.W_dec_z, self.E_dec_z, self.H_dec_z, self.b_dec_z,
                       self.W_dec_i, self.E_dec_i, self.H_dec_i, self.b_dec_i,
                       self.W_dec_f, self.E_dec_f, self.H_dec_f, self.b_dec_f,
                       self.W_dec_o, self.E_dec_o, self.H_dec_o, self.b_dec_o]

        Wb_nonlstm_enc = [self.Wf_enc_z, self.bf_enc_z]

        # bundle, todo: note we removed peephole from definition ...
        self.params = [self.UV_att, self.W_att, self.v_att,
                       self.W_y, self.H_y, self.b_y, self.E_y] + WHEb_dec

        if not nonlstm_encode:
            self.params += WHb_f_enc
            if bidir == 2:
                self.params += WHb_b_enc
        else:
            self.params += Wb_nonlstm_enc  ## special case, to test image capture, just twist the encode to be nonlstm

        # Used for dropout.
        trng = RandomStreams(SEED)
        use_noise = theano.shared(numpy_floatX(0.))

        # input parameter defined ....
        x_in = T.tensor3()     # input, since batched, dim rise to 3 : len * mb * nx
        x = x_in.astype(config.floatX)

        y_in = T.tensor3()   # ground truth labels ,  len * mb * ny
        y_target = y_in.astype(config.floatX)

        y_decinput = T.concatenate([y0, y_target], axis=0)[:-1, :,:].astype(config.floatX)   # decode input labels, shifted to right by one and start with eos


        lr = T.scalar('lr')

        def encode(x_t, h_tm1, c_tm1, W_enc_z, H_enc_z, b_enc_z, W_enc_i, H_enc_i, b_enc_i,
                                        W_enc_f, H_enc_f, b_enc_f, W_enc_o, H_enc_o, b_enc_o):
            g_t = T.tanh(T.dot(x_t, W_enc_z) + T.dot(h_tm1, H_enc_z) + T.dot(I_mb, b_enc_z))
            i_t = T.nnet.sigmoid(T.dot(x_t, W_enc_i) + T.dot(h_tm1, H_enc_i) + T.dot(I_mb, b_enc_i) ) # + T.dot(I_mb, ph_i.T) * c_tm1)
            f_t = T.nnet.sigmoid(T.dot(x_t, W_enc_f) + T.dot(h_tm1, H_enc_f) + T.dot(I_mb, b_enc_f) ) # + T.dot(I_mb, ph_f.T) * c_tm1
            c_t = g_t * i_t + c_tm1 * f_t
            o_t = T.nnet.sigmoid(T.dot(x_t, W_enc_o) + T.dot(h_tm1, H_enc_o) + T.dot(I_mb, b_enc_o) ) # + T.dot(I_mb, ph_o.T) * c_t
            h_t = T.tanh(c_t) * o_t
            return [h_t, c_t]

        def relu(x):
            return theano.tensor.switch(x<0, 0, x)

        if nonlstm_encode:
            hf = relu(T.dot(x, self.Wf_enc_z) + T.dot(I_mb, self.bf_enc_z))  # len * mb * nx
            xh = T.concatenate([x, hf], axis=2)  # since dim0 is the length of input,  so it is of len * batch * xh_dim
        else:
            [hf, cf], _ = theano.scan(fn=encode, sequences=x, outputs_info=[hf0, cf0],
                                  non_sequences=WHb_f_enc,
                                  n_steps=x.shape[0])
            xh = T.concatenate([x, hf], axis=2)  # since dim0 is the length of input,  so it is of len * batch * xh_dim

            if bidir == 2:
                [hb, cb], _ = theano.scan(fn=encode, sequences=x, outputs_info=[hb0, cb0],
                                      non_sequences=WHb_b_enc, go_backwards=True)

                xh = T.concatenate([x, hf, hb[::-1]], axis=2)  #same as above
                # note: scan is in input backward fashion, but output corresponding to an inverted order, thus use [::-1] to reverse it.

        # attention prepare, since the same across all place
        UVxh = T.dot(xh, self.UV_att)  #.dimshuffle(1, 0)  actually it does not matter (then shuffle dim by switch 1st and 2nd dim)
        # dim z=x+h, then dot of len*mb*z  and z*a=  len*mb*a
        if restriction is not None:
            restriction_matrix = theano.shared(name="restriction", value=restriction).astype(config.floatX)

        def stable_softmax(yin):
            e_yin = np.exp(yin - yin.max(axis=1, keepdims=True))
            return e_yin / e_yin.sum(axis=1, keepdims=True)

        def stable_softmax_nonzero(yin, zerosout):
            e_yin = np.exp(yin - yin.max(axis=1, keepdims=True)) #
            return e_yin / e_yin.sum(axis=1, keepdims=True) * zerosout
            #return T.nnet.softmax(yin - yin.max(axis=1, keepdims=True)) * zerosout

        def decode(y_tm1, sd_tm1, cd_tm1, xh, UVxh):
            beta_st = T.dot(sd_tm1, self.W_att) + UVxh  # note, dimension mismatch is fine, a*mb +  len * a * mb
            beta_t = T.dot(beta_st, self.v_att)   #1*len*mb      v_att is (a*1)  => len * mb * 1
            alpha_t = stable_softmax(beta_t.dimshuffle(1,0,2))  # .dimshuffle(1,0,)).dimshuffle(0, 1, 'x')
            z_t = T.batched_dot(xh.dimshuffle(1, 2, 0), alpha_t).flatten(2)   #dimshuffle(0,1,)
            # old version, depends on int of size
            #alpha_t = T.nnet.softmax(beta_t.reshape((mb, lt))).reshape((mb, lt, 1))  # compress to 2d for softmax then elivated to 3d
            #z_t = T.batched_dot(xh.dimshuffle(1, 2, 0), alpha_t).reshape((mb, xhdim))  # after it is mb * nxh * 1
            g_t = T.tanh(T.dot(z_t, self.W_dec_z) + T.dot(sd_tm1, self.H_dec_z) + T.dot(I_mb, self.b_dec_z)
                         + T.dot(y_tm1, self.E_dec_z))
            i_t = T.nnet.sigmoid(T.dot(z_t, self.W_dec_i) + T.dot(sd_tm1, self.H_dec_i) + T.dot(I_mb, self.b_dec_i)
                         + T.dot(y_tm1, self.E_dec_i))    # + T.dot(I_mb, ph_i.T) * c_tm1)
            f_t = T.nnet.sigmoid(T.dot(z_t, self.W_dec_f) + T.dot(sd_tm1, self.H_dec_f) + T.dot(I_mb, self.b_dec_f)
                         + T.dot(y_tm1, self.E_dec_f))   #+ T.dot(I_mb, ph_f.T) * c_tm1
            cd_t = g_t * i_t + cd_tm1 * f_t
            o_t = T.nnet.sigmoid(T.dot(z_t, self.W_dec_o) + T.dot(sd_tm1, self.H_dec_o) + T.dot(I_mb, self.b_dec_o)
                         + T.dot(y_tm1, self.E_dec_o))   # + T.dot(I_mb, ph_o.T) * c_t
            sd_t = T.tanh(cd_t) * o_t

            #sd_t = dropout(sd_t, use_noise, trng)
            if restriction is None:
                y_t = stable_softmax( ( T.dot(z_t, self.W_y) + T.dot(sd_t, self.H_y)
                    + T.dot(y_tm1, self.E_y) + T.dot(I_mb, self.b_y) ) )
            else:
                restriction_perbatch = restriction_matrix[T.argmax(y_tm1, axis=1)]
                y_t = stable_softmax_nonzero( (T.dot(z_t, self.W_y) + T.dot(sd_t, self.H_y)
                    + T.dot(y_tm1, self.E_y) + T.dot(I_mb, self.b_y)) , restriction_perbatch)

            return [sd_t, cd_t, y_t]

        [sd_dec, cd_dec, y_dec], _ = theano.scan(fn=decode,
                                   sequences=y_decinput, # dict(input=y_decinput, taps=[0]),
                                   outputs_info=[dict(initial=sd0, taps=[-1]), dict(initial=cd0, taps=[-1]), None ], #, dict(initial=y0, taps=[-1])],
                                   non_sequences=[xh, UVxh],
                                   n_steps=y_decinput.shape[0])

        p_y_given_x_sentence = y_dec[:, :, :]      # here size len x ny x mb
        y_pred = T.argmax(p_y_given_x_sentence, axis=2)

        # cost and gradients and learning rate
        sentence_cost = -T.mean(T.log(T.nonzero_values(p_y_given_x_sentence * y_target[:,:,:]) + np.float32(1e-8)))

        sentence_gradients = T.grad(sentence_cost, self.params)
        sentence_updates = OrderedDict((p, p - lr * g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[x_in, y_target], outputs=y_pred)
        self.sentence_train = theano.function(inputs=[x_in, y_target, lr],
                                              outputs=sentence_cost,
                                              updates=sentence_updates)

        self.only_encode = theano.function(inputs=[x_in], outputs=[xh, UVxh])
        self.only_decode_step = decode

        # by default it is sgd
        self.optm = optimizers.sgd
        self.f_grad_shared, self.f_update = self.optm(lr, dict(zip([s.name for s in self.params], self.params)),
                                                      sentence_gradients, x, y_target, sentence_cost)

    def train(self, x, y, learning_rate):
        #for (x_batch, y_batch) in train_batches:
            # here x_batch and y_batch are elements of train_batches and
            # therefore numpy arrays; function MSGD also updates the params
        #    print('Current loss is ', self.sentence_train(x_batch, y_batch, learning_rate))
        cost = self.sentence_train(x, y, learning_rate)
        return cost
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
            np.savetxt(os.path.join(folder,
                                    'lstm_' + param.name + '.npy'), param.get_value(), fmt='%10.15f')

    def load(self, folder):
        for param in self.params:
            param.set_value(np.loadtxt(os.path.join(folder,
                                   'lstm_' + param.name + '.npy'), param.get_value(), fmt='%10.15f'))

    def beamsearch(self, x, beam_size=1, max_search_len=20):
        xh, UVxh = self.only_encode(x)
        y_prediction = []
        for i in range(x.shape[1]):
            predictions = self.beamsearch_decode(xh, UVxh, i, beam_size, max_search_len)
            y_prediction += [predictions]
        print y_prediction # this is the beam version for all in the minibatch


    def beamsearch_decode(self, xh, UVxh, index, beam_size, max_search_len):
        h =np.zeros((1, self.nh_dec))
        c = np.zeros((1))
        Ws = []
        if beam_size > 1:
            # log probability, indices of words predicted in this beam so far, and the hidden and cell states
            beams = [(0.0, [], h, c)] 
            nsteps = 0
            while True:
                beam_candidates = []
                for b in beams:
                    ixprev = b[1][-1] if b[1] else 0 # start off with the word where this beam left off
                    if ixprev == 0 and b[1]:
                        # this beam predicted end token. Keep in the candidates but don't expand it out any more
                        beam_candidates.append(b)
                        continue
                    h1, c1, y1 = self.only_decode_step(Ws[ixprev], b[2], b[3], xh, UVxh) # y1 is already the softmax value of y
                    # decode(y_tm1, sd_tm1, cd_tm1, xh, UVxh):return [sd_t, cd_t, y_t]
                    #LSTMtick(x, h_prev, c_prev), return (Y, Hout, C) # return output, new hidden, new cell

                    y1 = y1.ravel() # make into 1D vector
                    maxy1 = np.amax(y1)
                    top_indices = np.argsort(-y1)  # we do -y because we want decreasing order
                    for i in xrange(beam_size):
                        wordix = top_indices[i]
                        beam_candidates.append((b[0] + y1[wordix], b[1] + [wordix], h1, c1))
                beam_candidates.sort(reverse = True) # decreasing order
                beams = beam_candidates[:beam_size] # truncate to get new beams
                nsteps += 1
                if nsteps >= max_search_len: # bad things are probably happening, break out
                    break
                # strip the intermediates
            predictions = [(b[0], b[1]) for b in beams]
        else:
          # greedy inference. lets write it up independently, should be bit faster and simpler
            ixprev = 0
            nsteps = 0
            predix = []
            predlogprob = 0.0
            while True:
                #(y1, h, c) = LSTMtick(Ws[ixprev], h, c)
                h, c, y1 = self.only_decode_step(Ws[ixprev], h, c, xh, UVxh)
                ixprev = np.amax(y1)
                ixlogprob = y1[ixprev]
                predix.append(ixprev)
                predlogprob += ixlogprob
                nsteps += 1
                if ixprev == 0 or nsteps >= max_search_len:
                    break
            predictions = [(predlogprob, predix)]

        return predictions


def sanitycheck():
    inputsize =4096
    outputsize= 8000
    minibatch = 8

    maxlen_input = 40
    bidirection = False
    restriction_matrix = None

    learning_rate = 0.2

    rnn = LSTM_att(nh_enc=100, nh_dec=100, nh_att=50,
                    nx=inputsize, ny=outputsize, mb=minibatch,
                    lt=maxlen_input, bidir=bidirection+1, nonlstm_encode=False,
                    restriction = restriction_matrix)

    batchinput_x = np.random.random((maxlen_input, minibatch, inputsize)).astype(dtype=np.float32)
    batchinput_y = np.ones((20, minibatch, outputsize)).astype(dtype=np.float32)

    loss = rnn.train(batchinput_x, batchinput_y, learning_rate )
    print loss


if __name__ == '__main__':
    sanitycheck()  # dimension check
    print "finished"
