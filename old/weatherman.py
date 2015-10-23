# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:58:08 2015

same as wm65.py in model1

@author: hongyuan
"""

import pickle
import time
import numpy as np
import theano
from theano import sandbox
import theano.tensor as T
import os
import scipy.io
from collections import defaultdict
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from utils import *
import jpype
import theano.typed_list


dtype=theano.config.floatX

#TODO: hyper-params
dropoutvalue = np.float32(1.0)
nmodel = 500
batchsize = 100

max_epoch = 50
beam = 1
maxdif = 5

#TODO: get the data
datafolder = '../../data/'
#dataforsharefolder = './'

traindata, devdata = getdata(datafolder)
#traindata = traindata[:1000]
#devdata = devdata[:50]
word2ind, ind2word, vocabmat = getvocab(datafolder)
reflist = getref(devdata)
#reflist = reflist[:50]
#
trainsize = len(traindata)
max_step = int(trainsize / batchsize)
print "finish reading data"
### 
#'''
allXcube = []
allXcubeback = []
allScube = []
allTargetcube = []
#
for step in range(max_step):
    batchdata = []
    for j in range(batchsize):
        batchdata.append(traindata[step*batchsize+j])
    Xcube, Scube, Targetcube = \
    getbatchdata(batchdata, vocabmat, word2ind, ind2word)
    allXcube.append(np.copy(Xcube))
    allXcubeback.append(np.copy(Xcube[::-1,:,:]))
    allScube.append(np.copy(Scube))
    allTargetcube.append(np.copy(Targetcube))
#'''
print "finish putting data into GPU"

###
###
compilestart = time.time()

#TODO: some preparation
#sigma = lambda x: 1 / (1 + T.exp(-x))
#act = T.tanh
#inputbias = np.float32(-0.0)
#outputbias = np.float32(-0.0)
#forgetbias = np.float32(0.0)
nen = nmodel # size of LSTM encoder
nde = nmodel # size of LSTM decoder
nbeta = nmodel
ny = len(vocabmat[:,0])
D = len(traindata[0]['info'][0,:])

print "finish preparation"

#TODO: Theano dropout
srng = RandomStreams(seed=0)
windows = srng.uniform((batchsize,nde)) < dropoutvalue
getwins = theano.function([],windows)

#TODO: definie encoder
def encoder(infomatf, infomatb, htm1matf, ctm1matf, htm1matb, ctm1matb, 
            Eenf, Eenb, Wenf, Wenb, benf, benb):
    # infomat is a matrix, having # batch * D
    dim = Eenf.shape[1] 
    #
    xtmatf = theano.dot(infomatf, Eenf)
    xtmatb = theano.dot(infomatb, Eenb)
    #
    pretranf = T.concatenate([xtmatf, htm1matf], axis=1)
    pretranb = T.concatenate([xtmatb, htm1matb], axis=1)
    #
    posttranf = theano.dot(pretranf, Wenf) + benf
    posttranb = theano.dot(pretranb, Wenb) + benb
    #
    itmatf = T.nnet.sigmoid(posttranf[:, 0:dim])
    ftmatf = T.nnet.sigmoid(posttranf[:, dim:(2*dim)])
    gtmatf = T.tanh(posttranf[:, (2*dim):(3*dim)])
    otmatf = T.nnet.sigmoid(posttranf[:, (3*dim):])
    ctmatf = ftmatf * ctm1matf + itmatf * gtmatf
    #
    htmatf = otmatf * T.tanh(ctmatf)
    #
    itmatb = T.nnet.sigmoid(posttranb[:, 0:dim])
    ftmatb = T.nnet.sigmoid(posttranb[:, dim:(2*dim)])
    gtmatb = T.tanh(posttranb[:, (2*dim):(3*dim)])
    otmatb = T.nnet.sigmoid(posttranb[:, (3*dim):])
    ctmatb = ftmatb * ctm1matb + itmatb * gtmatb
    #
    htmatb = otmatb * T.tanh(ctmatb)
    #
    return htmatf, ctmatf, htmatb, ctmatb
#
Eenf = theano.shared(sample_weights(D,nen),name='Eenf')
Eenb = theano.shared(sample_weights(D,nen),name='Eenb')
Wenf = theano.shared(sample_weights(2*nen, 4*nen), name='Wenf')
Wenb = theano.shared(sample_weights(2*nen, 4*nen), name='Wenb')
benf = theano.shared(np.zeros((4*nen,),dtype=dtype), name='benf')
benb = theano.shared(np.zeros((4*nen,),dtype=dtype), name='benb')

c0matenf = theano.shared(np.zeros((batchsize,nen), dtype=dtype), name = 'c0matenf')
h0matenf = theano.shared(np.zeros((batchsize,nen), dtype=dtype), name = 'h0matenf')
c0matenb = theano.shared(np.zeros((batchsize,nen), dtype=dtype), name = 'c0matenb')
h0matenb = theano.shared(np.zeros((batchsize,nen), dtype=dtype), name = 'h0matenb')
#
#
Xf = T.tensor3(dtype=dtype, name = 'Xf' )
Xb = T.tensor3(dtype=dtype, name = 'Xb' )
# X : # of lines (36) * # batch * D
[hcub_enf, ccub_enf, hcub_enb0, ccub_enb0], _ = \
theano.scan(fn = encoder,
            sequences = [dict(input=Xf,taps=[0]), dict(input=Xb,taps=[0])], 
            outputs_info = [dict(initial=h0matenf,taps=[-1]), dict(initial=c0matenf,taps=[-1]),
                            dict(initial=h0matenb, taps=[-1]), dict(initial=c0matenb, taps=[-1])],
            non_sequences = [Eenf, Eenb, Wenf, Wenb, benf, benb])
#

newH = T.concatenate( [Xf, hcub_enf, hcub_enb0[::-1,:,:]] ,axis=2)
#|--> # lines(36) * batchsize * (D+2nen)
print "finish decode"

#TODO: get the selection
Pbeta = theano.shared(sample_weights(D+nen+nen,nbeta), name='Pbeta')
qbeta = theano.shared(np.zeros((nbeta,),dtype=dtype), name='qbeta')
#
Beta = T.tanh( T.tensordot(newH, Pbeta,(2,0)) )#  |->  # lines * # batch * nbeta
Alpha = T.nnet.sigmoid(T.transpose(T.tensordot(Beta, qbeta, (2,0)),axes=(1,0)))
# |--> batchsize * lines
#newH1 = T.transpose(newH, axes=(1,0,2)) #|-> # batch * lines * (D+2nen)
#ztmat = T.sum((Alpha[:,:,None] * newH1), axis=1)
#

#TODO: define the decoder
def decoder(wordmat, stm1mat, cstm1mat, 
            newH, Alpha, 
            Wbeta, Ubeta, vbeta, Ede, Wde, bde, L0, L):
    #newH : # lines * # batch * (D+nen+nen)
    # wordmat : # batch * ny
    xtmat = theano.dot(wordmat, Ede)
    #
    xtmat = theano.dot(wordmat, Ede)
    beta1 = T.tensordot(newH,Ubeta,(2,0))
    beta2 = theano.dot(stm1mat, Wbeta)
    beta3 = T.tanh( beta1 + beta2 )
    beta4 = T.tensordot(beta3,vbeta,(2,0)) #  |->  # lines * # batch
    pre_alphamat = T.nnet.softmax(T.transpose(beta4, axes=(1,0)))
    #
    pre_alphamat2 = pre_alphamat * Alpha
    alphamat = pre_alphamat2 / pre_alphamat2.sum(axis=1, keepdims=True)
    #
    newH1 = T.transpose(newH, axes=(1,0,2))
    ztmat = T.sum((alphamat[:,:,None]*newH1),axis=1)
    # here we get the beta : beta = dot( act(dot(h,U)+dot(stm1,W)) , vbeta)
    # start decoding
    pretran = T.concatenate([xtmat, stm1mat, ztmat], axis=1)
    dim = Ede.shape[1]
    posttran = theano.dot(pretran, Wde) + bde
    #
    itmat = T.nnet.sigmoid(posttran[:, :dim])
    ftmat = T.nnet.sigmoid(posttran[:, dim:(2*dim)])
    gtmat = T.tanh(posttran[:, (2*dim):(3*dim)])
    otmat = T.nnet.sigmoid(posttran[:, (3*dim):])
    cstmat = ftmat * cstm1mat + itmat * gtmat
    stmat = otmat * T.tanh(cstmat)
    #
    winst = getwins()
    stfory = stmat * winst
    #
    fory = T.concatenate([stfory, ztmat],axis=1)
    yt0 = theano.dot((xtmat + theano.dot(fory, L)), L0) 
    # |-> # batch * ny
    ytmat = T.nnet.softmax(yt0)
    logytmat = T.log(ytmat + np.float32(1e-8) )
    #
    return stmat, cstmat, ytmat, logytmat

#
#
Wbeta = theano.shared(sample_weights(nde,nbeta), name='Wbeta')
Ubeta = theano.shared(sample_weights(D+nen+nen,nbeta), name='Ubeta')
vbeta = theano.shared(np.zeros((nbeta,),dtype=dtype), name='vbeta')
#
Ede = theano.shared(sample_weights(ny,nde), name='Ede')
Wde = theano.shared(sample_weights((2*nde+D+2*nen), 4*nde), name='Wde')
bde = theano.shared(np.zeros((4*nde,),dtype=dtype), name='bde')
#
L0 = theano.shared(sample_weights(nde,ny), name='L0')
L = theano.shared(sample_weights((nde+D+2*nen),nde), name='L')
#
cs0matde = theano.shared(np.zeros((batchsize, nde), dtype=dtype), name='cs0matde')
s0matde = theano.shared(np.zeros((batchsize, nde), dtype=dtype), name='s0matde')
#
Sentcube = T.tensor3(dtype=dtype, name='Sentcube') # |->  time-t * # batch * ny
[scub_de, cscub_de, ycub_de, logycub_de], _ = \
theano.scan(fn = decoder, 
            sequences=dict(input=Sentcube, taps=[0]), 
            outputs_info=[dict(initial=s0matde,taps=[-1]), dict(initial=cs0matde,taps=[-1]), 
                          None, None], 
            non_sequences=[newH, Alpha, 
                           Wbeta, Ubeta, vbeta,  
                           Ede, Wde, bde, L0, L])
#                   
print "finish decoder"
#TODO: define the loss function
Target = T.tensor3(dtype=dtype, name='Target')
#  time-t * # batch * ny
cost = -T.mean(T.sum( (Target*logycub_de), [2,0]))
#
print "finish the cost"

#TODO: Adam algo
alpha = theano.shared(np.float32(0.001),'alpha')
beta1 = theano.shared(np.float32(0.9),'beta1')
beta2 = theano.shared(np.float32(0.999), 'beta2')
eps = theano.shared(np.float32(0.00000001),'eps')
lam = theano.shared(np.float32(1.0 - 0.00000001), 'lam')
#
# adam - m
mEenf = theano.shared(np.zeros((D,nen),dtype=dtype),name='mEenf')
mEenb = theano.shared(np.zeros((D,nen),dtype=dtype),name='mEenb')
mWenf = theano.shared(np.zeros((2*nen, 4*nen),dtype=dtype), name='mWenf')
mWenb = theano.shared(np.zeros((2*nen, 4*nen),dtype=dtype), name='mWenb')
mbenf = theano.shared(np.zeros((4*nen,),dtype=dtype), name='mbenf')
mbenb = theano.shared(np.zeros((4*nen,),dtype=dtype), name='mbenb')
#
mPbeta = theano.shared(np.zeros((D+nen+nen,nbeta),dtype=dtype), name='mPbeta')
mqbeta = theano.shared(np.zeros((nbeta,),dtype=dtype), name='mqbeta')
#
mWbeta = theano.shared(np.zeros((nde,nbeta),dtype=dtype), name='mWbeta')
mUbeta = theano.shared(np.zeros((D+nen+nen,nbeta),dtype=dtype), name='mUbeta')
mvbeta = theano.shared(np.zeros((nbeta,),dtype=dtype), name='mvbeta')
#
mEde = theano.shared(np.zeros((ny,nde),dtype=dtype), name='mEde')
mWde = theano.shared(np.zeros(((2*nde+D+2*nen), 4*nde),dtype=dtype), name='mWde')
mbde = theano.shared(np.zeros((4*nde,),dtype=dtype), name='mbde')
#
mL0 = theano.shared(np.zeros((nde,ny),dtype=dtype), name='mL0')
mL = theano.shared(np.zeros(((nde+D+2*nen), nde),dtype=dtype), name='mL')
##
# adam - v
vEenf = theano.shared(np.zeros((D,nen),dtype=dtype),name='vEenf')
vEenb = theano.shared(np.zeros((D,nen),dtype=dtype),name='vEenb')
vWenf = theano.shared(np.zeros((2*nen, 4*nen),dtype=dtype), name='vWenf')
vWenb = theano.shared(np.zeros((2*nen, 4*nen),dtype=dtype), name='vWenb')
vbenf = theano.shared(np.zeros((4*nen,),dtype=dtype), name='vbenf')
vbenb = theano.shared(np.zeros((4*nen,),dtype=dtype), name='vbenb')
#
vPbeta = theano.shared(np.zeros((D+nen+nen,nbeta),dtype=dtype), name='vPbeta')
vqbeta = theano.shared(np.zeros((nbeta,),dtype=dtype), name='vqbeta')
#
vWbeta = theano.shared(np.zeros((nde,nbeta),dtype=dtype), name='vWbeta')
vUbeta = theano.shared(np.zeros((D+nen+nen,nbeta),dtype=dtype), name='vUbeta')
vvbeta = theano.shared(np.zeros((nbeta,),dtype=dtype), name='vvbeta')
#
vEde = theano.shared(np.zeros((ny,nde),dtype=dtype), name='vEde')
vWde = theano.shared(np.zeros(((2*nde+D+2*nen), 4*nde),dtype=dtype), name='vWde')
vbde = theano.shared(np.zeros((4*nde,),dtype=dtype), name='vbde')
#
vL0 = theano.shared(np.zeros((nde,ny),dtype=dtype), name='vL0')
vL = theano.shared(np.zeros(((nde+D+2*nen), nde),dtype=dtype), name='vL')
#
#
mparams = [mEenf, mEenb, mWenf, mWenb, mbenf, mbenb, 
           mPbeta, mqbeta, mWbeta, mUbeta, mvbeta, mEde, mWde, mbde, mL0, mL]
#
vparams = [vEenf, vEenb, vWenf, vWenb, vbenf, vbenb, 
           vPbeta, vqbeta, vWbeta, vUbeta, vvbeta, vEde, vWde, vbde, vL0, vL]
#

timestep = theano.shared(np.float32(1))
beta1_t = beta1*(lam**(timestep-1))

print "finish rate"
#TODO: take grads
params = [Eenf, Eenb, Wenf, Wenb, benf, benb, 
          Pbeta, qbeta, Wbeta, Ubeta, vbeta, Ede, Wde, bde, L0, L]
#
'''
gEenf, gEenb, gWenf, gWenb, gbenf, gbenb, 
gWbeta, gUbeta, gvbeta, gEde, gWde, gbde, gL0, gL = \
T.grad(cost, params)

gparams = [gEenf, gEenb, gWenf, gWenb, gbenf, gbenb, 
           gWbeta, gUbeta, gvbeta, gEde, gWde, gbde, gL0, gL]
'''
gparams = T.grad(cost, params)
#
print "finish grads"
#TODO: update
updates = []

for param, gparam, mparam, vparam in zip(params, gparams, mparams, vparams):
    #updates.append((param, param - gparam*learning_rate))
    newm0 = beta1_t * mparam + (1-beta1_t) * gparam
    newv0 = beta2 * vparam + (1-beta2) * (gparam**2)
    newm = newm0 / (1-(beta1**timestep) )
    newv = newv0 / (1-(beta2**timestep) )
    newparam0 = param - alpha*( newm/(T.sqrt(newv)+eps) )
    #
    #newparam = \
    #newparam0 * (T.clip( (normcap/T.sqrt(T.sum(newparam0**2))) ,np.float32(0.0),np.float32(1.0)))
    #
    updates.append((param, newparam0))
    #updates.append((param, newparam))
    updates.append((mparam, newm0))
    updates.append((vparam, newv0))
updates.append((timestep, timestep+1.0))

print "finish updates"


#TODO: define training function
#batchindex = T.lscalar(name='batchindex')
#currentindex = theano.shared
#
learn_model_fn = theano.function(inputs=[Xf, Xb, Sentcube, Target], 
                                 outputs = cost, 
                                 updates = updates)


#
print "finish THEANO preparation"
compileend = time.time()
compiletime = compileend-compilestart
print "the time needed for compilation is ", compiletime


#TODO: start training
print "start training"
train_errs = np.ndarray(max_epoch)

###
# prepare for the bleu -- using Java code
thepath = "../../code10/dist/generation.jar"
djavapath = "-Djava.class.path=%s"%os.path.abspath(thepath)
jpype.startJVM("/usr/lib/jvm/java-7-oracle/jre/lib/amd64/server/libjvm.so",
               "-ea",
               djavapath)
BleuScorer = jpype.JClass("cortex.BleuScorer")
bleuscorer = BleuScorer()
bleuscorer.setThreshold(maxdif)
###
f = open('log','w')
f.write('This is the training log. It records the training error and the validation bleu.\n')
f.write('epoch & training err & dev bleu & time in sec \n')
f.write('before that, the compilation time is '+str(compiletime))
f.write('\n')
f.close()

##
refSets = jpype.java.util.ArrayList()
refSet = jpype.java.util.ArrayList()
for ref0 in reflist:
    reference = jpype.java.util.ArrayList()
    for s in ref0.split():
        reference.add(s)
    refSet.add(reference)
refSets.add(refSet)

#
maxbleu = -1
###
for epi in range(max_epoch):
    print "training epoch", epi
    start = time.time()
    
    err = 0.
    # randomly sample training data, not in deterministic order
    idlist = np.random.permutation(max_step)
    for step0 in range(max_step):
        #
        step = idlist[step0]
        train_cost = \
        learn_model_fn(allXcube[step], allXcubeback[step], 
                       allScube[step], allTargetcube[step])
        err += train_cost
        print "finish the step ", step0, step
    train_errs[epi] = err / max_step
    print "finish epoch ", epi
    end = time.time()
    timetrain = end - start
    #
    #print "calculating BLEU on dev set"
    #
    # save the model first
    model = {}
    #
    # for encoder
    model['Eenf'] = Eenf.get_value()
    model['Eenb'] = Eenb.get_value()
    model['Wenf'] = Wenf.get_value()
    model['Wenb'] = Wenb.get_value()
    model['benf'] = benf.get_value()
    model['benb'] = benb.get_value()
    model['c0matenf'] = c0matenf.get_value()
    model['c0matenb'] = c0matenb.get_value()
    model['h0matenf'] = h0matenf.get_value()
    model['h0matenb'] = h0matenb.get_value()
    #
    model['Pbeta'] = Pbeta.get_value()
    model['qbeta'] = qbeta.get_value()
    model['Wbeta'] = Wbeta.get_value()
    model['Ubeta'] = Ubeta.get_value()
    model['vbeta'] = vbeta.get_value()
    model['Ede'] = Ede.get_value()
    model['Wde'] = Wde.get_value()
    model['bde'] = bde.get_value()
    model['L0'] = L0.get_value()
    model['L'] = L.get_value()
    #
    model['dropout'] = dropoutvalue
    #
    model['cs0matde'] = cs0matde.get_value()
    model['s0matde'] = s0matde.get_value()
    #
    fname = './track/model'+str(epi)+'.pickle'
    f = open(fname,'w')
    #f = open('model.pickle','w')
    pickle.dump(model,f)
    f.close()
    #
    print "calculating BLEU on dev set"
    #
    testlist = []
    teststart = time.time()
    testlist = gettextgs(devdata, model, ind2word, vocabmat)
    testend = time.time()   
    testtime = testend - teststart
    #
    # calculate the BLEU score -- threshold = 5
    tests = jpype.java.util.ArrayList()
    for test0 in testlist:
        test = jpype.java.util.ArrayList()
        for s in test0.split():
            test.add(s)
        tests.add(test)
    #
    thebs = bleuscorer.evaluateBleu(tests, refSets)
    thebleu = 100*thebs.getScore()
    #
    
    #
    f = open('log','a')
    f.write(str(epi))
    f.write(' & ')
    f.write(str(train_errs[epi]))
    f.write(' & ')
    f.write(str(thebleu))
    f.write(' & ')
    f.write(str(timetrain))
    f.write(' & ')
    f.write(str(testtime))
    f.write(' & ')
    #
    if thebleu > maxbleu:
        f.write('new best model here and id is '+str(epi))
        f.write('\n')
        maxbleu = thebleu
    else:
        f.write('\n')
    #
    f.close()
    time.sleep(5)
    
    print "finish the epoch ", epi
print "finish training"
    
jpype.shutdownJVM()











