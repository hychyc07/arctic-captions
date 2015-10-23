import cPickle as pkl
import gzip
import os
import sys
import time

import numpy

def prepare_data(caps, features, worddict, maxlen=None, n_words=10000, zero_pad=False):
    # x: a list of sentences
    seqs = []
    feat_list = []
    for cc in caps:
        seqs.append([worddict[w] if w in worddict and worddict[w] < n_words else 1 for w in cc[0].split()])
        feat_list.append(features[cc[1]])
    lengths = [len(s) for s in seqs]

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

    y = numpy.zeros((len(feat_list), feat_list[0].shape[1])).astype('float32')
    for idx, ff in enumerate(feat_list):
        y[idx,:] = numpy.array(ff.todense())
    y = y.reshape([y.shape[0], 14*14, 512])
    if zero_pad:
        y_pad = numpy.zeros((y.shape[0], y.shape[1]+1, y.shape[2])).astype('float32')
        y_pad[:,:-1,:] = y
        y = y_pad

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    capture = numpy.zeros((n_samples, maxlen)).astype('int64')
    capture_mask = numpy.zeros((n_samples, maxlen)).astype('float32')
    for idx, s in enumerate(seqs):
        capture[idx, :lengths[idx]] = s
        capture_mask[idx, :lengths[idx]+1] = 1.

    return capture, capture_mask, y

def load_data(load_train=True, load_dev=True, load_test=True, path='../../datasets/Flickr_8k/'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    #############
    # LOAD DATA #
    #############
    flicker_dataname = 'flickr_8k_align'
    if load_train:
        with open(path+ flicker_dataname +'.train.pkl', 'rb') as f:
            train = pkl.load(f)
    else:
        train = None
    print '... loading training data', len(train[0])

    if load_test:
        with open(path+ flicker_dataname +'.test.pkl', 'rb') as f:
            test = pkl.load(f)
    else:
        test = None
    print '... loading testing data', len(test[0])

    if load_dev:
        with open(path+ flicker_dataname +'.dev.pkl', 'rb') as f:
            valid = pkl.load(f)
    else:
        valid = None
    print '... loading validation data', len(valid[0])


    with open(path+'dictionary.pkl', 'rb') as f:
        worddict = pkl.load(f)

    return train, valid, test, worddict
