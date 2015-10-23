#!/usr/bin/env python
import sys, argparse,operator
import cPickle as pkl
import numpy as np
import scipy
import lmdb
import caffe
from caffe.io import caffe_pb2

# https://groups.google.com/forum/#!msg/caffe-users/6OOcM-XfvOI/Cs5VVdfDubEJ

caffe.set_mode_cpu()

def load_db(db_dir):
    lmdb_env = lmdb.open(db_dir)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    feats_set = []
    row = 0
    col = 0
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        feature = caffe.io.datum_to_array(datum)
        row = datum.height
        col = datum.width
        feats_set += [scipy.sparse.csr_matrix(feature.reshape(4096,))]
    print "load image features: ", len(feats_set)
    if len(feats_set) > 0:
        print "size: ", row, col, np.shape(feats_set[0])
    return feats_set

if __name__ == "__main__":
    folder= '/home/hychyc07/caffe/examples/YouTubeClips_Flow/'
    allfeat =load_db(db_dir=folder+'all_features')
    file_out = open(folder + 'feat.txt' , 'w' )
    for feat in allfeat:
        file_out.write(str(feat.toarray().tolist()[0]).replace(',','')[1:-1] + '\n')
    file_out.close()