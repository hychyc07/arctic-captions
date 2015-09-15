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

def load_db(db_dir, datasize): 
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
        feats_set += [scipy.sparse.csr_matrix(feature.reshape(14*14*512))]
    print "load image features: ", len(feats_set)
    print "size: ", row, col, np.shape(feats_set[0])
    return feats_set[0:datasize]

def load_filecap(args):

    # get all the file, it matches with the features we get
    filelists = []
    for files in open(args.datalists):
        filename = files[files.rfind('/')+1:-1]
        filelists += [filename]
    filelists = filelists[0: len(filelists)/50*50]
    print "load files: ", len(filelists), filelists[1:3]

    # get all the capture lists
    capture_map = {}
    capture_vocab = {}
    capturelists= []
    for caplines in open(args.caplists):
        lineinfo = caplines.split('\t')
        image_name = lineinfo[0].split('#')[0]
        image_cap = lineinfo[1].replace('\n','').lower()
        if image_name not in filelists:
            continue
        capturelists += [(image_cap,image_name)]
        for words in image_cap.replace(".",'').split():
            if words not in capture_vocab:
                capture_vocab[words] = 1
            else:
                capture_vocab[words] += 1

        #if image_name not in capture_map:
        #    capture_map[image_name] = [image_cap]
        #else:
        #    capture_map[image_name] += [image_cap]

    sorted_capture_vocab = sorted(capture_vocab.items(), key=operator.itemgetter(1))
    sorted_capture_vocab.reverse()
    print "capture_size:", len(capturelists)
    print " examples, ", capturelists[0]
    print "vocab_size:", len(capture_vocab)
    print " top10 vocab, ",sorted_capture_vocab[0:10]
    print " more than once vocab: ", len([words for words in sorted_capture_vocab if words[1]>1])

    return filelists, capturelists, sorted_capture_vocab

def main(args):
    ## capture load and vocab build
    print ">> now capture load and vocab build"
    filelists, capturelists, sorted_vocab = load_filecap(args)

    print ">> now start dumping vocab"
    dic_dump = {".": 0}  #1 is reserved for unknown
    for i in range(0,len(sorted_vocab)):
        dic_dump[sorted_vocab[i][0]] = i+2 #rank+1, reserve 0 and 1
    pkl.dump(dic_dump, open("dictionary.pkl", "wb"))
    print "<< finished"

    ## feature load
    print ">> now loading features from convnet: (wait, long time)"
    feats_set = load_db(args.lmdb, args.datasize)
    file_feature_map = dict(zip(filelists, feats_set))
    #for feature in feats_set:
    #    print feature[np.nonzero(feature)]

    print ">> now start dumping capture and feature "
    feat_dump = {}
    capturelists_dump = []
    for file in file_feature_map.keys():
        feat_dump[file] = file_feature_map[file] #scipy.sparse.csr_matrix(file_feature_map[file])
    for caps in capturelists:
        if caps[1] in feat_dump.keys():
            capturelists_dump += [ caps ]
    cap_feature_dump = [capturelists_dump, feat_dump]
    print "feature example", feat_dump.items()[1]
    pkl.dump(cap_feature_dump, open("flicker_30k_align.train.pkl", "wb"))
    pkl.dump(cap_feature_dump, open("flicker_30k_align.test.pkl", "wb"))
    pkl.dump(cap_feature_dump, open("flicker_30k_align.val.pkl", "wb"))

    print "<< finished"


if __name__ == "__main__":
    base_path = '/home/hychyc07/caffe/'
    default_feature_path = base_path + 'examples/flickr30k/flickr30k-images_1k_featuresvgg'
    default_filelists_path = base_path + 'examples/flickr30k/flickr30k-images_1k.txt'
    default_caplists_path = base_path + 'examples/flickr30k/results_20130124.token'

    parser = argparse.ArgumentParser()

    parser.add_argument('--datasize', type=int,default=3, required=False)
    parser.add_argument('--datalists', type=str, default=default_filelists_path, required=False)
    parser.add_argument('--caplists', type=str, default=default_caplists_path, required=False)
    parser.add_argument('--lmdb', type=str, default=default_feature_path, required=False)

    args = parser.parse_args()

    main(args)
