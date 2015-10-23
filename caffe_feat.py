import sys, time
import caffe
import numpy as np
import argparse
from collections import defaultdict

caffe_root='/home/hychyc07/caffe/'
debug = True

class caffe_feat:
    def __init__(self, proto, model, imgmean, targetsize=224, lmdb=None) :
        self.proto = proto
        self.model = model
        self.net = caffe.Net(proto, model, caffe.TEST)
        self.target_size=targetsize

        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        self.img_mean = imgmean
        self.transformer.set_mean('data', self.img_mean) # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

        if not debug:
            return
        if 'action' not in model:
            imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
            try:
                self.labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
            except:
                # !../data/ilsvrc12/get_ilsvrc_aux.sh
                self.labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
        else:
            self.labels = np.array(['unknown', 'Diving', 'GolfSwing', 'Kicking', 'Lifting',
                                    'Riding', 'Running', 'SkateBoarding', 'SwingBench', 'SwingSide', 'Walking'])

    def extract_feat_file(self, img_file=None):
        if img_file is None:
            image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
        else:
            image = caffe.io.load_image(caffe_root + str(img_file))
        time1 = time.time()
        self.extract_feat(image)
        time2 = time.time()
        if debug:
            print "time costs ", str(time2-time1), ' sec'
        #image : an image with type np.float32 in range [0, 1]
        #of size (H x W x 3) in RGB or
        #of size (H x W x 1) in grayscale.

    def extract_feat(self, image):
        self.net.blobs['data'].reshape(1, 3, self.target_size, self.target_size)
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image)
        out = self.net.forward()

        if debug:
            print("Predicted class is #{}.".format(out['prob'].argmax()))
            # sort top k predictions from softmax output
            top_k = self.net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
            print top_k, self.labels[top_k]

        if 'conv5_4' in self.net.blobs:
            feat_2d = self.net.blobs['conv5_4'].data[0]
            if debug:
                print feat_2d.shape
        else:
            feat_2d = None

        if 'fc7' in  self.net.blobs:
            feat_1d = self.net.blobs['fc7'].data[0]
            if debug:
                print feat_1d.shape
                print feat_1d.nonzero()
        else:
            feat_1d = None

        return feat_2d, feat_1d

def initialize_model(model, mode):
    if mode is None:
        mode = 'cpu'
    if 'cpu' in mode.lower():
        caffe.set_mode_cpu()
    if 'gpu' in mode.lower():
        caffe.set_device(0)
        caffe.set_mode_gpu()

    if model is None:
        model = 'cafferef'
    if 'cafferef' in model.lower():
        cnn_proto = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
        cnn_model = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
        imgnet_mean = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)
        cnn_imgmean = imgnet_mean
        cnn_imgsize = 227
    if 'vgg' in model.lower():
        if 'vgg16' in model.lower():
            cnn_proto = caffe_root + 'models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers_deploy.prototxt'
            cnn_model = caffe_root + 'models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel'
        if 'vgg19' in model.lower():
            cnn_proto = caffe_root + 'models/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers_deploy.prototxt'
            cnn_model = caffe_root + 'models/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.caffemodel'
        vgg_mean = np.array([103.939, 116.779, 123.68])
        cnn_imgmean = vgg_mean
        cnn_imgsize = 224
    if 'action' in model.lower():
        cnn_proto = caffe_root + 'models/action_cube/deploy_extractpred.prototxt'
        cnn_model = caffe_root + 'models/action_cube/action_cube.caffemodel'
        cnn_imgmean = np.array([128, 128, 128])
        cnn_imgsize = 227

    cf = caffe_feat(cnn_proto, cnn_model, cnn_imgmean, cnn_imgsize)
    return cf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--mode', type=str, required=False)
    args = parser.parse_args()
    cf = initialize_model(args.model, args.mode)
    cf.extract_feat_file()

