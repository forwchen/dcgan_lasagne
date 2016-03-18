import sys
import os
import time

import lasagne
import theano
import theano.tensor as T
import numpy as np
from lasagne.layers import get_output
from lasagne.regularization import regularize_network_params,l2
from PIL import Image

from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool

batch_size = 64


class Deconv2DLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_filters, filter_size, subsample=(1,1),
                border_mode=(0,0), W=lasagne.init.GlorotUniform(), **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.subsample = subsample
        self.border_mode = border_mode
        self.W = self.add_param(W, self.get_W_shape(), name='W')

    def get_W_shape(self):
        return (self.input_shape[1], self.num_filters) + self.filter_size

    def get_output_shape_for(self, input_shape):
        filter_shape = self.get_W_shape()

        return (input_shape[0], filter_shape[1], input_shape[2]*self.subsample[0], input_shape[3]*self.subsample[1])

    def get_output_for(self, input, **kwargs):
        return self.convolve(input)


    def convolve(self, input, **kwargs):
        img = gpu_contiguous(input)
        kerns = gpu_contiguous(self.W)
        out_shape = self.get_output_shape_for(img.shape)

        desc = GpuDnnConvDesc(border_mode=self.border_mode,
            subsample=self.subsample)(gpu_alloc_empty(out_shape[0], out_shape[1], out_shape[2], out_shape[3]).shape, kerns.shape)
        out_mem = gpu_alloc_empty(out_shape[0], out_shape[1], out_shape[2], out_shape[3])
        return GpuDnnConvGradI()(kerns, img, out_mem, desc)


def load_dataset():
    import gzip

    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')

    return X_train, X_test


def build_gen(input_var=None):

    net = lasagne.layers.InputLayer(shape=(batch_size, 100, 1, 1),
                                    input_var=input_var)

    net = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
            net, num_filters=128, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad='full'))

    net = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
            net, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad='full'))

    net = lasagne.layers.Upscale2DLayer(net, scale_factor=2)

    net = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
            net, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad='same'))

    net = lasagne.layers.Upscale2DLayer(net, scale_factor=2)

    net = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
            net, num_filters=1, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.sigmoid,
            pad='same'))

    print [l.__class__.__name__ for l in lasagne.layers.get_all_layers(net)]
    print net.output_shape

    return net

def build_gen2(input_var=None):
    net = lasagne.layers.InputLayer(shape=(batch_size, 100, 1, 1),
                                    input_var=input_var)

    net = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
            net, num_filters=128, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad='full'))

    net = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
            net, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad='full'))

    net = lasagne.layers.batch_norm(Deconv2DLayer(
            net, num_filters=32, filter_size=(5,5), subsample=(2,2),
            border_mode=(2,2)))

    net = lasagne.layers.NonlinearityLayer(net)

    net =  lasagne.layers.batch_norm(Deconv2DLayer(
            net, num_filters=1, filter_size=(5,5), subsample=(2,2),
            border_mode=(2,2)))

    net = lasagne.layers.NonlinearityLayer(net, nonlinearity=lasagne.nonlinearities.sigmoid)

    print [l.__class__.__name__ for l in lasagne.layers.get_all_layers(net)]
    print net.output_shape

    return net


def build_dis_init(input_var=None):

    net = lasagne.layers.InputLayer(shape=(batch_size, 1, 28, 28),
                                    input_var=input_var)

    net = lasagne.layers.Conv2DLayer(
            net, num_filters=32, filter_size=(5, 5), stride=2,
            nonlinearity=None,
            pad='same')

    net = lasagne.layers.BatchNormLayer(net)
    net = lasagne.layers.NonlinearityLayer(net)

    net = lasagne.layers.Conv2DLayer(
            net, num_filters=64, filter_size=(5, 5), stride=2,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad='same')

    net = lasagne.layers.BatchNormLayer(net)
    net = lasagne.layers.NonlinearityLayer(net)

    net = lasagne.layers.Conv2DLayer(
            net, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad='valid')

    net = lasagne.layers.BatchNormLayer(net)
    net = lasagne.layers.NonlinearityLayer(net)

    net = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(net, p=0.9),
            num_units=1,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    params = []
    for l in lasagne.layers.get_all_layers(net):
        for p in l.get_params(trainable=True):
            print type(p)
            params.append(p)

    print params
    return params


def build_dis(input_var=None, p_dis=None):

    net = lasagne.layers.InputLayer(shape=(batch_size, 1, 28, 28),
                                    input_var=input_var)

    net = lasagne.layers.Conv2DLayer(
            net, num_filters=32, filter_size=(5, 5), stride=2,
            nonlinearity=None,
            W=p_dis[0],
            b=p_dis[1],
            pad='same')

    net = lasagne.layers.BatchNormLayer(
            net,
            beta=p_dis[2],
            gamma=p_dis[3])

    net = lasagne.layers.NonlinearityLayer(net)

    net = lasagne.layers.Conv2DLayer(
            net, num_filters=64, filter_size=(5, 5), stride=2,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=p_dis[4],
            b=p_dis[5],
            pad='same')

    net = lasagne.layers.BatchNormLayer(
            net,
            beta=p_dis[6],
            gamma=p_dis[7])

    net = lasagne.layers.NonlinearityLayer(net)

    net = lasagne.layers.Conv2DLayer(
            net, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=p_dis[8],
            b=p_dis[9],
            pad='valid')

    net = lasagne.layers.BatchNormLayer(
            net,
            beta=p_dis[10],
            gamma=p_dis[11])

    net = lasagne.layers.NonlinearityLayer(net)

    net = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(net, p=0.9),
            num_units=1,
            W=p_dis[12],
            b=p_dis[13],
            nonlinearity=lasagne.nonlinearities.sigmoid)

    # print [l.__class__.__name__ for l in lasagne.layers.get_all_layers(net)]
    # print net.output_shape

    return net


def get_gen_loss(D2):
    return T.mean(T.nnet.relu(D2) - D2 + T.log(1.0 + T.exp(-T.abs_(D2))))

def get_dis_loss(D1, D2):
    return T.mean(T.nnet.relu(D1) - D1 + T.log(1.0 + T.exp(-T.abs_(D1)))) + \
        T.mean(T.nnet.relu(D2) - D2 + T.log(1.0 + T.exp(-T.abs_(D2))))

def iterate_minibatches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]



def main(num_epochs=2):
    X_train, X_test = load_dataset()
    real_data = T.tensor4('real')
    fake_data = T.tensor4('fake')

    p_dis = build_dis_init() # just to create weights

    D1 = build_dis(real_data, p_dis)
    G = build_gen2(fake_data)

    gen_data = get_output(G)
    D2 = build_dis(gen_data, p_dis)

    D_loss = get_dis_loss(get_output(D1), get_output(D2))
    G_loss = get_gen_loss(get_output(G))

    params_d = lasagne.layers.get_all_params(D1, trainable=True)

    # to make sure the params are shared between D1 & D2
    # params_d2 = lasagne.layers.get_all_params(D2, trainable=True)
    # print [id(p) for p in params_d]
    # print [id(p) for p in params_d2]

    params_g = lasagne.layers.get_all_params(G, trainable=True)

    D_loss += regularize_network_params(D1, l2)
    G_loss += regularize_network_params(G, l2)

    lr_g = 0.0002
    lr_d = 0.0002

    updates_d = lasagne.updates.adam(D_loss, params_d, learning_rate=lr_d)
    updates_g = lasagne.updates.adam(G_loss, params_g, learning_rate=lr_g)

    train_fn_g = theano.function([fake_data], G_loss, updates=updates_g)
    train_fn_d = theano.function([real_data, fake_data], D_loss, updates=updates_d)

    gen = theano.function([fake_data], gen_data)

    # return None
    k = 1
    print 'Start training...'
    for epoch in range(num_epochs):
        train_err_g = 0
        train_err_d = 0

        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, batch_size, shuffle=True):
            f_data = np.array(np.random.uniform(-1, 1, (batch_size, 100, 1, 1)), dtype=theano.config.floatX)

            if train_batches % (k+1) == 0:
                train_err_g += train_fn_g(f_data)
            else:
                train_err_d += train_fn_d(batch, f_data)

            train_batches += 1
            # if train_batches > 300:
            #     break

        # print "Epoch {} of {} took {:.3f}s".format(
        #     epoch + 1, num_epochs, time.time() - start_time)

        # print "  training loss g:\t\t{:.6f}".format(train_err_g / train_batches)
        # print "  training loss d:\t\t{:.6f}".format(train_err_d / train_batches)

    # generate 1 batch of samples to see how well the network learns
    rescale = 4
    for i in xrange(1):
        f_data = np.array(np.random.uniform(-1, 1, (batch_size, 100, 1, 1)), dtype=theano.config.floatX)
        print f_data[1]
        g_data = gen(f_data)
        print g_data[1]
        for j in xrange(batch_size):
            img = g_data[j].reshape(28, 28)*256
            img = img.repeat(rescale, axis = 0).repeat(rescale, axis = 1).astype(np.uint8())
            img = Image.fromarray(img)
            img.save('generated/'+str(i*batch_size+j)+'.png',format="PNG")
        break



if __name__ == '__main__':
    main()
    # build_gen2()
