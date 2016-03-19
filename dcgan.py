import sys
import os
import time

import lasagne
import theano
import theano.tensor as T
import numpy as np
from numpy.random import RandomState
from lasagne.layers import get_output
from lasagne.regularization import regularize_network_params,l2
from PIL import Image

from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_alloc_empty
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI


batch_size = 64
Z_dim = 1024


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
    net = lasagne.layers.InputLayer(shape=(batch_size, Z_dim, 1, 1),
                                    input_var=input_var)

    net = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
            net, num_filters=1024, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad='full'))

    net = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
            net, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad='full'))

    net = lasagne.layers.batch_norm(Deconv2DLayer(
            net, num_filters=64, filter_size=(5,5), subsample=(2,2),
            border_mode=(2,2)))

    net = lasagne.layers.NonlinearityLayer(net)

    net = Deconv2DLayer(
            net, num_filters=1, filter_size=(5,5), subsample=(2,2),
            border_mode=(2,2))

    net = lasagne.layers.NonlinearityLayer(net, nonlinearity=lasagne.nonlinearities.sigmoid)

    print [l.__class__.__name__ for l in lasagne.layers.get_all_layers(net)]
    print net.output_shape

    return net


def build_dis_init(input_var=None):

    net = lasagne.layers.InputLayer(shape=(batch_size, 1, 28, 28),
                                    input_var=input_var)

    net = lasagne.layers.Conv2DLayer(
            net, num_filters=64, filter_size=(5, 5), stride=2,
            nonlinearity=None,
            pad='same')

    # net = lasagne.layers.BatchNormLayer(net)
    net = lasagne.layers.NonlinearityLayer(net, lasagne.nonlinearities.LeakyRectify(0.2))

    net = lasagne.layers.Conv2DLayer(
            net, num_filters=128, filter_size=(5, 5), stride=2,
            #nonlinearity=lasagne.nonlinearities.rectify,
            nonlinearity=None,
            pad='same')

    net = lasagne.layers.BatchNormLayer(net)
    net = lasagne.layers.NonlinearityLayer(net, lasagne.nonlinearities.LeakyRectify(0.2))

    net = lasagne.layers.Conv2DLayer(
            net, num_filters=1024, filter_size=(5, 5),
            #nonlinearity=lasagne.nonlinearities.rectify,
            nonlinearity=None,
            pad='valid')

    net = lasagne.layers.BatchNormLayer(net)
    net = lasagne.layers.NonlinearityLayer(net, lasagne.nonlinearities.LeakyRectify(0.2))

    net = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(net, p=0.9),
            num_units=1,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    params = []
    for l in lasagne.layers.get_all_layers(net):
        for p in l.get_params(trainable=True):
            params.append(p)

    print [l.__class__.__name__ for l in lasagne.layers.get_all_layers(net)]
    print net.output_shape

    return params


def build_dis(input_var=None, p_dis=None):

    net = lasagne.layers.InputLayer(shape=(batch_size, 1, 28, 28),
                                    input_var=input_var)

    net = lasagne.layers.Conv2DLayer(
            net, num_filters=64, filter_size=(5, 5), stride=2,
            nonlinearity=None,
            W=p_dis[0],
            b=p_dis[1],
            pad='same')

    # net = lasagne.layers.BatchNormLayer(
    #         net,
    #         beta=p_dis[2],
    #         gamma=p_dis[3])

    net = lasagne.layers.NonlinearityLayer(net, lasagne.nonlinearities.LeakyRectify(0.2))

    net = lasagne.layers.Conv2DLayer(
            net, num_filters=128, filter_size=(5, 5), stride=2,
            nonlinearity=None,
            W=p_dis[2],
            b=p_dis[3],
            pad='same')

    net = lasagne.layers.BatchNormLayer(
            net,
            beta=p_dis[4],
            gamma=p_dis[5])

    net = lasagne.layers.NonlinearityLayer(net, lasagne.nonlinearities.LeakyRectify(0.2))

    net = lasagne.layers.Conv2DLayer(
            net, num_filters=1024, filter_size=(5, 5),
            nonlinearity=None,
            W=p_dis[6],
            b=p_dis[7],
            pad='valid')

    net = lasagne.layers.BatchNormLayer(
            net,
            beta=p_dis[8],
            gamma=p_dis[9])

    net = lasagne.layers.NonlinearityLayer(net, lasagne.nonlinearities.LeakyRectify(0.2))

    net = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(net, p=0.9),
            num_units=1,
            W=p_dis[10],
            b=p_dis[11],
            nonlinearity=lasagne.nonlinearities.sigmoid)

    # print [l.__class__.__name__ for l in lasagne.layers.get_all_layers(net)]
    # print net.output_shape

    return net


def get_gen_loss(D2):
    return T.nnet.binary_crossentropy(D2, T.ones(D2.shape)).mean()


def get_dis_loss(D1, D2):
    return T.nnet.binary_crossentropy(D1, T.ones(D1.shape)).mean() + \
        T.nnet.binary_crossentropy(D2, T.zeros(D2.shape)).mean()

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



def main(num_epochs=100):
    X_train, X_test = load_dataset()
    real_data = T.tensor4('real')
    fake_data = T.tensor4('fake')

    param_dis = build_dis_init() # just to create weights

    D1 = build_dis(real_data, param_dis)
    G = build_gen(fake_data)

    gen_data = get_output(G)
    D2 = build_dis(gen_data, param_dis)

    D_loss = get_dis_loss(get_output(D1), get_output(D2))
    G_loss = get_gen_loss(get_output(D2))

    params_g = lasagne.layers.get_all_params(G, trainable=True)
    params_d = lasagne.layers.get_all_params(D1, trainable=True)

    # to make sure the params are shared between D1 & D2
    # params_d2 = lasagne.layers.get_all_params(D2, trainable=True)
    # print [id(p) for p in params_d]
    # print [id(p) for p in params_d2]

    lr_g = 0.0002
    lr_d = 0.0002

    w_decay = 0.000025

    grad_g = T.grad(G_loss, params_g)
    for g, p in zip(grad_g, params_g):
        g += p * w_decay

    grad_d = T.grad(D_loss, params_d)
    for g, p in zip(grad_d, params_d):
        g += p * w_decay

    updates_d = lasagne.updates.adam(grad_d, params_d, learning_rate=lr_d, beta1=0.5)
    updates_g = lasagne.updates.adam(grad_g, params_g, learning_rate=lr_g, beta1=0.5)

    train_fn_g = theano.function([fake_data], G_loss, updates=updates_g)
    train_fn_d = theano.function([real_data, fake_data], D_loss, updates=updates_d)

    gen = theano.function([fake_data], gen_data)

    k = 1
    print 'Start training...'
    for epoch in range(num_epochs):
        train_err_g = 0
        train_err_d = 0

        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, batch_size, shuffle=True):
            f_data = np.array(np.random.uniform(-1, 1, (batch_size, Z_dim, 1, 1)), dtype=theano.config.floatX)

            if train_batches % (k+1) == 0:
                train_err_g += train_fn_g(f_data)
            else:
                train_err_d += train_fn_d(batch, f_data)

            train_batches += 1


        print "Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time)

        print "  training loss g:\t\t{:.6f}".format(train_err_g / train_batches)
        print "  training loss d:\t\t{:.6f}".format(train_err_d / train_batches)

        # generate s batches of samples to see how well the network learns at every epoch
        s = 1
        rescale = 4
        for i in xrange(s):
            f_data = np.array(np.random.uniform(-1, 1, (batch_size, Z_dim, 1, 1)), dtype=theano.config.floatX)
            g_data = gen(f_data)
            for j in xrange(batch_size):
                img = g_data[j].reshape(28, 28)*256
                img = img.repeat(rescale, axis = 0).repeat(rescale, axis = 1).astype(np.uint8())
                img = Image.fromarray(img)
                img.save('generated/'+str(epoch)+'_'+str(i*batch_size+j)+'.png',format="PNG")




if __name__ == '__main__':
    main()
