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

from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool

seed = 42
np_rng = RandomState(seed)
batch_size = 64

def l2norm(x, axis=1, e=1e-8, keepdims=True):
    return T.sqrt(T.sum(T.sqr(x), axis=axis, keepdims=keepdims) + e)

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

def clip_norm(g, c, n):
    if c > 0:
        g = T.switch(T.ge(n, c), g*c/n, g)
    return g

def clip_norms(gs, c):
    norm = T.sqrt(sum([T.sum(g**2) for g in gs]))
    return [clip_norm(g, c, norm) for g in gs]

class Regularizer(object):

    def __init__(self, l1=0., l2=0., maxnorm=0., l2norm=False, frobnorm=False):
        self.__dict__.update(locals())

    def max_norm(self, p, maxnorm):
        if maxnorm > 0:
            norms = T.sqrt(T.sum(T.sqr(p), axis=0))
            desired = T.clip(norms, 0, maxnorm)
            p = p * (desired/ (1e-7 + norms))
        return p

    def l2_norm(self, p):
        return p/l2norm(p, axis=0)

    def frob_norm(self, p, nrows):
        return (p/T.sqrt(T.sum(T.sqr(p))))*T.sqrt(nrows)

    def gradient_regularize(self, p, g):
        g += p * self.l2
        g += T.sgn(p) * self.l1
        return g

    def weight_regularize(self, p):
        p = self.max_norm(p, self.maxnorm)
        if self.l2norm:
            p = self.l2_norm(p)
        if self.frobnorm > 0:
            p = self.frob_norm(p, self.frobnorm)
        return p


class Update(object):

    def __init__(self, regularizer=Regularizer(), clipnorm=0.):
        self.__dict__.update(locals())

    def __call__(self, params, grads):
        raise NotImplementedError

class Adam(Update):

    def __init__(self, lr=0.001, b1=0.9, b2=0.999, e=1e-8, l=1-1e-8, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        t = theano.shared(floatX(1.))
        b1_t = self.b1*self.l**(t-1)

        for p, g in zip(params, grads):
            g = self.regularizer.gradient_regularize(p, g)
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)

            m_t = b1_t*m + (1 - b1_t)*g
            v_t = self.b2*v + (1 - self.b2)*g**2
            m_c = m_t / (1-self.b1**t)
            v_c = v_t / (1-self.b2**t)
            p_t = p - (self.lr * m_c) / (T.sqrt(v_c) + self.e)
            p_t = self.regularizer.weight_regularize(p_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t) )
        updates.append((t, t + 1.))
        return updates


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
    net = lasagne.layers.InputLayer(shape=(batch_size, 1024, 1, 1),
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
            print type(p)
            params.append(p)

    print params
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
    # return T.mean(T.nnet.relu(D2) - D2 + T.log(1.0 + T.exp(-T.abs_(D2))))
    return T.nnet.binary_crossentropy(D2, T.ones(D2.shape)).mean()


def get_dis_loss(D1, D2):
    # return T.mean(T.nnet.relu(D1) - D1 + T.log(1.0 + T.exp(-T.abs_(D1)))) + \
        # T.mean(T.nnet.relu(D2) - D2 + T.log(1.0 + T.exp(-T.abs_(D2))))

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



def main(num_epochs=20):
    X_train, X_test = load_dataset()
    real_data = T.tensor4('real')
    fake_data = T.tensor4('fake')

    p_dis = build_dis_init() # just to create weights

    D1 = build_dis(real_data, p_dis)
    G = build_gen(fake_data)

    gen_data = get_output(G)
    D2 = build_dis(gen_data, p_dis)

    D_loss = get_dis_loss(get_output(D1), get_output(D2))
    G_loss = get_gen_loss(get_output(D2))

    params_g = lasagne.layers.get_all_params(G, trainable=True)
    params_d = lasagne.layers.get_all_params(D1, trainable=True)

    # to make sure the params are shared between D1 & D2
    # params_d2 = lasagne.layers.get_all_params(D2, trainable=True)
    # print [id(p) for p in params_d]
    # print [id(p) for p in params_d2]

    lrt = sharedX(0.0002)
    d_updater = Adam(lr=lrt, b1=0.5, regularizer=Regularizer(l2=2.5e-5))
    g_updater = Adam(lr=lrt, b1=0.5, regularizer=Regularizer(l2=2.5e-5))
    updates_d = d_updater(params_d, D_loss)
    updates_g = g_updater(params_g, G_loss)

    # lr_g = 0.0002
    # lr_d = 0.0002
    # G_loss += regularize_network_params(G, l2)
    # D_loss += regularize_network_params(D1, l2)
    # updates_d = lasagne.updates.adam(D_loss, params_d, learning_rate=lr_d)
    # updates_g = lasagne.updates.adam(G_loss, params_g, learning_rate=lr_g)

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
            f_data = np.array(np.random.uniform(-1, 1, (batch_size, 1024, 1, 1)), dtype=theano.config.floatX)

            if train_batches % (k+1) == 0:
                train_err_g += train_fn_g(f_data)
            else:
                train_err_d += train_fn_d(batch, f_data)

            train_batches += 1


        print "Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time)

        print "  training loss g:\t\t{:.6f}".format(train_err_g / train_batches)
        print "  training loss d:\t\t{:.6f}".format(train_err_d / train_batches)

        # generate 1 batch of samples to see how well the network learns at every epoch
        rescale = 4
        for i in xrange(1):
            f_data = np.array(np.random.uniform(-1, 1, (batch_size, 1024, 1, 1)), dtype=theano.config.floatX)
            g_data = gen(f_data)
            for j in xrange(batch_size):
                img = g_data[j].reshape(28, 28)*256
                img = img.repeat(rescale, axis = 0).repeat(rescale, axis = 1).astype(np.uint8())
                img = Image.fromarray(img)
                img.save('generated/'+str(epoch)+'_'+str(i*batch_size+j)+'.png',format="PNG")
            break



if __name__ == '__main__':
    main()
