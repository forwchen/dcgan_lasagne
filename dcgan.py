import sys
import os
import time

import lasagne
import theano
import theano.tensor as T
import numpy as np
from lasagne.layers import get_output



batch_size = 64

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



def main(num_epochs=100):
    X_train, X_test = load_dataset()
    real_data = T.tensor4('real')
    fake_data = T.tensor4('fake')

    p_dis = build_dis_init() # just to create weights

    D1 = build_dis(real_data, p_dis)
    G = build_gen(fake_data)

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

    lr_g = 0.01
    lr_d = 0.001

    updates_d = lasagne.updates.adam(D_loss, params_d, learning_rate=lr_d)
    updates_g = lasagne.updates.adam(G_loss, params_g, learning_rate=lr_g)

    train_fn_g = theano.function([fake_data], G_loss, updates=updates_g)
    train_fn_d = theano.function([real_data, fake_data], D_loss, updates=updates_d)

    print 'Start training...'
    for epoch in range(num_epochs):
        train_err_g = 0
        train_err_d = 0

        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, batch_size, shuffle=True):
            f_data = np.array(np.random.uniform(-1, 1, (batch_size, 100, 1, 1)), dtype=theano.config.floatX)

            train_err_g += train_fn_g(f_data)
            train_err_d += train_fn_d(batch, f_data)

            train_batches += 1

        print "Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time)

        print "  training loss g:\t\t{:.6f}".format(train_err_g / train_batches)
        print "  training loss d:\t\t{:.6f}".format(train_err_d / train_batches)


if __name__ == '__main__':
    main()
