import numpy as np
import theano
import lasagne as las
from theano import tensor as T
from lasagne import layers as L
import custom_layer as cs


def build_net_v0():
    fov = 40    # field of view = patch length
    n_channels = 2
    n_classes = 4
    filt = [7, 6, 6]
    n_filt = [20, 25, 60, 30, n_classes]
    pool = [2, 2]
    dropout = [0.2, 0.2]

    # 40
    l_in = L.InputLayer((None, n_channels, fov, fov))

    l_1 = L.Conv2DLayer(l_in, n_filt[0], filt[0])
    l_2 = L.DropoutLayer(l_1, p=dropout[0])
    l_3 = L.MaxPool2DLayer(l_2, pool[0])
    l_4 = L.Conv2DLayer(l_3, n_filt[1], filt[1])
    l_5 = L.DropoutLayer(l_4, p=dropout[1])
    l_6 = L.MaxPool2DLayer(l_5, pool[1])
    l_7 = L.Conv2DLayer(l_6, n_filt[2], 5, filt[2])
    l_8 = L.Conv2DLayer(l_7, n_filt[3], 1)
    l_9 = L.Conv2DLayer(l_8, n_filt[4], 1,
                        nonlinearity=las.nonlinearities.sigmoid)
    return l_in, l_9, fov


def build_net_v1():
    '''
    cnn with 100 x 100 input and 4 classes out
    :return:
    '''
    fov = 100   # field of view = patch length
    n_channels = 2
    filt = [15, 10, 6, 6]
    n_filt = [20, 25, 60, 30, 15, 4]
    pool = [2, 2, 2]
    dropout = [0.2, 0.2, 0.2]

    l_in = L.InputLayer((None, n_channels, fov, 100))

    l_1 = L.Conv2DLayer(l_in, n_filt[0], filt[0])
    l_2 = L.DropoutLayer(l_1, p=dropout[0])
    l_3 = L.MaxPool2DLayer(l_2, pool[0])
    # 43
    l_4 = L.Conv2DLayer(l_3, n_filt[1], filt[1])
    l_5 = L.DropoutLayer(l_4, p=dropout[1])
    l_6 = L.MaxPool2DLayer(l_5, pool[1])
    # 17
    l_7 = L.Conv2DLayer(l_6, n_filt[2], filt[2])
    l_8 = L.DropoutLayer(l_7, p=dropout[2])
    l_9 = L.MaxPool2DLayer(l_8, pool[2])
    # 6
    l_10 = L.Conv2DLayer(l_9, n_filt[3], filt[3])
    l_11 = L.Conv2DLayer(l_10, n_filt[4], 1)

    l_12 = L.Conv2DLayer(l_11, n_filt[5], 1,
                         nonlinearity=las.nonlinearities.sigmoid)
    return l_in, l_12, fov


def build_ID_v0():
    fov = 40  # field of view = patch length
    n_channels = 2
    n_classes = 4
    filt = [7, 6, 6]
    n_filt = [20, 25, 60, 30, n_classes]
    pool = [2, 2]
    dropout = [0.5, 0.5]

    # init weights =

    # 40
    l_in = L.InputLayer((None, n_channels, fov, fov))

    # parallel 1
    l_1 = L.Conv2DLayer(l_in, n_filt[0], filt[0])
    l_2 = L.MaxPool2DLayer(l_1, pool[0])
    l_3 = L.DropoutLayer(l_2, p=dropout[0])
    # 17
    l_4 = L.Conv2DLayer(l_3, n_filt[1], filt[1])
    l_5 = L.MaxPool2DLayer(l_4, pool[1])
    l_6 = L.DropoutLayer(l_5, p=dropout[1])
    # 6
    l_7 = L.Conv2DLayer(l_6, n_filt[2], 5, filt[2])
    # 1

    # parallel 2
    l_slice = cs.SliceLayer(l_in)
    l_resh = las.layers.reshape(l_slice, (-1, 9, 1, 1))

    l_merge = las.layers.ConcatLayer([l_resh, l_7], axis=1)
    l_8 = L.Conv2DLayer(l_merge, n_filt[3], 1, W=gen_identity_filter([1,3,7,5]),
                        nonlinearity=las.nonlinearities.elu)
    l_9 = L.Conv2DLayer(l_8, n_filt[4], 1,
                        nonlinearity=las.nonlinearities.elu,
                        W=gen_identity_filter([0, 1, 2, 3]))
    return l_in, l_9, fov

def build_ID_v0_hydra():
    l_in, l_9, fov = build_ID_v0()
    l_in_direction = L.InputLayer((None, 1), input_var=T.matrix(dtype='int32'))
    l_10 = L.SliceLayer(l_9, indices=l_in_direction.input_var, axis=1)
    return l_in, l_in_direction, l_9, l_10, fov

def build_ID_v0_hybrid():
    fov = 40  # field of view = patch length
    n_channels = 2
    n_classes = 8
    filt = [7, 6, 6]
    n_filt = [20, 25, 60, 30, n_classes]
    pool = [2, 2]
    dropout = [0.5, 0.5]

    # init weights =

    # 40
    l_in = L.InputLayer((None, n_channels, fov, fov))

    # parallel 1
    l_1 = L.Conv2DLayer(l_in, n_filt[0], filt[0])
    l_2 = L.MaxPool2DLayer(l_1, pool[0])
    l_3 = L.DropoutLayer(l_2, p=dropout[0])
    # 17
    l_4 = L.Conv2DLayer(l_3, n_filt[1], filt[1])
    l_5 = L.MaxPool2DLayer(l_4, pool[1])
    l_6 = L.DropoutLayer(l_5, p=dropout[1])
    # 6
    l_7 = L.Conv2DLayer(l_6, n_filt[2], 5, filt[2])
    # 1

    # parallel 2
    l_slice = cs.SliceLayer(l_in)
    l_resh = las.layers.reshape(l_slice, (16, 9, 1, 1))

    l_merge = las.layers.ConcatLayer([l_resh, l_7], axis=1)
    l_8 = L.Conv2DLayer(l_merge, n_filt[3], 1, W=gen_identity_filter([1,3,7,5]),
                        nonlinearity=las.nonlinearities.elu)
    l_9 = L.Conv2DLayer(l_8, n_filt[4], 1,
                        nonlinearity=las.nonlinearities.rectify,
                        W=gen_identity_filter([0, 1, 2, 3]))
    return l_in, l_9, fov


def loss_updates_probs_v0(l_in, target, last_layer, L1_weight=10**-5):

    all_params = L.get_all_params(last_layer)

    l_out_train = L.get_output(last_layer, deterministic=False)
    l_out_valid = L.get_output(last_layer, deterministic=True)

    L1_norm = las.regularization.regularize_network_params(
        last_layer,
        las.regularization.l1)

    loss_train = T.mean(
        las.objectives.squared_error(l_out_train, target)) + \
                        L1_weight * L1_norm
    loss_valid = T.mean(
        las.objectives.squared_error(l_out_valid, target))

    updates = las.updates.adam(loss_train, all_params)

    loss_train_f = theano.function([l_in.input_var, target], loss_train,
                                   updates=updates)

    loss_valid_f = theano.function([l_in.input_var, target], loss_valid)
    probs_f = theano.function([l_in.input_var], l_out_valid)

    return loss_train_f, loss_valid_f, probs_f


def loss_updates_hydra_v0(l_in_data, l_in_direction, last_layer,
                          L1_weight=10**-5):

    all_params = L.get_all_params(last_layer)

    bs = l_in_data.input_var.shape[0]

    l_out_train = L.get_output(last_layer, deterministic=False)
    l_out_valid = L.get_output(last_layer, deterministic=True)

    L1_norm = las.regularization.regularize_network_params(
        last_layer,
        las.regularization.l1)


    loss_train = T.mean(
        las.objectives.squared_error(l_out_train[:bs/2], l_out_train[bs/2:])) + \
                        L1_weight * L1_norm
    loss_valid = T.mean(
        las.objectives.squared_error(l_out_train[:bs/2], l_out_train[bs/2:]))

    updates = las.updates.adam(loss_train, all_params)

    loss_train_f = theano.function([l_in_data.input_var, l_in_direction.input_var],
                                   loss_train,
                                   updates=updates)

    loss_valid_f = theano.function([l_in_data.input_var, l_in_direction.input_var],
                                   loss_valid)
    probs_f = theano.function([l_in_data.input_var, l_in_direction.input_var],
                              l_out_valid)

    return loss_train_f, loss_valid_f, probs_f


def loss_updates_probs_v1_hybrid(l_in, target, last_layer, L1_weight=10**-5):

    all_params = L.get_all_params(last_layer)

    l_out_train = L.get_output(last_layer, deterministic=False)
    l_out_valid = L.get_output(last_layer, deterministic=True)

    L1_norm = las.regularization.regularize_network_params(
        last_layer,
        las.regularization.l1)


    loss_train = \
        T.mean(0.01 * las.objectives.squared_error(
                            l_out_train[:, :4], target[:, :4]) + \
                las.objectives.squared_error(
                            l_out_train[:, 4:], target[:, 4:])) + \
                L1_weight * L1_norm
    loss_valid = T.mean(0.01 * las.objectives.squared_error(
                            l_out_train[:, :4], target[:, :4]) + \
                las.objectives.squared_error(
                            l_out_train[:, 4:], target[:, 4:]))

    updates = las.updates.adam(loss_train, all_params)

    loss_train_f = theano.function([l_in.input_var, target], loss_train,
                                   updates=updates)

    loss_valid_f = theano.function([l_in.input_var, target], loss_valid)
    probs_f = theano.function([l_in.input_var], l_out_valid)

    return loss_train_f, loss_valid_f, probs_f


def prob_funcs(l_in, last_layer):
    l_out_valid = L.get_output(last_layer, deterministic=True)
    probs_f = theano.function([l_in.input_var], l_out_valid)
    return probs_f


def gen_identity_filter(indices):
    def initializer(shape):
        W = np.random.normal(0, size=shape).astype(dtype=theano.config.floatX)
        W[range(len(indices)), indices, :, :] = 1.
        return W
    return initializer


if __name__ == '__main__':



    print gen_identity_filter((34, 60, 1, 1)).tolist()

