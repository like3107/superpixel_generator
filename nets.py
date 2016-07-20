import numpy as np
import theano
import lasagne as las
from theano import tensor as T
from lasagne import layers as L
from theano.sandbox import cuda as c


def genIdentityFilter(num_filters,num_input_channels,select_input_channel,select_output_channel,size):
    filter_rows = size
    filter_columns = size
    def initializer(shape):
        print shape
        W = np.zeros((num_filters, num_input_channels, filter_rows, filter_columns))
        W[select_output_channel,select_input_channel,size/2,size/2] = 1
        return W
    return initializer

def build_net_v0():
    filt = [7, 7, 5]
    n_filt = [20, 25, 60, 30, 4]
    pool = [2, 2]
    dropout = [0.5, 0.5]

    l_in = L.InputLayer((None, 3, 40, 40))
    l_1 = L.Conv2DLayer(l_in, n_filt[0], filt[0],W=genIdentityFilter(n_filt[0],
                                                    L.get_output_shape(l_in)[1],
                                                    0,
                                                    0,filt[0]))
    l_2 = L.DropoutLayer(l_1, p=dropout[0])
    l_3 = L.MaxPool2DLayer(l_2, pool[0])
    l_4 = L.Conv2DLayer(l_3, n_filt[1], filt[1],W=genIdentityFilter(n_filt[1],
                                                    L.get_output_shape(l_3)[1],
                                                    0,
                                                    0,filt[1]))
    l_5 = L.DropoutLayer(l_4, p=dropout[1])
    l_6 = L.MaxPool2DLayer(l_5, pool[1])
    # print L.get_output_shape(l_6)
    # print 
    # print genIdentityFilter(n_filt[2],L.get_output_shape(l_6)[1],
    #                                                 0,
    #                                                 0,filt[2]).shape
    # print L.get_output_shape(L.Conv2DLayer(l_6, n_filt[2], filt[2]))
    # print "WShape",L.Conv2DLayer(l_6, n_filt[2], filt[2],W=dummy).W.get_value().shape
    # l_7 = L.Conv2DLayer(l_6, n_filt[2], filt[2],W=genIdentityFilter(n_filt[2],
    l_7 = L.Conv2DLayer(l_6, n_filt[2], filt[2],W=genIdentityFilter(n_filt[2],
                                                    L.get_output_shape(l_6)[1],
                                                    0,
                                                    0,filt[2]))
    l_8 = L.Conv2DLayer(l_7, n_filt[3], 1,W=genIdentityFilter(n_filt[3],
                                                    L.get_output_shape(l_7)[1],
                                                    0,
                                                    0,1))

    l_9 = L.Conv2DLayer(l_8, n_filt[4], 1,
                        nonlinearity=las.nonlinearities.elu)

    # print L.get_output_shape(l_7)
    # print L.get_output_shape(l_8)
    # print L.get_output_shape(l_9)

    return l_in, l_9


def loss_updates_probs_v0(l_in, target, last_layer, L1_weight=10**-3):

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


