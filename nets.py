import numpy as np
import theano
import lasagne as las
from theano import tensor as T
from lasagne import layers as L
import custom_layer as cs


class NetBuilder():

    def __init__(self):
        self.net_name = None
        self.build_methods = \
            dict( (method,getattr(self, method)) \
                for method in dir(self) \
                if callable(getattr(self, method))\
                and method.startswith("build_"))

        self.loss_methods = \
            dict( (method,getattr(self, method)) \
                for method in dir(self) \
                if callable(getattr(self, method))\
                and method.startswith("loss_"))

    def get_net(self, netname):
        print 'building net: ', netname
        return self.build_methods["build_"+netname]

    def get_loss(self, lossname):
        return self.loss_methods["loss_"+lossname]

    def build_net_v0(self):
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


    def build_net_v1(self):
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


    def build_net_v5(self):
        fov = 40    # field of view = patch length
        n_channels = 4
        n_classes = 4
        filt = [7, 6, 6]
        n_filt = [20, 25, 60, 30, n_classes]
        pool = [2, 2]
        dropout = [0.5, 0.5]

        # 40
        l_in = L.InputLayer((None, n_channels, fov, fov))
        l_1 = L.Conv2DLayer(l_in, n_filt[0], filt[0],
                            nonlinearity=las.nonlinearities.rectify)
        l_2 = L.MaxPool2DLayer(l_1, pool[0])
        l_3 = L.DropoutLayer(l_2, p=dropout[0])
        # 17
        l_4 = L.Conv2DLayer(l_3, n_filt[1], filt[1],
                            nonlinearity=las.nonlinearities.rectify)
        l_5 = L.MaxPool2DLayer(l_4, pool[1])
        l_6 = L.DropoutLayer(l_5, p=dropout[1])
        # 6
        l_7 = L.Conv2DLayer(l_6, n_filt[2], 5, filt[2],
                            nonlinearity=las.nonlinearities.rectify)
        l_8 = L.Conv2DLayer(l_7, n_filt[3], 1,
                            nonlinearity=las.nonlinearities.rectify)
        l_9 = L.Conv2DLayer(l_8, n_filt[4], 1,
                            nonlinearity=las.nonlinearities.rectify,
                            b=np.random.random(4)*10+10.)
        return l_in, l_9, fov


    def build_net_v5_big(self):
        fov = 76    # field of view = patch length
        n_channels = 4
        n_classes = 4
        filt = [7, 6, 6, 5]
        n_filt = [20, 25, 60, 60, 30, n_classes]
        pool = [2, 2, 2]
        dropout = [0.5, 0.5, 0.5]

        # 80
        l_in = L.InputLayer((None, n_channels, fov, fov))
        l_1 = L.Conv2DLayer(l_in, n_filt[0], filt[0],
                            nonlinearity=las.nonlinearities.rectify)
        l_2 = L.MaxPool2DLayer(l_1, pool[0])
        l_3 = L.DropoutLayer(l_2, p=dropout[0])
        # 35
        l_4 = L.Conv2DLayer(l_3, n_filt[1], filt[1],
                            nonlinearity=las.nonlinearities.rectify)
        l_5 = L.MaxPool2DLayer(l_4, pool[1])
        l_6 = L.DropoutLayer(l_5, p=dropout[1])
        # 15
        l_7 = L.Conv2DLayer(l_6, n_filt[2], filt[2],
                            nonlinearity=las.nonlinearities.rectify)
        l_8 = L.MaxPool2DLayer(l_7, pool[2])
        l_9 = L.DropoutLayer(l_8, p=dropout[2])
        # 5
        l_10 = L.Conv2DLayer(l_9, n_filt[3], 5, filt[3],
                            nonlinearity=las.nonlinearities.rectify)
        # 1
        l_11 = L.Conv2DLayer(l_10, n_filt[4], 1,
                            nonlinearity=las.nonlinearities.rectify)
        l_12 = L.Conv2DLayer(l_11, n_filt[5], 1,
                            nonlinearity=las.nonlinearities.rectify,
                            b=np.random.random(4)*10+10.)
        return l_in, l_12, fov


    def build_net_v5_zstack(self):
        fov = 76    # field of view = patch length
        n_channels = 8# 2* claims +  3 * membrane + 3* raw
        n_classes = 4
        filt = [7, 6, 6, 5]
        n_filt = [20, 25, 60, 60, 30, n_classes]
        pool = [2, 2, 2]
        dropout = [0.5, 0.5, 0.5]

        # 80
        l_in = L.InputLayer((None, n_channels, fov, fov))
        l_1 = L.Conv2DLayer(l_in, n_filt[0], filt[0],
                            nonlinearity=las.nonlinearities.rectify)
        l_2 = L.MaxPool2DLayer(l_1, pool[0])
        l_3 = L.DropoutLayer(l_2, p=dropout[0])
        # 35
        l_4 = L.Conv2DLayer(l_3, n_filt[1], filt[1],
                            nonlinearity=las.nonlinearities.rectify)
        l_5 = L.MaxPool2DLayer(l_4, pool[1])
        l_6 = L.DropoutLayer(l_5, p=dropout[1])
        # 15
        l_7 = L.Conv2DLayer(l_6, n_filt[2], filt[2],
                            nonlinearity=las.nonlinearities.rectify)
        l_8 = L.MaxPool2DLayer(l_7, pool[2])
        l_9 = L.DropoutLayer(l_8, p=dropout[2])
        # 5
        l_10 = L.Conv2DLayer(l_9, n_filt[3], 5, filt[3],
                            nonlinearity=las.nonlinearities.rectify)
        # 1
        l_11 = L.Conv2DLayer(l_10, n_filt[4], 1,
                            nonlinearity=las.nonlinearities.rectify)
        l_12 = L.Conv2DLayer(l_11, n_filt[5], 1,
                            nonlinearity=las.nonlinearities.rectify,
                            b=np.random.random(4)*10+10.)
        return l_in, l_12, fov

    def build_net_v5_BN(self, n_classes = 4,
                              n_channels = 4,
                              n_filt = [20, 25, 60, 30]):
        fov = 40    # field of view = patch length
        filt = [7, 6, 6]
        n_filt += [n_classes]
        pool = [2, 2]

        # 40
        l_in = L.InputLayer((None, n_channels, fov, fov))
        l_1 = L.batch_norm(L.Conv2DLayer(l_in, n_filt[0], filt[0]))

        l_2 = L.MaxPool2DLayer(l_1, pool[0])
        # 17
        l_3 = L.batch_norm(L.Conv2DLayer(l_2, n_filt[1], filt[1]))
        l_4 = L.MaxPool2DLayer(l_3, pool[1])
        # 6
        l_5 = L.Conv2DLayer(l_4, n_filt[2], 5, filt[2],
                            nonlinearity=las.nonlinearities.rectify)
        l_6 = L.Conv2DLayer(l_5, n_filt[3], 1,
                            nonlinearity=las.nonlinearities.rectify)
        l_7 = L.Conv2DLayer(l_6, n_filt[4], 1,
                            nonlinearity=las.nonlinearities.rectify,
                            b=np.random.random(n_classes)*10+10.)
        return l_in, l_7, fov

    def build_net_v6_BN(self, n_classes = 4,
                              n_channels = 4,
                              n_filt = [20, 25, 60, 30]):
        fov = 30
        filt = [3, 5, 5]
        n_filt += [n_classes]
        pool = [2, 2]
        # 40
        l_in = L.InputLayer((None, n_channels, fov, fov))
        l_1 = L.batch_norm(L.Conv2DLayer(l_in, n_filt[0], filt[0]))

        l_2 = L.MaxPool2DLayer(l_1, pool[0])
        # 17
        l_3 = L.batch_norm(L.Conv2DLayer(l_2, n_filt[1], filt[1]))
        l_4 = L.MaxPool2DLayer(l_3, pool[1])
        # 6
        l_5 = L.Conv2DLayer(l_4, n_filt[2], 5, filt[2],
                            nonlinearity=las.nonlinearities.rectify)
        l_6 = L.Conv2DLayer(l_5, n_filt[3], 1,
                            nonlinearity=las.nonlinearities.rectify)
        l_7 = L.Conv2DLayer(l_6, n_filt[4], 1,
                            nonlinearity=las.nonlinearities.rectify,
                            b=np.random.random(n_classes)*10+10.)
        return l_in, l_7, fov

    def build_ID_v0(self):
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


    def build_ID_v1_multichannel(self):
        fov = 40  # field of view = patch length
        n_channels = 4
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


    def build_ID_v0_hydra(self):
        l_in, l_9, fov = self.build_ID_v0()
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov


    def build_ID_v01_hydra(self):
        l_in, l_9, fov = self.build_ID_v1_multichannel()
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov


    def build_ID_v5_hydra(self):
        l_in, l_9, fov = self.build_net_v5()
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov


    def build_ID_v5_hydra_big(self):
        l_in, l_9, fov = self.build_net_v5_big()
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov


    def build_ID_v5_hydra_zstack_o(self):
        l_in, l_9, fov = self.build_net_v5_BN(n_channels=8)
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov

    def build_ID_v5_hydra_zstack_down(self):
        l_in, l_9, fov = self.build_net_v5_BN(n_channels=10)
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov

    def build_ID_v5_hydra_down(self):
        l_in, l_9, fov = self.build_net_v5_BN(n_channels=4)
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov

    def build_ID_v6_hydra_zstack_down_big(self):
        l_in, l_9, fov = self.build_net_v5_BN(n_channels=10,n_filt = [30, 40, 60, 30])
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov

    def build_ID_v6_hydra_zstack_down(self):
        l_in, l_9, fov = self.build_net_v6_BN(n_channels=10)
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov
    

    def build_ID_v5_hydra_BN(self):
        l_in, l_9, fov = self.build_net_v5_BN()
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov

    def build_ID_v5_hydra_BN_grad(self):
        l_in, l_9, fov = self.build_net_v5_BN(n_classes=3)
        l_9_height = cs.GradientToHeight(l_9)
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9_height, l_in_direction])
        return l_in, l_in_direction, l_9_height, l_10, fov

    def build_ID_v0_hybrid(self):
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
        l_8 = L.Conv2DLayer(l_merge, n_filt[3], 1, W=self.gen_identity_filter([1,3,7,5]),
                            nonlinearity=las.nonlinearities.elu)
        l_9 = L.Conv2DLayer(l_8, n_filt[4], 1,
                            nonlinearity=las.nonlinearities.rectify,
                            W=self.gen_identity_filter([0, 1, 2, 3]))
        return l_in, l_9, fov


    def build_ID_v1_hybrid(self):
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

        l_8 = L.Conv2DLayer(l_7, n_filt[3], 1,
                            nonlinearity=las.nonlinearities.elu)
        l_9 = L.Conv2DLayer(l_8, n_filt[4], 1,
                            nonlinearity=las.nonlinearities.rectify)
        l_10 = L.Conv2DLayer(l_8, n_filt[4], 1,
                            nonlinearity=las.nonlinearities.sigmoid)
        return l_in, (l_9, l_10), fov


    def loss_updates_probs_v0(self, l_in, target, last_layer, L1_weight=10**-5,
                              update='adam'):

        all_params = L.get_all_params(last_layer, trainable=True)

        l_out_train = L.get_output(last_layer, deterministic=False)
        l_out_valid = L.get_output(last_layer, deterministic=True)

        L1_norm = las.regularization.regularize_network_params(
                last_layer,
                las.regularization.l1)

        loss_individual_batch = (l_out_train - target)**2

        loss_train = T.mean(loss_individual_batch)
        if L1_weight > 0:
            loss_train +=  L1_weight * L1_norm
        loss_valid = T.mean(loss_individual_batch)
        if update == 'adam':
            updates = las.updates.adam(loss_train, all_params)
        if update == 'sgd':
            updates = las.updates.sgd(loss_train, all_params, 0.0001)
        loss_train_f = theano.function([l_in.input_var, target],
                                       [loss_train, loss_individual_batch],
                                       updates=updates)

        loss_valid_f = theano.function([l_in.input_var, target], loss_valid)
        probs_f = theano.function([l_in.input_var], l_out_valid)

        return loss_train_f, loss_valid_f, probs_f


    def loss_updates_hydra_v0(self, l_in_data, l_in_direction, last_layer,
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


    def loss_updates_hydra_v5(self, l_in_data, l_in_direction, last_layer,
                              L1_weight=10**-5, margin=0):

        all_params = L.get_all_params(last_layer, trainable=True)

        bs = l_in_data.input_var.shape[0]

        l_out_train = L.get_output(last_layer, deterministic=False)
        l_out_valid = L.get_output(last_layer, deterministic=True)

        L1_norm = \
            las.regularization.regularize_network_params(last_layer,
                                                         las.regularization.l1)

        # typeII - typeI + m
        individual_batch = (l_out_train[bs/2:] - l_out_train[:bs/2] + margin)**2
        loss_train = T.mean(individual_batch)
        if L1_weight > 0:
            loss_train += L1_weight * L1_norm

        loss_valid = T.mean(individual_batch)

        updates = las.updates.adam(loss_train, all_params)

        # theano funcs
        loss_train_f = theano.function([l_in_data.input_var,
                                        l_in_direction.input_var],
                                       [loss_train, individual_batch],
                                       updates=updates)
        loss_valid_f = theano.function([l_in_data.input_var,
                                        l_in_direction.input_var],
                                       loss_valid)
        probs_f = theano.function([l_in_data.input_var, l_in_direction.input_var],
                                  l_out_valid)

        return loss_train_f, loss_valid_f, probs_f


    def loss_updates_probs_v1_hybrid(self, l_in, target, last_layer, L1_weight=10**-5):

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


    def loss_updates_probs_v2_hybrid(self, l_in, target, last_layers, L1_weight=10**-5):

        l_height, l_id = last_layers
        l_merge = las.layers.MergeLayer([l_height, l_id])
        all_params = L.get_all_params(l_merge)

        l_out_train_height = L.get_output(l_height, deterministic=False)
        l_out_train_id = L.get_output(l_id, deterministic=False)
        l_out_valid_height = L.get_output(l_height, deterministic=True)
        l_out_valid_id = L.get_output(l_id, deterministic=True)

        L1_norm = las.regularization.regularize_network_params(
            l_merge,
            las.regularization.l1)


        loss_train = \
            T.mean(las.objectives.squared_error(
                                        l_out_train_height, target[:, :4]) + \
                    50 * las.objectives.squared_error(l_out_train_id, target[:, 4:])) + \
                        L1_weight * L1_norm
        loss_valid = T.mean(las.objectives.squared_error(
                                        l_out_valid_height, target[:, :4]) + \
                    50 * las.objectives.squared_error(
                                        l_out_valid_id, target[:, 4:]))

        updates = las.updates.adam(loss_train, all_params)

        loss_train_f = theano.function([l_in.input_var, target], loss_train,
                                       updates=updates)

        loss_valid_f = theano.function([l_in.input_var, target], loss_valid)
        probs_f = theano.function([l_in.input_var], l_out_valid_height)

        return loss_train_f, loss_valid_f, probs_f

    def gen_identity_filter(self, indices):
        def initializer(shape):
            W = np.random.normal(0, size=shape).astype(dtype=theano.config.floatX)
            W[range(len(indices)), indices, :, :] = 1.
            return W
        return initializer

def prob_funcs(l_in, last_layer):
    l_out_valid = L.get_output(last_layer, deterministic=True)
    probs_f = theano.function([l_in.input_var], l_out_valid)
    return probs_f


def prob_funcs_hybrid(l_in, last_layers):
    l_height, l_id = last_layers
    l_out_valid = L.get_output(l_height, deterministic=True)
    probs_f = theano.function([l_in.input_var], l_out_valid)
    return probs_f




if __name__ == '__main__':

    l_in, l_in_direction, l_9, l_10, fov = self.build_ID_v0_hydra()
    las.layers.CustomRecurrentLayer()

