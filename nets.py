import numpy as np
import theano
import lasagne as las
from theano import tensor as T
from lasagne import layers as L
import custom_layer as cs
import utils as u


class NetBuilder:
    def __init__(self, options=None):
        self.net_name = None
        self.options = options

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

    def get_fov(self, netname):
        return self.build_methods["build_"+netname]()[-2]

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
                            nonlinearity=las.nonlinearities.rectify)
                            # b=np.random.random(n_classes)*10+10.)
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

    def build_net_v7_EAT(self, n_classes = 4,
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
                            b=np.random.random(n_classes)*10)

        l_6_eat = L.Conv2DLayer(l_5, n_filt[3], 1,
                            nonlinearity=las.nonlinearities.rectify)
        l_7_eat = L.Conv2DLayer(l_6_eat, n_filt[4], 1,
                            nonlinearity=las.nonlinearities.sigmoid)

        return l_in, l_7, l_7_eat, fov


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
                            nonlinearity=las.nonlinearities.rectify,
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
                            nonlinearity=las.nonlinearities.rectify,
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
        return l_in, l_in_direction, l_9, l_10, fov, None


    def build_ID_v5_hydra(self):
        l_in, l_9, fov = self.build_net_v5()
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov, None

    def build_net_v8_dilated(self, l_image_in=None):
        n_channels = self.options.network_channels
        if n_channels is None:
            n_channels = 1
        fov = None
        n_classes = 4
        filts =         [4,     3,      3,      3,      3,      3,      3,      1,      1]
        dils  =         [1,     1,      2,      4,      8,      16,     1,      1,      1]
        n_filts =       [32,    32,     64,    64,    128,    128,    256,    2048, n_classes]
        # debug
        batch_norms =   [True, True,   True,   True,   True,   True,   True,   False,   False]
        regs        =   [True, True,   True,   True,   True,   True,   True,   True,   False]
        # batch_norms = [False] * len(n_filt)
        ELU = las.nonlinearities.elu
        ReLU = las.nonlinearities.rectify
        act_fcts =      [ELU,  ELU,     ELU,    ELU,    ELU,    ELU,    ELU,    ReLU,   ReLU]
        names       =   ['conv','conv','conv','conv','conv', 'conv', 'conv', 'fc','fc']

        if l_image_in is None:
            l_in = L.InputLayer((None, n_channels, fov, fov))
        else:
            l_in = L.InputLayer((None, n_channels, fov, fov), input_var=l_image_in)
        l_prev = l_in
        i = 0
        l_seconds_last = None
        for filt, dil, n_filt, batch_norm, act_fct, reg, name in \
                zip(filts, dils, n_filts, batch_norms, act_fcts, regs, names):
            i += 1
            l_seconds_last = l_prev
            if batch_norm:
                l_next = L.batch_norm(
                    L.DilatedConv2DLayer(l_prev, n_filt, filt,
                                         dilation=(dil, dil), name=name,
                                         nonlinearity=act_fct))
            else:
                l_next = L.DilatedConv2DLayer(l_prev, n_filt, filt,
                                              dilation=(dil, dil), name=name,
                                              nonlinearity=act_fct)
            if not reg:
                l_next.params[l_next.W].remove('regularizable')
            l_prev = l_next
        fov = 68

        return l_in, (l_prev, l_seconds_last), fov

    def build_net_v8_dilated_ft(self):

        l_in_old, l_in_direction, l_out_old, l_out_direction, _, _ =\
            self.build_v8_hydra_dilated()

        u.load_network(self.options.load_init_net_path, l_out_old)
        # get pointer to last conv layer
        l_out_precomp = l_out_old
        while 'conv' not in l_out_precomp.name:
            l_out_precomp = l_out_precomp.input_layer

        # pointer to first fc layer
        first_fc = l_out_old
        while 'fc' in first_fc.name:
            previous = first_fc
            first_fc = first_fc.input_layer
        first_fc = previous

        n_channels = 2

        n_classes = 4
        fov = None
        filts =     [4, 3, 3, 8]
        dils =      [4, 8, 16, 1]
        n_filts =   [32, 32, 64, 64]
        names =     ['ft', 'ft', 'ft', 'ft']
        l_in = L.InputLayer((None, n_channels, fov, fov))
        l_prev = l_in
        for filt, dil, n_filt, name in zip(filts, dils, n_filts, names):
            # debug
            l_next = L.batch_norm(L.DilatedConv2DLayer(l_prev, n_filt, filt,
                                                       dilation=(dil, dil)),
                                  name=name)
            # l_next = L.DilatedConv2DLayer(l_prev, n_filt, filt,
            #                                            dilation=(dil, dil))
            l_prev = l_next
        l_claims_out = l_prev

        l_in_dense = las.layers.InputLayer((None, 256, 1, 1))
        l_merge = las.layers.ConcatLayer([l_in_dense, l_claims_out])

        W = np.random.random((2048, 320, 1, 1)).astype('float32') / 10000.
        W[:, :-64, 0, 0] = np.array(first_fc.W.eval()).swapaxes(0, 1)[:, :, 0, 0]
        W = theano.shared(W)

        fc_1 = las.layers.Conv2DLayer(l_merge, 2048, filter_size=1,
                                       name='fc', W=W, b=first_fc.b,
                                      nonlinearity=las.nonlinearities.rectify)
        l_out_cross = las.layers.Conv2DLayer(
                                fc_1, 4, filter_size=1,
                                name='fc',
                                W=l_out_old.W.dimshuffle(1,0,2,3),
                                b=l_out_old.b,
                                nonlinearity = las.nonlinearities.rectify)
        l_out_cross.params[l_out_cross.W].remove('regularizable')
        layers = {}
        layers['l_in_claims'] = l_in
        layers['l_in_precomp'] = l_in_old
        layers['l_in_dense'] = l_in_dense
        layers['l_in_old'] = l_in_old
        layers['l_out_cross'] = l_out_cross
        layers['l_out_precomp'] = l_out_precomp
        layers['l_claims_out'] = l_claims_out

        # print 'b new', l_out_cross.b.shape.eval()
        # print 'W new', l_out_cross.W.shape.eval()
        #
        # print 'b old', first_fc.b.eval()
        # print 'old out', first_fc.W.shape.eval()
        # print 'old out', first_fc.output_shape
        #
        # print 'lout', l_out_old.W.shape.eval()
        # print 'lout', l_out_old.b.shape.eval()
        # print 'lout', l_out_old.output_shape

        # print 'l_prev', l_prev.output_shape
        # print 'l_prev', l_prev.input_layer.input_layer.W.shape.eval()
        # print 'l_prev', l_prev.input_layer.beta.shape.eval()
        #
        # print 'last_conv', last_conv.output_shape
        # print 'last_conv', last_conv.input_layer.input_layer.W.shape.eval()
        #
        # print 'BN beta', last_conv.input_layer.beta.shape.eval()
        # print 'BN gamma', last_conv.input_layer.gamma.shape.eval()
        fov = 68
        return layers, fov


    def build_net_v8_hydra_dilated_ft_joint(self, l_image_in = None,
                                            l_claims_in = None):
        print 'building joint net'
        l_in_old, l_in_direction, l_out_old, l_out_direction, _, _ =\
            self.build_v8_hydra_dilated(l_image_in = l_image_in)

        u.load_network(self.options.load_init_net_path, l_out_old)
        # get pointer to last conv layer
        l_out_precomp = l_out_old
        while 'conv' not in l_out_precomp.name:
            l_out_precomp = l_out_precomp.input_layer

        # pointer to first fc layer
        first_fc = l_out_old
        while 'fc' in first_fc.name:
            previous = first_fc
            first_fc = first_fc.input_layer
        first_fc = previous

        n_channels = 2

        n_classes = 4
        fov = None
        filts =     [4, 3, 3, 8]
        dils =      [4, 8, 16, 1]
        n_filts =   [32, 32, 64, 64]
        names =     ['ft', 'ft', 'ft', 'ft']
        if l_claims_in is None:
            l_in = L.InputLayer((None, n_channels, fov, fov))
        else:
            l_in = L.InputLayer(shape=(None, n_channels, fov, fov),
                                input_var = l_claims_in)
        l_prev = l_in
        for filt, dil, n_filt, name in zip(filts, dils, n_filts, names):
            l_next = L.batch_norm(L.DilatedConv2DLayer(l_prev, n_filt, filt,
                                                       dilation=(dil, dil)),
                                  name=name)
            # debug
            # l_next = L.DilatedConv2DLayer(l_prev, n_filt, filt,
            #                                            dilation=(dil, dil))
            l_prev = l_next
        l_claims_out = l_prev

        # l_in_dense = las.layers.InputLayer((None, 256, 1, 1))
        l_merge = las.layers.ConcatLayer([l_out_precomp, l_claims_out])

        W = np.random.random((2048, 320, 1, 1)).astype('float32') / 10000.
        W[:, :-64, 0, 0] = np.array(first_fc.W.eval()).swapaxes(0, 1)[:, :, 0, 0]
        W = theano.shared(W)

        fc_1 = las.layers.Conv2DLayer(l_merge, 2048, filter_size=1,
                                       name='fc', W=W, b=first_fc.b,
                                      nonlinearity=las.nonlinearities.rectify)
        l_out_cross = las.layers.Conv2DLayer(
                                fc_1, 4, filter_size=1,
                                name='fc',
                                W=l_out_old.W.dimshuffle(1,0,2,3),
                                b=l_out_old.b,
                                nonlinearity = las.nonlinearities.rectify)

        l_out_cross.params[l_out_cross.W].remove('regularizable')
        layers = {}
        layers['l_in_claims'] = l_in
        layers['l_in_precomp'] = l_in_old
        # layers['l_in_dense'] = l_in_dense
        layers['l_in_old'] = l_in_old
        layers['l_out_cross'] = l_out_cross
        layers['l_out_precomp'] = l_out_precomp
        layers['l_claims_out'] = l_claims_out
        layers['l_merge'] = l_merge

        # print 'b new', l_out_cross.b.shape.eval()
        # print 'W new', l_out_cross.W.shape.eval()
        #
        # print 'b old', first_fc.b.eval()
        # print 'old out', first_fc.W.shape.eval()
        # print 'old out', first_fc.output_shape
        #
        # print 'lout', l_out_old.W.shape.eval()
        # print 'lout', l_out_old.b.shape.eval()
        # print 'lout', l_out_old.output_shape

        # print 'l_prev', l_prev.output_shape
        # print 'l_prev', l_prev.input_layer.input_layer.W.shape.eval()
        # print 'l_prev', l_prev.input_layer.beta.shape.eval()
        #
        # print 'last_conv', last_conv.output_shape
        # print 'last_conv', last_conv.input_layer.input_layer.W.shape.eval()
        #
        # print 'BN beta', last_conv.input_layer.beta.shape.eval()
        # print 'BN gamma', last_conv.input_layer.gamma.shape.eval()
        fov = 68
        return layers, fov


    def build_v8_hydra_dilated_ft(self):
        layers, fov = \
            self.build_net_v8_dilated_ft()
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_single_out = cs.BatchChannelSlicer([layers['l_out_cross'],
                                              l_in_direction])
        layers['l_in_direction'] = l_in_direction
        layers['l_single_out'] = l_single_out
        return layers, fov, None


    def build_v8_hydra_dilated_ft_joint(self, l_image_in = None, l_claims_in = None):
        layers, fov = \
            self.build_net_v8_hydra_dilated_ft_joint(l_image_in = l_image_in, l_claims_in = l_claims_in)
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_single_out = cs.BatchChannelSlicer([layers['l_out_cross'],
                                              l_in_direction])
        layers['l_in_direction'] = l_in_direction
        layers['l_single_out'] = l_single_out
        return layers, fov, None

    def build_ID_v5_hydra_big(self):
        l_in, l_9, fov = self.build_net_v5_big()
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov, None


    def build_ID_v5_hydra_zstack_o(self):
        l_in, l_9, fov = self.build_net_v5_BN(n_channels=8)
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov, None

    def build_ID_v5_hydra_down(self):
        l_in, l_9, fov = self.build_net_v5_BN(n_channels=6)
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov, None

    def build_ID_v5_hydra_zstack_down(self):
        l_in, l_9, fov = self.build_net_v5_BN(n_channels=10)
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov, None

    def build_ID_v6_hydra_zstack_down_big(self):
        l_in, l_9, fov = self.build_net_v5_BN(n_channels=10,n_filt = [30, 40, 60, 30])
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov, None

    def build_ID_v6_hydra_zstack_down(self):
        l_in, l_9, fov = self.build_net_v6_BN(n_channels=10)
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov, None

    def build_v8_hydra_dilated(self, l_image_in=None):
        l_in, (l_9, l_seconds_last), fov = self.build_net_v8_dilated(l_image_in=l_image_in)
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov, None

    def build_ID_v5_hydra_BN(self):
        l_in, l_9, fov = self.build_net_v5_BN()
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov, None

    def build_ID_v5_hydra_BN_grad(self):
        l_in, l_9, fov = self.build_net_v5_BN(n_classes=3)
        l_9_height = cs.GradientToHeight(l_9)
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9_height, l_in_direction])
        return l_in, l_in_direction, l_9_height, l_10, fov, None

    def build_ID_v7_down_EAT_BN(self):
        l_in, l_9, l_eat, fov = self.build_net_v7_EAT(n_channels=6)
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov, l_eat

    def build_ID_v7_zstack_down_EAT_BN(self):
        l_in, l_9, l_eat, fov = self.build_net_v7_EAT(n_channels=10)
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov, l_eat

    def build_ID_v8_EAT_BN(self):
        l_in, l_9, l_eat, fov = self.build_net_v7_EAT(n_channels=3)
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov, l_eat

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
            loss_train += L1_weight * L1_norm
        loss_valid = T.mean(loss_individual_batch)
        if update == 'adam':
            updates = las.updates.adam(loss_train, all_params)
        if update == 'sgd':
            updates = las.updates.sgd(loss_train, all_params, 0.0001)
        loss_train_f = theano.function([l_in.input_var, target],
                                       [loss_train, loss_individual_batch,
                                        l_out_train],
                                       updates=updates)

        loss_valid_f = theano.function([l_in.input_var, target], loss_valid)
        probs_f = theano.function([l_in.input_var], l_out_valid)

        return loss_train_f, loss_valid_f, probs_f

    def loss_updates_v7_EAT(self, l_in, target, last_layer, eat_in, eat_gt, eat_factors, L1_weight=10**-5,
                              update='adam'):

        all_params = L.get_all_params(last_layer, trainable=True)
        all_params_merge = L.get_all_params(eat_in, trainable=True)
        print all_params_merge
        [all_params_merge.remove(p) for p in all_params if p in all_params_merge]
        print all_params_merge

        l_out_train = L.get_output(last_layer, deterministic=False)
        l_out_valid = L.get_output(last_layer, deterministic=True)
        l_out_train_eat = L.get_output(eat_in, deterministic=False)
        l_out_valid_eat = L.get_output(eat_in, deterministic=True)

        L1_norm = las.regularization.regularize_network_params(
                last_layer,
                las.regularization.l1)

        loss_individual_batch = (l_out_train - target)**2
        loss_merging_batch = eat_factors*((l_out_train_eat - eat_gt)**2)

        loss_train = T.mean(loss_individual_batch)
        loss_merge = T.sum(loss_merging_batch)/T.sum(eat_factors)

        if L1_weight > 0:
            loss_train +=  L1_weight * L1_norm
        loss_valid = T.mean(loss_individual_batch)
        if update == 'adam':
            updates = las.updates.adam(loss_train, all_params)
        if update == 'sgd':
            updates = las.updates.sgd(loss_train, all_params, 0.0001)

        if update == 'adam':
            updates_merge = las.updates.adam(loss_merge, all_params_merge)
        if update == 'sgd':
            updates_merge = las.updates.sgd(loss_merge, all_params_merge, 0.0001)

        loss_train_f = theano.function([l_in.input_var, target],
                                       [loss_train, loss_individual_batch],
                                       updates=updates)
        loss_merge_f = theano.function([l_in.input_var, eat_gt, eat_factors],
                                       [loss_merge, loss_merging_batch],
                                       updates=updates_merge)

        loss_valid_f = theano.function([l_in.input_var, target], loss_valid)
        probs_f = theano.function([l_in.input_var], l_out_valid)
        eat_f = theano.function([l_in.input_var], l_out_valid_eat)

        return loss_train_f, loss_valid_f, probs_f, loss_merge_f, eat_f

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
        probs_f = theano.function([l_in.input_var], l_out_valid)

        return loss_train_f, loss_valid_f, probs_f


    def loss_updates_hydra_v5(self, l_in_data, l_in_direction, last_layer, l_out_cross,
                              L1_weight=10**-5, margin=0):

        all_params = L.get_all_params(last_layer, trainable=True)

        bs = l_in_data.input_var.shape[0]

        l_out_train = L.get_output(last_layer, deterministic=False)
        l_out_valid = L.get_output(last_layer, deterministic=True)
        l_out_prediciton = L.get_output(l_out_cross, deterministic=True)

        L1_norm = \
            las.regularization.regularize_network_params(last_layer,
                                                         las.regularization.l1)

        # typeII - typeI + m
        individual_batch = (l_out_train[bs/2:] - l_out_train[:bs/2] + margin)
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
        probs_f = theano.function([l_in_data.input_var], l_out_prediciton)

        return loss_train_f, loss_valid_f, probs_f

    def loss_updates_hydra_v8(self, layers,
                              L1_weight=10**-5, margin=0):

        all_params = L.get_all_params(layers['l_out_cross'], trainable=True)

        bs = layers['l_in_claims'].input_var.shape[0]
        # debug
        l_out_train = L.get_output(layers['l_single_out'], deterministic=True)
        l_out_prediciton = L.get_output(layers['l_out_cross'],
                                        deterministic=True)
        l_out_single_debug = L.get_output(layers['l_single_out'],
                                        deterministic=True)

        L1_norm = \
            las.regularization.regularize_network_params(layers['l_out_cross'],
                                                         las.regularization.l1)

        # typeII - typeI + m
        individual_batch = (l_out_train[bs/2:] - l_out_train[:bs/2] + margin)
        loss_valid = T.mean(individual_batch)
        if L1_weight > 0:
            loss_train = loss_valid + L1_weight * L1_norm

        print "using nesterov_momentum"
        # updates = las.updates.adam(loss_train, all_params)
        updates = las.updates.nesterov_momentum(loss_train, all_params, 0.001)

        # theano funcs
        # precompute convs on raw till dense layer
        out_precomp = las.layers.get_output(layers['l_out_precomp'],
                                            deterministic=True)
        fc_prec_conv_body = \
            theano.function([layers['l_in_precomp'].input_var],
                            out_precomp)


        if self.options.net_arch == 'v8_hydra_dilated_ft_joint':
            print 'joint loss used'

            if self.options.fc_prec:
                loss_train_f = theano.function([layers['l_in_claims'].input_var,
                                                layers['l_in_old'].input_var,
                                                layers[
                                                    'l_in_direction'].input_var],
                                               [loss_train, individual_batch,
                                                l_out_prediciton, l_out_train],
                                               updates=updates)
                print 'fc prec'
                l_in_from_prec = las.layers.InputLayer((None, 64, 1, 1))
                layers['l_merge'].input_layers[0] = l_in_from_prec
                layers['l_merge'].input_shapes[0] = l_in_from_prec.output_shape
                l_out_prediciton_prec = L.get_output(layers['l_out_cross'],
                                                     deterministic=True)
                probs_f = theano.function([layers['l_in_claims'].input_var,
                                           l_in_from_prec.input_var],
                                          l_out_prediciton_prec)
            else:
                loss_train_f = theano.function([layers['l_in_claims'].input_var,
                                                layers['l_in_old'].input_var,
                                                layers[
                                                    'l_in_direction'].input_var],
                                               [loss_train, individual_batch,
                                                l_out_prediciton, l_out_train],
                                               updates=updates)
                probs_f = theano.function([layers['l_in_claims'].input_var,
                                           layers['l_in_old'].input_var],
                                          l_out_prediciton)

            claim_out, debug_f, debug_singe_out = (None, None, None)
        else:
            # l_in_dense is output of precomputed fc_prec_conv_body
            loss_train_f = theano.function([layers['l_in_claims'].input_var,
                                            layers['l_in_dense'].input_var,
                                            layers['l_in_direction'].input_var],
                                           [loss_train, individual_batch,
                                            l_out_prediciton, l_out_train],
                                           updates=updates)
            loss_valid_f = theano.function([layers['l_in_claims'].input_var,
                                            layers['l_in_dense'].input_var,
                                            layers['l_in_direction'].input_var],
                                           loss_valid)
            probs_f = theano.function([layers['l_in_claims'].input_var,
                                       layers['l_in_dense'].input_var],
                                      l_out_prediciton)
            claim_out = las.layers.get_output(layers['l_claims_out'])
            debug_f = theano.function([layers['l_in_claims'].input_var],
                                      claim_out)
            debug_singe_out = theano.function([layers['l_in_claims'].input_var,
                                               layers['l_in_dense'].input_var,
                                               layers['l_in_direction'].input_var],
                                              [l_out_single_debug,
                                               l_out_prediciton])

        return probs_f, fc_prec_conv_body, loss_train_f, debug_f, debug_singe_out


    def loss_updates_hydra_coldwar(self, layers,
                              L1_weight=10**-5, margin=0):

        all_params = L.get_all_params(layers['l_out_cross'], trainable=True)

        bs = layers['l_in_claims'].input_var.shape[0]
        # debug
        l_out_train = L.get_output(layers['l_single_out'], deterministic=True)
        l_out_prediciton = L.get_output(layers['l_out_cross'],
                                        deterministic=True)
        l_out_single_debug = L.get_output(layers['l_single_out'],
                                        deterministic=True)

        L1_norm = \
            las.regularization.regularize_network_params(layers['l_out_cross'],
                                                         las.regularization.l1)

        # typeII - typeI + m
        individual_batch = -l_out_train[:bs/2]
        loss_valid = T.mean(individual_batch)
        if L1_weight > 0:
            loss_train = loss_valid + L1_weight * L1_norm

        updates = las.updates.adam(loss_train, all_params)

        # theano funcs
        # precompute convs on raw till dense layer
        out_precomp = las.layers.get_output(layers['l_out_precomp'])
        fc_prec_conv_body = \
            theano.function([layers['l_in_precomp'].input_var],
                            out_precomp)

        # l_in_dense is output of precomputed fc_prec_conv_body
        loss_train_f = theano.function([layers['l_in_claims'].input_var,
                                        layers['l_in_dense'].input_var,
                                        layers['l_in_direction'].input_var],
                                       [loss_train, individual_batch,
                                        l_out_prediciton, l_out_train],
                                       updates=updates)
        loss_valid_f = theano.function([layers['l_in_claims'].input_var,
                                        layers['l_in_dense'].input_var,
                                        layers['l_in_direction'].input_var],
                                       loss_valid)
        probs_f = theano.function([layers['l_in_claims'].input_var,
                                   layers['l_in_dense'].input_var],
                                  l_out_prediciton)
        claim_out = las.layers.get_output(layers['l_claims_out'])
        debug_f = theano.function([layers['l_in_claims'].input_var],
                                  claim_out)
        debug_singe_out = theano.function([layers['l_in_claims'].input_var,
                                           layers['l_in_dense'].input_var,
                                           layers['l_in_direction'].input_var],
                                          [l_out_single_debug,
                                           l_out_prediciton])

        return probs_f, fc_prec_conv_body, loss_train_f, debug_f, debug_singe_out

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

    # l_in, l_in_direction, l_9, l_10, fov = self.build_ID_v0_hydra()
    # las.layers.CustomRecurrentLayer()

    from theano.sandbox import cuda
    cuda.use('gpu0')


    real_global_claims = np.ones((3,1,4,4), dtype='float32')

    gobal_raw = theano.shared(np.zeros((3,6,4,4),dtype='float32'), borrow=False)
    gobal_claims = theano.shared(real_global_claims, borrow=False)

    gobal_claims_t = T.ftensor4()
    # gloabl_input = theano.shared(np.zeros((3,8,100,100),dtype='float32'))
    input_batch = np.zeros((3,8,40,40),dtype='float32')
    input_batch_t = theano.shared(input_batch)

    batches = [0,1,2,3]

    coords = T.ivector()
    b = T.iscalar()
    set_val_t = T.fscalar()
    # define graph for slice raw > input
    pad = 1

    # real_global_claims[1,0,1,1] += 17

    set_f = theano.function([b, coords, set_val_t], updates=[(gobal_claims,
                                T.set_subtensor(gobal_claims[b, 0, coords[0],
                                  coords[1]], set_val_t))]) 
    # set_f = theano.function([b, coords, gobal_claims_t], T.set_subtensor(gobal_claims_t[b, 0, coords[0],
                                  # coords[1]], 17))


    set_f(1, np.array((1, 0), dtype='int32'), 17)
    print 'global claims after set', np.where(gobal_claims.get_value() == 17)
    print "set", gobal_claims.get_value() 


    crop_raw_f = theano.function([b, coords], gobal_raw[None, b, :,
                                         coords[0]-pad:coords[0]+pad,
                                         coords[1]-pad:coords[1]+pad])

    claim_c = gobal_claims_t[None, b, :, coords[0]-pad:coords[0]+pad,
                                  coords[1]-pad:coords[1]+pad]
    me_id = T.fscalar()
    claim_me = T.eq(claim_c, me_id)
    claim_them_with_bg = T.neq(claim_c, me_id)
    claim_not_bg = T.neq(claim_c, 0)
    claim_them = claim_them_with_bg & claim_not_bg

    get_calims_f = theano.function([b, gobal_claims_t, coords, me_id], [claim_me, claim_them])

    raw_list = []
    claim_list = []
    me_idx = 1
    for b in [0,1,2]:
        # crop_f(1,np.array([20,20],dtype='int32'))
        raw_list.append(crop_raw_f(b,np.array([1,1], dtype='int32')))
        me, them = get_calims_f(b,gobal_claims.eval(),np.array([1,1], dtype='int32'), me_idx)
        claim_list.append(T.cast(T.concatenate((me,them),axis=1),dtype='float32'))

    print [x.dtype for x in raw_list]
    print [x.dtype for x in claim_list]

    raw_batch = T.concatenate(raw_list, axis=0)
    claim_batch = T.concatenate(claim_list, axis=0)
    input_batch = T.concatenate((raw_batch, claim_batch), axis=1)
 
    print input_batch.shape.eval()
