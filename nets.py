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

    def build_net_v8_dilated(self, l_image_in=None):
        n_channels = self.options.network_channels
        if n_channels is None:
            n_channels = 1
        fov = None
        n_classes = 4
        filts =         [4,     3,      3,      3,      3,      3,      3,      1,      1,      1]
        dils  =         [1,     1,      2,      4,      8,      16,     1,      1,      1,      1]
        n_filts =       [32,    32,     64,    64,    128,    128,    256,    2048,     128,    n_classes]
        # debug
        batch_norms =   [True, True,   True,   True,   True,   True,   True,   False,  False,   False]
        regs        =   [True, True,   True,   True,   True,   True,   True,   True,   True,    False]
        # batch_norms = [False] * len(n_filt)
        ELU = las.nonlinearities.elu
        ReLU = las.nonlinearities.rectify
        ident = las.nonlinearities.identity
        act_fcts =      [ELU,  ELU,     ELU,    ELU,    ELU,    ELU,    ELU,    ReLU, ReLU,   ReLU]
        names       =   ['conv','conv','conv','conv','conv', 'conv', 'conv', 'fc', 'fc', 'fc']
        assert(len(filts) == len(dils) and len(filts) == len(batch_norms) and len(filts) == len(regs) and
               len(filts) == len(n_filts) and len(names) == len(act_fcts) and len(filts) == len(names))
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
                    L.DilatedConv2DLayer(l_prev, n_filt, filt, dilation=(dil, dil), name=name, nonlinearity=act_fct))
            else:
                l_next = L.DilatedConv2DLayer(l_prev, n_filt, filt, dilation=(dil, dil), name=name,
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
                                nonlinearity = cs.elup1)
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

    def build_net_recurrent(self):

        n_channels = self.options.network_channels
        fov = 6

        l_in_inp = las.layers.InputLayer((None, n_channels, fov, fov))
        # input_claim = las.layers.InputLayer((None, n_channels, fov, fov))

        l_conv = las.layers.Conv2DLayer(l_in_inp, 10, 6)

        # out (b, t,  x, y)

        # inpt recurrent (b, time, channels)
        l_resh = las.layers.ReshapeLayer(l_conv, (3, 2, 10))
        l_in_hid = las.layers.InputLayer((None, 10))
        l_recurrent = las.layers.RecurrentLayer(l_resh, 10, hid_init=l_in_hid)

        out = las.layers.get_output(l_conv)
        out_conv_f = theano.function([l_in_inp.input_var], out)

        out_rec = las.layers.get_output(l_recurrent)
        out_rec_f = theano.function([l_in_inp.input_var,
                                     l_in_hid.input_var], out_rec)

        return out_conv_f, out_rec_f

    def build_net_v8_hydra_dilated_ft_joint(self, l_image_in = None,
                                            l_claims_in = None):
        self.sequ_len_s = T.iscalar()
        print 'building recurrent joint net'
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

        fc_1 = las.layers.Conv2DLayer(l_merge, 2048, filter_size=1, name='fc', W=W, b=first_fc.b,
                                      nonlinearity=las.nonlinearities.rectify)

        # recurrent part
        l_resh_pred = las.layers.ReshapeLayer(fc_1, (-1, self.sequ_len_s, 2048))
        rec_hidden = self.options.n_recurrent_hidden
        l_in_hid = las.layers.InputLayer((None, rec_hidden))
        l_in_rec_mask = las.layers.InputLayer((None, self.options.backtrace_length))
        W_hid_to_hid = np.random.random((128, 128)).astype('float32') / 10000.
        l_recurrent = las.layers.RecurrentLayer(l_resh_pred, rec_hidden,
                                                hid_init=l_in_hid, mask_input=l_in_rec_mask,
                                                W_in_to_hid=l_out_old.input_layer.W[:, :, 0, 0],
                                                W_hid_to_hid=W_hid_to_hid,
                                                b=l_out_old.input_layer.b)
        l_reshape_fc = las.layers.ReshapeLayer(l_recurrent, (-1, rec_hidden, 1, 1))

        # last layer
        l_out_cross = las.layers.Conv2DLayer(l_reshape_fc, 4, filter_size=1, name='fc',
                                             W=l_out_old.W.dimshuffle(1,0,2,3), b=l_out_old.b,
                                             nonlinearity=las.nonlinearities.rectify)
        l_out_cross.params[l_out_cross.W].remove('regularizable')
        layers = {}
        layers['l_in_claims'] = l_in
        layers['l_in_rec_mask'] = l_in_rec_mask
        layers['l_in_precomp'] = l_in_old
        layers['l_in_hid'] = l_in_hid
        layers['l_recurrent'] = l_recurrent
        layers['l_reshape'] = l_resh_pred
        layers['l_in_old'] = l_in_old
        layers['l_out_cross'] = l_out_cross
        layers['l_out_precomp'] = l_out_precomp
        layers['l_claims_out'] = l_claims_out
        layers['l_merge'] = l_merge

        fov = 68
        return layers, fov


    def build_net_v8_hydra_dilated_ft_joint_old(self, l_image_in = None,
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
                                nonlinearity=las.nonlinearities.identity)

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

    def build_v8_hydra_dilated(self, l_image_in=None):
        l_in, (l_9, l_seconds_last), fov = self.build_net_v8_dilated(l_image_in=l_image_in)
        l_in_direction = L.InputLayer((None,), input_var=T.vector(dtype='int32'))
        l_10 = cs.BatchChannelSlicer([l_9, l_in_direction])
        return l_in, l_in_direction, l_9, l_10, fov, None


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
        step = self.options.backtrace_length
        sum_height = l_out_train[::step]
        for t in range(1, ):
            sum_height += l_out_train[t::step]

        individual_batch = (sum_height[bs/2/step:] - sum_height[:bs/2/step])
        loss_valid = T.mean(individual_batch)
        if L1_weight > 0:
            loss_train = loss_valid + L1_weight * L1_norm

        print "using nesterov_momentum"
        if self.options.optimizer == "nesterov":
            updates = las.updates.nesterov_momentum(loss_train, all_params, 0.001)
        elif self.options.optimizer == "adam":
            updates = las.updates.adam(loss_train, all_params)
        else:
            raise Exception("unknown optimizer %s"%self.options.optimizer)

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
                loss_train_f = theano.function(
                    [layers['l_in_claims'].input_var,
                     layers['l_in_old'].input_var,
                     layers['l_in_direction'].input_var,
                     layers['l_in_hid'].input_var,
                     layers['l_in_rec_mask'].input_var,
                     self.sequ_len_s],
                   [loss_train, individual_batch,
                    l_out_prediciton, l_out_train],
                   updates=updates)
                print 'fc prec'
                l_in_from_prec = las.layers.InputLayer((None, 64, 1, 1))
                # disconnect graph temporarely
                layers['l_merge'].input_layers[0] = l_in_from_prec
                layers['l_merge'].input_shapes[0] = l_in_from_prec.output_shape

                l_out_prediciton_prec = L.get_output(layers['l_out_cross'],
                                                     deterministic=True)
                l_out_hidden = L.get_output(layers['l_recurrent'])
                probs_f = theano.function([layers['l_in_claims'].input_var,
                                           l_in_from_prec.input_var,
                                           layers['l_in_hid'].input_var,
                                           layers['l_in_rec_mask'].input_var,
                                           self.sequ_len_s],
                                          [l_out_prediciton_prec, l_out_hidden])

                layers['l_merge'].input_layers[0] = layers['l_out_precomp']
                layers['l_merge'].input_shapes[0] = \
                    layers['l_out_precomp'].output_shape
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
