import numpy as np


def calc_conv_reduction(edge_len, filter_size, stride=1, pooling=1, padding=0,
                        dilatation=1):
    filter_size = (dilatation - 1) * (filter_size - 1) + filter_size
    new_edge_len = \
        ((float(edge_len) - filter_size + padding) / stride + 1) / pooling
    if new_edge_len != int(new_edge_len):
        error = 'float: edge_len %f, filter_size %f, stride %f, pooling  %f, ' \
                'padding %f' % (edge_len, filter_size, stride, pooling, padding)
        raise Exception(error)
    return new_edge_len


        # fov = 45    # field of view = patch length
        # filt = [3, 5, 5]
        # n_filt = [20, 25, 60, 30, n_classes]
        # pool = [2, 2]

        # # 40
        # l_in = L.InputLayer((None, n_channels, fov, fov))
        # l_1 = L.batch_norm(L.Conv2DLayer(l_in, n_filt[0], filt[0]))

        # l_2 = L.MaxPool2DLayer(l_1, pool[0])
        # # 17
        # l_3 = L.batch_norm(L.Conv2DLayer(l_2, n_filt[1], filt[1]))
        # l_4 = L.MaxPool2DLayer(l_3, pool[1])
        # # 6
        # l_5 = L.Conv2DLayer(l_4, n_filt[2], 5, filt[2],
        #                     nonlinearity=las.nonlinearities.rectify)
        # l_6 = L.Conv2DLayer(l_5, n_filt[3], 1,
        #                     nonlinearity=las.nonlinearities.rectify)
        # l_7 = L.Conv2DLayer(l_6, n_filt[4], 1,
        #                     nonlinearity=las.nonlinearities.rectify,
        #                     b=np.random.random(n_classes)*10+10.)


def add_channel(net, additionl_channels):

    print net
    exit()


def calc_field_of_view(filt_sizes, stride=None, poolings=None,
                       dilatations=None):

    if poolings is None:
        poolings = np.ones(len(filt_sizes), dtype=int)
    if stride is None:
        stride = np.ones(len(filt_sizes), dtype=int)
    poolings, filt_sizes, stride, dilatations = np.array(poolings[::-1]), \
                                                np.array(filt_sizes[::-1]),\
                                                np.array(stride[::-1]), \
                                                np.array(dilatations[::-1])


    if dilatations is not None:
        filt_sizes += (filt_sizes - 1) * (dilatations - 1)
    new_edge_len = 1
    i = 0
    for p, f, s, d in zip(poolings, filt_sizes, stride, dilatations):
        i += 1
        print 'pool', p, 'filt size', f, 'stride', s
        # new_edge_len = f + 2 * (d - 1) + new_edge_len + (i - 1)
        new_edge_len = new_edge_len  + f + 2 * (d - 1)
        print "new", new_edge_len


if __name__ == '__main__':
    filts = [4, 3, 3, 3, 3,  3, 3, 1, 1]
    dils =  [1, 1, 2, 4, 8, 16, 1, 1, 1]

    # strides = [1, 1, 1]
    filts = [3, 3]
    dils = [1, 1]
    fov = 61
    print fov
    calc_field_of_view(filts, dilatations=dils)
    # for fs, p, dil in zip(filts, poolings, dilatations):
    #     fov = calc_conv_reduction(fov, fs, pooling=p, dilatation=dil)
    #     print fov