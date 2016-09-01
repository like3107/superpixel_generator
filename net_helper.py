import numpy as np


def calc_conv_reduction(edge_len, filter_size, stride=1, pooling=1, padding=0):
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

if __name__ == '__main__':

	filt = [3, 5, 5]
	pooling = [2,2,1]
	fov = 30
	print fov
	for fs,p in zip(filt,pooling):
		fov = calc_conv_reduction(fov , fs, pooling=p)
		print fov