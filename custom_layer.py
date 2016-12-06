import lasagne
las = lasagne
import numpy as np
import theano
from theano import tensor as T


class SliceLayer(lasagne.layers.Layer):

    def get_output_for(self, input, **kwargs):
        bs = self.input_shape[0]
        middle_ind = self.input_shape[2]/2
        output_t = input[:, 0:1,
                         middle_ind-1:middle_ind+2,
                         middle_ind - 1:middle_ind + 2]
        return output_t

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 1, 3, 3)


class BatchChannelSlicer(las.layers.MergeLayer):
    """
    input: 2 Las layers:
        1. layer which should be sliced along the channels (axis=1)
        2. vector of which indices should be used (one channel per batch)
    output: shape (b, 1, ...)
    """
    def __init__(self, incomings, **kwargs):
        super(BatchChannelSlicer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shape):
        if len(input_shape) > 2:
            return tuple((input_shape[0], 1, input_shape[2:]))
        else:
            return tuple((input_shape[0], 1))

    def get_output_for(self, inputs, **kwargs):
        l_in, slices = inputs
        batches = l_in.shape[0]
        return l_in[T.arange(batches), slices, None]


class CrossSlicer(las.layers.Layer):
    """
    input: 2 Las layers:
        1. layer which should be sliced along the channels (axis=1)
        2. vector of which indices should be used (one channel per batch)
    output: shape (b, 1, ...)
    """
    def __init__(self, incoming, **kwargs):
        super(CrossSlicer, self).__init__(incoming, **kwargs)
        self.slices_x = theano.shared(np.array([[0, 1, 2, 1]], dtype=np.int32))  # up left down right
        self.slices_y = theano.shared(np.array([[1, 0, 1, 2]], dtype=np.int32))  # up left down right

    def get_output_shape_for(self, input_shape):
        return tuple((input_shape[0], input_shape[1], None, None))

    def get_output_for(self, input, **kwargs):
        batches = input.shape[0]
        batches_list = T.extra_ops.repeat(T.arange(batches), 4, axis=0).flatten()
        slices_x = T.extra_ops.repeat(self.slices_x, batches, axis=0).flatten()
        slices_y = T.extra_ops.repeat(self.slices_y, batches, axis=0).flatten()
        input = input.swapaxes(1, 0)[:, batches_list, slices_x, slices_y].reshape((-1, batches, 2, 2)).swapaxes(0, 1)
        return input


class GradientToHeight(las.layers.Layer):
    """
    input: 2 Las layers:
        1. layer which should be sliced along the channels (axis=1)
        2. vector of which indices should be used (one channel per batch)
    output: shape (b, 1, ...)
    """
    def __init__(self, incoming, **kwargs):
        super(GradientToHeight, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        # return tuple((input_shape[0], 4, *input_shape[2:]))
        return tuple((input_shape[0], 4, 1, 1))

    def get_output_for(self, input, **kwargs):
        return T.stack([
            input[:,0,:,:]-input[:,1,:,:],
            input[:,0,:,:]+input[:,2,:,:],
            input[:,0,:,:]+input[:,1,:,:],
            input[:,0,:,:]-input[:,2,:,:]],axis=1)


# elu
def elup1(x):
    return T.switch(x > 0, x+1, T.exp(x))



if __name__ == '__main__':
    shape = (10, 2, 3, 3)
    l_in = las.layers.InputLayer(shape)
    l_1 = SliceLayer(l_in)
    l_slice = CrossSlicer(l_in)

    # l_dense = las.layers.DenseLayer(l_in, 3)
    l_out = las.layers.get_output(l_slice)

    a = np.arange(np.prod(shape)).reshape(shape).astype(theano.config.floatX)

    probs = theano.function([l_in.input_var], l_out)

    probs(a)
    print 'a', a
    print 'out', probs(a)
    print probs(a).shape
