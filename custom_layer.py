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


if __name__ == '__main__':
    shape = (10, 2, 4, 4)
    l_in = las.layers.InputLayer(shape)
    l_1 = SliceLayer(l_in)
    l_dense = las.layers.DenseLayer(l_in, 3)
    l_out = las.layers.get_output(l_dense)

    a = np.arange(np.prod(shape)).reshape(shape).astype(theano.config.floatX)

    probs = theano.function([l_in.input_var], l_out)

    probs(a)
    print 'a', a
    print
    print probs(a).shape
