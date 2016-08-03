import lasagne as las
from lasagne import layers as L
import theano
from theano import tensor as T
import numpy as np
import custom_layer as cs


l_net_out = L.InputLayer((None, 4,1,1))
l_in_dir = L.InputLayer((None, ), input_var=T.vector(dtype='int32'))
l_slice = cs.BatchChannelSlicer([l_net_out, l_in_dir])
l_out = L.get_output(l_slice)

get_out_f = theano.function([l_net_out.input_var,
                             l_in_dir.input_var], l_out,
                            on_unused_input='ignore')


net_out = np.arange((2*4)).reshape(2, 4, 1, 1).astype(theano.config.floatX)
dirs = np.random.randint(0, 4, size=(2)).astype(np.int32)
output = get_out_f(net_out, dirs)

print 'in', net_out
print 'dir in', dirs
print 'out'
print output.shape
print output

