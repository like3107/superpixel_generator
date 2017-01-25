
import numpy as np
import scipy.stats as sc
import theano
import theano.tensor as T

import lasagne.layers as L
import lasagne.nonlinearities as nonl
import lasagne.regularization as reg
import lasagne.objectives as lobj
from lasagne.updates import adam

batch_size = 10
seq_length = 12
input_channels = 3
input_var = T.tensor3(name="input", dtype=theano.config.floatX)
mask_var = T.matrix(name="mask", dtype=theano.config.floatX)
# seq_len_var = T.vector(name="sequence_lenghts", dtype="uint8")

input_l = L.InputLayer((batch_size, seq_length, input_channels), input_var)
mask_l = L.InputLayer((batch_size, seq_length), mask_var)

lstm = L.LSTMLayer(input_l, num_units=2, mask_input=mask_l)
print L.get_output_shape(lstm)
out = L.get_output(lstm)

func = theano.function([input_var, mask_var], out)

inputs = np.random.randn(batch_size,seq_length,input_channels).astype(theano.config.floatX)
masks = np.zeros((batch_size,seq_length)).astype('uint8')
input_len = np.random.random_integers(2,seq_length-1,batch_size).astype('uint8')
for i in range(batch_size):
    masks[i,:input_len[i]] = 1

print "Lengths: ", input_len
print "Masks: ", masks
print input_len
out = func(inputs, masks)
for i in range(batch_size):
    print "Len of sequence: ", input_len[i]
    print "Output:", out[i,:,:]