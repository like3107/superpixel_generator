import numpy as np


def calc_conv_reduction(edge_len, filter_size, stride=1, pooling=1, padding=0):
    new_edge_len = \
        ((float(edge_len) - filter_size + padding) / stride + 1) / pooling
    if new_edge_len != int(new_edge_len):
        error = 'float: edge_len %f, filter_size %f, stride %f, pooling  %f, ' \
                'padding %f' % (edge_len, filter_size, stride, pooling, padding)
        raise Exception(error)
    return new_edge_len


if __name__ == '__main__':
    print calc_conv_reduction(6, 6)