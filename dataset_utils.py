import h5py as h
import os
import numpy as np
from matplotlib import pyplot as plt


def load_h5(path, h5_key=None, group=None):
    print os.path.exists(path)
    f = h.File(path, 'r')
    if group is not None:
        g = f[group]
    else:   # no groups in file structure
        g = f
    if h5_key is None:     # no h5 key specified
        output = list()
        for key in g.keys():
            output.append(np.array([key]))

    elif isinstance(h5_key, basestring):   # string
        print g.keys()
        output = np.array(g[h5_key])

    elif isinstance(h5_key, list):          # list
        output = list()
        for key in h5_key:
            output.append(np.array(g[key]))
    f.close()

    return output


def save_h5(path, h5_key, data, overwrite='w-'):
    f = h.File(path, overwrite)
    f.create_dataset(h5_key, data=data)
    f.close()


def make_array_cumulative(array):
    ids = np.unique(array)
    cumulative_array = np.zeros_like(array)
    for i in range(len(ids)):
        print '\r %f %%' %( 100. * i / len(ids)),
        cumulative_array[array == ids[i]] = i
    return cumulative_array


if __name__ == '__main__':



    # loading of cremi
    path = './data/sample_A_20160501.hdf'
    a = load_h5(path, 'raw', 'volumes')
    save_h5('./data/raw_as.h5', 'raw', a, 'w')
    plt.imshow(a[5, :, :], cmap='Dark2')
    plt.show()












