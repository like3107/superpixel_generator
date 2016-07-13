import h5py as h
import numpy as np
from matplotlib import pyplot as plt
import theano


def load_h5(path, h5_key=None, group=None, group2=None):

    f = h.File(path, 'r')
    if group is not None:
        g = f[group]
        if group2 is not None:
            g = g[group2]
    else:   # no groups in file structure
        g = f
    if h5_key is None:     # no h5 key specified
        output = list()
        for key in g.keys():
            output.append(np.array(g[key]))

    elif isinstance(h5_key, basestring):   # string
        print g.keys()
        output = np.array(g[h5_key])

    elif isinstance(h5_key, list):          # list
        output = list()
        for key in h5_key:
            output.append(np.array(g[key]))
    else:
        raise Exception('h5 key type is not supported')
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
        print '\r %f %%' % (100. * i / len(ids)),
        cumulative_array[array == ids[i]] = i
    return cumulative_array


class BatchManV0:

    def __init__(self, raw_path, label_path,
                 raw_key=None, label_key=None, batch_size=10,
                 global_edge_len=110, patch_len=40):

        self.raw = load_h5(raw_path, h5_key=raw_key)[0]
        self.labels = load_h5(label_path, h5_key=label_key)[0]
        assert(self.raw.shape == self.labels.shape)
        assert(self.raw.shape[1] == self.raw.shape[2])

        self.n_slices = len(self.raw)
        self.bs = batch_size
        self.global_el = global_edge_len
        self.pl = patch_len
        self.pad = patch_len / 2
        self.rl = self.raw.shape[1]

        assert(patch_len <= global_edge_len)
        assert(global_edge_len < self.rl + self.pad)

        # TODO: implement padding
        if self.rl - self.global_el < 0:
            raise Exception('implement padding if this is the case')

        # private
        self.global_batch = None
        self.global_label_batch = None
        self.ids = None
        self.global_seeds = None
        self.global_gts = None
        self.batch_counter = None
        self.id_counter = None

    def prepare_batch(self):
        # initialize two global batches = region where CNNs compete
        # against each other
        self.global_batch = np.zeros((self.bs, 1,
                                     self.global_el, self.global_el),
                                     dtype=theano.config.floatX)
        self.global_label_batch = np.zeros((self.bs, 1,
                                            self.global_el, self.global_el),
                                           dtype=theano.config.floatX)

        ind_b = np.random.permutation(self.n_slices)[:self.bs]
        ind_x = np.random.randint(0, self.rl - self.global_el + 1,
                                  size=self.bs)
        ind_y = np.random.randint(0, self.rl - self.global_el + 1,
                                  size=self.bs)

        self.ids = []
        for batch in range(self.bs):
            self.global_batch[batch, :, :] = \
                self.raw[ind_b[batch],
                         ind_x[batch]:ind_x[batch] + self.global_el,
                         ind_y[batch]:ind_y[batch] + self.global_el]
            self.global_label_batch[batch, :, :] = \
                self.labels[ind_b[batch],
                            ind_x[batch]:ind_x[batch] + self.global_el,
                            ind_y[batch]:ind_y[batch] + self.global_el]
            self.ids.append(
                np.unique(self.global_label_batch[batch, 0,
                                                  self.pad:-self.pad-1,
                                                  self.pad:-self.pad-1])
                    .astype(int))

    def get_seeds(self):
        batch = -1
        self.global_seeds = []
        for ids in self.ids:    # iterates over batches
            batch += 1
            seeds = []
            for id in ids:
                regions = np.where(
                    self.global_label_batch[batch, 0,
                                            self.pad:-self.pad-1,
                                            self.pad:-self.pad-1] == id)
                rand_seed = np.random.randint(0, len(regions[0]))
                seeds.append([regions[0][rand_seed] + self.pad,
                              regions[1][rand_seed] + self.pad])
            self.global_seeds.append(seeds)

    def seeds_initial_gt(self):

        self.global_gts = list()
        batch = -1
        for seeds in self.global_seeds:
            batch += 1
            gts = list()
            k = -1
            for seed in seeds:
                k += 1
                gts.append(self.get_adjacent_gts(seed, self.ids[batch][k],
                                                 batch))
            self.global_gts.append(gts)

    def get_adjacent_gts(self, seed, id, batch):

        seeds_x = [max(self.pad, seed[0]-1),
                   seed[0],
                   min(seed[0]+1, self.global_el - self.pad-1),
                   seed[0]]
        seeds_y = [seed[1],
                   max(self.pad, seed[1]-1),
                   seed[1],
                   min(seed[1]+1, self.global_el - self.pad-1)]

        # boundary conditions
        ground_truth = \
            np.array(
                [self.global_label_batch[batch, 0, seeds_x, seeds_y] == id][0],
                     dtype=theano.config.floatX)

        return ground_truth

    def crop_raw(self, seed, batch_counter):

        print self.raw.shape
        raw = self.global_batch[batch_counter, :,
                       seed[0] - self.pad:seed[0] + self.pad,
                       seed[1] - self.pad:seed[1] + self.pad]
        label = self.global_label_batch[batch_counter, :,
                            seed[0] - self.pad:seed[0] + self.pad,
                            seed[1] - self.pad:seed[1] + self.pad]

        return raw, label

    def get_init_train_batch(self):
        print 'get init train batch'
        if self.batch_counter is None:
            self.batch_counter = 0
            self.id_counter = 0

        raw_batch = np.zeros((self.bs, 1, self.pl, self.pl),
                             dtype=theano.config.floatX)
        gts = np.zeros((self.bs, 4), dtype=theano.config.floatX)
        print 'end get init batch train'
        for b in range(self.batch_counter, self.bs):
            for segment in range(self.id_counter, len(self.ids[b])):
                seed = self.global_seeds[b][segment]
                print 'seed', seed, b
                raw, label = self.crop_raw(seed, b)
                raw = raw[0]
                label = label[0]
                gts = self.global_gts[b][segment]
                print 'raw shape', raw.shape
                plt.figure(figsize=(10, 10))
                raw[29, 29] = 0
                plt.imshow(raw, cmap='gray', interpolation='none')
                plt.figure(figsize=(10, 10))
                label[30, 30] = 0
                plt.imshow(label, interpolation='none')
                print gts
                plt.show()

if __name__ == '__main__':

    # loading of cremi
    # path = './data/sample_A_20160501.hdf'
    # a = load_h5(path, 'neuron_ids', 'volumes', 'labels')
    # a = make_array_cumulative(a)
    # save_h5('./data/label_a.h5', 'labels', a, 'w')
    # plt.imshow(a[5, :, :], cmap='Dark2')
    # plt.show()

    # loading from BM
    label_path = './data/labels_as.h5'
    raw_path = './data/raw_as.h5'

    bm = BatchManV0(raw_path, label_path, batch_size=10, patch_len=60,
                    global_edge_len=95)

    bm.prepare_batch()
    bm.get_seeds()
    bm.seeds_initial_gt()
    bm.get_init_train_batch()












