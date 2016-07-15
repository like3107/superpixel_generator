import h5py as h
import numpy as np
from matplotlib import pyplot as plt
import theano
from Queue import PriorityQueue


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
                 global_edge_len=110, patch_len=40, remain_in_territory=True):

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
        self.remain_in_territory = remain_in_territory

        assert(patch_len <= global_edge_len)
        assert(global_edge_len < self.rl + self.pad)

        # TODO: implement padding
        if self.rl - self.global_el < 0:
            raise Exception('implement padding if this is the case')

        # private
        self.global_batch = None
        self.global_label_batch = None
        self.global_claims = None

        self.priority_queue = None

    def prepare_global_batch(self):
        # initialize two global batches = region where CNNs compete
        # against each other
        self.global_batch = np.zeros((self.bs, 1,
                                     self.global_el, self.global_el),
                                     dtype=theano.config.floatX)
        self.global_label_batch = np.zeros((self.bs, 1,
                                            self.global_el, self.global_el),
                                           dtype=theano.config.floatX)

        # get indices for global batches in raw/ label cubes
        ind_b = np.random.permutation(self.n_slices)[:self.bs]
        ind_x = np.random.randint(0, self.rl - self.global_el + 1,
                                  size=self.bs)
        ind_y = np.random.randint(0, self.rl - self.global_el + 1,
                                  size=self.bs)

        # slice from the data cubes
        global_ids = []
        for batch in range(self.bs):
            self.global_batch[batch, :, :] = \
                self.raw[ind_b[batch],
                         ind_x[batch]:ind_x[batch] + self.global_el,
                         ind_y[batch]:ind_y[batch] + self.global_el]
            self.global_label_batch[batch, :, :] = \
                self.labels[ind_b[batch],
                            ind_x[batch]:ind_x[batch] + self.global_el,
                            ind_y[batch]:ind_y[batch] + self.global_el]
            global_ids.append(
                np.unique(
                    self.global_label_batch[batch, 0,
                                            self.pad:-self.pad-1,
                                            self.pad:-self.pad-1]).astype(int))
        return global_ids

    def get_seeds(self, global_ids):
        batch = -1
        global_seeds = []
        for ids in global_ids:    # iterates over batches
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
            global_seeds.append(seeds)
        return global_seeds

    def initialize_priority_queue(self, global_seeds, global_ids):
        b = -1  # batch counter
        self.priority_queue = []
        for seeds, ids in zip(global_seeds, global_ids):
            b += 1
            q = PriorityQueue()
            i = -1      # ids within slice
            for seed, id in zip(seeds, ids):
                i += 1
                q.put((-0.1, seed, id))
            self.priority_queue.append(q)

    def get_adjacent_gts(self, seed, batch, id=None):
        if id is None:
            id = self.global_label_batch[batch, 0, seed[0], seed[1]]
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
            np.array([self.global_label_batch[batch, 0,
                                              seeds_x, seeds_y] == id][0],
                     dtype=theano.config.floatX)
        return ground_truth

    def crop_raw(self, seed, batch_counter):
        raw = self.global_batch[batch_counter, :,
                                seed[0] - self.pad:seed[0] + self.pad,
                                seed[1] - self.pad:seed[1] + self.pad]
        return raw

    def init_train_batch(self):
        # extract slices and ids within raw and label cubes
        global_ids = self.prepare_global_batch()
        # extract starting points
        global_seeds = self.get_seeds(global_ids)
        # put seeds and ids in priority queue. All info to load batch is in pq
        self.initialize_priority_queue(global_seeds, global_ids)

        # remember where territory has been claimed before. 1 claimed, 0 free
        self.global_claims = np.ones((self.bs, self.global_el, self.global_el),
                                     dtype=int)
        self.global_claims[:, self.pad:-self.pad, self.pad:-self.pad] = 0

    def get_batches(self):
        seeds, ids = self.get_seeds_from_queue()
        raw_batch = np.zeros((self.bs, 1, self.pl, self.pl),
                             dtype=theano.config.floatX)
        gts = np.zeros((self.bs, 4, 1, 1), dtype=theano.config.floatX)
        for b in range(self.bs):
            raw_batch[b, 0, :, :] = self.crop_raw(seeds[b], b)
            gts[b, :, 0, 0] = self.get_adjacent_gts(seeds[b], b)
            self.global_claims[b, seeds[b][0], seeds[b][1]] = ids[b]
        return raw_batch, gts, seeds, ids

    def get_seeds_from_queue(self):
        seeds = []
        ids = []
        for b in range(self.bs):
            if self.priority_queue[b].empty():
                print 'batch', b
                raise Exception('priority queue is empty. This might be due to '
                                'unconnected cmoponents with the same ID')
            else:
                already_claimed = True
                out_of_territory = True
                while already_claimed or out_of_territory:
                    prob, seed, id = self.priority_queue[b].get()
                    if self.global_claims[b, seed[0], seed[1]] == 0:
                        already_claimed = False
                    else:
                        already_claimed = True
                    if self.remain_in_territory:
                        if self.global_label_batch[b, 0, seed[0], seed[1]] \
                                == id:
                            out_of_territory = False
                        else:
                            out_of_territory = True

                seeds.append(seed)
                ids.append(id)

        return seeds, ids

    def update_priority_queue(self, probabilities, seeds, ids):
        assert(len(probabilities) == len(seeds))
        assert(len(probabilities) == self.bs)

        for b in range(self.bs):
            counter = -1
            for x, y in zip([-1, 0, 1, 0], [0, -1, 0, 1]):
                counter += 1
                coords = [seeds[b][0] + x, seeds[b][1] + y]
                if self.global_claims[b, coords[0], coords[1]] == 0:
                    self.priority_queue[b].put((1 - probabilities[b][counter],
                                                [coords[0], coords[1]], ids[b]))


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
    bm.init_train_batch()

    # print bm.get_batches(10*[[45, 46]])












