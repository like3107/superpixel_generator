import h5py as h
import numpy as np
from matplotlib import pyplot as plt
import theano
from Queue import PriorityQueue
import utils as u


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
    if isinstance(h5_key, str):
        f.create_dataset(h5_key, data=data)
    if isinstance(h5_key, list):
        for key, values in zip(h5_key, data):
            f.create_dataset(key, data=values)
    f.close()


def mirror_cube(array, pad_length):
    assert (len(array.shape) == 3)
    mirrored_array = np.pad(array, ((0, 0), (pad_length, pad_length),
                                    (pad_length, pad_length)), mode='reflect')
    return mirrored_array


def make_array_cumulative(array):
    ids = np.unique(array)
    cumulative_array = np.zeros_like(array)
    for i in range(len(ids)):
        print '\r %f %%' % (100. * i / len(ids)),
        cumulative_array[array == ids[i]] = i
    return cumulative_array


class BatchManV0:
    def __init__(self, raw, label,
                 raw_key=None, label_key=None, batch_size=10,
                 global_edge_len=110, patch_len=40, padding_b=False):
        """
        batch loader. Use either for predict OR train. For valid and train use:
        get batches function.

        :param raw:
        :param label:
        :param raw_key:
        :param label_key:
        :param batch_size:
        :param global_edge_len:
        :param patch_len:
        :param padding_b:
        """
        if isinstance(raw, str):
            self.raw = load_h5(raw, h5_key=raw_key)[0]
        else:
            self.raw = raw
        if isinstance(label, str):
            self.labels = load_h5(label, h5_key=label_key)[0]
        else:
            self.labels = label

        self.padding_b = padding_b
        self.pad = patch_len / 2
        if self.padding_b:
            self.raw = mirror_cube(self.raw, self.pad)
        else:
            # crop label
            self.labels = self.labels[:, self.pad:-self.pad, self.pad:-self.pad]

        self.rl = self.raw.shape[1]     # includes padding
        self.n_slices = len(self.raw)
        self.bs = batch_size
        self.global_el = global_edge_len        # length of field raw (
        # includes padding)
        self.pl = patch_len

        assert(patch_len <= global_edge_len)
        assert(global_edge_len < self.rl + self.pad)

        if self.rl - self.global_el < 0:
            raise Exception('try setting padding to True')

        # private
        self.global_batch = None            # includes padding
        self.global_label_batch = None      # no padding
        self.global_claims = None           # includes padding

        self.priority_queue = None

    def prepare_global_batch(self):
        # initialize two global batches = region where CNNs compete
        # against each other

        # get indices for global batches in raw/ label cubes
        ind_b = np.random.permutation(self.n_slices)[:self.bs]
        # indices to raw, correct for label which edge len is -self.pl shorter
        ind_x = np.random.randint(0,
                                  self.rl - self.global_el,
                                  size=self.bs)
        ind_y = np.random.randint(0,
                                  self.rl - self.global_el,
                                  size=self.bs)

        # slice from the data cubes
        global_ids = []
        for b in range(self.bs):
            self.global_batch[b, :, :] = \
                self.raw[ind_b[b],
                         ind_x[b]:ind_x[b] + self.global_el,
                         ind_y[b]:ind_y[b] + self.global_el]
            # ind_x[b], ind_y[b] = (ind_x[b] - self.pl, ind_y[b] - self.pl)
            self.global_label_batch[b, :, :] = \
                self.labels[ind_b[b],
                            ind_x[b]:ind_x[b] + self.global_el - self.pl,
                            ind_y[b]:ind_y[b] + self.global_el - self.pl]
            global_ids.append(
                np.unique(
                    self.global_label_batch[b, :, :]).astype(int))
        return global_ids

    def get_seeds(self, global_ids):    # seed coords relative to global batch
        batch = -1
        global_seeds = []   # seeds relative to global_claims, global_batch
        #  map (- self.pad for global_label_label)
        for ids in global_ids:    # iterates over batches
            batch += 1
            seeds = []
            for id in ids:
                regions = np.where(
                    self.global_label_batch[batch, :, :] == id)
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

    def get_cross_coords(self, seed):
        seeds_x = [max(self.pad, seed[0] - 1),
                   seed[0],
                   min(seed[0] + 1, self.global_el - self.pad - 1),
                   seed[0]]

        seeds_y = [seed[1],
                   max(self.pad, seed[1] - 1),
                   seed[1],
                   min(seed[1] + 1, self.global_el - self.pad - 1)]
        return seeds_x, seeds_y

    def get_adjacent_gts(self, seed, batch, id):
        seeds_x, seeds_y = self.get_cross_coords(seed)

        seeds_x, seeds_y = (np.array(seeds_x) - self.pad,
                            np.array(seeds_y) - self.pad)
        assert (np.any(seeds_x >= 0) or np.any(seeds_y >= 0))
        assert (np.any(self.rl - self.pl > seeds_x) or
                np.any(self.rl - self.pl > seeds_y))

        # boundary conditions
        ground_truth = \
            np.array([self.global_label_batch[batch,
                                              seeds_x, seeds_y] == id][0],
                     dtype=theano.config.floatX)
        return ground_truth

    def crop_raw(self, seed, batch_counter):
        raw = self.global_batch[batch_counter,
                                seed[0] - self.pad:seed[0] + self.pad,
                                seed[1] - self.pad:seed[1] + self.pad]
        return raw

    def crop_mask_claimed(self, seed, b, id):
        labels = self.global_claims[b,
                                    seed[0] - self.pad:seed[0] + self.pad,
                                    seed[1] - self.pad:seed[1] + self.pad]
        claimed = np.zeros((self.pl, self.pl), dtype=theano.config.floatX) - 1
        claimed[labels != id] = 1       # the others
        claimed[labels == 0] = 0       # me
        return claimed

    def init_train_batch(self):
        self.global_batch = np.zeros((self.bs, self.global_el, self.global_el),
                                     dtype=theano.config.floatX)

        self.global_label_batch = np.zeros((self.bs, self.global_el - self.pl,
                                            self.global_el - self.pl))

        # remember where territory has been claimed before. !=0 claimed, 0 free
        self.global_claims = np.ones((self.bs, self.global_el, self.global_el))
        self.global_claims[:, self.pad:-self.pad, self.pad:-self.pad] = 0

        # extract slices and ids within raw and label cubes
        global_ids = self.prepare_global_batch()
        # extract starting points
        global_seeds = self.get_seeds(global_ids)
        # put seeds and ids in priority queue. All info to load batch is in pq
        self.initialize_priority_queue(global_seeds, global_ids)

    def get_batches(self):
        seeds, ids = self.get_seeds_from_queue()

        raw_batch = np.zeros((self.bs, 2, self.pl, self.pl),
                             dtype=theano.config.floatX)
        gts = np.zeros((self.bs, 4, 1, 1), dtype=theano.config.floatX)

        for b in range(self.bs):
            raw_batch[b, 0, :, :] = self.crop_raw(seeds[b], b)
            raw_batch[b, 1, :, :] = self.crop_mask_claimed(seeds[b], b, ids[b])
            gts[b, :, 0, 0] = self.get_adjacent_gts(seeds[b], b, ids[b])
            self.global_claims[b, seeds[b][0], seeds[b][1]] = ids[b]
        return raw_batch, gts, seeds, ids

    def get_seeds_from_queue(self):
        seeds = []
        ids = []
        for b in range(self.bs):
            already_claimed = True
            while already_claimed:
                if self.priority_queue[b].empty():
                    raise Exception('priority queue empty. All pixels labeled')
                prob, seed, id = self.priority_queue[b].get()
                if self.global_claims[b, seed[0], seed[1]] == 0:
                    already_claimed = False

            seeds.append(seed)
            ids.append(id)

        return seeds, ids

    def update_priority_queue(self, probabilities, seeds, ids):
        assert(len(probabilities) == len(seeds))
        assert(len(probabilities) == self.bs)

        for b in range(self.bs):
            counter = -1
            seeds_x, seeds_y = self.get_cross_coords(seeds[b])
            for x, y in zip(seeds_x, seeds_y):
                counter += 1
                if self.global_claims[b, x, y] == 0:
                    self.priority_queue[b].put((1 - probabilities[b][counter],
                                                [x, y], ids[b]))

    # validation of cube slice by slice
    def init_prediction(self, start, stop):
        self.global_batch = np.zeros((self.bs, self.rl, self.rl),
                                     dtype=theano.config.floatX)
        self.global_label_batch = np.zeros((self.bs, self.rl - self.pl,
                                            self.rl - self.pl),
                                           dtype=theano.config.floatX)
        # remember where territory has been claimed before. !=0 claimed, 0 free
        self.global_claims = np.ones((self.bs, self.rl, self.rl))
        self.global_claims[:, self.pad:-self.pad, self.pad:-self.pad] = 0

        self.global_batch[:, :, :] = self.raw[start:stop, :, :]
        self.global_label_batch[:, :, :] = self.labels[start:stop, :, :]

        global_ids = []
        for b in range(start, stop):
            global_ids.append(
                np.unique(
                    self.global_label_batch[b, :, :]).astype(int))

        # get seeds
        batch = -1
        global_seeds = []
        for ids in global_ids:    # iterates over batches
            batch += 1
            seeds = []
            for id in ids:
                regions = np.where(self.global_label_batch[batch, :, :] == id)
                rand_seed = np.random.randint(0, len(regions[0]))
                seeds.append([regions[0][rand_seed] + self.pad,
                              regions[1][rand_seed] + self.pad])
            global_seeds.append(seeds)

        self.initialize_priority_queue(global_seeds, global_ids)

    def get_pred_batch(self):
        seeds, ids = self.get_seeds_from_queue()
        raw_batch = np.zeros((self.bs, 2, self.pl, self.pl),
                             dtype=theano.config.floatX)

        gts = np.zeros((self.bs, 4, 1, 1), dtype=theano.config.floatX)
        for b in range(self.bs):
            raw_batch[b, 0, :, :] = self.crop_raw(seeds[b], b)
            raw_batch[b, 1, :, :] = self.crop_mask_claimed(seeds[b], b, ids[b])
            gts[b, :, 0, 0] = self.get_adjacent_gts(seeds[b], b, ids[b])
            self.global_claims[b, seeds[b][0], seeds[b][1]] = ids[b]
        return raw_batch, gts, seeds, ids




if __name__ == '__main__':

    # loading of cremi
    # path = './data/sample_A_20160501.hdf'
    # /da
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












