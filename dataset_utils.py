import h5py as h
import numpy as np
from matplotlib import pyplot as plt
import theano
from Queue import PriorityQueue
import utils as u
from scipy.ndimage import convolve, gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage.feature import peak_local_max

import h5py


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
            output.append(np.array(g[key], dtype=theano.config.floatX))

    elif isinstance(h5_key, basestring):   # string
        output = [np.array(g[h5_key], dtype=theano.config.floatX)]

    elif isinstance(h5_key, list):          # list
        output = list()
        for key in h5_key:
            output.append(np.array(g[key], dtype=theano.config.floatX))
    else:
        raise Exception('h5 key type is not supported')

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


def segmentation_to_membrane(input_path,output_path):
    """
    compute a binary mask that indicates the boundary of two touching labels
    input_path: path to h5 label file
    output_path: path to output h5 file (will be created) 
    Uses threshold of edge filter maps
    """
    with h5py.File(input_path, 'r') as label_h5:
        with h5py.File(output_path, 'w') as height_h5:
            boundary_stack = np.empty_like(label_h5['label']).astype(np.float32)
            height_stack = np.empty_like(label_h5['label']).astype(np.float32)
            for i in range(height_stack.shape[0]):
                im = np.float32(label_h5['label'][i])
                gx = convolve(im, np.array([-1., 0., 1.]).reshape(1, 3))
                gy = convolve(im, np.array([-1., 0., 1.]).reshape(3, 1))
                boundary_stack[i] =  np.float32((gx**2 + gy**2) > 0)
                height_stack[i] = distance_transform_edt(boundary_stack[i] == 0)
            height_h5.create_dataset("boundary",data=boundary_stack, dtype=np.float32)
            height_h5.create_dataset("height",data=height_stack, dtype=np.float32)


class BatchManV0:
    def __init__(self, raw, label, height_gt=None, height_gt_key=None,
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
        if isinstance(height_gt, str):
            self.height_gt = 255. - load_h5(height_gt, h5_key=height_gt_key)[0]

        else:
            self.height_gt = height_gt

        self.padding_b = padding_b
        self.pad = patch_len / 2
        if self.padding_b:
            self.raw = mirror_cube(self.raw, self.pad)
        else:
            # crop label
            self.labels = self.labels[:, self.pad:-self.pad, self.pad:-self.pad]
            self.height_gt = self.height_gt[:, self.pad:-self.pad,
                                            self.pad:-self.pad]

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
        self.global_height_gt_batch = None   # no padding
        self.global_heightmap_batch = None
        self.global_directionmap_batch = None
        self.global_time = 0
        self.global_timemap = None
        self.global_errormap = None
        self.global_error_list = None
        self.priority_queue = None

    def prepare_global_batch(self, return_gt_ids=True):
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
            if self.height_gt is not None:
                self.global_height_gt_batch[b, :, :] = \
                    self.height_gt[ind_b[b],
                                   ind_x[b]:ind_x[b] + self.global_el - self.pl,
                                   ind_y[b]:ind_y[b] + self.global_el - self.pl]

            self.global_label_batch[b, :, :] = \
                self.labels[ind_b[b],
                            ind_x[b]:ind_x[b] + self.global_el - self.pl,
                            ind_y[b]:ind_y[b] + self.global_el - self.pl]
            if return_gt_ids:
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
            for Id in ids:
                regions = np.where(
                    self.global_label_batch[batch, :, :] == Id)
                rand_seed = np.random.randint(0, len(regions[0]))
                seeds.append([regions[0][rand_seed] + self.pad,
                              regions[1][rand_seed] + self.pad])
            global_seeds.append(seeds)
        return global_seeds

    def get_seeds_by_minimum(self):
        """
        Seeds by minima of dist trf of thresh of memb prob
        :return:
        """
        self.global_id2gt = []
        bin_membrane = np.ones(self.global_label_batch.shape, dtype=np.bool)
        dist_trf = np.zeros_like(self.global_label_batch)
        global_seeds = []
        global_seed_ids = []
        for b in range(self.bs):
            bin_membrane[b, :, :][self.global_batch[b,
                                                    self.pad:-self.pad,
                                                    self.pad:-self.pad] > 0.3]\
                = 0
            dist_trf[b, :, :] = gaussian_filter(
                distance_transform_edt(bin_membrane[b, :, :]), 1)
            seeds = peak_local_max(dist_trf[b, :, :], exclude_border=0,
                                   threshold_abs=1, min_distance=10)
            global_seeds.append(seeds)
            global_seed_ids.append(range(len(seeds)))
            id2gt = {}
            id_counter = -1
            for seed in seeds:
                id_counter += 1
                id2gt[id_counter] = self.global_label_batch[b, seed[0], seed[1]]
            self.global_id2gt.append(id2gt)
        return global_seeds, global_seed_ids

    def initialize_priority_queue(self, global_seeds, global_ids):
        b = -1  # batch counter
        self.priority_queue = []
        for seeds, ids in zip(global_seeds, global_ids):
            b += 1
            q = PriorityQueue()
            i = -1      # ids within slice
            for seed, Id in zip(seeds, ids):
                i += 1
                q.put((-0.1, seed, Id, -1))
            self.priority_queue.append(q)

    def walk_cross_coords(self, seed):
        offsets = [0,-1,0,1,0]
        for direction,[offset_x,offset_y] in enumerate(zip([-1,0,1,0],[0,-1,0,1])):
            # check boundary conditions
            if ((self.pad <= seed[0] + offset_x < self.global_el - self.pad) and
                (self.pad <= seed[1] + offset_y < self.global_el - self.pad)):
                yield seed[0] + offset_x , seed[1] + offset_y, direction
            else:
                yield seed[0], seed[1], direction

    def initialize_path_priority_queue(self, global_seeds, global_ids):
        """
        Initialize one PQ per batch:
            PQ: ((priority, seed_x, seedy, seed_id, gt_id, direction,
                 error_indicator, time_put)
            Init: e.g. (-0.1, seed_x, seed_y, seed_id, 5958, None, 0
        :param global_seeds:
        :param global_ids:
        :param global_ids_gt:
        :return: [PQ_batch_0, PQ_batch_1,..., PQ_batch_N)
        """
        self.priority_queue = []
        b = -1
        for seeds, ids in zip(global_seeds, global_ids):
            b += 1
            q = PriorityQueue()
            i = -1      # ids within slice
            for seed, id in zip(seeds, ids):
                i += 1
                q.put((-0.1, seed[0], seed[1], id, self.global_id2gt[b][id],
                       None, 0))
            self.priority_queue.append(q)

    def get_cross_coords(self, seed):
        seeds_x = [max(self.pad, seed[0] - 1),
                   seed[0],
                   min(seed[0] + 1, self.global_el - self.pad - 1),
                   seed[0]]


    def get_path_to_root(self, start_position, batch):
        def update_postion(pos, direction):
            offsets = zip([-1,0,1,0],[0,-1,0,1])[int(direction)]
            new_pos = [pos[0]-offsets[0],pos[1]-offsets[1]]
            return new_pos

        counter = 0
        current_position = start_position
        current_direction = self.global_directionmap_batch[batch, current_position[0], current_position[1]]
        path = [start_position]
        while (current_direction != -1):
            current_position = update_postion(current_position, current_direction)
            current_direction = self.global_directionmap_batch[batch, current_position[0], current_position[1]]
            path.append(current_position)
            counter += 1
        return path


    def get_cross_coords(self, seed, global_offset=0):
        seeds_x , seeds_y, dirs = [], [] ,[]
        for seed_x, seed_y, d in self.walk_cross_coords(seed):
            seeds_x.append(seed_x+global_offset)
            seeds_y.append(seed_y+global_offset)
            dirs.append(d)

        return np.array(seeds_x), np.array(seeds_y), np.array(d)

    def get_adjacent_gts(self, seed, batch, id):
        seeds_x, seeds_y, _ = self.get_cross_coords(seed,global_offset = -self.pad)

        assert (np.any(seeds_x >= 0) or np.any(seeds_y >= 0))
        assert (np.any(self.rl - self.pl > seeds_x) or
                np.any(self.rl - self.pl > seeds_y))

        # boundary conditions
        ground_truth = \
            np.array([self.global_label_batch[batch,
                                              seeds_x, seeds_y] == id][0],
                     dtype=theano.config.floatX)
        return ground_truth

    def get_adjacent_heights(self, seed, batch):
        seeds_x, seeds_y, _ = self.get_cross_coords(seed,global_offset = -self.pad)

        assert (np.any(seeds_x >= 0) or np.any(seeds_y >= 0))
        assert (np.any(self.rl - self.pl > seeds_x) or
                np.any(self.rl - self.pl > seeds_y))

        # boundary conditions
        ground_truth = \
            self.global_height_gt_batch[batch, seeds_x, seeds_y].flatten()
        return ground_truth

    def crop_raw(self, seed, batch_counter):
        raw = self.global_batch[batch_counter,
                                seed[0] - self.pad:seed[0] + self.pad,
                                seed[1] - self.pad:seed[1] + self.pad]
        return raw

    def crop_height(self, seed, batch_counter):
        h = self.global_heightmap_batch[batch_counter, :,
                                        seed[0] - self.pad:seed[0] + self.pad,
                                        seed[1] - self.pad:seed[1] + self.pad]
        return h

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

    def init_train_heightmap_batch(self):

        self.global_batch = np.zeros((self.bs, self.global_el, self.global_el),
                                     dtype=theano.config.floatX)

        self.global_label_batch = np.zeros((self.bs, self.global_el - self.pl,
                                            self.global_el - self.pl))
        # remember where territory has been claimed before. !=0 claimed, 0 free
        self.global_claims = np.ones((self.bs, self.global_el, self.global_el))
        self.global_claims[:, self.pad:-self.pad, self.pad:-self.pad] = 0

        self.global_heightmap_batch = np.zeros_like(self.global_claims)
        self.global_height_gt_batch = np.zeros_like(self.global_label_batch)

        # extract slices and ids within raw and label cubes
        global_ids = self.prepare_global_batch()
        # extract starting points
        global_seeds = self.get_seeds(global_ids)
        # put seeds and ids in priority queue. All info to load batch is in pq
        self.initialize_priority_queue(global_seeds, global_ids)

    def init_train_path_batch(self):
        self.global_batch = np.zeros((self.bs, self.global_el, self.global_el),
                                     dtype=theano.config.floatX)

        self.global_label_batch = np.zeros((self.bs, self.global_el - self.pl,
                                            self.global_el - self.pl))
        # remember where territory has been claimed before. !=0 claimed, 0 free
        self.global_claims = np.ones((self.bs, self.global_el, self.global_el))
        self.global_claims[:, self.pad:-self.pad, self.pad:-self.pad] = 0

        self.global_heightmap_batch = np.zeros_like(self.global_claims)
        self.global_height_gt_batch = np.zeros_like(self.global_label_batch)
        self.global_timemap = np.empty_like(self.global_claims)
        self.global_timemap.fill(np.inf)

        self.global_errormap = np.zeros_like(self.global_claims)
        global_error_list = [[] for b in range(self.bs)]

        self.global_directionmap_batch = np.ones_like(self.global_claims)
        self.global_directionmap_batch *= -1

        self.prepare_global_batch(return_gt_ids=False)
        global_seeds, global_seed_ids = self.get_seeds_by_minimum()
        self.initialize_path_priority_queue(global_seeds, global_seed_ids)


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

    def get_heightmap_batches(self):
        seeds, ids = self.get_seeds_from_queue()

        raw_batch = np.zeros((self.bs, 2, self.pl, self.pl),
                             dtype=theano.config.floatX)
        gts = np.zeros((self.bs, 4, 1, 1), dtype=theano.config.floatX)

        for b in range(self.bs):
            raw_batch[b, 0, :, :] = self.crop_raw(seeds[b], b)
            raw_batch[b, 1, :, :] = self.crop_mask_claimed(seeds[b], b, ids[b])
            gts[b, :, 0, 0] = self.get_adjacent_heights(seeds[b], b)
            self.global_claims[b, seeds[b][0], seeds[b][1]] = ids[b]
        return raw_batch, gts, seeds, ids

    def get_seeds_from_queue(self):
        seeds = []
        ids = []
        self.global_time += 1
        for b in range(self.bs):
            already_claimed = True
            while already_claimed:
                if self.priority_queue[b].empty():
                    raise Exception('priority queue empty. All pixels labeled')
                prob, seed, id, id_gt, direction, error_ind, time_put = self.priority_queue[b].get()
                if self.global_claims[b, seed[0], seed[1]] == 0:
                    already_claimed = False

            self.global_directionmap_batch[b, seed[0], seed[1]] = direction
            self.global_timemap[b, seed[0], seed[1]] = time_put

            # check for errors in boundary crossing
            if(error_ind):
                self.global_errormap[b, seeds[0], seed[1]] = 1

            # check for errors in touching regions
            for x, y, direction in  self.walk_cross_coords(seeds[b]):
                n_label = self.id2gt[b][self.global_label_batch[b, x, y]]
                if (self.global_claims[b, x, y] == 1 # neighbor is claimed
                    # and check if claimed by other label
                    and n_label != id_gt):

                    # check for slow intrusion( neighbor is intruder )
                    if(n_label != self.global_label_batch[b, x, y]):
                        global_error_list[b].append((self.global_timemap[b, x, y], seed[0]-x, y-seed[1]))

                    # check for fast intrusion( current is intruder )
                    if(id_gt != self.global_label_batch[b, x, y]):
                        global_error_list[b].append((self.global_timemap[b, x, y], x-seed[0], y-seed[1]))

            seeds.append(seed)
            ids.append(id)

        return seeds, ids

    def update_priority_queue(self, height, seeds, ids):
        assert(len(height) == len(seeds))
        assert(len(height) == self.bs)

        for b in range(self.bs):
            for x, y, direction in  self.walk_cross_coords(seeds[b]):

                d_prev = self.global_heightmap_batch[b, x, y]
                d_j = max(height[b][direction], d_prev)

                if (d_prev > 0):
                    d_j = min(d_j, d_prev)

                self.global_heightmap_batch[b, x, y] = d_j

                if self.global_claims[b, x, y] == 0:
                    self.priority_queue[b].put((d_j, [x, y], ids[b], direction))

    def update_priority_path_queue(self, heights_batch, seeds, ids):
        directions = [0,1,2,3]
        for b, seed, id, heights in zip(range(self.bs), seeds, ids,
                                        heights_batch):
            # if possibly wrong
            new_seeds_x, new_seeds_y = self.get_cross_coords(seed)
            for x, y, height, direction in \
                    zip(new_seeds_x, new_seeds_y, heights, directions):
                error_indicator = False
                # check whether crossing into other gt id region
                if self.global_label_batch[b, x, y] != self.global_id2gt[b][id]\
                        and self.global_label_batch[b, seed[0], seed[1]] == \
                    self.global_id2gt[b][id]:
                    error_indicator = True
                    # Init: e.g.(-0.1, seed_x, seed_y, seed_id, None, 0

                self.priority_queue[b].put(height, x, y, id, direction,
                                           self.global_time)


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

    # segmentation_to_membrane('./data/volumes/label_a.h5',"./data/volumes/height_a.h5")
    # segmentation_to_membrane('./data/volumes/label_b.h5',"./data/volumes/height_b.h5")
    
    # bm = BatchManV0(raw_path, label_path, batch_size=10, patch_len=60,
    #                 global_edge_len=95)
    # bm.init_train_batch()

    net_name = 'cnn_ID2_trash'
    label_path = './data/volumes/label_a.h5'
    label_path_val = './data/volumes/label_b.h5'
    height_gt_path = './data/volumes/height_a.h5'
    height_gt_key = 'height'
    height_gt_path_val = './data/volumes/height_b.h5'
    height_gt_key_val = 'height'
    raw_path = './data/volumes/membranes_a.h5'
    raw_path_val = './data/volumes/membranes_b.h5'
    save_net_path = './data/nets/' + net_name + '/'
    load_net_path = './data/nets/cnn_ID_2/net_300000'      # if load true
    tmp_path = '/media/liory/ladata/bla'        # debugging
    batch_size = 16         # > 4
    global_edge_len = 300
    patch_len= 40

    bm = BatchManV0(raw_path, label_path,
                       height_gt=height_gt_path,
                       height_gt_key=height_gt_key,
                       batch_size=batch_size,
                       patch_len=patch_len, global_edge_len=global_edge_len,
                       padding_b=False)
    bm.init_train_path_batch()

    # print bm.get_batches(10*[[45, 46]])












