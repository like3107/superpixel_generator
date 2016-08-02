import h5py as h
import numpy as np
import random
from os.path import exists
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import theano
from Queue import PriorityQueue
import utils as u
from scipy.ndimage import convolve, gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage.feature import peak_local_max

import h5py


def load_h5(path, h5_key=None, group=None, group2=None):
    if not exists(path):
        error = 'path: %s does not exist, check' % path
        raise Exception(error)
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
                boundary_stack[i], height_stack[i] = \
                    segmenation_to_membrane_core(im)
            height_h5.create_dataset("boundary",data=boundary_stack, dtype=np.float32)
            height_h5.create_dataset("height",data=height_stack, dtype=np.float32)


def segmenation_to_membrane_core(label_image):
    gx = convolve(label_image, np.array([-1., 0., 1.]).reshape(1, 3))
    gy = convolve(label_image, np.array([-1., 0., 1.]).reshape(3, 1))
    boundary= np.float32((gx ** 2 + gy ** 2) > 0)
    height = distance_transform_edt(boundary == 0)
    return boundary, height


class BatchManV0:
    def __init__(self, raw, label, height_gt=None, height_gt_key=None,
                 raw_key=None, label_key=None, batch_size=10,
                 global_edge_len=110, patch_len=40, padding_b=False,
                 train_b=True, find_errors=False,
                 gt_seeds_b=False):
        """
        batch loader. Use either for predict OR train. For valId and train use:
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

        self.train_b = train_b
        if not train_b:     # tmp
            assert (padding_b)
        if isinstance(raw, str):
            self.raw = load_h5(raw, h5_key=raw_key)[0]
        else:
            self.raw = raw
        if self.train_b:
            if isinstance(label, str):
                self.labels = load_h5(label, h5_key=label_key)[0]
            else:
                self.labels = label
            if isinstance(height_gt, str):
                self.height_gt = load_h5(height_gt, h5_key=height_gt_key)[0]
                maximum = np.max(self.height_gt)
                self.height_gt = maximum - self.height_gt
            else:
                self.height_gt = height_gt
        else:
            self.height_gt = None
        # either pad raw or crop labels -> labels are always shifted by self.pad
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
        self.global_el = global_edge_len        # length of field, global_batch
        # includes padding)
        self.pl = patch_len

        assert(patch_len <= global_edge_len)
        assert(global_edge_len <= self.rl)
        if not train_b:
            assert (self.rl == self.global_el)

        if self.rl - self.global_el < 0:
            raise Exception('try setting padding to True')

        # private
        self.global_batch = None                # includes padding, nn input
        self.global_claims = None               # includes padding, tri-map, inp
        self.global_directionmap_batch = None   # no padding
        self.global_label_batch = None          # no padding
        self.global_height_gt_batch = None      # no padding
        self.global_heightmap_batch = None      # no padding
        self.global_timemap = None              # no padding
        self.global_errormap = None             # no padding
        self.gt_seeds_b = gt_seeds_b
        self.global_seeds = None                # !!ALL!! coords include padding
        self.global_time = 0
        self.global_error_dict = None
        self.priority_queue = None
        self.crossing_errors = None
        self.find_errors = find_errors

    def prepare_global_batch(self, return_gt_ids=True, start=None):
        # initialize two global batches = region where CNNs compete
        # against each other

        # get indices for global batches in raw/ label cubes
        if start is None:
            ind_b = np.random.permutation(self.n_slices)[:self.bs]
        else:
            ind_b = np.arange(start + self.bs)

        # indices to raw, correct for label which edge len is -self.pl shorter
        ind_x = np.random.randint(0,
                                  self.rl - self.global_el + 1,
                                  size=self.bs)
        ind_y = np.random.randint(0,
                                  self.rl - self.global_el + 1,
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
            if self.train_b:
                self.global_label_batch[b, :, :] = \
                    self.labels[ind_b[b],
                                ind_x[b]:ind_x[b] + self.global_el - self.pl,
                                ind_y[b]:ind_y[b] + self.global_el - self.pl]
            if return_gt_ids and self.train_b:
                global_ids.append(
                    np.unique(
                        self.global_label_batch[b, :, :]).astype(int))
                return global_ids

    def get_gt_seeds(self):    # seed coords relative to global batch
        """
        return seeds by minimum of dist trf of thresholded membranes
        :param sigma: smothing of dist trf
        :param dist_trf:
        :return:
        """
        global_seeds = []   # seeds relative to global_claims, global_batch
        self.global_id2gt = []

        global_ids = []
        dist_trf = np.zeros_like(self.global_label_batch)
        #  map (- self.pad for global_label_label)
        for b in range(self.bs):
            global_ids.append(np.unique(
                self.global_label_batch[b, :, :]).astype(int))

            _, dist_trf[b, :, :] = \
                segmenation_to_membrane_core(self.global_label_batch[b, :, :])
        id_old = 0

        for b, ids in zip(range(self.bs), global_ids):    # iterates over batches
            seeds = []
            id2gt = {}
            for Id in ids:      # ids within each slice
                assert (Id != id_old)
                id_old = Id
                id2gt[Id] = Id
                regions = np.where(
                    self.global_label_batch[b, :, :] == Id)
                seed_ind = np.argmax(dist_trf[b][regions])
                seed = np.array([regions[0][seed_ind], regions[1][seed_ind]]) \
                                 + self.pad
                seeds.append([seed[0], seed[1]])
            self.global_id2gt.append(id2gt)
            global_seeds.append(seeds)
        self.global_seeds = global_seeds
        return global_seeds, global_ids

    def get_seeds_by_minimum(self, sigma=2, min_dist=8, thresh=0.3):
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
                                                    self.pad:-self.pad] > thresh]\
                = 0
            dist_trf[b, :, :] = gaussian_filter(
                distance_transform_edt(bin_membrane[b, :, :]), sigma=sigma)
            # seeds in coord system of labels
            seeds = np.array(
                peak_local_max(dist_trf[b, :, :], exclude_border=0,
                                threshold_abs=1, min_distance=min_dist))
            seeds += self.pad
            if len(seeds) == 0:
                print 'WARNING no seeds found (no minima in global batch). ' \
                      'Setting seed to middle'
                seeds = np.array([self.global_el / 2, self.global_el / 2])

            global_seeds.append(seeds)
            global_seed_ids.append(range(1, len(seeds)+1))
            id2gt = {}
            for id_counter, seed in enumerate(seeds):
                id2gt[id_counter+1] = \
                    self.global_label_batch[b,
                                            seed[0]-self.pad,
                                            seed[1]-self.pad]
            self.global_id2gt.append(id2gt)
        self.global_seeds = global_seeds
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

    def initialize_path_priority_queue(self, global_seeds, global_ids):
        """
        Initialize one PQ per batch:
            PQ: ((height, seed_x, seedy, seed_id, direction, error_indicator,
            time_put)
            Init: e.g. (-0.1, seed_x, seed_y, seed_id, None, 0
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
            for seed, Id in zip(seeds, ids):
                i += 1
                q.put((-1., seed[0], seed[1], Id, -1, False, 0))
            self.priority_queue.append(q)

    def walk_cross_coords(self, center):
        # walk in coord system of global label batch: x : 0 -> global_el - pl
        # use center if out of bounds
        center_x, center_y = center
        for direction, [offset_x, offset_y] in \
                enumerate(zip([-1, 0, 1, 0], [0, -1, 0, 1])):
            # check boundary conditions
            if (self.pad <= center_x + offset_x < self.global_el - self.pad) and\
                    (self.pad <= center_y + offset_y < self.global_el - self.pad):
                yield center_x + offset_x, center_y + offset_y, direction
            else:
                yield center_x, center_y, direction

    def get_path_to_root(self, start_position, batch):
        def update_postion(pos, direction):
            offsets = zip([-1, 0, 1, 0], [0, -1, 0, 1])[int(direction)]
            new_pos = [pos[0]-offsets[0], pos[1]-offsets[1]]
            return new_pos

        current_position = start_position
        current_direction = \
            self.global_directionmap_batch[batch,
                                           current_position[0]-self.pad,
                                           current_position[1]-self.pad]
        yield start_position, current_direction
        while (current_direction != -1):
            current_position = update_postion(current_position,
                                              current_direction)
            current_direction = \
                self.global_directionmap_batch[batch,
                                               current_position[0]-self.pad,
                                               current_position[1]-self.pad]
            yield current_position, current_direction

    def find_type_I_error(self): # 1st crossing from own gt ID into other ID
        for error_I in self.global_error_dict.values():
            if not "crossing" in error_I:
                start_position = error_I["large_pos"]
                batch = error_I["batch"]
                for pos, d in self.get_path_to_root(start_position, batch):
                    # debug
                    self.global_errormap[batch, 2, pos[0]-self.pad,
                                         pos[1]-self.pad] = True
                    # debug
                    if self.global_errormap[batch, 0, pos[0]-self.pad,
                                            pos[1]-self.pad]:
                        original_error = np.array(pos)
                        print 'found crossing'

                        error_I["crossing"] = original_error
                        error_I["crossing_time"] = self.global_timemap[batch,
                                                                       pos[0],
                                                                       pos[1]]
                        error_I["crossing_direction"] = d
           # debug
        self.draw_debug_image("walk_"+str(len(self.global_error_dict)),
                              save=True)
        # self.draw_error_reconst("reconst_"+str(len(self.global_error_dict)))

    def get_cross_coords(self, seed, global_offset=0):
        seeds_x, seeds_y, dirs = [], [], []
        for seed_x, seed_y, d in self.walk_cross_coords(seed):
            seeds_x.append(seed_x+global_offset)
            seeds_y.append(seed_y+global_offset)
            dirs.append(d)

        return np.array(seeds_x), np.array(seeds_y), np.array(d)

    def get_adjacent_gts(self, seed, batch, Id):
        seeds_x, seeds_y, _ = self.get_cross_coords(seed,
                                                    global_offset=-self.pad)

        assert (np.any(seeds_x >= 0) or np.any(seeds_y >= 0))
        assert (np.any(self.rl - self.pl > seeds_x) or
                np.any(self.rl - self.pl > seeds_y))

        # boundary conditions
        ground_truth = \
            np.array([self.global_label_batch[batch,
                                              seeds_x, seeds_y] == Id][0],
                     dtype=theano.config.floatX)
        return ground_truth

    def get_adjacent_heights(self, seed, batch):
        seeds_x, seeds_y, _ = self.get_cross_coords(seed,
                                                    global_offset=-self.pad)

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

    def crop_mask_claimed(self, seed, b, Id):
        labels = self.global_claims[b,
                                    seed[0] - self.pad:seed[0] + self.pad,
                                    seed[1] - self.pad:seed[1] + self.pad]
        claimed = np.zeros((self.pl, self.pl), dtype=theano.config.floatX) - 1
        claimed[labels != Id] = 1       # the others
        claimed[labels == 0] = 0       # me
        return claimed

    def crop_timemap(self, center, b):
        # assert(center[0]-self.pad > 0)
        # assert(center[1]-self.pad > 0)
        # assert(center[0]+self.pad < self.global_el - self.pl)
        # assert(center[1]+self.pad < self.global_el - self.pl)

        return self.global_timemap[b,center[0]-self.pad:center[0]+self.pad,
                                   center[1]-self.pad:center[1]+self.pad]

    def crop_time_mask(self, centers, timepoint, batches):
        """
        compute mask that is 1 if a voxel was not accessed before timepoint
        and zero otherwise
        """
        mask = np.zeros((len(batches), self.pl, self.pl),dtype=bool)
        # fore lists to np.array so we can do array arithmetics
        centers = np.array(centers)
        for i,b in enumerate(batches):
            mask[i,:,:][self.crop_timemap(centers[i], b)>timepoint[i]] = 1
        return mask

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

        self.global_heightmap_batch = np.zeros_like(self.global_label_batch)
        self.global_height_gt_batch = np.zeros_like(self.global_label_batch)

        # extract slices and ids within raw and label cubes
        global_ids = self.prepare_global_batch()    # seeds in label coord syst
        # extract starting points
        global_seeds = self.get_seeds(global_ids)
        # put seeds and ids in priority queue. All info to load batch is in pq
        self.initialize_priority_queue(global_seeds, global_ids)

    def init_train_path_batch(self):
        self.global_batch = np.zeros((self.bs, self.global_el, self.global_el),
                                     dtype=theano.config.floatX)
        self.global_label_batch = np.zeros((self.bs, self.global_el - self.pl,
                                            self.global_el - self.pl),
                                           dtype=theano.config.floatX)
        # remember where territory has been claimed before. !=0 claimed, 0 free
        self.global_claims = np.ones((self.bs, self.global_el, self.global_el))
        self.global_claims[:, self.pad:-self.pad, self.pad:-self.pad] = 0

        self.global_heightmap_batch = np.zeros_like(self.global_label_batch)
        self.global_height_gt_batch = np.zeros_like(self.global_label_batch)
        self.global_timemap = np.empty_like(self.global_batch, dtype=np.int)
        self.global_timemap.fill(np.inf)
        self.global_time = 0
        self.global_errormap = np.zeros((self.bs, 3,self.global_el - self.pl,
                                            self.global_el - self.pl),
                                        dtype=np.bool)

        self.global_error_dict = {}

        self.global_directionmap_batch = np.zeros_like(self.global_label_batch)\
                                         - 1
        self.prepare_global_batch(return_gt_ids=False)
        # also initializes id_2_gt lookup, seeds in coord syst of label
        if self.gt_seeds_b:
            global_seeds, global_seed_ids = self.get_gt_seeds()
        else:
            global_seeds, global_seed_ids = self.get_seeds_by_minimum()
        self.initialize_path_priority_queue(global_seeds, global_seed_ids)
        return global_seeds # debug only, remove me, tmp

    def get_batches(self):
        centers, ids = self.get_centers_from_queue()

        raw_batch = np.zeros((self.bs, 2, self.pl, self.pl),
                             dtype=theano.config.floatX)
        gts = np.zeros((self.bs, 4, 1, 1), dtype=theano.config.floatX)

        for b in range(self.bs):
            raw_batch[b, 0, :, :] = self.crop_raw(centers[b], b)
            raw_batch[b, 1, :, :] = self.crop_mask_claimed(centers[b], b, ids[b])
            gts[b, :, 0, 0] = self.get_adjacent_gts(centers[b], b, ids[b])
            self.global_claims[b, centers[b][0], centers[b][1]] = ids[b]
        return raw_batch, gts, centers, ids

    def get_path_batches(self):
        centers, ids = self.get_centers_from_queue()
        raw_batch = np.zeros((self.bs, 2, self.pl, self.pl),
                             dtype=theano.config.floatX)
        gts = np.zeros((self.bs, 4, 1, 1), dtype=theano.config.floatX)
        for b in range(self.bs):
            raw_batch[b, 0, :, :] = self.crop_raw(centers[b], b)
            raw_batch[b, 1, :, :] = self.crop_mask_claimed(centers[b], b, ids[b])
            gts[b, :, 0, 0] = self.get_adjacent_heights(centers[b], b)
            # check whether already pulled
            assert(self.global_claims[b, centers[b][0], centers[b][1]] == 0)
            self.global_claims[b, centers[b][0], centers[b][1]] = ids[b]
        return raw_batch, gts, centers, ids

    def get_path_error_batch(self):
        centers, ids = self.get_centers_from_queue()
        total_number_of_errors = len(self.global_error_dict)
        print "total_number_of_errors",total_number_of_errors
        if total_number_of_errors > 0:
            print "FOUND ONE"
            print "ERRORLIST ",self.global_error_dict
            # self.draw_debug_image("error_list_found"+str(self.global_time))
        return raw_batch, gts, centers, ids

    def get_heightmap_batches(self):
        centers, ids = self.get_centers_from_queue()
        raw_batch = np.zeros((self.bs, 2, self.pl, self.pl),
                             dtype=theano.config.floatX)
        gts = np.zeros((self.bs, 4, 1, 1), dtype=theano.config.floatX)
        for b in range(self.bs):
            raw_batch[b, 0, :, :] = self.crop_raw(centers[b], b)
            raw_batch[b, 1, :, :] = self.crop_mask_claimed(centers[b], b, ids[b])
            gts[b, :, 0, 0] = self.get_adjacent_heights(centers[b], b)
            self.global_claims[b, centers[b][0], centers[b][1]] = ids[b]
        return raw_batch, gts, centers, ids


    def get_centers_from_queue(self):
        centers = []
        ids = []
        # remember if crossed once and always carry this information along every
        # path for error indicator
        self.error_indicator_pass = np.zeros(self.bs, dtype=np.bool)
        def error_index(b, id1, id2):
            return (b, min(id1, id2), max(id1, id2))

        self.global_time += 1
        for b in range(self.bs):
            # pull from pq at free pixel position
            already_claimed = True
            while already_claimed:
                if self.priority_queue[b].empty():
                    raise Exception('priority queue empty. All pixels labeled')
                height, center_x, center_y, Id, direction, error_indicator, \
                    time_put = self.priority_queue[b].get()
                if self.global_claims[b, center_x, center_y] == 0:
                    already_claimed = False
            assert (self.global_claims[b, center_x, center_y] == 0)
            assert(self.pad <= center_x < self.global_el - self.pad)
            assert(self.pad <= center_y < self.global_el - self.pad)
            self.global_directionmap_batch[b,
                                           center_x - self.pad,
                                           center_y - self.pad] = direction
            self.global_timemap[b,
                                center_x,
                                center_y] = time_put

            # check for errors in boundary crossing (type I)
            if error_indicator:
                self.error_indicator_pass[b] = True     # remember to pass on
                self.global_errormap[b, 1, center_x-self.pad, center_y-self.pad] \
                    = 1

            elif self.global_id2gt[b][Id] != \
                    self.global_label_batch[b, center_x-self.pad,
                                            center_y-self.pad]:
                self.global_errormap[b, 0, center_x-self.pad,
                                     center_y-self.pad] \
                    = 1
                self.error_indicator_pass[b] = True

            # check for errors in neighbor regions
            if self.find_errors:
                for x, y, direction in self.walk_cross_coords([center_x,
                                                               center_y]):
                    # print self.global_errormap[b,x-self.pad,y-self.pad]
                    c = int(self.global_claims[b, x, y])
                    if c > 0 and not error_index(b,Id,c) in self.global_error_dict:
                        claimId = int(self.global_id2gt[b][c])
                        gtId = int(self.global_id2gt[b][Id])
                        if claimId > 0 and claimId != gtId:  # neighbor is claimed
                            # print "neighbor is claimed"
                            center_introder_b = \
                                self.global_errormap[b, 1, center_x-self.pad,
                                                    center_y-self.pad]
                            neighbor_introduer_b = \
                                self.global_errormap[b, 1, x-self.pad,
                                                     y-self.pad]
                            if center_introder_b and not neighbor_introduer_b:
                                # print "fast intrusion"
                                self.global_error_dict[error_index(b,Id,c)]  = {"batch":b,
                                  "touch_time":self.global_timemap[b, x, y],
                                  "large_pos":[center_x, center_y],
                                  "large_direction":direction,
                                  "large_id":Id,
                                  "small_pos":[x, y],
                                  "small_id":c}
                                self.find_type_I_error()
                            elif not center_introder_b and neighbor_introduer_b:
                                # print "slow intrusion"
                                self.global_error_dict[error_index(b,Id,c)] = {"batch":b,
                                  "touch_time":self.global_timemap[b, x, y],
                                  "large_pos":[x,y],
                                  "large_direction":(direction + 2)%4, # turns direction by 180 degrees
                                  "large_id":c,
                                  "small_pos":[center_x, center_y],
                                  "small_id":Id}
                                self.find_type_I_error()
                            elif center_introder_b and neighbor_introduer_b:
                                # TODO: TYPE 3 error tmp
                                # raise Exception('error type 3 found')
                                print 'type 3 error not yet implemented'
                                # self.find_type_I_error()

            centers.append((center_x, center_y))
            ids.append(Id)

        return centers, ids

    # copy of update priority path priority queue without all path error stuff
    def get_centers_from_queue_prediction(self):

        centers, ids = [], []
        for b in range(self.bs):
            # pull from pq at free pixel position
            already_claimed = True
            while already_claimed:
                if self.priority_queue[b].empty():
                    raise Exception('priority queue empty. All pixels labeled')
                height, center_x, center_y, Id, direction, error_indicator, \
                    time_put = self.priority_queue[b].get()
                if self.global_claims[b, center_x, center_y] == 0:
                    already_claimed = False
            assert (self.global_claims[b, center_x, center_y] == 0)
            assert(self.pad <= center_x < self.global_el - self.pad)
            assert(self.pad <= center_y < self.global_el - self.pad)

            centers.append((center_x, center_y))
            ids.append(Id)
        return centers, ids

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
                    self.priority_queue[b].put((d_j, x, y, ids[b], direction))

    def update_priority_path_queue(self, heights_batch, centers, ids):
        directions = [0, 1, 2, 3]
        for b, center, Id, heights in zip(range(self.bs), centers, ids,
                                        heights_batch[:, :, 0, 0]):
            # if possibly wrong
            new_seeds_x, new_seeds_y, _ = self.get_cross_coords(center)

            # check for touching errors
            for x, y, height, direction in \
                    zip(new_seeds_x, new_seeds_y, heights, directions):
                if self.error_indicator_pass[b]:
                    error_indicator = True
                else:
                    error_indicator = False

                if self.global_claims[b, x, y] == 0:

                    height_prev = self.global_heightmap_batch[b, x-self.pad,
                                                              y-self.pad]
                    height_j = max(heights[direction], height_prev)
                    if height_prev > 0:
                        height_j = min(height_j, height_prev)
                    self.global_heightmap_batch[b, x-self.pad, y-self.pad] = \
                        height_j
                    self.priority_queue[b].put((height, x, y,
                                                Id, direction,
                                               error_indicator, self.global_time))

    def update_priority_path_queue_prediction(self, heights_batch, centers, ids):
        directions = [0, 1, 2, 3]
        for b, center, Id, heights in zip(range(self.bs), centers, ids,
                                        heights_batch[:, :, 0, 0]):
            # if possibly wrong
            new_seeds_x, new_seeds_y, _ = self.get_cross_coords(center)

            # check for touching errors
            for x, y, height, direction in \
                    zip(new_seeds_x, new_seeds_y, heights, directions):

                if self.global_claims[b, x, y] == 0:

                    height_prev = self.global_heightmap_batch[b, x-self.pad,
                                                              y-self.pad]
                    height_j = max(heights[direction], height_prev)
                    if height_prev > 0:
                        height_j = min(height_j, height_prev)
                    self.global_heightmap_batch[b, x-self.pad, y-self.pad] = \
                        height_j
                    self.priority_queue[b].put((height, x, y,
                                                Id, direction,
                                               False, self.global_time))

    def reconstruct_input_at_timepoint(self, timepoint, centers, ids, batches):
        raw_batch = np.zeros((len(batches), 2, self.pl, self.pl),
                             dtype=theano.config.floatX)
        for i,b in enumerate(batches):
            raw_batch[i, 0, :, :] = self.crop_raw(centers[i], b)
            raw_batch[i, 1, :, :] = self.crop_mask_claimed(centers[i], b, ids[i])

        mask = self.crop_time_mask(centers, timepoint, batches)
        raw_batch[:, 1, :, :][mask] = 0
        return raw_batch

    # validation of cube slice by slice
    def init_prediction(self, start, stop):
        self.global_batch = np.zeros((self.bs, self.rl, self.rl),
                                     dtype=theano.config.floatX)
        # remember where territory has been claimed before. !=0 claimed, 0 free
        self.global_claims = np.ones((self.bs, self.rl, self.rl))
        self.global_claims[:, self.pad:-self.pad, self.pad:-self.pad] = 0
        self.global_batch[:, :, :] = self.raw[start:stop, :, :]
        self.prepare_global_batch(return_gt_ids=False, start=start)

        global_seeds, global_seed_ids = self.get_seeds_by_minimum()
        self.initialize_path_priority_queue(global_seeds, global_seed_ids)

    def get_pred_batch(self):
        centers, ids = self.get_centers_from_queue_prediction()
        raw_batch = np.zeros((self.bs, 2, self.pl, self.pl),
                             dtype=theano.config.floatX)
        for b in range(self.bs):
            raw_batch[b, 0, :, :] = self.crop_raw(centers[b], b)
            raw_batch[b, 1, :, :] = self.crop_mask_claimed(centers[b], b,
                                                           ids[b])
            assert (self.global_claims[b, centers[b][0], centers[b][1]] == 0)
            self.global_claims[b, centers[b][0], centers[b][1]] = ids[b]
        # check whether already pulled
        # self.global_claims[b, centers[b][0], centers[b][1]] = ids[b]
        return raw_batch, centers, ids

    def reconstruct_path_error_inputs(self):
        error_I_timelist = []
        error_I_pos_list = []
        error_I_id_list = []
        error_batch_list = []
        error_II_pos_list = []
        error_II_time_list = []
        error_II_id_list = []
        error_I_direction = []
        error_II_direction = []

        for error in self.global_error_dict.values():
            error_batch_list.append(error["batch"])
            error_I_timelist.append(error["crossing_time"])
            error_I_direction.append(error["crossing_direction"])
            error_I_pos_list.append(error["crossing"])
            error_I_id_list.append(error["large_id"])
            error_II_pos_list.append(error["small_pos"])
            error_II_direction.append(error["large_direction"])
            error_II_time_list.append(error["touch_time"])
            error_II_id_list.append(error["small_id"])

        reconst_e1 = self.reconstruct_input_at_timepoint(error_I_timelist,
                                                         error_I_pos_list,
                                                         error_I_id_list,
                                                         error_batch_list)
        reconst_e2 = self.reconstruct_input_at_timepoint(error_II_time_list,
                                                         error_II_pos_list,
                                                         error_II_id_list,
                                                         error_batch_list)
        return reconst_e1, reconst_e2 , np.array(error_I_direction), np.array(error_II_direction)

    def draw_error_reconst(self, image_name, path='./data/nets/debug/images/', save=True):
        for e_idx, error in self.global_error_dict.items():
            plot_images = []
            if not "draw_file" in error:
                reconst_e1 = self.reconstruct_input_at_timepoint( [error["crossing_time"]], [error["crossing"]], [error["large_id"]], [error["batch"]])
                reconst_e2 = self.reconstruct_input_at_timepoint( [error["touch_time"]], [error["small_pos"]], [error["small_id"]], [error["batch"]])

                # plot_images.append({"title":"Raw Input",
                #                     'im':reconst_e1[i, 0, :, :]})
                # plot_images.append({"title":"timemap",
                #                     'im':self.crop_timemap(np.array(error["crossing"]), error_I_batch_list[i])})
                plot_images.append({"title":"Ground Truth Label",
                        "cmap":"rand",
                        'im':self.global_label_batch[error["batch"], error["crossing"][0] - 2*self.pad:error["crossing"][0],
                                        error["crossing"][1] - 2*self.pad:error["crossing"][1]]})
                plot_images.append({"title":"reconst claims at t="+str(error["crossing_time"]),
                                    'cmap':"rand",
                                    'im':reconst_e1[0, 1, :, :]})
                plot_images.append({"title":"final claims",
                                    'cmap':"rand",
                                    'im':self.global_claims[error["batch"],
                                        error["crossing"][0] - self.pad:error["crossing"][0] + self.pad,
                                        error["crossing"][1] - self.pad:error["crossing"][1] + self.pad]})

                plot_images.append({"title":"E2 Ground Truth Label",
                        "cmap":"rand",
                        'im':self.global_label_batch[error["batch"], error["small_pos"][0] - 2*self.pad:error["small_pos"][0],
                                        error["small_pos"][1] - 2*self.pad:error["small_pos"][1]]})
                plot_images.append({"title":"E2 reconst claims at t="+str(error["touch_time"]),
                                    'cmap':"rand",
                                    'im':reconst_e2[0, 1, :, :]})
                plot_images.append({"title":"E2 final claims",
                                    'cmap':"rand",
                                    'im':self.global_claims[error["batch"],
                                        error["small_pos"][0] - self.pad:error["small_pos"][0] + self.pad,
                                        error["small_pos"][1] - self.pad:error["small_pos"][1] + self.pad]})
                print "plotting ",image_name+'_'+str(e_idx)
                error["draw_file"] = image_name+'_'+str(e_idx)
                u.save_images(plot_images, path=path, name=image_name+'_'+str(e_idx))
            else:
                print "skipping ",e_idx


    def draw_debug_image(self, image_name, path='./data/nets/debug/images/',
                         save=True, b=0):
        plot_images = []
        plot_images.append({"title":"Claims",
                            'cmap':"rand",
                            'im':self.global_claims[b, self.pad:-self.pad-1,
                                 self.pad:-self.pad-1]})
        plot_images.append({"title":"Raw Input",
                            'im':self.global_batch[b, self.pad:-self.pad-1,
                                 self.pad:-self.pad-1]})
        plot_images.append({"title":"Heightmap Prediciton",
                            'im':self.global_heightmap_batch[b, :, :]})
        plot_images.append({"title":"Heightmap Ground Truth",
                            'im':self.global_height_gt_batch[b, :, :],
                            'scatter':np.array(self.global_seeds[b])-self.pad})
        print self.global_error_dict
        print [np.array(e["crossing"])-self.pad for e in self.global_error_dict.values() if "crossing" in e]
        print [np.array(e["crossing"])-self.pad for e in self.global_error_dict.values() if "crossing" in e and e["batch"] == b]
        plot_images.append({"title":"Ground Truth Label",
                            'scatter':np.array([np.array(e["crossing"])-self.pad for e in self.global_error_dict.values() if "crossing" in e and e["batch"] == 4]),
                            "cmap":"rand",
                            'im':self.global_label_batch[b, :, :]})
        plot_images.append({"title":"Error Map",
                            'im':self.global_errormap[b, 0, :, :]})
        plot_images.append({"title":"path Map",
                            'scatter':np.array([np.array(e["large_pos"])-self.pad for e in self.global_error_dict.values() if e["batch"] == b]),
                            'im':self.global_errormap[b, 2, :, :]})
        plot_images.append({"title":"Direction Map",
                            'im':self.global_directionmap_batch[b, :, :]})
        plot_images.append({"title":"Time Map ",
                    'im':self.global_timemap[b, :, :]})

        if save:
            u.save_images(plot_images, path=path, name=image_name)
        else:
            print 'show'
            plt.show()


class BatchMemento:
    """
    Remembers slices for CNN in style bc01
    """
    def __init__(self, bs, memory_size, memorize_direction=True,
                 random_return=False):
        self.bs = bs
        self.ms = memory_size
        self.memory = None
        self.full_b = False
        self.counter = 0
        self.memorize_dir_b = memorize_direction
        self.direction_memory = None

    def add_to_memory(self, mini_b, dir_memory):
        if self.memory is None:
            self.memory = np.zeros(([self.ms] + list(mini_b.shape[-3:])),
                                   dtype=theano.config.floatX)
            self.direction_memory = np.zeros(self.ms, dtype=theano.config.floatX)
        if len(mini_b.shape) == 3:  # if single slice

            slices_to_add = 1
        else:
            slices_to_add = mini_b.shape[0]

        if self.counter+slices_to_add > self.ms:
            np.roll(self.memory, self.ms - self.counter+slices_to_add, axis=0)
            np.roll(self.direction_memory, self.ms - self.counter+slices_to_add, axis=0)
            self.counter = self.ms - slices_to_add

        self.memory[self.counter:self.counter+slices_to_add, :, :, :] = \
            mini_b
        self.direction_memory[self.counter:self.counter+slices_to_add] = \
            mini_b
        self.counter += slices_to_add

    def is_ready(self):
        return self.counter >= self.bs

    def get_batch(self):
        assert(self.is_ready())
        return self.memory[self.counter-self.bs:self.counter], \
               self.direction_memory[self.counter-self.bs:self.counter]

    def clear_memory(self):
        self.memory = None
        self.direction_memory = None


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
    batch_size = 8         # > 4
    global_edge_len = 300
    patch_len= 40

    bm = BatchManV0(raw_path, label_path,
                    height_gt=height_gt_path,
                    height_gt_key=height_gt_key,
                    batch_size=batch_size,
                    patch_len=patch_len, global_edge_len=global_edge_len,
                    padding_b=True,find_errors=True)
    gt_seeds_b = True
    seeds = bm.init_train_path_batch()
    seeds = np.array(seeds[4])
    heights = np.random.random(size=batch_size)
    b = 4
    raw_batch, gts, centers, ids = bm.get_path_batches()
    name = 'debug'
    for i in range(500000):
        if i % 100 == 0:
            print i

        if i % 1000 == 0:
            bm.draw_debug_image(name + '_deb_%i' %i, save=True)
            print i
        raw_batch, gts, centers, ids = bm.get_path_batches()

        if i % 5000 == 0 and i != 0:
            bm.init_train_path_batch()
            raw_batch, gts, centers, ids = bm.get_path_batches()
            name += 'asdf'
        probs = np.zeros((batch_size, 4, 1,1))
        for c in range(batch_size):
            d = 0
            for x, y, _ in bm.walk_cross_coords(centers[b]):
                x -= bm.pad
                y -= bm.pad
                # probs[c, d] = bm.global_height_gt_batch[b, x, y]
                # probs[c, d] = i
                probs[c, d] = random.random()
                d += 1
        bm.update_priority_path_queue(probs, centers, ids)
















