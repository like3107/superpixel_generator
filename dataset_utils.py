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
        if isinstance(raw, str):
            self.raw = load_h5(raw, h5_key=raw_key)[0]
        else:
            self.raw = raw
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
        assert(global_edge_len < self.rl)

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
        self.global_seeds = None                # coords include padding
        self.global_time = 0
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
            seeds = peak_local_max(dist_trf[b, :, :], exclude_border=0,
                                   threshold_abs=1, min_distance=min_dist)
            global_seeds.append(np.array(seeds) + self.pad)
            global_seed_ids.append(range(1, len(seeds)+1))
            id2gt = {}
            id_counter = 0
            for seed in seeds:
                id_counter += 1
                id2gt[id_counter] = self.global_label_batch[b,
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
                q.put((0, seed[0], seed[1], Id, -1, False, 0))
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
            offsets = zip([-1,0,1,0],[0,-1,0,1])[int(direction)]
            new_pos = [pos[0]-offsets[0], pos[1]-offsets[1]]
            return new_pos

        current_position = start_position
        current_direction = \
            self.global_directionmap_batch[batch,
                                           current_position[0],
                                           current_position[1]]
        path = [start_position]
        while (current_direction != -1):
            current_position = update_postion(current_position,
                                              current_direction)
            current_direction = \
                self.global_directionmap_batch[batch,
                                               current_position[0],
                                               current_position[1]]
            path.append(current_position)
        return path

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

    def crop_time_mask(self, center, timepoint, batches):
        time_crop = self.global_timemap[batches,
                                        center[0]-self.pad:center[0]+self.pad,
                                        center[1]-self.pad:center[1]+self.pad]
        mask = np.zeros_like(time_crop)
        mask[time_crop<=timepoint] = 1
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
        self.global_timemap = np.empty_like(self.global_label_batch)
        self.global_timemap.fill(np.inf)
        self.global_time = 0

        self.global_errormap = np.zeros(self.global_label_batch.shape,
                                        dtype=np.bool)
        self.global_error_list = [[] for _ in range(self.bs)]

        self.global_directionmap_batch = np.zeros_like(self.global_label_batch)\
                                         - 1
        self.prepare_global_batch(return_gt_ids=False)
        # also initializes id_2_gt lookup, seeds in coord syst of label
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
            # if b == 4: # tmp
            #     print 'claiming', centers[b][0], centers[b][1], ids[b]
            # check whether already pulled
            assert(self.global_claims[b, centers[b][0], centers[b][1]] == 0)

            self.global_claims[b, centers[b][0], centers[b][1]] = ids[b]
        return raw_batch, gts, centers, ids

    def get_path_error_batch(self):
        centers, ids = self.get_centers_from_queue()
        total_number_of_errors = np.sum([len(a) for a in self.global_error_list])
        print "total_number_of_errors",total_number_of_errors
        if total_number_of_errors > 0:
            print "FOUND ONE"
            print "ERRORLIST ",self.global_error_list
            # self.draw_debug_image("error_list_found"+str(self.global_time))
            exit()
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
        self.global_time += 1
        for b in range(self.bs):
            already_claimed = True
            while already_claimed:
                if self.priority_queue[b].empty():
                    raise Exception('priority queue empty. All pixels labeled')
                height, center_x, center_y, Id, direction, error_ind, time_put = \
                    self.priority_queue[b].get()
                if self.global_claims[b, center_x, center_y] == 0:
                    already_claimed = False
            # print 'b', b, 'size', center_x, center_y
            assert (self.global_claims[b, center_x, center_y] == 0)
            assert(self.pad <= center_x < self.global_el - self.pad)
            assert(self.pad <= center_y < self.global_el - self.pad)
            # tmp debug
            # if b == 4:
            #     print 'pulling....'
            #     print 'height', height, 'centerx', center_x, 'y', center_y, 'id', Id, \
            #         'direction', direction, 'error ind', error_ind, 'tput', time_put

            self.global_directionmap_batch[b,
                                           center_x - self.pad,
                                           center_y - self.pad] = direction
            self.global_timemap[b,
                                center_x - self.pad,
                                center_y - self.pad] = time_put

            # check for errors in boundary crossing
            if error_ind:
                # print "error at ",center_x-self.pad,center_y-self.pad,b,np.sum(self.global_errormap),np.sum(self.global_errormap[b])
                self.global_errormap[b,
                                     center_x-self.pad,
                                     center_y-self.pad] = 1

            # tmp uncomment
            # check for errors in neighbor regions
            for x, y, direction in self.walk_cross_coords([center_x, center_y]):
                neighbor_label = [self.global_label_batch[b, x-self.pad,
                                                          y-self.pad]]
                if (self.global_claims[b, x, y] != 0 and  # neighbor is claimed
                   neighbor_label != self.global_id2gt[b][Id]):
                    # and check if claimed by other gt label (over-segmenting is
                    # ok)

                    # check for slow intrusion( neighbor is intruder )
                    if neighbor_label != self.global_label_batch[b,
                                                                 x-self.pad,
                                                                 y-self.pad]:
                        self.global_error_list[b].append(
                            (self.global_timemap[b, x-self.pad, y-self.pad],
                             x-center_x,
                             y-center_y))
                    # check for fast intrusion( current is intruder )
                    if neighbor_label != self.global_label_batch[b,
                                                                 x-self.pad,
                                                                 y-self.pad]:
                        self.global_error_list[b].append(
                            (self.global_timemap[b, x-self.pad, y-self.pad],
                             x-center_x,
                             y-center_y))
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
            new_seeds_x, new_seeds_y, _ = \
                self.get_cross_coords(center)

            # check for touching errors
            for x, y, height, direction in \
                    zip(new_seeds_x, new_seeds_y, heights, directions):
                error_indicator = False
                # check whether crossing into other gt id region
                if self.global_label_batch[b, x-self.pad, y-self.pad]\
                        != self.global_id2gt[b][Id]\
                    and self.global_label_batch[b,
                                                center[0]-self.pad,
                                                center[1]-self.pad] == \
                        self.global_id2gt[b][Id]:
                        error_indicator = True

                if self.global_claims[b, x, y] == 0:

                    height_prev = self.global_heightmap_batch[b, x-self.pad,
                                                              y-self.pad]
                    height_j = max(heights[direction], height_prev)
                    if height_prev > 0:
                        height_j = min(height_j, height_prev)
                    self.global_heightmap_batch[b, x-self.pad, y-self.pad] = height_j
                    # if b == 4:
                    #     print 'pusing'
                    #     print 'x', x, 'y', y, 'b', b
                    self.priority_queue[b].put((height, x, y,
                                                Id, direction,
                                               error_indicator, self.global_time))

    def reconstruct_input_at_timepoint(self, timepoint, centers, id, batches):

        raw_batch = np.zeros((len(batches), 2, self.pl, self.pl),
                             dtype=theano.config.floatX)
        raw_batch[range(len(batches)), 0, :, :] = self.crop_raw(centers, batches)
        raw_batch[range(len(batches)), 1, :, :] = self.crop_mask_claimed(centers, batches, id)
        
        self.crop_time_mask()

        return raw_batch, gts, centers, ids

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
            for Id in ids:
                regions = np.where(self.global_label_batch[batch, :, :] == Id)
                rand_seed = np.random.randint(0, len(regions[0]))
                seeds.append([regions[0][rand_seed] + self.pad,
                              regions[1][rand_seed] + self.pad])
            global_seeds.append(seeds)

        self.initialize_priority_queue(global_seeds, global_ids)

    def get_pred_batch(self):
        seeds, ids = self.get_centers_from_queue()
        raw_batch = np.zeros((self.bs, 2, self.pl, self.pl),
                             dtype=theano.config.floatX)

        gts = np.zeros((self.bs, 4, 1, 1), dtype=theano.config.floatX)
        for b in range(self.bs):
            raw_batch[b, 0, :, :] = self.crop_raw(seeds[b], b)
            raw_batch[b, 1, :, :] = self.crop_mask_claimed(seeds[b], b, ids[b])
            gts[b, :, 0, 0] = self.get_adjacent_gts(seeds[b], b, ids[b])
            self.global_claims[b, seeds[b][0], seeds[b][1]] = ids[b]
        return raw_batch, gts, seeds, ids

    def draw_debug_image(self, image_name, path='./data/nets/debug/images/', save=True):
        plot_images = []
        plot_images.append({"title":"Claims",
                            'cmap':"rand",
                            'im':self.global_claims[4, self.pad:-self.pad-1,
                                 self.pad:-self.pad-1]})
        plot_images.append({"title":"Raw Input",
                            'im':self.global_batch[4, self.pad:-self.pad-1,
                                 self.pad:-self.pad-1]})
        plot_images.append({"title":"Heightmap Prediciton",
                            'im':self.global_heightmap_batch[4, :, :]})
        plot_images.append({"title":"Heightmap Ground Truth",
                            'im':self.global_height_gt_batch[4, :, :],
                            'scatter':np.array(self.global_seeds[4])-self.pad})
        plot_images.append({"title":"Ground Truth Label",
                            "cmap":"rand",
                            'im':self.global_label_batch[4, :, :]})
        plot_images.append({"title":"Error Map",
                            'im':self.global_errormap[4, :, :]})
        plot_images.append({"title":"Direction Map",
                            'im':self.global_directionmap_batch[4, :, :]})
        plot_images.append({"title":"Time Map ",
                    'im':self.global_timemap[4, :, :]})

        if save:
            u.save_images(plot_images, path=path, name=image_name)
        else:
            print 'show'
            plt.show()


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
    seeds = bm.init_train_path_batch()
    seeds = np.array(seeds[4])
    heights = np.random.random(size=batch_size)
    b = 4
    raw_batch, gts, centers, ids = bm.get_path_batches()
    name = 'debug'
    for i in range(500000):
        if i % 100 == 0:
            print i

        if i % 5000 == 0:
            bm.draw_debug_image(name + '_deb_%i' %i, save=True)
            print i
        raw_batch, gts, centers, ids = bm.get_path_batches()

        if i % 65000 == 0 and i != 0:
            bm.init_train_path_batch()
            raw_batch, gts, centers, ids = bm.get_path_batches()
            name += 'asdf'
        probs = np.zeros((batch_size, 4, 1,1))
        for c in range(batch_size):
            d = 0
            for x, y, _ in bm.walk_cross_coords(centers[b]):
                x -= bm.pad
                y -= bm.pad
                probs[c, d] = bm.global_height_gt_batch[b, x, y]
                d += 1
        bm.update_priority_path_queue(probs, centers, ids)

    # print bm.get_batches(10*[[45, 46]])












