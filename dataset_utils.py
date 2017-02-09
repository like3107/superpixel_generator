# import matplotlib
# try:
#     matplotlib.use('Qt4Agg')
# except:
# matplotlib.use('Agg')

import h5py as h
import numpy as np
import random
from os import makedirs
from os.path import exists
from ws_timo import wsDtseeds
from skimage import measure
from matplotlib import pyplot as plt
from Queue import PriorityQueue
from scipy.ndimage.measurements import watershed_ift
import utils as u
from scipy import ndimage
from scipy import stats
from scipy.ndimage import convolve, gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, binary_dilation
from skimage.feature import peak_local_max
from skimage.morphology import label, watershed
from itertools import product
# from sklearn.metrics import adjusted_rand_score
import h5py
# from cv2 import dilate, erode
import data_provider
import time
import copy
from theano import tensor as T

from IPython import embed


class HoneyBatcherPredict(object):
    def __init__(self, options):
        """
        batch loader. Use either for predict. For valId and train use:
        get batches function.

        :param options:
        """

        # either pad raw or crop labels -> labels are always shifted by self.pad
        self.padding_b = options.padding_b
        self.pl = int(options.patch_len)
        self.pad = options.patch_len / 2
        self.seed_method = options.seed_method
        self.bs = options.batch_size

        self.batch_data_provider = data_provider.get_dataset_provider(options.dataset)(options)

        self.batch_shape = self.batch_data_provider.get_batch_shape()
        self.image_shape = self.batch_data_provider.get_image_shape()
        self.label_shape = self.batch_data_provider.get_label_shape()
        self.global_input_batch = np.zeros(self.batch_shape, dtype=np.float32)
        self.global_label_batch = np.zeros(self.label_shape, dtype=np.int)
        self.global_height_gt_batch = np.zeros(self.label_shape, dtype=np.float32)

        # length of field, global_batch # includes padding)


        # private
        self.n_channels = options.network_channels + options.claim_channels
        self.options = options
        self.lowercomplete_e = options.lowercomplete_e
        self.max_penalty_pixel = options.max_penalty_pixel

        self.global_claims = np.empty(self.image_shape)
        self.global_claims.fill(-1.)
        self.global_claims[:, self.pad:-self.pad, self.pad:-self.pad] = 0
        self.global_heightmap_batch = np.empty(self.label_shape)            # post pq
        self.global_seed_ids = None
        self.global_seeds = None  # !!ALL!! coords include padding
        self.priority_queue = None
        self.coordinate_offset = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.int)
        x = np.linspace(-self.pad,self.pad,2*self.pad + 1)
        mg = np.meshgrid(x,x)
        xv, yv = np.meshgrid(x,x)
        self.manhattan_dist = np.abs(xv)+np.abs(yv)
        self.manhattan_dist -= np.amax(self.manhattan_dist)
        self.direction_array = np.arange(4)
        self.error_indicator_pass = np.zeros((self.bs))
        self.global_time = 0
        self.edge_map_gt = None
        self.global_prediction_map_FC = None
        self.preselect_batches = None
        self.hard_regions = None
        self.global_prediction_map_nq = None

        self.timo_min_len = 5
        self.timo_sigma = 0.3
        self.SeedMan = SeedMan()
        assert(self.pl == self.pad * 2 + 1)

    def get_seed_ids(self):
        assert (self.global_seeds is not None)  # call get seeds first
        self.global_seed_ids =  [np.arange(start=1, stop=len(s)+1) for s in self.global_seeds]

    def initialize_priority_queue(self):
        """
        Initialize one PQ per batch:
            call get_seeds and get_seed_ids first
            PQ: ((height, seed_x, seedy, seed_id, direction, error_indicator,
            time_put)
            Init: e.g. (-0.1, seed_x, seed_y, seed_id, None, 0
        :param global_seeds:
        :param global_ids:
        :param global_ids_gt:
        :return: [PQ_batch_0, PQ_batch_1,..., PQ_batch_N)
        """
        self.priority_queue = []
        for b, (seeds, ids) in enumerate(zip(self.global_seeds,
                                             self.global_seed_ids)):
            q = PriorityQueue()
            for seed, Id in zip(seeds, ids):
                q.put((-np.inf, 0., seed[0], seed[1], Id, -1, False, 0))
            self.priority_queue.append(q)

    def walk_cross_coords(self, center):
        # walk in coord system of global label batch: x : 0 -> global_el - pl
        # use center if out of bounds
        coords_x, coords_y, directions = self.get_cross_coords(center)
        for x, y, d in zip(coords_x, coords_y, directions):
            yield x, y, d

    def get_cross_coords(self, center):
        assert (center[0] >= self.pad and center[0] < self.image_shape[-2] - self.pad)
        assert (center[1] >= self.pad and center[1] < self.image_shape[-1] - self.pad)
        coords = self.coordinate_offset + center
        np.clip(coords[:, 0], self.pad, self.label_shape[1] + self.pad - 1, out=coords[:,0])
        np.clip(coords[:, 1], self.pad, self.label_shape[2] + self.pad - 1, out=coords[:,1])
        return coords[:, 0], coords[:, 1], self.direction_array

    def get_cross_coords_offset(self, center):
        coords = self.coordinate_offset + center - self.pad
        np.clip(coords[:, 0], 0, self.label_shape[1] - 1, out=coords[:, 0])
        np.clip(coords[:, 1], 0, self.label_shape[2] - 1, out=coords[:, 1])
        return coords[:, 0], coords[:, 1], self.direction_array

    def crop_input(self, center, b, out=None):
        if out is None:
            return self.global_input_batch[b, :,
                                           center[0] - self.pad:center[0] + self.pad + 1,
                                           center[1] - self.pad:center[1] + self.pad + 1]
        else:
            out[:] = self.global_input_batch[b, :,
                                             center[0] - self.pad:center[0] + self.pad + 1,
                                             center[1] - self.pad:center[1] + self.pad + 1]

    def crop_mask_claimed(self, center, b, Id, out=None):
        labels = self.global_claims[b,
                                    center[0] - self.pad:center[0] + self.pad + 1,
                                    center[1] - self.pad:center[1] + self.pad + 1]
        if out is None:
            out = np.zeros((self.options.claim_channels, self.pl, self.pl), dtype='float32')
        else:
            out[:self.options.claim_channels].fill(0)
        out[0, :, :][(labels != Id) & (labels != 0)] = 1  # the others
        out[0, :, :][labels == -1] = 0                    # the others
        out[1, :, :][labels == Id] = 1                    # me

        if len(out) > 2:
            out[2, :, :][labels <= 0] = 1
        if len(out) > 3:
            min_pos = np.argmin(self.manhattan_dist * out[0])
            ncol = labels.shape[1]
            if labels[min_pos/ncol, min_pos%ncol] > 0:
                out[3, :, :][labels == labels[min_pos/ncol, min_pos%ncol]] = 1

        if self.options.claim_aug == "height":
            h = np.array(self.global_heightmap_batch[b,
                                                    max(0,center[0]-2*self.pad):center[0]+1,
                                                    max(0,center[1]-2*self.pad):center[1]+1])
            h[h<0] = 0
            h[h==np.inf] = 0
            hx = max(self.pad-center[0],0)
            hy = max(self.pad-center[1],0)
            hsx, hsy = h.shape
            out[0, hx:hx+hsx, hy:hy+hsy] *= h
            out[1, hx:hx+hsx, hy:hy+hsy] *= h
            if len(out) > 3:
                out[3, hx:hx+hsx, hy:hy+hsy] *= h
        elif self.options.claim_aug == "raw":
            raw = self.global_input_batch[b, 0,
                         center[0] - self.pad:center[0] + self.pad + 1,
                         center[1] - self.pad:center[1] + self.pad + 1]
            out[0] *= raw
            out[1] *= raw
            if len(out) > 2:
                out[2] *= raw
            if len(out) > 3:
                out[3] *= raw
        
        return out

    def crop_height_map(self, center, b):
        height = self.global_heightmap_batch[b,
                                             center[0] - self.pad:center[0] + self.pad + 1,
                                             center[1] - self.pad:center[1] + self.pad + 1]
        return height

    def prepare_global_batch(self):
        return self.batch_data_provider.prepare_input_batch(self.global_input_batch,
                                                            preselect_batches=self.preselect_batches)

    def set_preselect_batches(self, batches):
        assert(self.bs == len(batches))
        self.preselect_batches = batches

    def reset_claims(self):
        self.global_claims.fill(-1.)
        self.global_claims[:, self.pad:-self.pad, self.pad:-self.pad] = 0

    def init_batch(self, start=None, allowed_slices = None):
        # remember where territory has been claimed before. !=0 claimed, 0 free
        self.reset_claims()

        self.global_heightmap_batch.fill(np.inf)        # post [-><-]

        self.prepare_global_batch()
        self.get_seed_coords()
        self.get_seed_ids()
        self.initialize_priority_queue()

        self.global_prediction_map = np.empty((self.bs, self.label_shape[1], self.label_shape[2], 4))   # pre [<-->]
        self.global_prediction_map.fill(np.inf)
        self.global_prediction_map_nq = np.empty((self.bs, self.label_shape[1], self.label_shape[2], 4))   # pre [<-->]
        self.global_prediction_map_nq.fill(np.inf)

    def get_seed_coords(self, sigma=1.0, min_dist=4, thresh=0.2):
        """
        Seeds by minima of dist trf of thresh of memb prob
        :return:
        """
        # print "using seed method:", self.seed_method
        if self.seed_method == "gt":
            self.get_seed_coords_gt(self.options.s_minsize)
        elif self.seed_method == "over":
            self.get_seed_coords_grid()
        elif self.seed_method == "timo":
            self.get_seed_coords_timo()
        elif self.seed_method == "file":
            self.batch_data_provider.get_seed_coords_from_file(self.global_seeds)
        else:
            raise Exception("no valid seeding method defined")

    def get_seed_coords_timo(self, sigma=1.0, thresh=0.2):
        """
        Seeds by minima of dist trf of thresh of memb prob
        :return:
        """
        self.global_seeds = []
        for b in range(self.bs):
            seeds = \
                self.SeedMan.get_seed_coords_timo(self.global_input_batch[b, 0, self.pad:-self.pad, self.pad:-self.pad],
                                                  min_memb_size=self.timo_min_len,
                                                  sigma=self.timo_sigma,
                                                  min_dist=min_dist,
                                                  thresh=thresh)
            self.global_seeds.append(seeds)

    def get_seed_coords_grid(self, gridsize=7):
        """
        Seeds by grid
        :return:
        """
        self.global_seeds = []
        shape = self.label_shape[1:3]
        offset_x = ((shape[0]) % gridsize) /2
        offset_y = ((shape[1]) % gridsize) /2
        for b in range(self.bs):
            seeds_b = self.SeedMan.get_seed_coords_grid(gridsize=gridsize)
            self.global_seeds.append(seeds_b)

    def get_seed_coords_gt(self, minsize = 0):
        self.global_seeds = []
        # label_image = label_image.astype(np.uint32)
        for b, label_image in enumerate(self.global_label_batch):
            seeds = self.SeedMan.get_seed_coords_gt(label_image, self.pad, minsize=minsize)
            self.global_seeds.append(seeds)


    def get_ws_segmentation(self):
        return self.batch_data_provider.get_timo_segmentation(self.global_label_batch, self.global_input_batch,
                                                              self.global_seeds)

    def find_hard_regions(self):
        self.hard_regions = self.batch_data_provider.find_timo_errors(\
            self.global_label_batch, self.global_input_batch, self.global_seeds)
        l = self.global_label_batch.shape[1]
        for b, seeds in enumerate(self.global_seeds):
            pos = np.array(seeds) - self.pad
            pos = np.clip(pos, 1, l - 1)
            for p in pos:
                self.hard_regions[b, p[0] - 1:p[0] + 2, p[1] - 1:p[1] + 2] = 1

    def get_centers_from_queue(self):
        centers = np.empty((self.bs, 2),dtype='int32')
        ids = []
        heights = []

        for b in range(self.bs):
            height, _, center_x, center_y, Id, direction, error_indicator, time_put = self.get_center_i_from_queue(b)
            centers[b, :] = [center_x, center_y]
            ids.append(Id)
            heights.append(height)

        return centers, ids, heights

    def get_center_i_from_queue(self, b):
        # pull from pq at free pixel position
        already_claimed = True
        while already_claimed:
            if self.priority_queue[b].empty():
                self.serialize_to_h5("empty_queue_state")
                self.draw_debug_image("empty_queue")
                raise Exception('priority queue empty. All pixels labeled')
            height, _, center_x, center_y, Id, direction, error_indicator, \
                time_put = self.priority_queue[b].get()
            if self.global_claims[b, center_x, center_y] == 0:
                already_claimed = False

        assert (self.global_claims[b, center_x, center_y] == 0)
        return height, _, center_x, center_y, Id, direction, error_indicator, \
                time_put

    def get_network_input(self, center, b, Id, out):
        self.crop_mask_claimed(center, b, Id, out=out[0:self.options.claim_channels])
        self.crop_input(center, b, out=out[self.options.claim_channels:])
        return out

    def get_batches(self):
        centers, ids, heights = self.get_centers_from_queue()
        # TODO: use input batch
        raw_batch = np.zeros((self.bs, self.n_channels, self.pl, self.pl), dtype='float32')
        for b, (center, height, Id) in enumerate(zip(centers, heights, ids)):
            self.get_batch_i(b, center, height, Id, raw_batch)
        return raw_batch, centers, ids

    def get_batch_i(self,  b, center, height, Id, raw_batch):
        assert (self.global_claims[b, center[0], center[1]] == 0)
        self.set_claims(b, center, Id)
        self.get_network_input(center, b, Id, raw_batch[b, :, :, :])
        # check whether already pulled
        self.global_heightmap_batch[b, center[0] - self.pad, center[1] - self.pad] = height

    def set_claims(self, b, center, idx):
        self.global_claims[b, center[0], center[1]] = idx

    def update_priority_queue(self, heights_batch, centers, ids, **kwargs):
        assert(np.all(np.isfinite(heights_batch)))
        for b, center, Id, height in zip(range(self.bs), centers, ids, heights_batch):
            self.global_prediction_map_nq[b, center[0] - self.pad, center[1] - self.pad, :] = height
            self.update_priority_queue_i(b, center, Id, height, **kwargs)

    def update_priority_queue_i(self, b, center, Id, height, **kwargs):
        # if possibly wrong
        cross_x, cross_y, cross_d = self.get_cross_coords(center)
        lower_bound = self.global_heightmap_batch[b, center[0] - self.pad,
                                                     center[1] - self.pad] + self.lowercomplete_e
        if lower_bound == np.inf:
            print "encountered inf for prediction center !!!!", \
                b, center, Id, height, lower_bound
            raise Exception('encountered inf for prediction center')

        self.max_new_old_pq_update(b, cross_x, cross_y, height, lower_bound, Id, cross_d, center,
                                   input_time=self.global_time)

    def max_new_old_pq_update(self, b, x, y, heights, lower_bound, Id,
                               direction, center, input_time=0, add_all=False):
        # check if there is no other lower prediction
        is_lowest = self.check_is_lowest(b, heights, x, y, add_all)

        heights[heights < lower_bound] = lower_bound
        self.global_heightmap_batch[b, x - self.pad, y - self.pad][is_lowest] = heights[is_lowest]
        self.global_prediction_map[b, center[0] - self.pad, center[1] - self.pad, :] = heights
        for cx, cy, cd, hj, il in zip(x, y, direction, heights, is_lowest):
            if il:
                self.priority_queue[b].put((hj, np.random.random(), cx, cy, Id, cd, self.error_indicator_pass[b],
                                            input_time))

    def check_is_lowest(self, b, heights, x, y, add_all):
        return ((heights < self.global_heightmap_batch[b, x - self.pad, y - self.pad]) | add_all )\
                        & (self.global_claims[b, x, y] == 0)

    def get_num_free_voxel(self):
        return np.sum([self.global_claims[0] == 0])

    def get_image_crops(self, b):
        return self.global_input_batch[b, :, self.pad:-self.pad, self.pad:-self.pad], \
               self.global_claims[b, self.pad:-self.pad, self.pad:-self.pad]

    def draw_debug_image(self, image_name,
                         path='./../data/debug/images/',
                         save=True, b=0, inherite_code=False):

        batch, claims = self.get_image_crops(b)

        plot_images = []
        for channel in range(self.batch_shape[1]):
            plot_images.append({"title": "Input %d" % channel,
                                'im': batch[channel],
                                'interpolation': 'none'})
        if not inherite_code:
            plot_images.append({"title": "Claims",
                                'cmap': "rand",
                                'im': batch[0],
                                'interpolation': 'none'})
        if np.min(self.global_heightmap_batch) != np.inf:
            plot_images.append({"title": "Min Heightmap",
                                'cmap': "gray",
                                'im': self.global_heightmap_batch[b, :, :],
                                'interpolation': 'none'})
        if self.global_prediction_map_FC is not None:
            plot_images.append({"title": "Heightmap Prediciton",
                            'im': self.global_prediction_map_FC[b, 0, :, :],
                            'interpolation': 'none',
                            'cmap':'gray'})
        # if self.edge_map_gt is not None:
        #     for i in range(4):
        #         plot_images.append({"title": "Edge Map %i" %i,
        #                         'im': self.edge_map_gt[b, i, :, :],
        #                         'interpolation': 'none',
        #                         'cmap':'gray'})
        for i in range(4):
            plot_images.append({"title": "Network Output %i" % i,
                                'im': self.global_prediction_map_nq[b, :, :, i],
                                'interpolation': 'none',
                                'cmap': 'gray'})
        if not inherite_code:
            if save:
                u.save_images(plot_images, path=path, name=image_name)
            else:
                print 'show'
                plt.show()
        else:
            return plot_images


class HoneyBatcherPath(HoneyBatcherPredict):
    def __init__(self,  options):
        super(HoneyBatcherPath, self).__init__(options)

        # private
        self.add_height_b = False
        # All no padding
        self.global_directionmap_batch = np.zeros(self.image_shape, dtype=np.int) - 1       # post PQ
        self.global_timemap = np.empty(self.image_shape, dtype=np.int)
        self.global_errormap = np.zeros(self.image_shape, dtype=np.int)

        self.n_batch_errors = options.n_batch_errors
        self.error_selections = None
        self.global_error_dict = None
        self.crossing_errors = None
        self.find_errors_b = options.fine_tune_b and not options.rs_ft
        self.error_indicator_pass = None

        print 'du pl pad label raw', self.pl, self.pad, self.label_shape, self.image_shape

    def get_seed_ids(self):
        super(HoneyBatcherPath, self).get_seed_ids()
        self.global_id2gt = []
        for b, (ids, seeds) in enumerate(zip(self.global_seed_ids,
                                             self.global_seeds)):
            id2gt = {}
            for Id, seed in zip(ids, seeds):
                id2gt[Id] = self.global_label_batch[b, seed[0] - self.pad, seed[1] - self.pad]
            self.global_id2gt.append(id2gt)

    def crop_timemap(self, center, b):
        return self.global_timemap[b,
                                   center[0] - self.pad:center[0] + self.pad + 1,
                                   center[1] - self.pad:center[1] + self.pad + 1]

    def crop_time_mask(self, centers, timepoint, batches):
        """
        compute mask that is 1 if a voxel was not accessed before timepoint
        and zero otherwise
        """
        mask = np.zeros((len(batches), self.pl, self.pl), dtype=bool)
        # fore lists to np.array so we can do array arithmetics
        centers = np.array(centers)
        for i, b in enumerate(batches):
            mask[i, :, :][self.crop_timemap(centers[i], b) > timepoint[i]] = 1
        return mask

    def prepare_global_batch(self):
        rois = super(HoneyBatcherPath, self).prepare_global_batch()
        self.batch_data_provider.prepare_label_batch(self.global_label_batch, self.global_height_gt_batch, rois)
        return rois

    def init_batch(self, start=None, allowed_slices = None):
        super(HoneyBatcherPath, self).init_batch(start=start, allowed_slices=allowed_slices)
        # load new global batch data
        self.global_timemap.fill(0)     # zeros for masking in recurrent path reconstruction
        self.global_time = 0
        self.global_errormap = np.zeros((self.bs, 3, self.label_shape[1], self.label_shape[2]), dtype=np.bool)
        self.global_error_dict = {}
        # direction map post pq: [-><-]
        self.global_directionmap_batch = np.zeros(self.batch_data_provider.get_label_shape(), dtype=np.int) - 1

    def get_batches(self):
        raw_batch, centers, ids = super(HoneyBatcherPath, self).get_batches()
        gts = np.zeros((self.bs, 4, 1, 1), dtype='float32')
        for b in range(self.bs):
            if self.add_height_b:
                gts[b, :, 0, 0] = self.get_adjacent_heights(centers[b], b, ids[b])
            else:
                gts[b, :, 0, 0] = self.get_adjacent_heights(centers[b], b)
        assert (not np.any(gts < 0))
        assert (np.any(np.isfinite(raw_batch)))
        assert (not np.any(raw_batch < 0))
        assert (np.any(np.isfinite(centers)))
        assert (np.any(np.isfinite(ids)))
        assert (np.any(np.isfinite(gts)))
        return raw_batch, gts, centers, ids

    def get_adjacent_heights(self, seed, batch, Id=None):
        seeds_x, seeds_y, _ = self.get_cross_coords_offset(seed)
        # boundary conditions
        assert (np.any(seeds_x >= 0) or np.any(seeds_y >= 0))
        # assert (np.any(self.rl - self.pl > seeds_x) or
                # np.any(self.rl - self.pl > seeds_y))
        ground_truth = self.global_height_gt_batch[batch, seeds_x, seeds_y].flatten()
        # increase height relative to label (go up even after boundary crossing)
        if Id is not None:      #  == if self.add_height
            mask = [self.global_label_batch[batch, seeds_x,  seeds_y] != self.global_id2gt[batch][Id]]
            if np.any(mask):
                if self.error_indicator_pass[batch] != 0:
                    # center on wrong label
                    ground_truth[mask] = self.error_indicator_pass[batch]
                else:       # center on correct id
                    ground_truth[mask] = (self.pad + 1) * self.scaling
                # \
                #         (self.pad + 1) * self.scaling + \
                #         np.random.randint(0, self.scaling)
        return ground_truth

    def get_centers_from_queue(self):
        self.error_indicator_pass = np.zeros(self.bs, dtype=np.int)
        self.global_time += 1
        centers, ids, heights = \
            super(HoneyBatcherPath, self).get_centers_from_queue()
        return centers, ids, heights

    def get_center_i_from_queue(self, b):
        height, _, center_x, center_y, Id, direction, error_indicator, time_put = \
            super(HoneyBatcherPath, self).get_center_i_from_queue(b)
        self.global_directionmap_batch[b, center_x - self.pad, center_y - self.pad] = direction
        self.global_timemap[b, center_x, center_y] = self.global_time

        # pass on if type I error already occured
        if error_indicator > 0:
            # went back into own territory --> reset error counter
            if self.global_id2gt[b][Id] == self.global_label_batch[b, center_x - self.pad, center_y - self.pad]:
                self.error_indicator_pass[b] = 0.
            else:   # remember to pass on
                self.error_indicator_pass[b] = 1
            self.global_errormap[b, 1, center_x - self.pad, center_y - self.pad] = 1
        # check for type I errors
        elif self.global_id2gt[b][Id] != \
                self.global_label_batch[b, center_x - self.pad, center_y - self.pad]:
            self.global_errormap[b, :2, center_x - self.pad, center_y - self.pad] = 1
            self.error_indicator_pass[b] = 1

        # check for errors in neighbor regions, type II
        # TODO: remove find_errors_b
        # if self.find_errors_b:
        #     self.check_type_II_errors(center_x, center_y, Id, b)
        # print 'b', b, 'height', height, 'centerxy', center_x, center_y, 'Id', Id, \
        #     direction, error_indicator, time_put
        return height, _, center_x, center_y, Id, direction, error_indicator, \
                    time_put

    def update_position(self, pos, direction):
        """
        update position by following the minimal spanning tree backwards
        for this reason: subtract direction for direction offset
        """
        assert(direction >= 0)
        offsets = self.coordinate_offset[int(direction)]
        new_pos = [pos[0] - offsets[0], pos[1] - offsets[1]]
        return new_pos

    def get_path_to_root(self, start_position, batch):

        current_position = start_position
        current_direction = self.global_directionmap_batch[batch, current_position[0]-self.pad,
                                                                  current_position[1]-self.pad]
        yield start_position, current_direction
        while current_direction != -1:
            current_position = self.update_position(current_position, current_direction)
            current_direction = self.global_directionmap_batch[batch, current_position[0]-self.pad,
                                                                      current_position[1]-self.pad]
            yield current_position, current_direction

    def locate_global_error_path_intersections(self):
        self.err_counter = -1

        def error_index(b, id1, id2, time=None):
            if time is None:
                return b, min(id1, id2), max(id1, id2)
            else:
                return len(self.global_error_dict.keys())
                return b, min(id1, id2), max(id1, id2), time
        def get_error_dict(b, x, y, center_x, center_y, small_direction, slow_intruder, touch_x, touch_y):
            self.err_counter += 1
            new_error = \
                {"Id":self.err_counter,
                 "batch": b,
                 # get time from center it was predicted from
                 "touch_time": self.global_timemap[b, touch_x, touch_y],
                 "large_pos": [center_x, center_y],
                 "large_direction": direction,
                 "large_id": self.global_claims[b, center_x, center_y],
                 "large_gtid": claim_projection[center_x, center_y],
                 "importance": 1,
                 "small_pos": [x, y],
                 "small_direction": small_direction,
                 "small_gtid": claim_projection[x, y],
                 "small_id": self.global_claims[b, x, y],
                 "slow_intruder": slow_intruder,
                 "plateau":False,
                 "used":False,
                 "first_rec":True} # debug
            assert (new_error["large_gtid"] != new_error["small_gtid"])
            assert (new_error["large_id"] != new_error["small_id"])
            return new_error

        for b in range(self.bs):

            # project claim id to ground truth id by lookup
            gtmap = np.array([0] + self.global_id2gt[b].values())
            claim_projection = gtmap[self.global_claims[b].astype(int)]
            claim_projection[self.pad - 1,:] = claim_projection[self.pad, :]
            claim_projection[-self.pad, :] = claim_projection[-self.pad - 1, :]
            claim_projection[:, self.pad - 1] = claim_projection[:, self.pad]
            claim_projection[:, -self.pad] = claim_projection[:, -self.pad - 1]
            not_found = np.zeros_like(claim_projection)
            
            # plot_images = []
            # plot_images.append({"title": "claim",
            #                     'cmap': "rand",
            #                     'im': self.global_claims[b,
            #                                              self.pad:-self.pad,
            #                                              self.pad:-self.pad]})
            # plot_images.append({"title": "gt",
            #                     'cmap': "rand",
            #                     'im': self.global_label_batch[b]})
            # plot_images.append({"title": "mix",
            #                     'cmap': "rand",
            #                     'im': claim_projection[self.pad:-self.pad,
            #                                            self.pad:-self.pad]})
            # plot_images.append({"title": "overflow",
            #                         'cmap': "gray",
            #                         'im': self.global_errormap[b, 1]})

            # find where path crosses region
            gx = convolve(claim_projection + 1, np.array([-1., 0., 1.]).reshape(1, 3))
            gy = convolve(claim_projection + 1, np.array([-1., 0., 1.]).reshape(3, 1))
            boundary = np.float32((gx ** 2 + gy ** 2) > 0)
            # find all boundary crossings
            path_fin_map = np.logical_and(boundary[self.pad:-self.pad, self.pad:-self.pad], self.global_errormap[b, 0])


            # plot_images.append({"title": "path_fin_0",
            #             'cmap': "gray",
            #             'im': path_fin_map})
            #
            np.logical_and(path_fin_map, (self.global_claims[b, self.pad:-self.pad, self.pad:-self.pad] > 0),
                           out=path_fin_map)
            # plot_images.append({"title": "path_fin_1",
            #                         'cmap': "gray",
            #                         'im': path_fin_map})
            # plot_images.append({"title": "boundary",
            #                     'cmap': "gray",
            #                     'im': boundary[self.pad:-self.pad,
            #                                    self.pad:-self.pad]})
            # u.save_images(plot_images, path="./../data/debug/",
            #               name="path_test_"+str(b)+".png")

            wrong_path_ends = np.transpose(np.where(path_fin_map)) + self.pad
            for center_x, center_y in wrong_path_ends:
                # check around intruder, claim pred = large pred = intruder
                claim_height = self.global_heightmap_batch[b, center_x - self.pad, center_y - self.pad]
                large_time = self.global_timemap[b, center_x, center_y]
                smallest_pred = np.inf
                small_height_old = np.inf
                touch_time_old = np.inf
                fast_intruder_found = False
                new_error = {}
                for x, y, direction in self.walk_cross_coords([center_x, center_y]):
                    assert (x - self.pad >= 0)
                    assert (center_x - self.pad >= 0)
                    assert (y - self.pad >= 0)
                    assert (center_y - self.pad >= 0)

                    if np.all([center_x, center_y] == [x, y]):
                        continue    # ignore predictions which are out of bounds

                    label_large = self.global_label_batch[b, center_x - self.pad, center_y - self.pad]
                    claim_projection_large = claim_projection[center_x, center_y]
                    label_small = self.global_label_batch[b, x - self.pad, y - self.pad]
                    claim_projection_small = claim_projection[x, y]

                    # if type3 true: A and B meet in C --> no penalty
                    type_3_error = (label_small != claim_projection_small)

                    # is neighborpixel same ID, if no -> error
                    if claim_projection_small != claim_projection_large and not type_3_error:
                        reverse_direction = self.reverse_direction(direction)
                        # on center pixel of intruder
                        prediction = self.global_prediction_map[b, x - self.pad,  y - self.pad, reverse_direction]
                        small_time = self.global_timemap[b, x, y]
                        if prediction < claim_height:
                            raise Exception('this violates PQ structure')

                        # fast intruder
                        # intruder runs into resident (fast intruder)
                        # find lowest prediction on intruder claimed pixel
                        elif prediction < smallest_pred and small_time < large_time:
                            fast_intruder_found = True      # prioritized
                            smallest_pred = prediction
                            new_error = get_error_dict(b, x, y, center_x, center_y, reverse_direction, False,
                                                       center_x, center_y)

                        # slow intruder find lowest claimed height
                        elif not fast_intruder_found:
                            small_height = self.global_heightmap_batch[b, x - self.pad, y - self.pad]
                            if small_height >= claim_height and  small_height <= small_height_old:
                                small_height_old = small_height
                                assert (small_time > large_time)
                                small_direction = self.global_directionmap_batch[b, x - self.pad, y - self.pad]
                                new_error = get_error_dict(b, x, y, center_x, center_y, small_direction, True,
                                                           x, y)

                if new_error != {}:
                    e_index = error_index(b, new_error["small_gtid"], new_error["large_gtid"], time=time.time())
                    self.global_error_dict[e_index] = new_error
                        # plot_images[-2] = \
                        #     {"title": "Path Map",
                        #      'scatter': np.array(
                        #          [np.array(e["large_pos"]) - self.pad
                        #              for e in self.global_error_dict.values()
                        #              if e["batch"] == b] +
                        #          [np.array(e["small_pos"]) - self.pad
                        #              for e in self.global_error_dict.values()
                        #              if e["batch"] == b]),
                        #      'im': self.global_errormap[b, 2, :, :],
                        #      'interpolation': 'none'}
                        #
                        # plot_images[-1] = {"title": "not found",
                        #                    'im': not_found,
                        #                    'interpolation': 'none'}
                        # # tmp debug
                        # print 'waring savign debug plot'
                        # u.save_images(plot_images, path="./../data/debug/",
                        #               name="path_test_"+str(b)+"_"+str(e_index[1])+
                        #                    str(e_index[2])+".png")
                # else:
                #     print "no match found for path end type 3"

                    # not_found[center_x-self.pad, center_y-self.pad] = 1
                    # for x, y, direction in self.walk_cross_coords([center_x,
                    #                                                center_y]):
                    #     print  claim_projection[x, y] ," should not be ", \
                    #         claim_projection[center_x, center_y]
                    #     reverse_direction = self.reverse_direction(direction)
                    #     print "prediction = ", \
                    #         self.global_prediction_map[b,
                    #                                    x-self.pad,
                    #                                    y-self.pad,
                    #                                    reverse_direction]
                    # raise Exception("no match found for path end")

    def weight_importance_by_length(self):
        for k in self.global_error_dict:
            self.global_error_dict[k]['importance'] = self.global_error_dict[k]['e1_length']

    def weight_importance_by_hard_regions(self):
        for k in self.global_error_dict:
            self.global_error_dict[k]['importance'] = self.global_error_dict[k]['e1_length']
            batch = self.global_error_dict[k]['batch']
            pos1 = np.array(self.global_error_dict[k]['e1_pos']) - self.pad
            pos2 = np.array(self.global_error_dict[k]['e2_pos']) - self.pad
            if self.hard_regions[batch, pos1[0], pos1[1]] or self.hard_regions[batch, pos2[0], pos2[1]]:
                self.global_error_dict[k]['importance'] *= 1000

    def weight_importance_by_overflow(self):
        for k in self.global_error_dict:
            b = self.global_error_dict[k]['batch']
            s_gt_id = self.global_error_dict[k]['small_gtid']
            underflow = np.sum(self.global_errormap[b, 1][self.global_label_batch[b]==s_gt_id])
            area = np.sum(self.global_label_batch[b]==s_gt_id)
            area = max(area, 200)
            self.global_error_dict[k]['importance'] = float(underflow) / area

    def get_plateau_indicator(self):
        return self.global_prediction_map_nq  < self.global_prediction_map

    def find_global_error_paths(self):
        print 'searching for hard regions'

        for b in range(self.bs):
            self.global_errormap[b, 0] = np.logical_and(self.global_errormap[b, 1],
                                                        binary_dilation(binary_erosion(self.global_errormap[b, 1])))

        self.find_hard_regions()
        self.locate_global_error_path_intersections()
        # now errors have been found so start and end of paths shall be found
        self.set_plateau_indicator()
        self.find_type_I_error()
        self.find_source_of_II_error()
        if self.options.weight_fct == "hard":
            self.weight_importance_by_hard_regions()
        elif self.options.weight_fct == "overflow":
            self.weight_importance_by_overflow()
        elif self.options.weight_fct == "length":
            self.weight_importance_by_length()
        elif self.options.weight_fct == "none":
            for k in self.global_error_dict:
                self.global_error_dict[k]['importance'] = 1.
        else:
            raise Exception('Error: unknown weighting scheme %s'%self.options.weight_fct)

    def plot_h1h2_errors(self, image_file, hist_file):

        h_pairs = []
        for error in self.global_error_dict.values():
            h1 = self.global_heightmap_batch[error["batch"],
                                                    error["e1_pos"][0] - self.pad,
                                                    error["e1_pos"][1] - self.pad]
            h2 = self.global_heightmap_batch[error["batch"],
                                                    error["e2_pos"][0] - self.pad,
                                                    error["e2_pos"][1] - self.pad]
            h_pairs.append([h1,h2])
            # print error["e1_pos"], h1, error["e2_pos"], h2

        data = np.array(h_pairs)
        col = ['g' if 'used' in e and e['used'] else 'r' for e in self.global_error_dict.values()]
        plt.xlabel('h1 error')
        plt.ylabel('h2 error')
        limit = max(np.max(data), 35)

        plt.xlim(xmin=0, xmax=limit*1.1)
        plt.ylim(ymin=0, ymax=limit*1.1)

        plt.scatter(data[:,0], data[:,1], c=col)
        plt.savefig(image_file)
        plt.clf()
        hmap = np.array(self.global_heightmap_batch)
        hmap[hmap<0] = 0
        plt.hist(hmap.flatten())
        plt.savefig(hist_file)
        plt.clf()
        # from IPython import embed; embed()
        return np.mean(h1), np.mean(h2)

    def set_plateau_indicator(self):
        self.global_plateau_indicator = self.global_prediction_map_nq  < self.global_prediction_map

    # crossing from own gt ID into other ID
    def find_type_I_error(self):
        for error_I in self.global_error_dict.values():
            if "e1_pos" not in error_I:
                current_pos = error_I["large_pos"]
                batch = error_I["batch"]
                # keep track of output direction
                current_direction = error_I["large_direction"]
                prev_in_other_region = self.global_errormap[batch, 1, current_pos[0] - self.pad,
                                                                      current_pos[1] - self.pad]

                e1_length = 0
                for pos, d in self.get_path_to_root(current_pos, batch):
                    # debug
                    # shortest path of error type II to root (1st crossing)
                    self.global_errormap[batch, 2, pos[0] - self.pad, pos[1] - self.pad] = True
                    # debug
                    # remember type I error on path
                    in_other_region = self.global_errormap[batch, 1, pos[0]-self.pad, pos[1]-self.pad]
                    #  detect transition from "others" region to "me" region
                    if prev_in_other_region and not in_other_region:
                        original_error = np.array(pos)
                        error_I["e1_pos"] = current_pos
                        error_I["e1_time"] = self.global_timemap[batch, pos[0], pos[1]]
                        error_I["e1_direction"] = current_direction
                        error_I["e1_length"] = e1_length
                        assert (error_I["large_id"] == self.global_claims[batch, pos[0], pos[1]])
                        # plot_images = [{"title": "error",
                        #         'cmap': "gray",
                        #         'im': self.global_errormap[batch, 0]}]
                        # plot_images.append({"title": "error",
                        #         'cmap': "gray",
                        #         'im': self.global_errormap[batch, 1]})
                        # plot_images.append({"title": "error",
                        #         'cmap': "gray",
                        #         'im': self.global_errormap[batch, 2]})
                        # u.save_images(plot_images, name="find_type_debug.png" ,path="./../data/debug/")
                        # print "error_I[large_gtid]",error_I["large_gtid"]
                        # print "gl",self.global_label_batch[batch, pos[0]-self.pad, pos[1]-self.pad]
                        assert (error_I["large_gtid"] == self.global_label_batch[batch, pos[0]-self.pad,
                                                                                        pos[1]-self.pad])
                        assert(current_direction >= 0)
                    current_pos = pos
                    current_direction = d
                    prev_in_other_region = in_other_region
                    e1_length += 1

                e1_length = error_I["e1_length"]

    def find_end_of_plateau(self, start_pos, prediction_dir, batch, error_dict=None):
        assert(prediction_dir >= 0)
        node_direction = self.global_directionmap_batch[batch, start_pos[0] - self.pad, start_pos[1] - self.pad]
        step_back_pos = self.update_position(start_pos, node_direction)

        if not self.global_plateau_indicator[batch, step_back_pos[0] - self.pad, step_back_pos[1] - self.pad,
                                             node_direction]:
            return start_pos, prediction_dir
        else:
            if error_dict is not None:
                error_dict["plateau"] = True
            assert(node_direction != -1)
            return self.find_end_of_plateau(step_back_pos, node_direction, batch, error_dict=None)

    def find_source_of_II_error(self):
        for error in self.global_error_dict.values():
            if "e2_pos" not in error:
                batch = error["batch"]
                if error["slow_intruder"]:
                    start_position = error["small_pos"]
                    start_direction = error["small_direction"]
                    assert (self.global_directionmap_batch[batch,
                                                           start_position[0] - self.pad,
                                                           start_position[1] - self.pad] == start_direction)
                    small_height = self.global_heightmap_batch[batch,
                                                               start_position[0] - self.pad,
                                                               start_position[1] - self.pad]

                    # go one step back
                    # start_position = self.update_position(start_position, start_direction)
                    error["e2_pos"], error["e2_direction"] = self.find_end_of_plateau(start_position, start_direction,
                                                                                      batch, error)
                    # slow intruder never reaches root
                    assert(self.global_directionmap_batch[error["batch"], error["small_pos"][0] - self.pad,
                                                                          error["small_pos"][1] - self.pad] != -1)
                else:       # fast intuder
                    e2_direction = error["small_direction"]
                    e2_pos = error["large_pos"]
                    assert(self.global_directionmap_batch[batch,
                                                          e2_pos[0] - self.pad,
                                                          e2_pos[1] - self.pad] != e2_direction)
                    error["e2_pos"], error["e2_direction"] = e2_pos, e2_direction
                assert (error["e2_direction"] >= 0)
                pos = error["e2_pos"]
                dir = error["e2_direction"]
                prev_pos = self.update_position(pos, dir) if dir != -1 else pos
                error["e2_time"] = self.global_timemap[batch, prev_pos[0], prev_pos[1]]

    def reconstruct_input_at_timepoint(self, errors, key_time, key_center, key_id, key_dir):
        raw_batch = np.zeros((len(errors), self.n_channels, self.pl, self.pl), dtype='float32')
        centers = []
        timepoints = []
        batches = []
        for i, error in enumerate(errors):
            batch, center, Id = [error['batch'], error[key_center], error[key_id]]
            # assert (self.global_timemap[batch, center[0], center[1]] == timepoints[i])
            # center = self.update_position(center, error[key_dir])
            batches.append(batch)
            centers.append(center)

            timepoints.append(error[key_time])
            self.get_network_input(center, batch, Id, raw_batch[i, :, :, :])
        mask = self.crop_time_mask(centers, timepoints, batches)
        raw_batch[:, 0, :, :][mask] = 0
        raw_batch[:, 1, :, :][mask] = 0
        if self.options.claim_channels > 2:
            raw_batch[:, 2, :, :][mask] = 1
        if self.options.claim_channels > 3:
            # simple masking of time does not work here, since the nearest neighbor
            # can change over time
            raise NotImplementedError()
        self.mask = mask
        return raw_batch

    def check_error(self, error):
        return not 'used' in error or not error['used']
        # return not 'used' in error and error['e1_length'] > 5

    def reverse_direction(self, direction):
        assert(direction >= 0)
        return (direction + 2) % 4

    def count_new_path_errors(self):
        return len([v for v in self.global_error_dict.values() if self.check_error(v)])

    def select_errors(self):
        # take approx this many errors or all
        unused_set = [k for k in self.global_error_dict.keys() if not self.global_error_dict[k]['used']]
        probs = np.array([self.global_error_dict[k]['importance'] for k in unused_set], dtype=float)
        probs /= np.sum(probs)
        selection = np.random.choice(unused_set, size=min(self.n_batch_errors, len(probs)), p=probs,
                                     replace=False)

        for k in selection:
            self.global_error_dict[k]['used'] = True
        return selection

    def reconstruct_path_error_inputs(self, backtrace_length=0):
        selection = self.select_errors()
        reconst_es = []
        error_selections = []
        for err_type, id_type in zip(['e1', 'e2'], ['large', 'small']):
            error_selection = [global_error_dict[e] for e in selection]
            reconst_e = self.reconstruct_input_at_timepoint(error_selection, err_type + "_time", err_type + "_pos",
                                                            id_type + "_id", err_type + "_direction")[:, :, 1:-1, 1:-1]
            error_selections.append(error_selection)
        return reconst_es[0], reconst_es[1]

    def serialize_to_h5(self, h5_filename, path="./../data/debug/serial/"):
        if not exists(path):
            makedirs(path)
        with h5py.File(path+'/'+h5_filename, 'w') as out_h5:
            out_h5.create_dataset("global_timemap", data=self.global_timemap ,compression="gzip")
            out_h5.create_dataset("global_errormap", data=self.global_errormap ,compression="gzip")
            out_h5.create_dataset("global_claims", data=self.global_claims ,compression="gzip")
            out_h5.create_dataset("global_input", data=self.global_input_batch ,compression="gzip")
            out_h5.create_dataset("global_heightmap_batch", data=self.global_heightmap_batch ,compression="gzip")
            out_h5.create_dataset("global_height_gt_batch", data=self.global_height_gt_batch ,compression="gzip")
            out_h5.create_dataset("global_prediction_map", data=self.global_prediction_map ,compression="gzip")
            out_h5.create_dataset("global_prediction_map_nq", data=self.global_prediction_map_nq ,compression="gzip")
            out_h5.create_dataset("global_directionmap_batch", data=self.global_directionmap_batch ,compression="gzip")

            for error_name,error in self.global_error_dict.items():
                for n,info in error.items():
                    out_h5.create_dataset("error/"+str(error_name)+"/"+n,data=np.array(info))

    def save_quick_eval(self, name, path, score=True):
        print "name, path",name, path
        if not exists(path):
            makedirs(path)
        with h5py.File(path+'/'+name+'pred.h5', 'w') as out_h5:
            out_h5.create_dataset("claims",
                        data=self.global_claims[:, self.pad:-self.pad,
                                  self.pad:-self.pad ], compression="gzip")
        with h5py.File(path+'/'+name+'_gt.h5', 'w') as out_h5:
            out_h5.create_dataset("gt_labels",
                        data=self.global_label_batch, compression="gzip")

        if score:
            import validation_scripts as vs

            print "path",path+'/'+name+'pred.h5', "gt_path",path+'/'+name+'_gt.h5'

            val_score = vs.validate_segmentation(pred_path=path+'/'+name+'pred.h5', gt_path=path+'/'+name+'_gt.h5')
            print val_score
            import json
            with open(path+'/'+name+'_score.json', 'w') as f:
                f.write(json.dumps(val_score))

    def draw_batch(self, raw_batch, image_name, path='./../data/debug/', save=True, gt=None, probs=None):
        plot_images = []
        n_batches = min(10, raw_batch.shape[0])     # for visibility
        for b in range(n_batches):
            plot_images.append({"title": "claim others", 'cmap': "rand", 'im': raw_batch[b, 0],
                                'interpolation': 'none'})
            plot_images.append({"title": "claim me", 'cmap': "rand", 'im': raw_batch[b, 1], 'interpolation': 'none'})
            plot_images.append({"title": "membrane", 'im': raw_batch[b, 2], 'interpolation': 'none'})
            plot_images.append({"title": "raw", 'im': raw_batch[b, 3], 'interpolation': 'none'})
        u.save_images(plot_images, path=path, name=image_name, column_size=4)

    def draw_error_reconst(self, image_name, path='./../data/debug/', save=True):
        for e_idx, error in self.global_error_dict.items():
            plot_images = []
            if not "draw_file" in error:
                reconst_e1 = \
                    self.reconstruct_input_at_timepoint([error["e1_time"]], [error["e1_pos"]], [error["large_id"]],
                                                        [error["batch"]])
                reconst_e2 = \
                    self.reconstruct_input_at_timepoint([error["e2_time"]], [error["e2_pos"]], [error["small_id"]],
                                                        [error["batch"]])

                plot_images.append({"title": "Ground Truth Label",
                                    "cmap": "rand",
                                    'im': self.global_label_batch[error["batch"],
                                          error["e1_pos"][0] - 2 * self.pad:
                                          error["e1_pos"][0],
                                          error["e1_pos"][1] - 2 * self.pad:
                                          error["e1_pos"][1]]})
                plot_images.append(
                    {"title": "reconst claims at t=" + str(error["e2_time"]),
                     'cmap': "rand",
                     'im': reconst_e1[0, 1, :, :]})
                plot_images.append({"title": "final claims",
                                    'cmap': "rand",
                                    'im': self.global_claims[error["batch"],
                                          error["e1_pos"][0] - self.pad:
                                          error["e1_pos"][0] + self.pad,
                                          error["e1_pos"][1] - self.pad:
                                          error["e1_pos"][1] + self.pad]})

                plot_images.append({"title": "E2 Ground Truth Label",
                                    "cmap": "rand",
                                    'im': self.global_label_batch[error["batch"],
                                          error["e2_pos"][0] - 2 * self.pad:
                                          error["e2_pos"][0],
                                          error["e2_pos"][1] - 2 * self.pad:
                                          error["e2_pos"][1]]})
                plot_images.append(
                    {"title": "E2 reconst claims at t=" + str(error["e1_time"]),
                     'cmap': "rand",
                     'im': reconst_e2[0, 1, :, :]})
                plot_images.append({"title": "E2 final claims",
                                    'cmap': "rand",
                                    'im': self.global_claims[error["batch"],
                                          error["e2_pos"][0] - self.pad:
                                          error["e2_pos"][0] + self.pad,
                                          error["e2_pos"][1] - self.pad:
                                          error["e2_pos"][1] + self.pad]})
                print "plotting ", image_name + '_' + str(e_idx)
                error["draw_file"] = image_name + '_' + str(e_idx)
                u.save_images(plot_images, path=path,
                              name=image_name + '_' + str(e_idx) + '.png')
            else:
                print "skipping ", e_idx

    def draw_debug_image(self, image_name, path='./../data/debug/', save=True, b=0, inheritance=False,
                         plot_height_pred=False):

        batch, claims = self.get_image_crops(b)
        plot_images = super(HoneyBatcherPath, self).\
            draw_debug_image(image_name=image_name,
                             path=path,
                             save=False,
                             b=b,
                             inherite_code=True)
        plot_images.append({"title": "Error Map",
                            'im': self.global_errormap[b, 0, :, :],
                             'interpolation': 'none'})

        e2_pos = np.array([np.array(e["e2_pos"]) - self.pad
                  for e in self.global_error_dict.values()
                  if e["batch"] == b and e["used"]])
        e2_color = ["g" if e["used"] else 'r'\
                 for e in self.global_error_dict.values() if e["batch"] == b  and e["used"]]
        e2_importance_color = ["g" if e["importance"] > 1000 else 'r'\
                 for e in self.global_error_dict.values() if e["batch"] == b  and e["used"]]


        plot_images.append({"title": "Claims",
                            'cmap': "rand",
                            'scatter':e2_pos,
                            'scatter_color': e2_color,
                            'im': claims,
                            'interpolation': 'none'})

        e1_pos = np.array([np.array(e["e1_pos"]) - self.pad
                  for e in self.global_error_dict.values()
                  if e["batch"] == b  and e["used"]])
        e1_color = ["g" if e["used"] else 'r'\
                 for e in self.global_error_dict.values() if e["batch"] == b  and e["used"]]
        plot_images.append({"title": "Ground Truth Label",
                            'scatter': e1_pos,
                            'scatter_color': e1_color,
                            "cmap": "rand",
                            'im': self.global_label_batch[b, :, :],
                            'interpolation': 'none'})

        plot_images.append({"title": "Overflow Map",
                            'im': self.global_errormap[b, 1, :, :],
                            'interpolation': 'none'})

        plot_images.append({"title": "Heightmap GT",
                            'im': self.global_height_gt_batch[b, :, :],
                            'scatter': np.array(self.global_seeds[b]) - self.pad,
                            'interpolation': 'none'})

        plot_images.append({"title": "Height Differences",
                            'im': self.global_heightmap_batch[b, :, :] -
                                  self.global_height_gt_batch[b, :, :],
                            'interpolation': 'none'})

        plot_images.append({"title": "Direction Map",
                            'im': self.global_directionmap_batch[b, :, :],
                            'interpolation': 'none'})

        plot_images.append({"title": "Path Map",
                            'scatter': np.array(
                                [np.array(e["large_pos"]) - self.pad for e in
                                 self.global_error_dict.values() if
                                 e["batch"] == b]),
                            'im': self.global_errormap[b, 2, :, :],
                            'interpolation': 'none'})

        if not self.hard_regions is None:
            plot_images.append({"title": "Hard Regions",
                    'im': self.hard_regions[b, :, :],
                    'scatter':e2_pos,
                    'scatter_color': e2_importance_color,
                    'interpolation': 'none'})

        timemap = np.array(self.global_timemap[b, self.pad:-self.pad, self.pad:-self.pad])
        timemap[timemap < 0] = 0
        plot_images.append({"title": "Time Map ",
                                'im': timemap})

        if save:
            print "saving image to ",path,image_name
            u.save_images(plot_images, path=path, name=image_name)
        else:
            print 'show'
            plt.show()

    def draw_error_paths(self, image_name, path='./../data/debug/'):
        def draw_id_bar(axis, ids, gt_label_image):
            # ax2 = axis.twinx()
            # ax2.plot(ids, linewidth=3)
            if len(ids) > 0:
                max_idx = np.max(ids)
                last_id = ids[0]
                current_back = 0
                current_front = 0
                for idx in ids:
                    if idx != last_id:
                        color = gt_label_image.cmap(gt_label_image.norm(last_id))
                        axis.axvspan(current_back, current_front,
                                     color=color)
                        axis.text(current_back, 0.5, str(int(last_id)),
                                  fontsize=12, rotation=90,va='bottom')
                        last_id = idx
                        current_back = current_front
                    current_front += 1
                color = gt_label_image.cmap(gt_label_image.norm(idx))
                axis.text(current_back, 0.5, str(int(idx)), fontsize=12,
                          rotation=90, va='bottom')
                axis.axvspan(current_back, current_front,color=color)

        def fill_gt(axis, ids, cmap):
            print "fill with ", ids
            for i, x in enumerate(ids):
                # polygon = plt.Rectangle((i-0.5,0),1,-1,color=cmap(x))
                # axis.add_patch(polygon)
                axis.axvspan(i - 0.5, i + 0.5, color=cmap(x % 256), alpha=0.5)

        cmap = u.random_color_map()
        MAXLENGTH = 200

        for nume, error in enumerate(self.global_error_dict.values()):
            f, ax = plt.subplots(nrows=2)
            gt_label_image = ax[1].imshow(self.global_label_batch[error["batch"], :, :],
                         interpolation=None, cmap=cmap)
            pred = {}
            height = {}
            gt_id = {}
            gt_height = {}

            x_min_max_coord = [10000000, 0]
            y_min_max_coord = [10000000, 0]

            color_sl = {"small_pos": "r", "large_pos": "g"}

            for e_name in ["small_pos", "large_pos"]:
                startpos = error[e_name]
                pred[e_name] = []
                height[e_name] = []
                gt_id[e_name] = []
                gt_height[e_name] = []

                prev_direction = None
                # prev_pos = None

                pos_xy = []

                for pos, d in self.get_path_to_root(startpos, error["batch"]):
                    pos_xy.append(pos)
                    used_direction = \
                        self.global_directionmap_batch[error["batch"],
                                                             pos[0] - self.pad,
                                                             pos[1] - self.pad]
                    if prev_direction != None:
                        pred[e_name].append(
                            self.global_prediction_map[error["batch"],
                                                       pos[0] - self.pad,
                                                       pos[1] - self.pad,
                                                       prev_direction])
                    height[e_name].append(
                        self.global_heightmap_batch[error["batch"],
                                                    pos[0] - self.pad,
                                                    pos[1] - self.pad])
                    gt_id[e_name].append(self.global_label_batch[error["batch"],
                                                                 pos[0] - self.pad,
                                                                 pos[1] - self.pad])
                    gt_height[e_name].append(
                        self.global_height_gt_batch[error["batch"],
                                                    pos[0] - self.pad,
                                                    pos[1] - self.pad])
                    prev_direction = d

                pred[e_name].append(0)
                pos_xy = np.array(pos_xy) - self.pad
                x_min_max_coord[0] = min(x_min_max_coord[0], np.min(pos_xy[:, 0]))
                x_min_max_coord[1] = max(x_min_max_coord[1], np.max(pos_xy[:, 0]))
                y_min_max_coord[0] = min(y_min_max_coord[0], np.min(pos_xy[:, 1]))
                y_min_max_coord[1] = max(y_min_max_coord[1], np.max(pos_xy[:, 1]))
                ax[1].plot(pos_xy[:, 1], pos_xy[:, 0], marker=',',
                              color=color_sl[e_name])

            pred["small_pos"].reverse()
            height["small_pos"].reverse()
            gt_id["small_pos"].reverse()
            gt_height["small_pos"].reverse()

            height["small_pos"].append(
                self.global_prediction_map[error["batch"],
                                           error["small_pos"][0] - self.pad,
                                           error["small_pos"][1] - self.pad,
                                           error["small_direction"]])

            height["large_pos"].insert(0,
                self.global_prediction_map[error["batch"],
                                           error["large_pos"][0] - self.pad,
                                           error["large_pos"][1] - self.pad,
                                           error["large_direction"]])

            ax[0].plot(pred["small_pos"][-MAXLENGTH:], "r:")
            ax[0].plot(np.arange(len(pred["large_pos"][:MAXLENGTH])) + \
                       len(pred["small_pos"][-MAXLENGTH:]) \
                       , pred["large_pos"][:MAXLENGTH], "g:")
            ax[0].plot(height["small_pos"][-MAXLENGTH:], "r-")

            ax[0].plot(np.arange(len(height["large_pos"][:MAXLENGTH])) + \
                       len(height["small_pos"][-MAXLENGTH:]) - 2 \
                       , height["large_pos"][:MAXLENGTH], "g-")

            ax[0].plot(gt_height["small_pos"][-MAXLENGTH:], "k-")
            ax[0].plot(np.arange(len(gt_height["large_pos"][:MAXLENGTH])) + \
                       len(gt_height["small_pos"][-MAXLENGTH:]) \
                       , gt_height["large_pos"][:MAXLENGTH], "k-")

            ids = gt_id["small_pos"][-MAXLENGTH:] + gt_id["large_pos"][:MAXLENGTH]
            # fill_gt(ax, ids, cmap)
            draw_id_bar(ax[0], ids, gt_label_image)
            ax[0].axvline(len(pred["small_pos"][-MAXLENGTH:]) - 0.5, color='k',
                          linestyle='-')

            x_min_max_coord[0] -= self.pad
            y_min_max_coord[0] -= self.pad
            x_min_max_coord[1] += self.pad
            y_min_max_coord[1] += self.pad

            ax[1].set_xlim(y_min_max_coord)
            ax[1].set_ylim(x_min_max_coord)

            f.savefig(path + image_name + '_e%07d' % nume, dpi=200)
            plt.close(f)


class HoneyBatcherE(HoneyBatcherPath):
    def get_plateau_indicator(self):
        return self.global_prediction_map_nq  > 0

    def update_priority_queue(self, heights_batch, centers, ids):
        for b, center, Id, height in zip(range(self.bs), centers, ids,
                                          heights_batch[:, :, 0, 0]):
            # if possibly wrong
            cross_x, cross_y, cross_d = self.get_cross_coords(center)
            lower_bound = self.global_heightmap_batch[b,
                                                      center[0] - self.pad,
                                                      center[1] - self.pad] + \
                                                      self.lowercomplete_e
            if lower_bound < 0:
                lower_bound = 0
            if lower_bound == np.inf:
                print "encountered inf for prediction center !!!!", \
                    b, center, Id, height, lower_bound
                raise Exception('encountered inf for prediction center')
            # debug
            self.global_prediction_map_nq[
                b, center[0] - self.pad, center[1] - self.pad, :] = \
                height
            self.max_new_old_pq_update(b, cross_x, cross_y, height+lower_bound,
                                        lower_bound, Id, cross_d, center,
                                           input_time=self.global_time)


class HoneyBatcherBatcher(HoneyBatcherPath):
    """
    batch control class that propagates hidden states along the minimal spanning tree
    """
    def __init__(self, options):
        super(HoneyBatcherPath, self).__init__(options)
        self.global_hidden_states = None


class HoneyBatcherPatchFast(HoneyBatcherPath):
    def __init__(self, options):
        super(HoneyBatcherPatchFast, self).__init__(options)
        self.n_channels = 2
        self.edge_map_gt = None
    def get_network_input(self, center, b, Id, out):
        self.crop_mask_claimed(center, b, Id, out=out[0:2])
        return out

    def reconstruct_input_at_timepoint(self, timepoints, centers, ids, batches):
        raw_batch = np.zeros((len(batches), self.n_channels, self.pl, self.pl),
                             dtype='float32')
        for i, (b, center, Id) in enumerate(zip(batches, centers, ids)):
            raw_batch[i, :, :, :] = \
                self.get_network_input(center, b, Id, raw_batch[i, :, :, :])
            assert (self.global_timemap[b, center[0], center[1]] == timepoints[i])
        mask = self.crop_time_mask(centers, timepoints, batches)
        raw_batch[:, 0, :, :][mask] = 0
        raw_batch[:, 1, :, :][mask] = 0
        if self.options.claim_channels > 2:
            raw_batch[:, 2, :, :][mask] = 1
        if self.options.claim_channels > 3:
            # simple masking of time does not work here, since the nearest neighbor
            # can change over time
            raise NotImplementedError()
        return raw_batch


class HoneyBatcherRec(HoneyBatcherPath):
    def __init__(self, options):
        super(HoneyBatcherRec, self).__init__(options)
        self.global_hidden_states = np.empty((options.batch_size, 4, self.label_shape[1], self.label_shape[2],
                                                  options.n_recurrent_hidden))      # pre pq [<-->]
        self.n_recurrent_hidden = options.n_recurrent_hidden
        self.initial_hiddens = None

    def init_batch(self, start=None, allowed_slices=None):
        super(HoneyBatcherRec, self).init_batch(start=start, allowed_slices=None)
        self.global_hidden_states.fill(np.nan)

    def update_priority_queue_i(self, b, center, Id, height, hidden_states=None):
        self.global_hidden_states[b, :, center[0] - self.pad, center[1] - self.pad, :] = hidden_states[b, :, :]
        super(HoneyBatcherRec, self).update_priority_queue_i(b, center, Id, height)

    # def initialize_hiddens(self, callback):
    #     self.initial_hiddens = {}
    #     sequ_len = 1
    #     rnn_mask = np.ones((self.bs, sequ_len), dtype=np.float32)
    #     for b, (seeds, ids) in enumerate(zip(self.global_seeds,
    #                                          self.global_seed_ids)):
    #         for seed, Id in zip(seeds, ids):
    #             raw_batch = np.zeros((1, self.n_channels, self.pl, self.pl), dtype='float32')
    #             self.get_network_input(seed, b, Id, raw_batch[0])
    #             new_hidden = callback(raw_batch[:, 0:self.options.claim_channels,1:-1, 1:-1], raw_batch[:, self.options.claim_channels:,1:-1, 1:-1],
    #                              np.zeros((4, self.n_recurrent_hidden), dtype=np.float32), rnn_mask, 1)
    #             print (b, seed[0], seed[1]),"new_hidden",new_hidden
    #             self.initial_hiddens[(b, seed[0], seed[1])] = np.mean(new_hidden[0],axis=0)
    #             print (b, seed[0], seed[1]),"new_hidden stored",np.mean(new_hidden[0],axis=0)

    def get_hidden(self, b, center):
        try:
            direction = self.global_directionmap_batch[b, center[0] - self.pad, center[1] - self.pad]
        except Exception as e:
            print e.message, e.args
            embed()
        if direction == -1:  # seed
            if self.initial_hiddens is not None:
                print "getting hidden state"
                return self.initial_hiddens[(b, center[0],center[1])]
            else:
                return np.zeros((self.n_recurrent_hidden), dtype=np.float32)
        else:
            origin = self.update_position(center, self.global_directionmap_batch[b, center[0] - self.pad,
                                                                                    center[1] - self.pad])
            return self.global_hidden_states[b, direction, origin[0] - self.pad, origin[1] - self.pad, :]

    def get_batches(self):
        raw_batch, gts, centers, ids = super(HoneyBatcherRec, self).get_batches()
        hiddens = np.zeros((self.bs,  self.n_recurrent_hidden), dtype='float32')
        # load hidden states
        for b, center in enumerate(centers):
            hiddens[b] = self.get_hidden(b, center)
        return raw_batch, gts, centers, ids, hiddens

    def reverse_path(self, path, mask_key):
        count_non_mask = np.sum([mask_key not in e for e in path])
        path[:count_non_mask] = path[:count_non_mask][::-1]
        return path

    def backtrace_error(self, selection, backtrace_length, err_type, id_type):
        error_selection = []
        hidden_coords = []
        for sel in selection:
            current_error = copy.copy(self.global_error_dict[sel])
            new_path_error = []
            new_path_error.append(current_error)
            for t_back in range(1, backtrace_length):
                current_error = self.backtrace_error_step(current_error, err_type + "_time", err_type + "_pos",
                                                          id_type + "_id", err_type + "_direction", err_type + "_mask")
                new_path_error.append(current_error)
            new_path_error = self.reverse_path(new_path_error, err_type + "_mask")
            error_selection += new_path_error
            init_err_pos = current_error[err_type + "_pos"]
            # TODO: check me some time :)
            # init_err_dir = current_error[err_type + "_direction"]
            # origin = self.update_position(init_err_pos, init_err_dir)
            hidden_coords.append([current_error["batch"], init_err_pos])
        return error_selection, hidden_coords

    def backtrace_error_step(self, error, key_time, key_center, key_id, key_direction, key_mask):
        bt_error = copy.copy(error)
        batch, center, Id = [error['batch'], error[key_center], error[key_id]]

        if not error["slow_intruder"] and error["first_rec"] and 'e2' in key_time:
            direction = error[key_direction]        # first e2 of fast intruder (last in time) is not claimed
            error["first_rec"] = False
        else:
            direction = self.global_directionmap_batch[batch, center[0] - self.pad, center[1] - self.pad]

        # debug
        if direction == -1:
            print 'err type', error, key_time, key_center, key_id, key_direction, key_mask
            raise Exception("")

        # # debug
        # if -1 in bt_pos or self.image_shape[-1] in bt_pos:
        #     print 'err type', error, key_time, key_center, key_id, key_direction, key_mask
        #     embed()
        bt_pos = self.update_position(center, direction)
        bt_direction = self.global_directionmap_batch[batch, bt_pos[0] - self.pad, bt_pos[1] - self.pad]

        if bt_direction == -1:      # stop 1 step before reaching seed (seed is given)
            bt_error[key_mask] = True
            return bt_error
        else:
            bt_error[key_center] = bt_pos
            old_direction = self.global_directionmap_batch[batch, bt_pos[0] - self.pad, bt_pos[1] - self.pad]
            old_old_pos = self.update_position(bt_pos, old_direction)
            bt_error[key_time] = self.global_timemap[batch, old_old_pos[0], old_old_pos[1]]
            # debug
            if bt_error[key_time] < 0:
                print 'err negative time type', error, key_time, key_center, key_id, key_direction, key_mask
                embed()
            if np.any(old_old_pos < 0) or old_old_pos[0] >= self.image_shape[-2] - self.pad or old_old_pos[1] >= self.image_shape[-1] - self.pad:
                print 'err type', error, key_time, key_center, key_id, key_direction, key_mask
                embed()
            bt_error[key_direction] = old_direction
            return bt_error

    def reconstruct_path_error_inputs(self, backtrace_length=0):
        selection = self.select_errors()

        rnn_masks = []
        rnn_hidden_inits = []
        reconst_es = []
        error_selections = []
        for err_type, id_type in zip(['e1', 'e2'], ['large', 'small']):
            if backtrace_length == 0:
                error_selection = [global_error_dict[e] for e in selection]
            else:
                error_selection, hidden_coords = self.backtrace_error(selection, backtrace_length, err_type, id_type)

            rnn_hidden_init = [self.get_hidden(b, hidden_coord) for b, hidden_coord in hidden_coords]
            rnn_mask = [err_type + '_mask' not in e for e in error_selection]
            rnn_mask = np.array(rnn_mask, dtype=np.bool).reshape((-1, backtrace_length))

            reconst_e = self.reconstruct_input_at_timepoint(error_selection, err_type + "_time", err_type + "_pos",
                                                            id_type + "_id", err_type + "_direction")[:, :, 1:-1, 1:-1]
            rnn_masks.append(rnn_mask)
            reconst_es.append(reconst_e)
            error_selections.append(error_selection)
            rnn_hidden_inits.append(np.array(rnn_hidden_init))

            # debug
            # if err_type == 'e1':
            #     self.e1_pos = [err['e1_pos'] for err in error_selection]
            #     self.e1_b = [err['batch'] for err in error_selection]

        self.rnn_masks = rnn_masks
        self.reconst_es = reconst_es
        self.error_selections = error_selections
        self.rnn_hidden_inits = rnn_hidden_inits
        self.selection = selection
        return reconst_es[0], reconst_es[1], rnn_masks[0], rnn_masks[1], rnn_hidden_inits[0], rnn_hidden_inits[1]


class HoneyBatcherERec(HoneyBatcherRec):
    def update_priority_queue_i(self, b, center, Id, edge, hidden_states=None):
        prev_height = self.global_heightmap_batch[b, center[0] - self.pad, center[1] - self.pad]
        if prev_height == -np.inf :
            height = edge
        else:
            height = edge + prev_height
        super(HoneyBatcherERec, self).update_priority_queue_i(b, center, Id, height, hidden_states)

    def set_plateau_indicator(self):
        self.global_plateau_indicator = self.global_prediction_map_nq  == 0


class SeedMan(object):
    def __init__(self):
        self.global_seeds = None

    def get_seed_coords_gt(self, label_image, offset=0, minsize=0):
        # perform connected components to remove disconnected (same id) regions  # analysis.labelImage(label_image[b], out=label_image[b])
        label_image = measure.label(label_image)
        seed_ids = np.unique(label_image[:, :]).astype(int)

        # add border to labels
        padded_label = data_provider.pad_cube(label_image[:, :], 1, value=np.max(seed_ids) + 1)

        dist_trf = data_provider.segmenation_to_membrane_core(padded_label)[1][1:-1, 1:-1]
        seeds = []

        for Id in seed_ids:  # ids within each slice
            if minsize > 0 and np.sum(label_image[:, :] == Id) < minsize:
                print "removing seed for small(<%i) region with id %i" % (minsize, Id)
                continue
            regions = np.where(label_image[:, :] == Id)
            seed_ind = np.argmax(dist_trf[regions])
            seed = np.array([regions[0][seed_ind], regions[1][seed_ind]]) + offset
            seeds.append([seed[0], seed[1]])
        return seeds

    def get_seed_coords_grid(self, gridsize=7):
        seeds = [(x + self.pad, y + self.pad) for x, y in product(xrange(offset_x, shape[0], gridsize),
                                                                  xrange(offset_y, shape[1], gridsize))]
        return seeds

    def get_seed_coords_timo(self, image, min_memb_size=5, sigma=1.0, thresh=0.2):
        x, y = wsDtseeds(image, thresh, self.timo_min_len, sigma, groupSeeds=True)
        seeds = [[x_i + self.pad, y_i + self.pad] for x_i, y_i in zip(x, y)]


# TODO: make this loopy (maybe with lambdas and slices ???)
def augment_batch(batch, gt=None, direction=None):

    augment_shape = list(batch.shape)
    bs = batch.shape[0]
    augment_shape[0] *= 7
    augmented_batch = np.empty(augment_shape, dtype='float32')

    # original
    ac = 0
    augmented_batch[ac*bs:(ac+1)*bs] = batch
    # flip x
    ac = 1
    augmented_batch[ac*bs:(ac+1)*bs] = batch[:, :, ::-1, :]
    # flip y
    ac = 2
    augmented_batch[ac*bs:(ac+1)*bs] = batch[:,:,:,::-1]
    # turn 180
    ac = 3
    augmented_batch[ac*bs:(ac+1)*bs] = batch[:,:,::-1,::-1]

    transpose = np.transpose(batch, (0,1,3,2))
    ac = 4
    augmented_batch[ac*bs:(ac+1)*bs] = transpose
    # rot 90 by transpose and flipx
    ac = 5
    augmented_batch[ac*bs:(ac+1)*bs] = transpose[:,:,::-1,:]
    # rot -90 by transpose and flipy 
    ac = 6
    augmented_batch[ac*bs:(ac+1)*bs] = transpose[:,:,::-1,:]

    if gt is not None:
        # apply inverse transform on gt batch
        augment_gt_shape = list(gt.shape)
        augment_gt_shape[0] *= 7
        augmented_gt = np.empty(augment_gt_shape, dtype='float32')

        dgt = {}
        dgt["left"] = gt[:,0,:,:]
        dgt["down"] = gt[:,1,:,:]
        dgt["right"] = gt[:,2,:,:]
        dgt["up"] = gt[:,3,:,:]
        # original
        ac = 0
        augmented_gt[ac*bs:(ac+1)*bs] = gt
        # flip x
        ac = 1
        augmented_gt[ac*bs:(ac+1)*bs] = np.stack([dgt["right"],
                                                     dgt["up"],
                                                     dgt["left"],
                                                     dgt["down"]
                                                    ]).swapaxes(0,1)
        # flip y
        ac = 2
        augmented_gt[ac*bs:(ac+1)*bs] = np.stack([dgt["left"],
                                                     dgt["down"],
                                                     dgt["right"],
                                                     dgt["up"]
                                                    ]).swapaxes(0,1)
        # turn 180
        ac = 3
        augmented_gt[ac*bs:(ac+1)*bs] = np.stack([dgt["right"],
                                                     dgt["down"],
                                                     dgt["left"],
                                                     dgt["up"]
                                                    ]).swapaxes(0,1)
        # transpose
        ac = 4
        augmented_gt[ac*bs:(ac+1)*bs] = np.stack([dgt["up"],
                                                     dgt["left"],
                                                     dgt["down"],
                                                     dgt["right"]
                                                    ]).swapaxes(0,1)
        # rot 90 by transpose and flipx
        ac = 5
        augmented_gt[ac*bs:(ac+1)*bs] = np.stack([dgt["down"],
                                                  dgt["left"],
                                                  dgt["up"],
                                                  dgt["right"]
                                                    ]).swapaxes(0,1)
        # rot -90 by transpose and flipy 
        ac = 6
        augmented_gt[ac*bs:(ac+1)*bs] = np.stack([dgt["up"],
                                                  dgt["right"],
                                                  dgt["down"],
                                                  dgt["left"]
                                                    ]).swapaxes(0,1)
        return augmented_batch, augmented_gt

    if direction != None:
        # apply inverse transform to direction
        augment_dir_shape = list(direction.shape)
        augment_dir_shape[0] *= 7
        augmented_dir = np.empty(augment_dir_shape, dtype=np.int32)

            # original
        ac = 0
        augmented_dir[ac*bs:(ac+1)*bs] = direction
        # flip x
        ac = 1
        augmented_dir[ac*bs:(ac+1)*bs] = direction
        augmented_dir[ac*bs:(ac+1)*bs][direction == 0] = 2
        augmented_dir[ac*bs:(ac+1)*bs][direction == 2] = 0
        # flip y
        ac = 2
        augmented_dir[ac*bs:(ac+1)*bs] = direction
        augmented_dir[ac*bs:(ac+1)*bs][direction == 1] = 3
        augmented_dir[ac*bs:(ac+1)*bs][direction == 3] = 1

        # turn 180
        ac = 3
        augmented_dir[ac*bs:(ac+1)*bs] = (direction + 2) % 4

        ac = 4
        augmented_dir[ac*bs:(ac+1)*bs][direction == 0] = 1
        augmented_dir[ac*bs:(ac+1)*bs][direction == 1] = 0
        augmented_dir[ac*bs:(ac+1)*bs][direction == 2] = 3
        augmented_dir[ac*bs:(ac+1)*bs][direction == 3] = 2

        # rot 90 by transpose and flipx
        ac = 5
        augmented_dir[ac*bs:(ac+1)*bs] = (direction + 1) % 4
        # rot -90 by transpose and flipy 
        ac = 6
        augmented_dir[ac*bs:(ac+1)*bs] = (direction + 3) % 4
        return augmented_batch, augmented_dir

    return augmented_batch


def average_ouput(output, ft=False):
    aug_out_shape = list(output.shape)
    aug_out_shape[0] = 7
    aug_out_shape.insert(0,-1)
    augm_out = output.reshape(aug_out_shape)
    output_shape = list(augm_out.shape)
    del output_shape[1]
    mean_out = np.empty(output_shape)

    if ft:
        mean_out = np.mean(augm_out,axis=1)
    else:
        c = np.arange(7)
        mean_out[:,0,:,:] = np.mean(augm_out[:,c,[0,2,0,2,1,1,3],:,:],axis=1)
        mean_out[:,1,:,:] = np.mean(augm_out[:,c,[1,1,3,3,0,2,0],:,:],axis=1)
        mean_out[:,2,:,:] = np.mean(augm_out[:,c,[2,0,2,0,3,3,1],:,:],axis=1)
        mean_out[:,3,:,:] = np.mean(augm_out[:,c,[3,3,1,1,2,0,2],:,:],axis=1)
    return mean_out


def grad_to_height(grad):
    height = np.empty((grad.shape[0],4,1,1),dtype=grad.dtype)
    height[:,0,:,:] = grad[:,0,:,:]-grad[:,1,:,:]
    height[:,1,:,:] = grad[:,0,:,:]+grad[:,2,:,:]
    height[:,2,:,:] = grad[:,0,:,:]+grad[:,1,:,:]
    height[:,3,:,:] = grad[:,0,:,:]-grad[:,2,:,:]
    return height


def height_to_grad(height):
    grad = np.empty((height.shape[0],3,1,1),dtype=height.dtype)
    grad[:,0,:,:] = height.mean(axis=1)
    grad[:,1,:,:] = height[:,2,:,:]-height[:,0,:,:]
    grad[:,2,:,:] = height[:,1,:,:]-height[:,3,:,:]
    return grad


def height_to_fc_height_gt(height):
    fc_height_shape = list(height.shape)
    fc_height_shape.insert(1, 4)
    fc_height = np.zeros((fc_height_shape), dtype='float32')
    # cross_coords = [[-1, 0], [0, -1], [1, 0], [0, 1]]
    # top
    fc_height[:, 0, 1:, :] = height[:, :-1, :]
    fc_height[:, 0, 0, :] = height[:, 0, :]
    # left
    fc_height[:, 1, :, 1:] = height[:, :, :-1]
    fc_height[:, 1, :, 0] = height[:, :, 0]
    # bottom
    fc_height[:, 2, :-1, :] = height[:, 1:, :]
    fc_height[:, 2, -1, :] = height[:, -1, :]
    # right
    fc_height[:, 3, :, :-1] = height[:, :, 1:]
    fc_height[:, 3, :, -1] = height[:, :, -1]
    return fc_height


def height_to_fc_edge_gt(height):
    edge_height_shape = list(height.shape)
    edge_height_shape.insert(1, 4)
    global_edges = np.zeros((edge_height_shape), dtype='float32')
    # cross_coords = [[-1, 0], [0, -1], [1, 0], [0, 1]] top, left, bottom, right
    # top
    edges = height[:, :-1, :] - height[:, 1:, :]
    global_edges[:, 0, 1:, :] = edges
    global_edges[:, 0, 0, :] = 0.
    # left
    edges = height[:, :, :-1] - height[:, :, 1:]
    global_edges[:, 1, :, 1:] = edges
    global_edges[:, 1, :, 0] = 0
    # bottom
    edges = height[:, 1:, :] - height[:, :-1, :]
    global_edges[:, 2, :-1, :] = edges
    global_edges[:, 2, -1, :] = 0
    # right
    edges = height[:, :, 1:] - height[:, :, :-1]
    global_edges[:, 3, :, :-1] = edges
    global_edges[:, 3, :, -1] = 0
    np.absolute(global_edges, global_edges)
    return global_edges


class HoneyBatcherGonzales(HoneyBatcherPath):
    def __init__(self, options):
        super(HoneyBatcherGonzales, self).__init__(options)
        shape = list(self.batch_shape)
        self.global_input_batch = options.theano.shared(np.ones(shape, dtype='float32'))
        shape[1] = 1

        self.global_claims = options.theano.shared(np.ones(shape, dtype='float32'))
        coord = T.ivector()
        batch = T.iscalar()
        set_val_t = T.iscalar()

        self.set_claim_f = options.theano.function([batch, coord, set_val_t], 
            updates=[(self.global_claims,
                      T.set_subtensor(self.global_claims[batch, 0, coord[0],
                                                   coord[1]], set_val_t))])

        check_val_t = T.iscalar()
        cx = T.iscalar()
        cy = T.iscalar()
        self.check_claims = options.theano.function([batch, cx, cy, check_val_t], 
                            T.eq(self.global_claims[batch, 0, cx, cy], check_val_t),
                            allow_input_downcast=True)

        self.cout_free_voxels = options.theano.function([],T.sum(T.eq(self.global_claims[0, 0],0)))

        raw_list = []
        claim_list = []
        mes = T.ivector()
        coords = T.imatrix()
        self.shared_input_coord_list = [coords, mes]

        for b in range(self.bs):
            crop_raw = self.global_input_batch[None, b, :,
                                             coords[b, 0]-self.pad:coords[b, 0] + self.pad,
                                             coords[b, 1]-self.pad:coords[b, 1] + self.pad]

            claim_c = self.global_claims[None, b, :, coords[b, 0] - self.pad:coords[b, 0] + self.pad,
                                          coords[b, 1] - self.pad:coords[b, 1] + self.pad]

            claim_me = T.eq(claim_c, mes[b])
            claim_them_with_bg = T.neq(claim_c, mes[b])
            claim_not_bg = T.neq(claim_c, 0)
            claim_them = claim_them_with_bg & claim_not_bg
            raw_list.append(crop_raw)
            claim_list.append(T.cast(T.concatenate((claim_me,claim_them),axis=1),dtype='float32'))

        self.shared_input_batch = T.concatenate(raw_list, axis=0)
        self.shared_claims_batch = T.concatenate(claim_list, axis=0)


    def prepare_global_batch(self):
        input_x = np.empty(self.batch_shape, dtype='float32')
        rois = self.batch_data_provider.prepare_input_batch(\
                                            input_x)
        self.global_input_batch.set_value(input_x)
        self.batch_data_provider.prepare_label_batch(self.global_label_batch,
                                                     self.global_height_gt_batch,
                                                     rois)

    def get_network_input(self, center, b, Id, out):
        raise NotImplementedError

    def set_claims(self, b, center, idx):
        self.set_claim_f(b, np.array(center, dtype='int32'), idx)

    def reset_claims(self):
        shape = list(self.batch_shape)
        shape[1] = 1
        reset_c = np.empty(shape, dtype='float32')
        reset_c.fill(-1)
        reset_c[:, 0, self.pad:-self.pad, self.pad:-self.pad] = 0
        self.global_claims.set_value(reset_c)

    def check_claims_c(self, b, center_x, center_y):
        return not np.isfinite(self.global_heightmap_batch[b,
                                        center_x - self.pad,
                                        center_y - self.pad])

    def get_center_i_from_queue(self, b):
        already_claimed = True
        while already_claimed:
            if self.priority_queue[b].empty():
                self.serialize_to_h5("empty_queue_state")
                self.draw_debug_image("empty_queue")
                raise Exception('priority queue empty. All pixels labeled')
            height, _, center_x, center_y, Id, direction, error_indicator, \
                time_put = self.priority_queue[b].get()

            if self.check_claims_c(b, center_x, center_y):
                already_claimed = False

        return height, _, center_x, center_y, Id, direction, error_indicator, \
                time_put

    def get_batches(self):
        centers, ids, heights = self.get_centers_from_queue()
        for b, (center, height, Id) in enumerate(zip(centers, heights, ids)):
            self.set_claims(b, center, Id)
            # check whether already pulled
            self.global_heightmap_batch[b,
                                        center[0] - self.pad,
                                        center[1] - self.pad] = height

        raw_batch = None
        gts = None
        return raw_batch, gts, centers, ids

    def get_num_free_voxel(self):
        return int(self.cout_free_voxels())

    def get_image_crops(self, b):
        claims = np.array(self.global_claims.get_value(borrow=False),dtype='float32')[b, 0, self.pad:-self.pad, self.pad:-self.pad]
        return self.global_input_batch.get_value()[b, :, self.pad:-self.pad, self.pad:-self.pad], claims

    def check_is_lowest(self, b, heights, x, y, add_all):
        # claim_c = self.gen_calim_c(b, x, y)
        return ((heights < self.global_heightmap_batch[b,
                                           x - self.pad,
                                           y - self.pad]))
                        # & claim_c

    def gen_calim_c(self, b, x, y):
        return np.array([self.check_claims(b, cx, cy, 0) for cx,cy in zip(x,y)],dtype=bool)


class MergeDict(dict):
    def __missing__(self, key):
        return key

    def get_merge_id(self, idx):
        t = idx
        while self[t] != t:
            t = self[t]
        return t

    def merge(self, id1, id2):
        self[max(id1, id2)] = min(id1, id2)
        return min(id1, id2), max(id1, id2)


class HungryHoneyBatcher(HoneyBatcherPath):

    def init_batch(self, **kwargs):
        super(HungryHoneyBatcher, self).init_batch(**kwargs)
        self.merge_dict = [MergeDict() for b in range(self.bs)]
        self.merg_count_neg = 0
        self.merg_count_pos = 0


    def get_merging_gt(self, centers, ids):
        merging_gt = np.zeros((self.bs,4,1,1))
        merging_factor = np.zeros((self.bs,4,1,1))
        merging_ids = np.zeros((self.bs,4,1,1))

        #TODO: make this fast with fewer array accesses by removing batch loop
        for b, center, Id, in zip(range(self.bs), centers, ids):
            # check if neigbour merge is possible
            cross_x, cross_y, cross_d = self.get_cross_coords(center)
            # self.global_claims[b,cross_x,cross_y] -
            claimed = self.global_claims[b,cross_x,cross_y] > 0
            neighbor_id = self.global_claims[b,cross_x,cross_y]
            different_id = neighbor_id - Id != 0
            merge_partner = np.logical_and(claimed, different_id)
            # print "c,d",claimed,different_id,self.global_claims[b,cross_x,cross_y]

            # fast numpy check to avoid loop
            if np.any(merge_partner):
                gt_A = self.global_label_batch[b][self.global_claims[b,self.pad:-self.pad,self.pad:-self.pad] == Id]
                # print self.global_label_batch[b].shape, self.global_claims[b, self.pad:-self.pad, self.pad:-self.pad].shape
                for i,idx,mp in zip(range(4), neighbor_id, merge_partner):
                    if mp:
                        gt_B = self.global_label_batch[b][self.global_claims[b, self.pad:-self.pad, self.pad:-self.pad] == idx]
                        gt_AB = np.concatenate((gt_A,gt_B))
                        # union region score
                        regions = np.ones((gt_A.shape[0]+ gt_B.shape[0]))
                        # rand_union = adjusted_rand_score(gt_AB,regions)
                        VOI_union = ground_truth.variationOfInformation(
                                                gt_AB, regions, False)[0]
                        # split region score
                        regions[:gt_A.shape[0]] = 2
                        # rand_split = adjusted_rand_score(gt_AB,regions)
                        VOI_split = ground_truth.variationOfInformation(
                                                gt_AB, regions, False)[0]
                        # print "AB ",gt_AB,regions
                        # print "A ",gt_A
                        # print "B ",gt_B
                        # print "ids",Id,idx
                        merging_factor[b,i,0,0] = 1
                        merging_ids[b,i,0,0] = idx
                        # combare variation of information (lower is better)
                        # if VOI_split > VOI_union:

                        # check if majority of GT pixels agree
                        most_A = np.argmax(np.bincount(gt_A.astype(np.int)))
                        most_B = np.argmax(np.bincount(gt_B.astype(np.int)))
                        if most_B == most_A:
                            merging_gt[b,i,0,0] = 1
                            self.merg_count_pos += 1
                        else:
                            merging_gt[b,i,0,0] = 0
                            self.merg_count_neg += 1

        return merging_gt, merging_factor, merging_ids

    def get_batches(self):
        raw_batch, gts, centers, ids = super(HungryHoneyBatcher, self).get_batches()
        merging_gt, merging_factor, merging_ids = self.get_merging_gt(centers, ids)
        return raw_batch, gts, centers, ids, merging_gt, merging_factor, merging_ids

    def update_merge(self, merge_probs, merging_factor, merging_ids, ids):
        # print np.where(merging_factor != 0)
        # print np.transpose(np.where(merging_factor != 0))
        for b, direction, _, _ in np.transpose(np.where(merging_factor != 0)):
            # greedy merge if p > 0.5
            # print "merge_probs",merge_probs[b,direction,0,0],merging_factor[b,direction,0,0]
            if merge_probs[b,direction,0,0] >= 0.7:
                print "merging ",ids[b], "and", merging_ids[b,direction,0,0]
                self.merge_regions(b, ids[b],merging_ids[b,direction,0,0])


    def merge_regions(self, batch, id1, id2):
        merge_id, old_id = self.merge_dict[batch].merge(id1, id2)

        claims = self.global_claims[batch]
        claims[claims == old_id] = merge_id

    def get_centers_from_queue(self):
        centers, ids, heights = \
            super(HungryHoneyBatcher, self).get_centers_from_queue()
        # translate ids if merged
        translated_ids = [self.merge_dict[b].get_merge_id(ids[b]) for b in range(self.bs)]
        return centers, translated_ids, heights


if __name__ == '__main__':
    # generate_dummy_data2(2, edge_len=50, patch_len=40,
    #                      save_path='./../data/debug/bla')
    #

    # path = '/media/liory/DAF6DBA2F6DB7D67/cremi/final/CREMI-pmaps-padded/'
    # names = ['B+_last']

    # prpare_seg_path_wrapper(path, names)
    #
    # pass
    # path = '/media/liory/DAF6DBA2F6DB7D67/cremi/cremi_testdata'
    # prepare_aligned_test_data(path)
    # path = './data/volumes/'
    # cut_reprs(path)

    #
    # fig, ax = plt.subplots(4, 10)
    # for j in range(10):
    #     a = np.ones((1, 10, 40, 40))
    #     a = create_holes(a, 40)
    #
    #     for n, i in enumerate([0, 8, 6, 7]):
    #         ax[n, j].imshow(a[0, i], cmap='gray')
    # plt.show()

    # bm = HoneyBatcherPath('./data/volumes/membranes_second.h5' ,\
    #                     label = './data/volumes/label_second.h5' ,\
    #                     height_gt = './data/volumes/height_second.h5' ,\
    #                     raw = './data/volumes/raw_second.h5' ,\
    #                     batch_size = 1 ,\
    #                     patch_len = 40 ,\
    #                     global_edge_len = 1210 ,\
    #                     padding_b = False ,\
    #                     find_errors_b = False ,\
    #                     clip_method = 'clip' ,\
    #                     seed_method = True ,\
    #                     z_stack = True ,\
    #                     downsample = True ,\
    #                     scale_height_factor = 60.0 ,\
    #                     perfect_play = False ,\
    #                     add_height_b = False ,\
    #                     max_penalty_pixel = 3)
    #
    # sigma = 1
    # for ml in np.arange(1,10,2):
    #     # for sigma in np.arange(0.1,2,0.1):
    #     bm.timo_sigma = sigma
    #     bm.timo_sigma = 1.1
    #     bm.timo_min_len = ml
    #     bm.init_batch(allowed_slices=[70])
    #     bm.draw_debug_image('seed_%f_%i.png' % (sigma, ml), path='./data/nets/debug/images/')
    #
    #
    # path = '/media/liory/DAF6DBA2F6DB7D67/cremi/data/labeled/'
    # generate_quick_eval_big_FOV_z_slices(path, suffix='_first')
    #
    # # generate_dummy_data(20, 300, 40, save_path='')
    #
    # # loading of cremi
    # # path = './data/sample_A_20160501.hdf'
    # # /da
    # # a = make_array_cumulative(a)
    # # save_h5('./data/label_a.h5', 'labels', a, 'w')
    # # plt.imshow(a[5, :, :], cmap='Dark2')
    # # plt.show()
    #
    # # loading from BM
    #
    segmentation_to_membrane('../data/volumes/label_honeycomb_2.h5',"../data/volumes/height_honeycomb_2.h5")
    # # segmentation_to_membrane('./data/volumes/label_b.h5',"./data/volumes/height_b.h5")
    #
    # # bm = BatchManV0(raw_path, label_path, batch_size=10, patch_len=60,
    # #                 global_edge_len=95)
    # # bm.init_train_batch()
    #
    # # net_name = 'cnn_ID2_trash'
    # # label_path = './data/volumes/label_a.h5'
    # # label_path_val = './data/volumes/label_b.h5'
    # # height_gt_path = './data/volumes/height_a.h5'
    # # height_gt_key = 'height'
    # # height_gt_path_val = './data/volumes/height_b.h5'
    # # height_gt_key_val = 'height'
    # # raw_path = './data/volumes/membranes_a.h5'
    # # raw_path_val = './data/volumes/membranes_b.h5'
    # # save_net_path = './data/nets/' + net_name + '/'
    # # load_net_path = './data/nets/cnn_ID_2/net_300000'      # if load true
    # # tmp_path = '/media/liory/ladata/bla'        # debugging
    # # batch_size = 5         # > 4
    # # global_edge_len = 300
    # # patch_len= 40
    # #
    # # bm = BatchManV0(raw_path, label_path,
    # #                 height_gt=height_gt_path,
    # #                 height_gt_key=height_gt_key,
    # #                 batch_size=batch_size,
    # #                 patch_len=patch_len, global_edge_len=global_edge_len,
    # #                 padding_b=True,find_errors=True)
    # # gt_seeds_b = True
    # # seeds = bm.init_train_path_batch()
    # # seeds = np.array(seeds[4])
    # heights = np.random.random(size=batch_size)
    # b = 4
    # raw_batch, gts, centers, ids = bm.get_path_batches()
    # name = 'debug'
    # for i in range(500000):
    #     if i % 100 == 0:
    #         print i
    #
    #     if i % 1000 == 0:
    #         bm.draw_debug_image(name + '_deb_%i' %i, save=True)
    #         print i
    #     raw_batch, gts, centers, ids = bm.get_path_batches()
    #
    #     if i % 5000 == 0 and i != 0:
    #         bm.init_train_path_batch()
    #         raw_batch, gts, centers, ids = bm.get_path_batches()
    #     probs = np.zeros((batch_size, 4, 1,1))
    #     for c in range(batch_size):
    #         d = 0
    #         for x, y, _ in bm.walk_cross_coords(centers[b]):
    #             x -= bm.pad
    #             y -= bm.pad
    #             # probs[c, d] = bm.global_height_gt_batch[b, x, y]
    #             # probs[c, d] = i
    #             probs[c, d] = random.random()
    #             d += 1
    #     bm.update_priority_path_queue(probs, centers, ids)



