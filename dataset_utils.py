import matplotlib
# try:
#     matplotlib.use('Qt4Agg')
# except:
matplotlib.use('Agg')

import h5py as h
import numpy as np
import random
from os import makedirs
from os.path import exists
from ws_timo import wsDtseeds
from matplotlib import pyplot as plt
from Queue import PriorityQueue
from scipy.ndimage.measurements import watershed_ift
import utils as u
from scipy import ndimage
from scipy import stats
from scipy.ndimage import convolve, gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion
from skimage.feature import peak_local_max
from skimage.morphology import label, watershed
from itertools import product
# from sklearn.metrics import adjusted_rand_score
import h5py
# from cv2 import dilate, erode
import data_provider


class HoneyBatcherPredict(object):
    def __init__(self, options):
        """
        batch loader. Use either for predict. For valId and train use:
        get batches function.

        :param options:
        """

        # either pad raw or crop labels -> labels are always shifted by self.pad
        self.padding_b = options.padding_b
        self.pad = options.patch_len / 2
        self.seed_method = options.seed_method
        self.bs = options.batch_size
        self.batch_data_provider = data_provider.PolygonDataProvider(options)

        self.batch_shape = self.batch_data_provider.get_batch_shape()
        self.image_shape = self.batch_data_provider.get_image_shape()
        self.label_shape = self.batch_data_provider.get_label_shape()
        print "image_shape",self.image_shape
        print "label_shape",self.label_shape
        self.global_input_batch = np.zeros(self.batch_shape,
                                           dtype=np.float32)
        self.global_label_batch = np.zeros(self.label_shape,
                                           dtype=np.int)
        self.global_height_gt_batch = np.zeros(self.label_shape,
                                           dtype=np.float32)

        # length of field, global_batch # includes padding)
        self.pl = options.patch_len

        # private
        self.n_channels = options.network_channels

        self.lowercomplete_e = options.lowercomplete_e 
        self.max_penalty_pixel = options.max_penalty_pixel

        self.global_claims = np.empty(self.image_shape)
        self.global_heightmap_batch = np.empty(self.label_shape)
        self.global_seed_ids = None
        self.global_seeds = None  # !!ALL!! coords include padding
        self.priority_queue = None
        self.coordinate_offset = np.array([[-1,0],[0,-1],[1,0],[0,1]],dtype=np.int)
        self.direction_array = np.arange(4)
        self.error_indicator_pass = np.zeros((self.bs))
        self.global_time = 0

        self.timo_min_len = 5
        self.timo_sigma = 0.3
        # debug
        self.max_batch = 0
        self.counter = 0

    def get_seed_ids(self):
        assert (self.global_seeds is not None)  # call get seeds first
        self.global_seed_ids = \
            [np.arange(start=1, stop=len(s)+1) for s in self.global_seeds]

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
                q.put((0., 0., seed[0], seed[1], Id, -1, False, 0))
            self.priority_queue.append(q)

    def walk_cross_coords(self, center):
        # walk in coord system of global label batch: x : 0 -> global_el - pl
        # use center if out of bounds
        coords_x, coords_y, directions = self.get_cross_coords(center)
        for x, y, d in zip(coords_x, coords_y, directions):
            yield x, y, d

    def get_cross_coords(self, center):
        coords = self.coordinate_offset + center
        np.clip(coords[:,0],self.pad,\
            self.label_shape[1] + self.pad - 1, out=coords[:,0])
        np.clip(coords[:,1],self.pad,\
            self.label_shape[2] + self.pad - 1, out=coords[:,1])
        return coords[:,0], coords[:,1], self.direction_array

    def get_cross_coords_offset(self, center):
        coords = self.coordinate_offset + center - self.pad
        np.clip(coords[:,0],0,\
            self.label_shape[1] - 1, out=coords[:,0])
        np.clip(coords[:,1],0,\
            self.label_shape[2] - 1, out=coords[:,1])
        return coords[:,0], coords[:,1], self.direction_array

    def crop_input(self, seed, b):
        return self.global_input_batch[b, :,
                                     seed[0] - self.pad:seed[0] + self.pad,
                                     seed[1] - self.pad:seed[1] + self.pad]

    def crop_mask_claimed(self, seed, b, Id):
        labels = self.global_claims[b,
                 seed[0] - self.pad:seed[0] + self.pad,
                 seed[1] - self.pad:seed[1] + self.pad]
        claimed = np.zeros((2, self.pl, self.pl), dtype='float32')
        claimed[0, :, :][(labels != Id) & (labels != 0)] = 1  # the others
        claimed[0, :, :][labels == -1] = 0  # the others
        claimed[1, :, :][labels == Id] = 1  # me
        return claimed

    def crop_height_map(self, seed, b):
        height = self.global_heightmap_batch[b,
              seed[0] - self.pad:seed[0] + self.pad,
              seed[1] - self.pad:seed[1] + self.pad]
        return height

    def prepare_global_batch(self):
        return self.batch_data_provider.prepare_input_batch(\
                                            self.global_input_batch)

    def init_batch(self, start=None, allowed_slices = None):

        # remember where territory has been claimed before. !=0 claimed, 0 free
        self.global_claims.fill(-1.)
        self.global_claims[:, self.pad:-self.pad, self.pad:-self.pad] = 0
        self.global_heightmap_batch.fill(np.inf)

        self.prepare_global_batch()
        self.get_seed_coords()
        self.get_seed_ids()
        self.initialize_priority_queue()

    def get_seed_coords_timo(self, sigma=1.0, min_dist=4, thresh=0.2):
        """
        Seeds by minima of dist trf of thresh of memb prob
        :return:
        """
        self.global_seeds = []
        for b in range(self.bs):
            x, y = wsDtseeds(
                self.global_input_batch[b, 0,
                    self.pad:-self.pad, self.pad:-self.pad],
                    thresh, self.timo_min_len, self.timo_sigma, groupSeeds=True)
            seeds = \
                [[x_i + self.pad, y_i + self.pad] for x_i, y_i in zip(x, y)]
            self.global_seeds.append(seeds)

    def get_seed_coords_grid(self, gridsize = 7):
        """
        Seeds by grid
        :return:
        """
        self.global_seeds = []
        shape = self.label_shape[1:3]
        print shape
        offset_x = ((shape[0]) % gridsize) /2
        offset_y = ((shape[1]) % gridsize) /2
        for b in range(self.bs):
            seeds_b = [(x+self.pad,y+self.pad) for x,y in \
                        product(xrange(offset_x,shape[0],gridsize),
                        xrange(offset_y,shape[1],gridsize))]
            self.global_seeds.append(seeds_b)

    def get_seed_coords_gt(self):
        self.global_seeds = []
        seed_ids = []
        dist_trf = np.zeros_like(self.global_label_batch)
        for b in range(self.bs):
            seed_ids.append(np.unique(
                self.global_label_batch[b, :, :]).astype(int))

            # add border to labels
            padded_label = data_provider.pad_cube(
                                    self.global_label_batch[b, :, :],
                                    1,
                                    value=seed_ids[-1]+1)

            dist_trf[b, :, :] = \
                data_provider.segmenation_to_membrane_core(
                    padded_label)[1][1:-1,1:-1]

        for b, ids in zip(range(self.bs),
                          seed_ids):  # iterates over batches
            seeds = []
            for Id in ids:  # ids within each slice
                regions = np.where(
                    self.global_label_batch[b, :, :] == Id)
                seed_ind = np.argmax(dist_trf[b][regions])
                seed = np.array([regions[0][seed_ind],
                                 regions[1][seed_ind]]) + self.pad
                seeds.append([seed[0], seed[1]])
            self.global_seeds.append(seeds)

    def get_centers_from_queue(self):
        heights = []
        centers = []
        ids = []
        for b in range(self.bs):
            height, _, center_x, center_y, Id, direction, error_indicator, \
                            time_put = self.get_center_i_from_queue(b)
            centers.append((center_x, center_y))
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
        out[0:2] = self.crop_mask_claimed(center, b, Id)
        out[2:] = self.crop_input(center, b)


    def get_batches(self):
        centers, ids, heights = self.get_centers_from_queue()
        # TODO: use input batch
        raw_batch = np.zeros((self.bs, self.n_channels, self.pl, self.pl),
                             dtype='float32')
        for b, (center, height, Id) in enumerate(zip(centers, heights, ids)):
            assert (self.global_claims[b, center[0], center[1]] == 0)
            self.global_claims[b, center[0], center[1]] = Id
            self.get_network_input(center, b, Id, raw_batch[b, :, :, :])
            # check whether already pulled
            self.global_heightmap_batch[b,
                                        center[0] - self.pad,
                                        center[1] - self.pad] = height
        return raw_batch, centers, ids

    def update_priority_queue(self, heights_batch, centers, ids):
        for b, center, Id, heights in zip(range(self.bs), centers, ids,
                                          heights_batch[:, :, 0, 0]):
            # if possibly wrong
            cross_x, cross_y, cross_d = self.get_cross_coords(center)
            lower_bound = self.global_heightmap_batch[b,
                                                      center[0] - self.pad,
                                                      center[1] - self.pad] + \
                                                      self.lowercomplete_e
            if lower_bound == np.inf:
                print "encountered inf for prediction center !!!!", \
                    b, center, Id, heights, lower_bound
                raise Exception('encountered inf for prediction center')
            self.max_new_old_pq_update(b, cross_x, cross_y, heights, lower_bound,
                                       Id, cross_d,
                                       input_time=self.global_time)

    def max_new_old_pq_update(self, b, x, y, height, lower_bound, Id,
                               direction, input_time=0, add_all=False):
        # check if there is no other lower prediction
        is_lowest = \
            ((height < self.global_heightmap_batch[b,
                                                   x - self.pad,
                                                   y - self.pad]) | add_all )\
                    & (self.global_claims[b, x, y] == 0)
        height[height < lower_bound] = lower_bound
        self.global_heightmap_batch[b, x  - self.pad, y - self.pad][is_lowest] \
            = height[is_lowest]
        self.global_prediction_map[b, x - self.pad, y - self.pad, direction] = \
            height
        for cx, cy, cd, hj, il in zip(x, y, direction, height, is_lowest):
            if il:
                self.priority_queue[b].put((hj, np.random.random(), cx, cy,
                                            Id, cd,
                                            self.error_indicator_pass[b],
                                            input_time))

    def get_num_free_voxel(self):
        return np.sum(self.global_claims[0] == 0)

    def draw_debug_image(self, image_name,
                         path='./../data/debug/images/',
                         save=True, b=0, inherite_code=False):
        plot_images = []
        # TODO: loop over input
        for channel in range(self.batch_shape[1]):
            plot_images.append({"title": "Input %d" % channel,
                                'im': self.global_input_batch[b, channel,
                                     self.pad:-self.pad, self.pad:-self.pad],
                                'interpolation': 'none'})

        plot_images.append({"title": "Claims",
                            'cmap': "rand",
                            'im': self.global_claims[b, self.pad:-self.pad,
                                  self.pad:-self.pad],
                            'interpolation': 'none'})
        plot_images.append({"title": "Heightmap Prediciton",
                            'im': self.global_heightmap_batch[b, :, :],
                            'interpolation': 'none'})
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

        if isinstance(options.label_path, str):
            self.labels = data_provider.load_h5(options.label_path)[0]
        else:
            self.labels = options.label
            if self.slices is not None:
                self.labels = self.labels[self.slices]

        if "height_gt_path" in options:
            self.height_gt = data_provider.load_h5(options.height_gt_path)[0]
        else:
            self.height_gt = options.height_gt
            if self.slices is not None:
                self.height_gt = self.height_gt[self.slices]

        if self.height_gt is not None:
            if options.clip_method=='clip':
                np.clip(self.height_gt, 0, options.patch_len / 2, out=self.height_gt)
            elif isinstance(options.clip_method, basestring) and \
                                len(options.clip_method) > 3 and \
                                options.clip_method[:3] == 'exp':
                dist = float(options.clip_method[3:])
                self.height_gt = \
                    np.exp(np.square(self.height_gt) / (-2) / dist ** 2)
            maximum = np.max(self.height_gt)
            self.height_gt *= -1.
            self.height_gt += maximum
            if options.scale_height_factor is not None:
                self.height_gt *= options.scale_height_factor
                self.scaling = options.scale_height_factor
            else:
                self.scaling = 1.

        if not self.padding_b:
            # crop label
            self.labels = self.labels[:, self.pad:-self.pad, self.pad:-self.pad]
            self.height_gt = self.height_gt[:,
                                            self.pad:-self.pad,
                                            self.pad:-self.pad]

        # private
        self.add_height_b = False
        # All no padding
        self.global_directionmap_batch = np.zeros(self.image_shape,
                                                  dtype=np.int)
        self.global_timemap = np.empty(self.image_shape, dtype=np.int)
        self.global_errormap = np.zeros(self.image_shape, dtype=np.int)

        self.global_error_dict = None
        self.crossing_errors = None
        self.find_errors_b = options.fine_tune_b and not options.rs_ft
        self.error_indicator_pass = None

    def get_seed_ids(self):
        super(HoneyBatcherPath, self).get_seed_ids()
        self.global_id2gt = []
        for b, (ids, seeds) in enumerate(zip(self.global_seed_ids,
                                             self.global_seeds)):
            id2gt = {}
            for Id, seed in zip(ids, seeds):
                id2gt[Id] = self.global_label_batch[b,
                                                    seed[0] - self.pad,
                                                    seed[1] - self.pad]
            self.global_id2gt.append(id2gt)

    def crop_timemap(self, center, b):
        return self.global_timemap[b, center[0]-self.pad:center[0]+self.pad,
                                   center[1]-self.pad:center[1]+self.pad]

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
        self.batch_data_provider.prepare_label_batch(self.global_label_batch,
                                                     self.global_height_gt_batch,
                                                     rois)

    def init_batch(self, start=None, allowed_slices = None):
        super(HoneyBatcherPath, self).init_batch(start=start,
                                                 allowed_slices=allowed_slices)
        # load new global batch data
        self.global_timemap.fill(np.inf)
        self.global_time = 0
        self.global_errormap = np.zeros((self.bs, 3,
                                         self.label_shape[1],
                                         self.label_shape[2]),
                                        dtype=np.bool)
        self.global_prediction_map = np.empty((self.bs,
                                               self.label_shape[1],
                                               self.label_shape[2], 4))
        self.global_prediction_map.fill(np.inf)

        self.global_error_dict = {}
        self.global_directionmap_batch = \
            np.zeros_like(self.global_label_batch) - 1

    def get_seed_coords(self, sigma=1.0, min_dist=4, thresh=0.2):
        """
        Seeds by minima of dist trf of thresh of memb prob
        :return:
        """
        if self.seed_method == "gt":
            self.get_seed_coords_gt()
        elif self.seed_method == "over":
            self.get_seed_coords_grid()
        elif self.seed_method == "timo":
            self.get_seed_coords_timo()
        elif self.seed_method == "file":
            self.batch_data_provider.get_seed_coords_from_file(self.global_seeds)
        else:
            raise Exception("no valid seeding method defined")


    def get_batches(self):
        raw_batch, centers, ids = super(HoneyBatcherPath, self).get_batches()
        gts = np.zeros((self.bs, 4, 1, 1), dtype='float32')
        for b in range(self.bs):
            if self.add_height_b:
                gts[b, :, 0, 0] = self.get_adjacent_heights(centers[b], b,
                                                            ids[b])
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
        ground_truth = \
            self.global_height_gt_batch[batch, seeds_x, seeds_y].flatten()
        # increase height relative to label (go up even after boundary crossing)
        if Id is not None:      #  == if self.add_height
            mask = [self.global_label_batch[batch,
                                           seeds_x,
                                           seeds_y] != \
                    self.global_id2gt[batch][Id]]
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

        height, _, center_x, center_y, Id, direction, error_indicator, \
            time_put = super(HoneyBatcherPath, self).get_center_i_from_queue(b)

        self.global_directionmap_batch[b,
                                       center_x - self.pad,
                                       center_y - self.pad] = direction
        self.global_timemap[b, center_x, center_y] = self.global_time

        # pass on if type I error already occured
        if error_indicator > 0:
            # went back into own territory --> reset error counter
            if self.global_id2gt[b][Id] == \
                self.global_label_batch[b,
                                        center_x - self.pad,
                                        center_y - self.pad]:
                self.error_indicator_pass[b] = 0.
            else:   # remember to pass on
                self.error_indicator_pass[b] = \
                    min(error_indicator + self.scaling,
                        (self.pad + self.max_penalty_pixel) *
                        self.scaling +       # don't go to high
                        np.random.randint(self.scaling) / 5.)
            self.global_errormap[b, 1,
                                 center_x - self.pad,
                                 center_y - self.pad] = 1
        # check for type I errors
        elif self.global_id2gt[b][Id] != \
                self.global_label_batch[b, center_x - self.pad,
                                        center_y - self.pad]:
            self.global_errormap[b, :2, center_x - self.pad,
                                 center_y - self.pad] = 1
            self.error_indicator_pass[b] = self.scaling * (self.pad + 1)

        # check for errors in neighbor regions, type II
        # TODO: remove find_errors_b
        if self.find_errors_b:
            self.check_type_II_errors(center_x, center_y, Id, b)
        # print 'b', b, 'height', height, 'centerxy', center_x, center_y, 'Id', Id, \
        #     direction, error_indicator, time_put
        return height, _, center_x, center_y, Id, direction, error_indicator, \
                    time_put

    def update_position(self, pos, direction):
        """
        update position by following the minimal spanning tree backwards
        for this reason: subtract direction for direction offset
        """
        offsets = self.coordinate_offset[int(direction)]
        new_pos = [pos[0] - offsets[0], pos[1] - offsets[1]]
        return new_pos

    def get_path_to_root(self, start_position, batch):

        current_position = start_position
        current_direction = \
            self.global_directionmap_batch[batch,
                                           current_position[0]-self.pad,
                                           current_position[1]-self.pad]
        yield start_position, current_direction
        while current_direction != -1:
            current_position = self.update_position(current_position,
                                                    current_direction)
            current_direction = \
                self.global_directionmap_batch[batch,
                                               current_position[0]-self.pad,
                                               current_position[1]-self.pad]
            yield current_position, current_direction

    def locate_global_error_path_intersections(self):
        for b in range(self.bs):
            # plot_images = []

            # project claim id to ground truth id by lookup
            gtmap = np.array([0]+self.global_id2gt[b].values())
            claim_projection=gtmap[self.global_claims[b].astype(int)]
            claim_projection[self.pad-1,:]=claim_projection[self.pad,:]
            claim_projection[-self.pad,:]=claim_projection[-self.pad-1,:]
            claim_projection[:,self.pad-1]=claim_projection[:,self.pad]
            claim_projection[:,-self.pad]=claim_projection[:,-self.pad-1]
            not_found = np.zeros_like(claim_projection)
            #
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
            #                         'cmap': "grey",
            #                         'im': self.global_errormap[b, 1]})

            # find where path crosses region
            gx = convolve(claim_projection + 1, np.array([-1., 0., 1.]).reshape(1, 3))
            gy = convolve(claim_projection + 1, np.array([-1., 0., 1.]).reshape(3, 1))
            boundary = np.float32((gx ** 2 + gy ** 2) > 0)
            # find all boundary crossings
            path_fin_map =  np.logical_and(boundary[self.pad:-self.pad,
                                                    self.pad:-self.pad],
                                           self.global_errormap[b, 1])


            # plot_images.append({"title": "path_fin_0",
            #             'cmap': "grey",
            #             'im': path_fin_map})
            np.logical_and(path_fin_map,
                           (self.global_claims[b,
                                               self.pad:-self.pad,
                                               self.pad:-self.pad] > 0),
                           out=path_fin_map)
            # plot_images.append({"title": "path_fin_1",
            #                         'cmap': "grey",
            #                         'im': path_fin_map})
            # plot_images.append({"title": "boundary",
            #                     'cmap': "grey",
            #                     'im': boundary[self.pad:-self.pad,
            #                                    self.pad:-self.pad]})
            # u.save_images(plot_images, path="./../data/debug/",
            #               name="path_test_"+str(b)+".png")
            # plot_images.append([])
            # plot_images.append([])
            wrong_path_ends = np.transpose(np.where(path_fin_map)) + self.pad
            for center_x, center_y in wrong_path_ends:
                # print "path for ", center_x, center_y
                def error_index(b, id1, id2)    :
                    return b, min(id1, id2), max(id1, id2)

                def get_error_dict(b, x, y, center_x, center_y,
                                   reverse_direction, slow_intruder,
                                   touch_x, touch_y):
                    new_error = \
                            {"batch": b,
                            # get time from center it was predicted from
                            "touch_time": self.global_timemap[b,
                                                              touch_x, touch_y],
                            "large_pos": [center_x, center_y],
                            "large_direction": direction,
                            "large_id": self.global_claims[b,
                                                           center_x,
                                                           center_y],
                            "large_gtid": claim_projection[center_x,
                                                           center_y],
                            "small_pos": [x, y],
                            "small_direction": reverse_direction,
                            "small_gtid": claim_projection[x, y],
                            "small_id": self.global_claims[b, x, y],
                            "slow_intruder": slow_intruder}
                    assert (new_error["large_gtid"] != new_error["small_gtid"])
                    assert (new_error["large_id"] != new_error["small_id"])
                    return new_error

                # check around intruder, claim pred = large pred = intruder
                claim_height = self.global_heightmap_batch[b,
                                                         center_x - self.pad,
                                                         center_y - self.pad]
                large_time = self.global_timemap[b, center_x, center_y]
                small_pred = np.inf
                small_height_old = np.inf
                touch_time_old = np.inf
                fast_intruder_found = False
                new_error = {}
                for x, y, direction in self.walk_cross_coords([center_x,
                                                               center_y]):
                    assert (x - self.pad >= 0)
                    assert (center_x - self.pad >= 0)
                    assert (y - self.pad >= 0)
                    assert (center_y - self.pad >= 0)

                    # only penalize on on lowest prediction
                    if claim_projection[x, y] != \
                            claim_projection[center_x, center_y]:
                        reverse_direction = (direction + 2) % 4
                        prediction = \
                            self.global_prediction_map[b,
                                                       center_x - self.pad,
                                                       center_y - self.pad,
                                                       reverse_direction]
                        small_time = self.global_timemap[b, x, y]
                        if prediction < claim_height:
                            raise Exception('this violates PQ structure')
                        # intruder runs into resident (fast intruder)
                        elif prediction < small_pred and \
                                small_time < large_time:
                            fast_intruder_found = True      # prioritized
                            small_pred = prediction
                            new_error = get_error_dict(b, x, y, center_x,
                                                      center_y,
                                                      reverse_direction,
                                                      False,
                                                      center_x, center_y)
                        # slow intruder
                        elif not fast_intruder_found:   # prioritized
                            small_height = \
                                self.global_heightmap_batch[b,
                                                            x - self.pad,
                                                            y - self.pad]
                            if small_height >= claim_height and \
                                    small_height <= small_height_old:
                                small_height_old = small_height
                                assert (large_time < small_time)
                                new_error = get_error_dict(b, x, y, center_x,
                                                           center_y,
                                                           reverse_direction,
                                                           True,
                                                           x, y)
                # self.global_errormap[b, 2, x-self.pad,y-self.pad] = -1
                if new_error != {}:
                    # print new_error
                    e_index = error_index(b, new_error["small_gtid"],
                                          new_error["large_gtid"])
                    # only one error per ID pair (the earliest one)
                    save_error = False
                    if not e_index in self.global_error_dict:
                        save_error = True
                    else:
                        if self.global_error_dict[e_index]["touch_time"] > \
                                new_error["touch_time"]:
                            save_error = True
                    if save_error:
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
                else:
                    print "no match found for path end"
                    not_found[center_x-self.pad, center_y-self.pad] = 1
                    for x, y, direction in self.walk_cross_coords([center_x,center_y]):
                        print  claim_projection[x, y] ," should not be ", \
                            claim_projection[center_x, center_y]
                        reverse_direction = (direction + 2) % 4
                        print "prediction = ", \
                            self.global_prediction_map[b,
                                                       x-self.pad,
                                                       y-self.pad,
                                                       reverse_direction]
                    raise Exception("no match found for path end")

    def find_global_error_paths(self):
        self.locate_global_error_path_intersections()
        # now errors have been found so start and end of paths shall be found
        self.find_type_I_error()
        self.find_source_of_II_error()

    # crossing from own gt ID into other ID
    def find_type_I_error(self):
        for error_I in self.global_error_dict.values():
            if "e1_pos" not in error_I:
                start_position = error_I["large_pos"]
                batch = error_I["batch"]
                # keep track of output direction
                current_direction = error_I["large_direction"]
                prev_in_other_region = \
                    self.global_errormap[batch, 1,
                                         start_position[0] - self.pad,
                                         start_position[1] - self.pad] 

                e1_length = 0
                for pos, d in self.get_path_to_root(start_position, batch):
                    # debug
                    # shortest path of error type II to root (1st crossing)
                    self.global_errormap[batch, 2,
                                         pos[0] - self.pad,
                                         pos[1] - self.pad] = True
                    # debug
                    # remember type I error on path
                    in_other_region = \
                        self.global_errormap[batch, 1,
                                             pos[0]-self.pad,
                                             pos[1]-self.pad]
                    #  detect transition from "others" region to "me" region
                    if prev_in_other_region and not in_other_region:
                        original_error = np.array(pos)
                        # print 'found crossing. type II linked to type I. Error #',\
                        #     self.counter
                        error_I["e1_pos"] = original_error
                        error_I["e1_time"] = self.global_timemap[batch,
                                                                 pos[0],
                                                                 pos[1]]
                        error_I["e1_direction"] = current_direction
                        error_I["e1_length"] = e1_length
                        assert(current_direction >= 0)
                    current_direction = d
                    prev_in_other_region = in_other_region
                    e1_length += 1

                e1_length = error_I["e1_length"]
                self.counter += 1

    def find_end_of_plateau(self, start_position, start_direction, batch):
        current_height = self.global_heightmap_batch[batch,
                                                     start_position[0]-self.pad,
                                                     start_position[1]-self.pad]
        current_height -= self.lowercomplete_e
        current_direction = start_direction
        for pos, d in self.get_path_to_root(start_position, batch):
            # check if the slope is smaller than  zero
            if self.global_heightmap_batch[batch, pos[0]-self.pad, \
                                           pos[1]-self.pad] \
                                    < current_height:
                return pos,current_direction
            if d >= 0:
                # not at the end of the path
                current_height -= self.lowercomplete_e
                current_direction = d

        return pos, start_direction


    def find_source_of_II_error(self):
        for error in self.global_error_dict.values():
            if "e2_pos" not in error:
                batch = error["batch"]
                start_position = error["small_pos"]
                start_direction = error["small_direction"]
                if error["slow_intruder"]:
                    start_direction = \
                        self.global_directionmap_batch[
                            batch,
                            start_position[0] - self.pad,
                            start_position[1] - self.pad]
                    start_position = self.update_position(start_position,
                                                          start_direction)

                    error["e2_pos"], error["e2_direction"] = \
                                    self.find_end_of_plateau(start_position,
                                    start_direction,
                                    batch)
                else:
                    error["e2_pos"], error["e2_direction"] = \
                        start_position,  start_direction
                assert (error["e2_direction"] >= 0)
                error["e2_time"] = self.global_timemap[batch,
                                                       error["e2_pos"][0],
                                                       error["e2_pos"][1]]

    def check_type_II_errors(self, center_x, center_y, Id, b):
        def error_index(b, id1, id2):
            return b, min(id1, id2), max(id1, id2)
        for x, y, direction in self.walk_cross_coords([center_x,
                                                       center_y]):
            c = int(self.global_claims[b, x, y])  # neighbor label
            if c > 0:
                claimId = int(self.global_id2gt[b][c])
                gtId = int(self.global_id2gt[b][Id])
                if not error_index(b, gtId, claimId) \
                    in self.global_error_dict:
                    if claimId > 0 and claimId != gtId:  # neighbor claimed
                        center_intruder_b = \
                            self.global_errormap[b, 1, center_x - self.pad,
                                                 center_y - self.pad]
                        neighbor_intruder_b = \
                            self.global_errormap[b, 1, x - self.pad,
                                                 y - self.pad]
                        if center_intruder_b and not neighbor_intruder_b:
                            # print "fast intrusion"
                            self.current_type = 'fastI'
                            self.global_error_dict[error_index(b, gtId, claimId)] = \
                                {"batch": b,
                                 "touch_time": self.global_timemap[b, x, y],
                                 "large_pos": [center_x, center_y],
                                 "large_direction": direction,
                                 "large_id": Id,
                                 "large_gtid": gtId,
                                 "small_pos": [x, y],
                                 "small_direction": (direction + 2) % 4,
                                 "small_gtid": claimId,
                                 "small_id": c}
                            self.find_type_I_error()
                            self.find_source_of_II_error()
                        elif not center_intruder_b and neighbor_intruder_b:
                            # print "slow intrusion"
                            self.current_type = 'slowI'
                            self.global_error_dict[error_index(b, gtId, claimId)] = \
                                {"batch": b,
                                 "touch_time": self.global_timemap[b, x, y],
                                 "large_pos": [x, y],
                                 # turns direction by 180 degrees
                                 "large_direction": (direction + 2) % 4,
                                 "large_id": c,
                                 "large_gtid": claimId,
                                 "small_direction": direction,
                                 "small_pos": [center_x, center_y],
                                 "small_gtid": gtId,
                                 "small_id": Id}
                            self.find_type_I_error()
                            self.find_source_of_II_error()
                        elif center_intruder_b and neighbor_intruder_b:
                            # raise Exception('error type 3 found')
                            # print 'type 3 error not yet implemented'
                            # self.find_type_I_error()
                            # self.find_source_of_II_error()
                            pass

    def reconstruct_input_at_timepoint(self, timepoint, centers, ids, batches):
        raw_batch = np.zeros((len(batches), self.n_channels, self.pl, self.pl),
                             dtype='float32')
        for i, (b, center, Id) in enumerate(zip(batches, centers, ids)):
            self.get_network_input(center, b, Id, raw_batch[i, :, :, :])

        mask = self.crop_time_mask(centers, timepoint, batches)
        raw_batch[:, 1, :, :][mask] = 0
        raw_batch[:, 2, :, :][mask] = 0

        return raw_batch

    def check_error(self, error):
        return not 'used' in error
        # return not 'used' in error and error['e1_length'] > 5


    def count_new_path_errors(self):
        return len([v for v in self.global_error_dict.values()
                    if self.check_error(v)])

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
            # print "errorlength",error['e1_length']
            if self.check_error(error):
                error["used"] = True
                error_batch_list.append(error["batch"])
                error_I_timelist.append(error["e1_time"])
                error_I_direction.append(error["e1_direction"])
                error_I_pos_list.append(error["e1_pos"])
                error_I_id_list.append(error["large_id"])
                error_II_pos_list.append(error["e2_pos"])
                error_II_direction.append(error["e2_direction"])
                error_II_time_list.append(error["e2_time"])
                error_II_id_list.append(error["small_id"])

        reconst_e1 = self.reconstruct_input_at_timepoint(error_I_timelist,
                                                         error_I_pos_list,
                                                         error_I_id_list,
                                                         error_batch_list)
        reconst_e2 = self.reconstruct_input_at_timepoint(error_II_time_list,
                                                         error_II_pos_list,
                                                         error_II_id_list,
                                                         error_batch_list)

        return reconst_e1, reconst_e2, np.array(error_I_direction), np.array(
            error_II_direction)

    def serialize_to_h5(self, h5_filename, path="./../data/debug/serial/"):
        if not exists(path):
            makedirs(path)
        with h5py.File(path+'/'+h5_filename, 'w') as out_h5:
            out_h5.create_dataset("global_timemap",
                        data=self.global_timemap ,compression="gzip")
            out_h5.create_dataset("global_errormap",
                        data=self.global_errormap ,compression="gzip")
            out_h5.create_dataset("global_claims",
                        data=self.global_claims ,compression="gzip")
            out_h5.create_dataset("global_input",
                        data=self.global_input_batch ,compression="gzip")
            out_h5.create_dataset("global_heightmap_batch",
                        data=self.global_heightmap_batch ,compression="gzip")
            out_h5.create_dataset("global_height_gt_batch",
                        data=self.global_height_gt_batch ,compression="gzip")
            out_h5.create_dataset("global_prediction_map",
                        data=self.global_prediction_map ,compression="gzip")
            out_h5.create_dataset("global_directionmap_batch",
                        data=self.global_directionmap_batch ,compression="gzip")

            for error_name,error in self.global_error_dict.items():
                for n,info in error.items():
                    out_h5.create_dataset("error/"+str(error_name[0])+"_"+str(error_name[1])+"_"+str(error_name[2])+"/"+n,data=np.array(info))

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

            val_score = vs.validate_segmentation(pred_path=path+'/'+name+'pred.h5',
                                            gt_path=path+'/'+name+'_gt.h5')
            print val_score

            import json
            with open(path+'/'+name+'_score.json', 'w') as f:
                f.write(json.dumps(val_score))

    def draw_batch(self, raw_batch, image_name,
                   path='./../data/debug/',
                   save=True, gt=None, probs=None):
        plot_images = []
        for b in range(raw_batch.shape[0]):
            plot_images.append({"title": "membrane",
                                'im': raw_batch[b, 0],
                                'interpolation': 'none'})
            plot_images.append({"title": "raw",
                                'im': raw_batch[b, 1],
                                'interpolation': 'none'})
            plot_images.append({"title": "claim others",
                                'cmap': "rand",
                                'im': raw_batch[b, 2],
                                'interpolation': 'none'})
            plot_images.append({"title": "claim me",
                                'cmap': "rand",
                                'im': raw_batch[b, 3],
                                'interpolation': 'none'})
        u.save_images(plot_images, path=path, name=image_name, column_size=4)

    def draw_error_reconst(self, image_name, path='./../data/debug/',
                           save=True):
        for e_idx, error in self.global_error_dict.items():
            plot_images = []
            if not "draw_file" in error:
                reconst_e1 = self.reconstruct_input_at_timepoint([error["e1_time"]],
                                                                 [error["e1_pos"]],
                                                                 [error[
                                                                      "large_id"]],
                                                                 [error["batch"]])
                reconst_e2 = self.reconstruct_input_at_timepoint([error["e2_time"]],
                                                                 [error["e2_pos"]],
                                                                 [error[
                                                                      "small_id"]],
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

    def draw_debug_image(self, image_name, path='./../data/debug/',
                         save=True, b=0, inheritance=False):
        plot_images = super(HoneyBatcherPath, self).\
            draw_debug_image(image_name=image_name,
                             path=path,
                             save=False,
                             b=b,
                             inherite_code=True)

        plot_images.insert(2,{"title": "Error Map",
                            'im': self.global_errormap[b, 0, :, :],
                             'interpolation': 'none'})

        plot_images.insert(3,{"title": "Ground Truth Label",
                            'scatter': np.array(
                                [np.array(e["e1_pos"]) - self.pad for e in
                                 self.global_error_dict.values() if
                                 "e1_pos" in e and e["batch"] == 4]),
                            "cmap": "rand",
                            'im': self.global_label_batch[b, :, :],
                            'interpolation': 'none'})

        plot_images.insert(5,{"title": "Overflow Map",
                            'im': self.global_errormap[b, 1, :, :],
                            'interpolation': 'none'})
        
        plot_images.insert(6,{"title": "Heightmap GT",
                            'im': self.global_height_gt_batch[b, :, :],
                            'scatter': np.array(self.global_seeds[b]) - self.pad,
                            'interpolation': 'none'})

        plot_images.insert(8,{"title": "Height Differences",
                            'im': self.global_heightmap_batch[b, :, :] -
                                  self.global_height_gt_batch[b, :, :],
                            'interpolation': 'none'})

        plot_images.insert(9,{"title": "Direction Map",
                            'im': self.global_directionmap_batch[b, :, :],
                            'interpolation': 'none'})

        plot_images.insert(10,{"title": "Path Map",
                            'scatter': np.array(
                                [np.array(e["large_pos"]) - self.pad for e in
                                 self.global_error_dict.values() if
                                 e["batch"] == b]),
                            'im': self.global_errormap[b, 2, :, :],
                            'interpolation': 'none'})

        timemap = np.array(self.global_timemap[b, self.pad:-self.pad,
                                               self.pad:-self.pad])
        timemap[timemap < 0] = 0
        plot_images.insert(11,{"title": "Time Map ",
                                'im': timemap})

        if save:
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
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #



