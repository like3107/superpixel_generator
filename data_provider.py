import h5py as h
import numpy as np
import random
from os import makedirs
import utils as u
from os.path import exists
import cairo
import math
import sys
from scipy import ndimage
from scipy import stats
from scipy.ndimage import convolve, gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion
from skimage.feature import peak_local_max
from skimage.morphology import label, watershed
from scipy.spatial import Voronoi as voronoi
from voronoi_polygon import voronoi_finite_polygons_2d
# import png
from trainer_config_parser import get_options
import ws_timo_gtseeds

def segmenation_to_membrane_core(label_image):
    gx = convolve(label_image, np.array([-1., 0., 1.]).reshape(1, 3))
    gy = convolve(label_image, np.array([-1., 0., 1.]).reshape(3, 1))
    boundary= np.float32((gx ** 2 + gy ** 2) > 0)
    height = distance_transform_edt(boundary == 0)
    return boundary, height.astype(np.float32)


def generate_gt_height(label_image, max_height, clip_method='clip'):
    _, height = segmenation_to_membrane_core(label_image)
    if clip_method=='clip':
        np.clip(height, 0, max_height, out=height)
        maximum = np.max(height)
        height *= -1.
        height += maximum 
    elif clip_method=='exp':
        np.square(height, out=height)
        height /= (-2 * (max_height/10) ** 2)
        np.exp(height, out=height)
    return height


def get_dataset_provider(datasetname):
    print 'using dataset: ', datasetname
    return getattr(sys.modules[__name__], datasetname+"DataProvider")


class DataProvider(object):
    def __init__(self, options):
        self.options = options
        self.bs = options.batch_size
        self.set_slices(options)
        self.pad = options.patch_len /2
        self.pl = options.patch_len
        self.load_data(options)
        self.n_slices = range(self.full_input.shape[0])
        self.options.global_input_len = self.options.global_edge_len

        if options.padding_b:
            self.full_input = mirror_cube(self.full_input, self.pad)
            self.options.global_input_len += self.options.patch_len

        self.rl_x = self.full_input.shape[2]
        self.rl_y = self.full_input.shape[3]
        print "patch_len, global_edge_len, self.rlxy", options.patch_len, \
            options.global_edge_len, self.rl_x, self.rl_y

        # assert (patch_len <= global_edge_len)
        assert (options.global_input_len <= self.rl_x)
         # (options.global_input_len <=  self.rl_y)

    def set_slices(self, o):
        print o
        if "slices" in o:
            self.slices = o.slices
        else:
            self.slices = None
            print "no slices selected"

    def get_batch_shape(self):
        data_shape = list(self.full_input.shape)
        data_shape[0] = self.bs
        if self.options.global_edge_len > 0:
            data_shape[2] = self.options.global_input_len
            data_shape[3] = self.options.global_input_len

        return data_shape

    def get_image_shape(self):
        data_shape = list(self.get_batch_shape())
        del data_shape[1]
        return data_shape

    def get_label_shape(self):
        data_shape =  list(self.label.shape)
        data_shape[0] = self.bs

        if self.options.global_edge_len > 0:
            data_shape[1] = self.options.global_edge_len
            data_shape[2] = self.options.global_edge_len

        if not self.options.padding_b:
            data_shape[1] -= self.options.patch_len - 1
            data_shape[2] -= self.options.patch_len - 1


        return data_shape

    def prepare_input_batch(self, input, preselect_batches=None):
        # initialize two global batches = region where CNNs compete
        # against each other
        # get indices for global batches in raw/ label cubes
        if preselect_batches is not None:
            print self.bs, preselect_batches
            assert(self.bs == len(preselect_batches))
            ind_b = preselect_batches
        elif self.options.quick_eval:
            print "using fixed batches"
            ind_b = np.arange(self.bs)
            print ind_b
        else:
            ind_b = np.random.permutation(self.n_slices)[:self.bs]

        # indices to raw, correct for label which edge len is -self.pl shorter
        if self.options.global_edge_len > 0:
            if self.options.quick_eval:
                print 'using fixed indices '
                ind_x = np.empty(self.bs, dtype=int)
                ind_x.fill(int(0))
                ind_y = np.empty(self.bs, dtype=int)
                ind_y.fill(int(0))
            else:
                ind_x = np.random.randint(0,
                                          self.rl_x - self.options.global_input_len + 1,
                                          size=self.bs)
                ind_y = np.random.randint(0,
                                          self.rl_y - self.options.global_input_len + 1,
                                          size=self.bs)
            for b in range(self.bs):
                input[b, :, :, :] = \
                    self.full_input[ind_b[b], :,
                             ind_x[b]:ind_x[b] + self.options.global_input_len,
                             ind_y[b]:ind_y[b] + self.options.global_input_len]
            return [ind_b, ind_x, ind_y]
        else:
            print input.shape, self.full_input.shape
            input[range(self.bs)] = self.full_input[ind_b]
            return [ind_b, None, None]

    def prepare_label_batch(self, label, height, rois):
        if self.options.global_edge_len > 0:
            ind_b, ind_x, ind_y = rois
            if not self.options.padding_b:
                ind_x += self.options.patch_len / 2
                ind_y += self.options.patch_len / 2
            for b in range(self.bs):
                label_inp_len = self.options.global_input_len - self.options.patch_len + 1
                height[b, :, :] = self.height_gt[ind_b[b],
                                                 ind_x[b]:ind_x[b] + label_inp_len,
                                                 ind_y[b]:ind_y[b] + label_inp_len]
                label[b, :, :] = self.label[ind_b[b],
                                            ind_x[b]:ind_x[b] + label_inp_len,
                                            ind_y[b]:ind_y[b] + label_inp_len]
        else:
            for b in range(self.bs):
                if self.options.padding_b:
                    label[b] = self.label[b,:,:]
                    height[b] = self.height_gt[b,:,:]
                else:
                    label[b] = self.label[b, self.pad:-self.pad,
                                             self.pad:-self.pad]
                    height[b] = self.height_gt[b,self.pad:-self.pad,
                                                 self.pad:-self.pad]

    def get_seed_coords_from_file(global_seeds):
        # clear global_seeds but keep empty list reference
        seeds = load_h5(str(self.options.seed_file_path))
        del global_seeds[:]
        for b in range(self.bs):
            self.global_seeds.append(seeds[b]+self.pad)

    def load_data(self, options):
        # print self.options.input_data_path
        # self.full_input = load_h5(self.options.input_data_path,
        self.full_input = load_h5(str(self.options.input_data_path),
                                    h5_key=None,
                                    slices=self.slices)[0]
        self.height_gt = load_h5(self.options.height_gt_path,
                                    h5_key=None,
                                    slices=self.slices)[0]
        self.label = load_h5(self.options.label_path,
                                    h5_key=None,
                                    slices=self.slices)[0]


class CremiDataProvider(DataProvider):
    def load_data(self, options):
        self.full_input = load_h5(str(self.options.input_data_path),
                                h5_key=None,
                                slices=self.slices)[0]
        self.height_gt = load_h5(self.options.height_gt_path,
                                    h5_key=None,
                                    slices=self.slices)[0]
        max_height = self.pad
        # generate height with clipping method from distance transform
        if self.options.clip_method=='clip':
            np.clip(self.height_gt, 0, max_height, out=self.height_gt)
            maximum = np.max(self.height_gt)
            self.height_gt *= -1.
            self.height_gt += maximum
        elif self.options.clip_method=='exp':
            np.square(self.height_gt, out=self.height_gt)
            self.height_gt /= (-2 * (max_height/10) ** 2)
            np.exp(self.height_gt, out=self.height_gt)

        self.label = load_h5(self.options.label_path,
                                    h5_key=None,
                                    slices=self.slices)[0]

    def find_timo_errors(self, label_batch, input_batch, seeds):
        alpha = 0.9
        # timo parameters 
        threshold_dist_trf = 0.3
        thres_memb_cc = 15
        thresh_seg_cc = 0
        sigma_dist_trf = 2
        somethingunimportant = 0
        groupSeeds = False
        segmentation = np.zeros_like(label_batch)

        for b in range(self.bs):
            seed_mat = np.zeros(label_batch[b, :, :].shape).astype(np.bool)
            seed_plus_pad = np.array(seeds[b])
            seed = seed_plus_pad - self.pad
            # print 'init', seed[:, 0], seed[:, 1]
            seed_mat[seed[:, 0], seed[:, 1]] = 1.
            segmentation[b, :, :], _ = \
            ws_timo_gtseeds.wsDtSegmentation(alpha*input_batch[b, 1, self.pad:-self.pad, self.pad:-self.pad]\
                                    +(1-alpha)*input_batch[b, 0, self.pad:-self.pad, self.pad:-self.pad],
                                  threshold_dist_trf, thres_memb_cc,
                                  thresh_seg_cc, sigma_dist_trf,
                                  somethingunimportant,
                                  seed_mat=seed_mat,
                                  groupSeeds=groupSeeds)

            # note: this id translation requires that 
            # self.global_claims[b, seed_plus_pad[:, 0], seed_plus_pad[:, 1]]
            # is a range(1, #seeds_per_batch b)
            # requires seed_method == "gt"
            assert(self.options.seed_method == "gt")
            # map to gt id
            gt_id_map = np.zeros((len(seed)+1))
            for i, s in enumerate(segmentation[b, seed[:, 0], seed[:, 1]]):
                gt_id_map[s] = i+1
            segmentation[b] = gt_id_map[segmentation[b]]
        return segmentation != label_batch

class PolygonDataProvider(DataProvider):
    def __init__(self, options):
        self.size = (options.global_edge_len if options.global_edge_len != 0 else 500)
        self.linewidth = 3
        self.exception_counter = 0
        super(PolygonDataProvider, self).__init__(options)

    def get_dashes(self):
        # return np.random.randint(10, 60, size=40)
        # 1st dash length 2nd hole

        return [self.options.dash_len, self.options.hole_length]

    def prepare_input_batch(self, input):
        # load_data creates a new batch
        self.exception_counter = 0
        self.load_data(self.options)
        # TODO: check if this needs to be called every time
        if self.options.padding_b:
            self.full_input = mirror_cube(self.full_input, self.pad)
        return super(PolygonDataProvider, self).prepare_input_batch(input)

    def draw_circle(self):
        data = np.zeros((self.bs, self.size, self.size, 4), dtype=np.uint8)
        for b in range(self.bs):
            surface = cairo.ImageSurface.create_for_data(
                    data[b], cairo.FORMAT_ARGB32, self.size, self.size)
            cr = cairo.Context(surface)
            cr.set_source_rgb(1.0, 0, 0)
            cr.paint()
            cr.set_line_width (self.linewidth)
            cr.set_dash(self.get_dashes())
            cr.arc(self.size/2, self.size/2, self.size/4, 0, 2*math.pi) 
            cr.close_path()
            cr.set_source_rgb (0, 1., 0);
            cr.fill_preserve();
            cr.set_operator(cairo.OPERATOR_ADD)
            cr.set_source_rgb(0., 0., 1.0)
            cr.stroke();
            self.full_input = data[b,:,:,0]

        self.make_dataset(data)
        # with h.File("circle.h5","w") as out:
        #     out.create_dataset("test",data=data) 
        #     out.create_dataset("data",data=self.full_input.astype(np.float32)) 
        #     out.create_dataset("label",data=self.label.astype(np.float32)) 
        #     out.create_dataset("height",data=self.height_gt.astype(np.float32)) 

    def draw_voronoi(self, num_seeds = 5):
        
        self.label = np.zeros((self.bs,self.size, self.size), dtype=np.uint8)
        self.full_input = np.zeros((self.bs, 1, self.size, self.size),
                                   dtype=np.float32)
        
        b = 0
        while b < self.bs:
            seeds = np.random.randint(0, self.size, size=num_seeds*2).reshape((-1, 2))
            vor = voronoi(seeds)
            # print seeds, vor.vertices
            regions, vertices = voronoi_finite_polygons_2d(vor, radius=10000)

            data_border = np.zeros((self.size, self.size, 4), dtype=np.uint8)
            surface_border = cairo.ImageSurface.create_for_data(
                        data_border, cairo.FORMAT_ARGB32, self.size, self.size)
            cr_b = cairo.Context(surface_border)
            cr_b.set_source_rgb(1.0, 1.0, 1.0)
            if self.options.dashes_on_b:
                cr_b.set_dash(self.get_dashes())
            coord_pairs = []

            for i, region in enumerate(regions):
                data = np.zeros((self.size, self.size, 4), dtype=np.uint8)
                surface = cairo.ImageSurface.create_for_data(
                        data, cairo.FORMAT_ARGB32, self.size, self.size)
                cr = cairo.Context(surface)
                polygon = vertices[region]
                cr.move_to(polygon[-1][0],polygon[-1][1])
                cr_b.move_to(polygon[-1][0],polygon[-1][1])
                lcoord = (polygon[-1][0],polygon[-1][1])
                cr.set_source_rgb(1, 0, 0)

                for p,r in zip(polygon,region):
                    cr.line_to(p[0],p[1])
                    if (lcoord[0],lcoord[1],p[0],p[1]) in coord_pairs:
                        cr_b.move_to(p[0],p[1])
                    else:
                        cr_b.line_to(p[0],p[1])
                        coord_pairs.append((lcoord[0],lcoord[1],p[0],p[1]))
                        coord_pairs.append((p[0],p[1],lcoord[0],lcoord[1]))
                        if self.options.dashes_on_b:
                            cr_b.set_dash(self.get_dashes())
                    lcoord = (p[0],p[1])
                cr.close_path()
                cr_b.stroke()
                cr.fill_preserve()
                cr.new_sub_path()
                self.label[b][data[:,:,2] > 100] = i+1

            self.full_input[b,0,:,:] = data_border[:,:,0]

            # failsafe if some pixels are not claimed
            for x,y in np.transpose(np.where(self.label[b]==0)):
                if x < self.size/2:
                    self.label[b,x,y] = self.label[b,x+1,y]
                else:
                    self.label[b,x,y] = self.label[b,x-1,y]
            if np.all(self.label[b] > 0):
                b += 1
            else:
                print "generating new voronoi cells batch=",b,

        self.make_height_gt()
        # with h.File("voronoi.h5","w") as out:
        #     out.create_dataset("labels",data=self.label)
        #     out.create_dataset("full_input",data=self.full_input)
        #     out.create_dataset("height",data=self.height_gt)

    def draw_debug(self, num_seeds=5):

        self.label = np.zeros((self.bs, self.size, self.size), dtype=np.uint8)
        self.full_input = np.zeros((self.bs, 1, self.size, self.size),
                                   dtype=np.float32)

        b = 0
        self.full_input[:, 0, self.size /2 , :] = 1
        self.full_input[:, 0, self.size /2-6:self.size /2+6 , :] = 1
        self.label[:, self.size /2:, :] = 1
        self.make_height_gt()
        # with h.File("voronoi.h5","w") as out:
        #     out.create_dataset("labels",data=self.label)
        #     out.create_dataset("full_input",data=self.full_input)
        #     out.create_dataset("height",data=self.height_gt)


    def make_dataset(self, data, labels=[0., 1.]):
        self.full_input = data[:, np.newaxis,:,:,0].astype(np.float32)
        self.full_input /= 256.

        self.label = np.zeros_like(data[:, :,:,1],dtype=np.float32)
        thresholds = [(l1+l0)/2 for (l0,l1) in zip(sorted(labels),sorted(labels[1:]))]

        for b in range(self.bs):

            mask = self.label[b] >= thresholds[0]
            self.label[b][mask] = 0

            for i, l in enumerate(thresholds):
                mask = data[b, :,:,1] > l*256
                self.label[b][mask] = i+1

        self.make_height_gt()

    def make_height_gt(self):
        self.height_gt = np.empty_like(self.label, dtype=np.float32)

        for b in range(self.bs):
            self.height_gt[b] = generate_gt_height(self.label[b],
                                       self.options.patch_len / 2,
                                       clip_method=self.options.clip_method)


    def draw_polygon(self):
        data = np.zeros((self.bs, self.size, self.size, 4), dtype=np.uint8)
        surface = cairo.ImageSurface.create_for_data(
                    data, cairo.FORMAT_ARGB32, self.size, self.size)
        cr = cairo.Context(surface)
        cr.set_source_rgb(1.0, 0, 0)
        cr.paint()
        cr.set_line_width (self.linewidth)
        # cr.set_dash(self.get_dashes()); 
        xm, ym = self.size, self.size
        cr.move_to (xm/2., ym/10.)
        cr.line_to(xm/1.3,ym/1.3) 
        cr.rel_line_to(-xm/2.5, 0)
        cr.curve_to (xm/5., ym/2., xm/5., ym/2., xm/2.,ym/2.) 
        cr.close_path()
        cr.set_source_rgb(0, 1., 0)
        cr.fill_preserve()
        cr.set_operator(cairo.OPERATOR_ADD)
        cr.set_source_rgb(0., 0., 1.0)
        cr.stroke()

        self.make_dataset(data)

    def draw_passage(self, passage_size=0.1):
        data = np.zeros((self.bs, self.size, self.size, 4), dtype=np.uint8)
        surface = cairo.ImageSurface.create_for_data(
                    data, cairo.FORMAT_ARGB32, self.size, self.size)
        cr = cairo.Context(surface)
        cr.set_source_rgb(1.0, 0.5, 0)
        cr.paint()
        cr.set_line_width (self.linewidth)
        # cr.set_dash(self.get_dashes());
        xm, ym = self.size, self.size
        cr.move_to (0, 0)
        cr.line_to(0.2*xm,(1-passage_size) * ym/2) 
        cr.line_to((1.-0.2)*xm,(1-passage_size) * ym/2) 
        cr.line_to(1.*xm,0.)
        cr.set_source_rgb (0, 1., 0);
        cr.fill_preserve();
        cr.set_operator(cairo.OPERATOR_ADD)
        cr.set_source_rgb(0., 0.0, 1.0)
        cr.stroke();

        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.new_sub_path() 
        cr.move_to (1.*xm, 1.*ym)
        cr.line_to((1-0.2)*xm,(1+passage_size) * ym/2)
        cr.line_to((0.2)*xm,(1+passage_size) * ym/2) 
        cr.line_to(0,1*ym) 
        cr.set_source_rgb (0, 0., 0);
        cr.fill_preserve();
        cr.set_operator(cairo.OPERATOR_ADD)
        cr.set_source_rgb(0., 0., 1.0)
        cr.stroke();

        self.make_dataset(data, labels = [0.0, 0.5, 1.] )

    def load_data(self, options):
        try:
            self.draw_voronoi()
            # self.draw_debug()
        except KeyError, e:
            print "Error during batch creation, retry ... Key Error %s" % str(e)
            self.exception_counter += 1
            if self.exception_counter < 20:
                self.load_data(options)
            else:
                exit()

    def load_test_data(self, options):
        np.random.seed(1337)
        orig_bs = self.bs
        self.bs = 1
        try:
            self.draw_voronoi()

        except KeyError, e:
            print "Error during batch creation, retry ... Key Error %s" % str(e)
            self.exception_counter += 1
            if self.exception_counter < 20:
                self.load_data(options)
            else:
                exit()


    def save_data(self, options):
        # export as h5 and png
        save_h5(self.options.save_net_path+"voronoi_test_labels.h5",
                            'data', data=self.label, overwrite='w')
        save_h5(self.options.save_net_path+"voronoi_test_height.h5",
                            'data', data=self.height_gt, overwrite='w')
        save_h5(self.options.save_net_path+"voronoi_test_input.h5",
                            'data', data=self.full_input, overwrite='w')

        from skimage import io
        inshape = self.full_input.shape
        print inshape
        k = np.zeros((inshape[2],inshape[3],3))
        k[:,:,2] = self.full_input[0,0]/256.

        # png.from_array(k, 'L')\
        #         .save(self.options.save_net_path+"voronoi_test_image.png")


        io.imsave(self.options.save_net_path+"test.png",k)

        self.bs = orig_bs

def cut_reprs(path):
    label_path = path + 'label_first_repr_big_zstack_cut.h5'
    memb_path = path + 'membranes_first_repr_big_zstack.h5'
    raw_path = path + 'raw_first_repr_big_zstack.h5'

    label = load_h5(label_path)[0][:, :300, :300].astype(np.uint64)
    memb = load_h5(memb_path)[0][:, :300, :300]
    raw = load_h5(raw_path)[0][:, :300, :300]

    save_h5(path + 'label_first_repr_zstack_cut.h5', 'data', data=label, overwrite='w')
    # save_h5(path + 'membranes_first_repr_zstack.h5', 'data', data=memb)
    # save_h5(path + 'raw_first_repr_zstack.h5', 'data', data=raw)


def load_h5(path, h5_key=None, group=None, group2=None, slices=None):
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
            output.append(np.array(g[key], dtype='float32'))
    elif isinstance(h5_key, basestring):   # string
        output = [np.array(g[h5_key], dtype='float32')]
    elif isinstance(h5_key, list):          # list
        output = list()
        for key in h5_key:
            output.append(np.array(g[key], dtype='float32'))
    else:
        raise Exception('h5 key type is not supported')
    if slices is not None:
        output = [output[0][slices]]
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
    
    pad_info = tuple((array.ndim-2)*[(0,0)]+\
                    [(pad_length, pad_length),
                    (pad_length, pad_length)])

    mirrored_array = np.pad(array, pad_info, mode='reflect')
    return mirrored_array


def pad_cube(array, pad_length, value=0):

    pad_info = tuple((array.ndim-2)*[(0,0)]+\
                [(pad_length, pad_length),
                (pad_length, pad_length)])
    padded_array = np.pad(array,
                            pad_info,
                            mode='constant',
                            constant_values=value)
    return padded_array


def make_array_cumulative(array):
    ids = np.unique(array)
    cumulative_array = np.zeros_like(array)
    for i in range(len(ids)):
        print '\r %f %%' % (100. * i / len(ids)),
        cumulative_array[array == ids[i]] = i
    return cumulative_array


def prpare_seg_path_wrapper(path, names):
    for name in names:
        segmentation = load_h5(path + 'seg_%s.h5' % name)[0]
        segmentation = prepare_data_mc(segmentation)
        save_h5(path + 'seg_%s_MC.h5' % name, 'data',
                data=segmentation.astype(np.uint32), overwrite='w')


def prepare_data_mc(segmentation):
    """
    input 2D segmentation in (x,y,z) returns 2D in (z, x, y) and non repeating
    ids
    :param segmentation:
    :return:
    """
    max_id = 0
    for slice in segmentation:
        slice += max_id
        max_id = np.max(slice) + 1
    segmentation = np.swapaxes(segmentation, 0, 2).astype(np.uint64)
    segmentation = segmentation[:, 75:-75, :]
    print 'seg fi', segmentation.shape
    return segmentation


def prepare_aligned_test_data(path):
    print 'preparing test data for prediction'
    for version in ['A+', 'C+']:
        print 'version', version
        memb_probs_path = path + '/sample%s_cantorFinal_pmaps_corrected.h5' % version
        raw_path = path + '/sample%s_raw_corrected.h5' % version

        memb_probs = load_h5(memb_probs_path)[0]
        raw = load_h5(raw_path)[0]
        print 'before: memb, raw', memb_probs.shape, raw.shape
        memb_probs = memb_probs.swapaxes(0, 2)
        raw = raw.swapaxes(0, 2)
        if version == 'A+':
            raw = np.pad(raw, ((0, 0), (38, 37), (0, 0)), 'reflect')
            memb_probs = np.pad(memb_probs, ((0, 0), (38, 37), (0, 0)), 'reflect')
        if version == 'B+':
            raw = np.pad(raw, ((0, 0), (75, 75), (0, 0)), 'reflect')
            memb_probs = np.pad(memb_probs, ((0, 0), (75, 75), (0, 0)), 'reflect')
        print 'after: memb, raw', memb_probs.shape, raw.shape

        save_h5(path + '/membranes_test_%s.h5' % version, 'data', data=memb_probs,
                overwrite='w')
        save_h5(path + '/raw_test_%s.h5' % version, 'data', data=raw,
                overwrite='w')


def cut_consti_data(vol_path, names=['raw', 'label', 'membranes', 'height'],
                    h5_key=None, label=False):
    def cut_consti_single(vol_path, name, h5_key=None,
                        label=False):
        all_data = np.empty((125*3, 1250, 1250), dtype=np.float32)
        for suffix, start in zip(['a', 'b', 'c'], range(0, 3*125, 125)):
            print 's', suffix, start, name
            all_data[start:start+125, :, :] = \
                    load_h5(vol_path + name + '_' + suffix + '.h5',
                            h5_key=h5_key)[0]
        first = range(0, 50) + range(125, 125+50) + range(2*125, 2*125+50)
        print 'first', first
        second = range(50, 125) + range(125+50, 2*125) + range(2*125+50, 3*125)
        print 'second', second
        if label:
            all_data = all_data.astype(np.uint64)
        save_h5(vol_path + name + '_first.h5', 'data', data=all_data[first, :, :],
                overwrite='w')
        save_h5(vol_path + name + '_second.h5', 'data', data=all_data[second, :, :],
                overwrite='w')

    for name in names:
        label_b = False
        h5_key = None
        if name == 'label':
            label_b = True
        if name == 'height':
            h5_key = 'height'
        print 'aneme', name
        cut_consti_single(vol_path, name, h5_key=h5_key,
                          label=label_b)


def generate_quick_eval(vol_path, names=['raw', 'label', 'membranes', 'height'],
                        h5_keys=[None,None, None, None],
                        label=False, suffix='_second',
                        n_slices=64, edg_len=300, n_slices_load=3 * 75,
                        inp_el=1250):

    represent_data = np.empty((4, n_slices, edg_len, edg_len))

    all_data = np.empty((4, n_slices_load, inp_el, inp_el))

    for i, (key, name) in enumerate(zip(h5_keys, names)):
        all_data[i, :, :, :] = load_h5(vol_path + name + suffix + '.h5',
                           h5_key=key)[0]

    slices = np.random.permutation(n_slices_load)[:n_slices]
    starts_x = np.random.randint(0, inp_el - edg_len, size=n_slices)
    starts_y = np.random.randint(0, inp_el - edg_len, size=n_slices)

    for i, (start_x, start_y, slice) in enumerate(zip(starts_x, starts_y, slices)):
        print 'i', i, slice, start_x, start_y
        represent_data[:, i, :, :] \
            = all_data[:, slice,
                       start_x:start_x+edg_len,
                       start_y:start_y+edg_len]

    for data, name in zip(represent_data, names):
        if name == 'label':
            data = data.astype(np.uint64)
        save_h5(vol_path + name + suffix + '_repr.h5', 'data',
                data=data, overwrite='w')

    print represent_data.shape


def generate_quick_eval_big_FOV_z_slices(
                        vol_path,
                        names=['raw', 'label','membranes', 'height'],
                        h5_keys=[None,None, None, None],
                        label=False, suffix='_first',
                        n_slices=16, edg_len=300, n_slices_load=3 * 50,
                        inp_el=1250, mode='valid', load=True, stack=True):
    print 'mode', mode, 'stack', stack, 'suffix', suffix
    if stack:
        factor = 3
    else:
        factor = 1
    represent_data = np.empty((4, factor * n_slices, edg_len, edg_len))

    all_data = np.empty((4, n_slices_load, inp_el, inp_el))

    for i, (key, name) in enumerate(zip(h5_keys, names)):
        all_data[i, :, :, :] = load_h5(vol_path + name + suffix + '.h5',
                           h5_key=key)[0]
    if not load:
        if mode == 'valid':
            z_inds = range(40, 50) + range(90, 100) + range(140, 150)
        elif mode == 'test':
            z_inds = range(0, 40) + range(50, 90) + range(100, 140)
        else:
            mode = 'second'
            assert (n_slices_load == 3 * 75)
            z_inds = range(n_slices_load)

        slices = sorted(np.random.choice(z_inds, size=n_slices,
                                         replace=False))
        starts_x = np.random.randint(0, inp_el - edg_len, size=n_slices)
        starts_y = np.random.randint(0, inp_el - edg_len, size=n_slices)
    else:
        slices, starts_x, starts_y = \
            load_h5(path + 'indices_%s.h5' % mode)[0].astype(int)

    if stack:
        save_inds_x = range(1, n_slices_load * 3, 3)
    else:
        save_inds_x = range(0, n_slices_load)
    for i, start_x, start_y, slice in zip(save_inds_x,
                                          starts_x, starts_y, slices):
        print 'i', i, slice, start_x, start_y

        represent_data[:, i, :, :] \
            = all_data[:, slice,
                       start_x:start_x+edg_len,
                       start_y:start_y+edg_len]
        if suffix == '_first':
            if slice == 0 or slice == 50 or slice == 100:
                below = slice
            else:
                below = slice - 1
            if slice == 49 or slice == 99 or slice == 149:
                above = slice
            else:
                above = slice + 1
        if suffix == '_second':
            if slice == 0 or slice == 75 or slice == 150:
                below = slice
            else:
                below = slice - 1
            if slice == 74 or slice == 149 or slice == 224:
                above = slice
            else:
                above = slice + 1

        if stack:
            inds_save = range(i-1, i+2)
            inds_load = [below, slice, above]
        else:
            inds_save = [i]
            inds_load = [slice]
        represent_data[:, inds_save, :, :] \
            = all_data[:, inds_load,
                       start_x:start_x+edg_len,
                       start_y:start_y+edg_len]

    # save data
    if not load:
        save_h5(path + 'indices_%s.h5' % mode, h5_key='zxy',
            data=[slices, starts_x, starts_y], overwrite='w')
    for data, name in zip(represent_data, names):
        if name == 'label':
            if stack:
                data = data.astype(np.uint64)[1::3]
            else:
                data = data.astype(np.uint64)
        if stack:
            stack_name = '_zstack'
        else:
            stack_name = ''
        save_h5(vol_path + name + '_%s' % mode + '_repr%s.h5' % stack_name, 'data',
                data=data, overwrite='w')


def segmentation_to_membrane(input_path,output_path):
    """
    compute a binary mask that indicates the boundary of two touching labels
    input_path: path to h5 label file
    output_path: path to output h5 file (will be created) 
    Uses threshold of edge filter maps
    """
    with h.File(input_path, 'r') as label_h5:
        with h.File(output_path, 'w') as height_h5:
            boundary_stack = np.empty_like(label_h5['data']).astype(np.float32)
            height_stack = np.empty_like(label_h5['data']).astype(np.float32)
            for i in range(height_stack.shape[0]):
                im = np.float32(label_h5['data'][i])
                boundary_stack[i], height_stack[i] = \
                    segmenation_to_membrane_core(im)
            height_h5.create_dataset("boundary",data=boundary_stack, dtype=np.float32)
            height_h5.create_dataset("height",data=height_stack, dtype=np.float32)


def create_holes(batch, fov):
    x, y = np.mgrid[0:fov:1, 0:fov:1]
    for b in range(batch.shape[0]):
        if np.random.random() > 0.5:
            if np.random.random() > 0.9:
                batch[b, 0, :, :] = 0
                batch[b, 8, fov / 4:-fov / 4, fov / 4:-fov / 4] = 0
            else:
                pos = np.dstack((x, y))
                rand_mat = np.diag(np.random.rand(2)) * 800
                rv = stats.multivariate_normal(np.random.randint(0, 40, 2),
                                               rand_mat)
                gauss = rv.pdf(pos).astype(np.float32)
                gauss /= np.max(gauss)
                gauss = 1. - gauss
                gauss_d = gauss[::2, ::2]
                batch[b, 0, :, :] *= gauss
                batch[b, 8, fov/4:-fov/4, fov/4:-fov/4] *= gauss_d

        if np.random.random() > 0.5:
            if np.random.random() > 0.9:
                batch[b, 6, :, :] = 0
            else:
                pos = np.dstack((x, y))
                rand_mat = np.diag(np.random.random(2)) * 800

                rv = stats.multivariate_normal(np.random.randint(0, 40, 2),
                                               rand_mat)
                gauss = rv.pdf(pos).astype(np.float32)
                gauss /= np.max(gauss)
                gauss = 1. - gauss
                batch[b, 6, :, :] *= gauss
        if np.random.random() > 0.5:
            if np.random.random() > 0.9:
                batch[b, 7, :, :] = 0
            else:
                pos = np.dstack((x, y))
                rand_mat = np.diag(np.random.random(2)) * 800
                rv = stats.multivariate_normal(np.random.randint(0, 40, 2),
                                               rand_mat)

                gauss = rv.pdf(pos).astype(np.float32)
                gauss /= np.max(gauss)
                gauss = 1. - gauss
                batch[b, 7, :, :] *= gauss
    return batch


def create_holes2(image, edge_len, n_holes=10):
    x, y = np.mgrid[0:edge_len:1, 0:edge_len:1]
    pos = np.dstack((x, y))
    for h in range(n_holes):
        rand_mat = np.diag(np.random.rand(2)) * edge_len * 2
        rv = stats.multivariate_normal(np.random.randint(0, edge_len, 2),
                                       rand_mat)
        gauss = rv.pdf(pos).astype(np.float32)
        gauss /= np.max(gauss)
        gauss = 1. - gauss
        image[:, :] *= gauss
    return image


def generate_dummy_data3(batch_size, edge_len, patch_len=40, save_path=None,
                         nz=64):
    batch_size = nz
    raw = np.zeros((batch_size, edge_len, edge_len), dtype='float32')
    label_gt = np.empty_like(raw)
    dist_trf = np.zeros_like(raw)

    # get membrane gt
    boundary = random_lines2(n_lines=3, bs=batch_size, edge_len=edge_len)
    boundary[:, 5, :] = 1
    invers_memb = np.ones_like(boundary)
    invers_memb[boundary == 1] = 0

    for b in range(boundary.shape[0]):

        raw[b, :, :] = boundary[b]
        raw[b, :, :] = create_holes2(boundary[b, :, :].copy(),
                                               edge_len)
        raw[b, :, :] /= np.max(raw[b, :, :])
        dist_trf[b] = distance_transform_edt(invers_memb[b])
        label_gt[b] = label(boundary[b], background=1, connectivity=1)
    seeds = u.get_seed_coords(label_gt, ignore_0=True)
    gt_new = np.zeros_like(raw)
    marker = np.zeros_like(raw).astype(np.int32)
    footprint = ndimage.generate_binary_structure(2, 1)

    for b in range(boundary.shape[0]):
        ims_seeds = np.array(seeds[b]) - 20
        for i, im_seed in enumerate(ims_seeds):
            marker[b, im_seed[0], im_seed[1]] = i + 1
        dist_trf[b] = (- dist_trf[b] + np.max(dist_trf[b])).astype(np.uint16)
        gt_new[b, :, :] = watershed(dist_trf[b], marker[b, :, :])
        # p = []
        # seed = np.array(seeds[b])
        # p.append({"title":"boundary",
        #           'im':boundary[b],
        #           'interpolation':'none',
        #           'scatter':seed - 20})
        # p.append({"title":"GT new",
        #           'im':gt_new[b],
        #           'cmap':'rand',
        #           'interpolation':'none',
        #           'scatter':seed - 20})
        # p.append({"title":"marker",
        #           'im':marker[b],
        #           'interpolation':'none',
        #           'scatter':seed - 20})
        # p.append({"title": "dist",
        #           'im': dist_trf[b] + edge_len**2,
        #           'interpolation': 'none',
        #           'scatter': seed - 20})
        # p.append({"title": "dist input",
        #           'im': dist_im,
        #           'interpolation': 'none',
        #           'scatter': seed - 20})
        # u.save_images(p, './../data/debug/', 'gt_no_holes')
        #
        # print seeds
        # exit()
    raw[raw < 0.1] = 0
    gt = gt_new
    membrane_prob = raw

    return raw, membrane_prob, dist_trf, gt


def random_lines2(n_lines, bs=None, edge_len=None, input_array=None):
    if input_array is None:
        input_array = np.zeros((bs, edge_len, edge_len))
    else:
        bs = input_array.shape[0]
        edge_len = input_array.shape[1]

    for b in range(bs):
        for i in range(n_lines):
            bott_left = np.random.randint(0, 2)
            if bott_left == 0:
                start = (np.random.randint(0, edge_len), 0)
            else:
                start = (0, np.random.randint(0, edge_len))
            top_right = np.random.randint(0, 2)
            if top_right == 0:
                end = (edge_len, np.random.randint(0, edge_len))
            else:
                end = (np.random.randint(0, edge_len), edge_len)

            x_points, y_points = u.get_line(start, end)
            x_points, y_points = \
                x_points[x_points < edge_len], y_points[x_points < edge_len]
            x_points, y_points = \
                x_points[y_points < edge_len], y_points[y_points < edge_len]
            input_array[b, x_points, y_points] = 1

    return input_array

class TestPolygonDataProvider(PolygonDataProvider):
    def load_data(self, options):
        self.load_test_data(options)

if __name__ == '__main__':
    # class opt():
    #     def __init__(self):
    #         self.batch_size = 10
    #         self.patch_len = 40
    #         self.network_channels = 1
    #         self.global_edge_len = 0
    #         self.clip_method='clip'
    #         self.padding_b=False
    #         self.net_name = "data_provider_test"
    #         self.save_net_path = './../data/nets/' + self.net_name
    #         if not exists(self.save_net_path):
    #             makedirs(self.save_net_path)
    # options = opt()
    # p = PolygonDataProvider(options)
    # inputx = np.zeros(p.get_batch_shape())
    # # print p.prepare_input_batch(inputx)
    # # for i in range(1000000):
    # #     print i
    # #     p.prepare_input_batch(inputx)
    # p.load_test_data(options)
    # # p.draw_circle()

    options = get_options()
    cdp = CremiDataProvider(options)
    cdp.load_data()
    print cdp.full_input.shape
    print cdp.label.shape
    print cdp.height_gt.shape
