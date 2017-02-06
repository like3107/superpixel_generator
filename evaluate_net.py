import train_scripts
import nets
import utils as u
from trainer_config_parser import get_options
from copy import copy
import progressbar
import dataset_utils as du


class Predictor(train_scripts.FCRecFinePokemonTrainer):
    def __init__(self, val_options):
        self.options = None
        self.options = get_options_from_net_file(val_options)        # sets self.options
        self.options = set_prediction_options(self.options, val_options)    # changes relevant self.options for validation
        print 'using options', self.options
        super(Predictor, self).__init__(self.options)
        self.bm.set_preselect_batches(range(len(self.options.slices)))

    def init_BM(self):
        self.BM = du.HoneyBatcherRec


    def predict(self):
        self.bm.init_batch()
        self.free_voxel = self.free_voxel_empty
        inputs = self.update_BM_FC()
        # precompute fc part
        self.precomp_input = self.fc_prec_conv_body(inputs)
        # select all sliced batches (in order)
        bar = progressbar.ProgressBar(max_value=self.free_voxel)
        print "predicting z-slices ", self.options.slices
        while (self.free_voxel > 0):
            self.iterations += 1
            self.free_voxel -= 1
            self.update_BM()
            # self.save_net()

            if self.free_voxel % ((self.free_voxel_empty-1) / 5) == 0:
                self.draw_debug(image_path=self.options.validation_save_path + '/images/' + 'slice_%04i' %
                                                                                            self.options.slices[0])
            if self.free_voxel % 100 == 0:
                bar.update(self.free_voxel_empty - self.free_voxel)
        # save to h5


class GottaCatchemAllPredictor(train_scripts.GottaCatchemAllTrainer):
    def __init__(self, val_options):
        self.options = None
        print 'val options', val_options
        self.options = get_options_from_net_file(val_options)        # sets self.options
        self.options = set_prediction_options(self.options, val_options)    # changes relevant self.options for validation
        print 'using options', self.options
        super(GottaCatchemAllPredictor, self).__init__(self.options)
        self.bm.set_preselect_batches(range(len(self.options.slices)))
        print "using options", self.options

    def predict(self):
        self.iterations += 1
        self.epoch += 1
        inputs, _, heights = self.update_BM()
        height_pred = self.prediction_f(inputs)
        # this is intensive surgery to the BM
        self.bm.global_heightmap_batch = height_pred
        print 'heigt ', self.bm.global_heightmap_batch


def get_options_from_net_file(net_options):
    net_path = net_options.load_net_path
    options = u.load_options(net_path, copy(net_options))      # use net_options as placeholder
    return options


def set_prediction_options(options, val_options):

    # options to keep from val script
    options.seed_method = val_options.seed_method

    options.gpu = val_options.gpu
    options.slices = val_options.slices
    options.batch_size = len(val_options.slices)
    options.load_net_b = val_options.load_net_b
    options.load_net_path = val_options.load_net_path
    options.save_net_path = val_options.save_net_path
    options.global_edge_len = val_options.global_edge_len
    options.net_name = val_options.net_name
    options.padding_b = val_options.padding_b
    options.input_data_path = val_options.input_data_path
    options.height_gt_path = val_options.height_gt_path
    options.label_path = val_options.label_path
    options.s_minsize = 0
    options.padding_b = val_options.padding_b
    options.fully_conf_valildation_b = val_options.fully_conf_valildation_b
    # default options
    # options.net_arch = 'v8_hydra_dilated_ft_joint'
    options.validation_b = True
    options.augment_pretraining = False
    options.augment_ft = False
    options.quick_eval = True
    options.fc_prec = True
    return options

if __name__ == '__main__':

    options = get_options()
    options.slices = [1,2,3,4,5,6,7]
    pred = Predictor(options)
    pred.predict()