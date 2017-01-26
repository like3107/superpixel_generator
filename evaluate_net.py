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
        self.get_options_from_net_file(val_options)        # sets self.options
        self.set_prediction_options(val_options)    # changes relevant self.options for validation
        print 'using options', self.options
        super(Predictor, self).__init__(self.options)
        self.bm.set_preselect_batches(range(len(self.options.slices)))
        print "using options", self.options

    def init_BM(self):
        self.BM = du.HoneyBatcherRec

    def get_options_from_net_file(self, net_options):
        net_path = net_options.load_net_path
        self.options = u.load_options(net_path, copy(net_options))      # use net_options as placeholder

    def set_prediction_options(self, val_options):

        # options to keep from val script
        self.options.seed_method = val_options.seed_method

        self.options.gpu = val_options.gpu
        self.options.slices = val_options.slices
        self.options.batch_size = len(val_options.slices)
        self.options.load_net_b = val_options.load_net_b
        self.options.load_net_path = val_options.load_net_path
        self.options.save_net_path = val_options.save_net_path
        self.options.global_edge_len = val_options.global_edge_len
        self.options.net_name = val_options.net_name
        self.options.padding_b = val_options.padding_b
        self.options.input_data_path = val_options.input_data_path
        self.options.height_gt_path = val_options.height_gt_path
        self.options.label_path = val_options.label_path
        self.options.s_minsize = 0
        # default options
        self.options.net_arch = 'v8_hydra_dilated_ft_joint'
        self.options.validation_b = True
        self.options.augment_pretraining = False
        self.options.augment_ft = False
        self.options.quick_eval = True
        self.options.fc_prec = True

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


if __name__ == '__main__':

    options = get_options()
    options.slices = [1,2,3,4,5,6,7]
    pred = Predictor(options)
    pred.predict()