import train_scripts
import nets
import utils as u
from trainer_config_parser import get_options
from copy import copy
import progressbar

class Predictor(train_scripts.FinePokemonTrainer):
    def __init__(self, options):
        self.get_options_from_net_file(options)
        self.set_prediction_options(options)
        self.builder = nets.NetBuilder(self.options)
        self.define_loss()
        self.network_i_choose_you()
        # options.patch_len = 68
        self.init_BM()
        self.bm = self.BM(self.options)
        self.bm.set_preselect_batches(range(len(self.options.slices)))
        self.bm.init_batch()
        self.iterations = -1
        self.update_steps = 10
        self.free_voxel_empty = self.bm.get_num_free_voxel()
        self.free_voxel = self.free_voxel_empty
        self.observation_counter = self.options.save_counter
        self.update_history = []
        self.loss_history = []
        self.prepare_paths()
        print "using options", self.options

    def get_options_from_net_file(self, options):
        self.options = copy(options)
        net_path = options.load_net_path
        # self.options.__dict__.clear()
        # u.load_options(net_path, self.options)
        self.options = options
        # debug temporary options
        self.options.net_arch = 'v8_hydra_dilated_ft_joint'


        self.options.gpu = options.gpu
        self.options.slices = options.slices
`
        self.options.batch_size = len(options.slices)
        self.options.load_net_b = options.load_net_b
        self.options.load_net_path = options.load_net_path
        self.options.save_net_path = options.save_net_path
        self.options.global_edge_len = options.global_edge_len
        self.options.quick_eval = options.quick_eval
        self.options.net_name = options.net_name
        self.options.padding_b = options.padding_b
        self.options.input_data_path = options.input_data_path
        self.options.height_gt_path = options.height_gt_path
        self.options.label_path = options.label_path

    def set_prediction_options(self, options):
        self.options.seed_method = options.seed_method

    def predict(self):
        # select all sliced batches (in order)
        bar = progressbar.ProgressBar(max_value=self.free_voxel)
        print "predicting z-slices ", self.options.slices
        while (self.free_voxel > 0):
            self.iterations += 1
            self.free_voxel -= 1
            self.update_BM()
            # self.save_net()

            if self.free_voxel % ((self.free_voxel_empty-1) / 5) == 0:
                self.draw_debug(image_path=self.options.save_net_path+\
                                    'slice_%04i'%self.options.slices[0])
            if self.free_voxel % 100 == 0:
                bar.update(self.free_voxel_empty - self.free_voxel)


        # save to h5


if __name__ == '__main__':

    options = get_options()
    options.slices = [1,2,3,4,5,6,7]
    pred = Predictor(options)
    pred.predict()