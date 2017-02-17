import train_scripts
import nets
import utils as u
from trainer_config_parser import get_options
from copy import copy
import progressbar
import dataset_utils as du
import numpy as np
import data_provider


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
        self.options = get_options_from_net_file(val_options)        # sets self.options
        self.options = set_prediction_options(self.options, val_options)    # changes relevant self.options for validation
        from theano.sandbox import cuda as c
        import theano
        c.use(self.options.gpu)
        self.options.theano = theano
        self.prepare_paths()
        # options.patch_len = 68
        self.builder = nets.NetBuilder(self.options)
        self.define_loss()
        self.network_i_choose_you()

        self.options.patch_len = self.builder.fov
        self.bs = self.options.batch_size
        self.slices_total = options.slices_total

        # super(GottaCatchemAllPredictor, self).__init__(self.options)
        self.iterations = -1
        self.epoch = 0

        print 'initializing data provider'
        self.batch_data_provider = data_provider.get_dataset_provider(self.options.dataset)(self.options)
        self.batch_shape = self.batch_data_provider.get_batch_shape()
        self.label_shape = self.batch_data_provider.get_label_shape()
        print 'dp initialized'


    def get_batch(self, start, end):
        self.preselect_batches = range(start, end)
        self.global_input_batch = np.zeros(self.batch_shape, dtype=np.float32)
        self.batch_data_provider.prepare_input_batch(self.global_input_batch,
                                                     preselect_batches=self.preselect_batches)

    def predict(self):

        bar = progressbar.ProgressBar(max_value=self.slices_total / self.bs)
        height_pres = np.zeros((self.slices_total, 1, self.label_shape[-1], self.label_shape[-1]))
        for b in range(0, self.slices_total, self.bs):
            self.iterations += 1
            self.epoch += 1
            self.get_batch(b, b + self.bs)
            height_pres[b:b+self.bs, :, :] = self.prediction_f(self.global_input_batch)
            bar.update(self.iterations)
            if self.iterations % 100 == 0:
                data_provider.save_h5(self.options.save_edges_path + '/inter_edges.h5', 'data', height_pres,
                                      overwrite='w', compression='gzip')
        data_provider.save_h5(self.options.save_edges_path + '/edges.h5', 'data', height_pres,
                              overwrite='w', compression='gzip')
        return height_pres




def get_options_from_net_file(net_options):
    net_path = net_options.load_net_path
    options = u.load_options(net_path, copy(net_options))      # use net_options as placeholder
    return options


def set_prediction_options(options, val_options):

    # options to keep from val script
    options.seed_method = val_options.seed_method

    options.gpu = val_options.gpu
    options.slices = val_options.slices
    options.batch_size = val_options.batch_size
    options.load_net_b = val_options.load_net_b
    options.load_net_path = val_options.load_net_path
    options.save_net_path = val_options.save_net_path
    options.global_edge_len = val_options.global_edge_len
    options.net_name = val_options.net_name
    options.padding_b = val_options.padding_b
    options.input_data_path = val_options.input_data_path
    options.height_gt_path = val_options.height_gt_path
    options.label_path = val_options.label_path
    options.s_minsize = val_options.s_minsize
    options.padding_b = val_options.padding_b
    options.fully_conf_valildation_b = val_options.fully_conf_valildation_b
    # default options
    # options.net_arch = 'v8_hydra_dilated_ft_joint'
    options.clip_method = 'clip'
    options.validation_b = True
    options.augment_pretraining = False
    options.augment_ft = False
    options.quick_eval = True
    options.fc_prec = True
    return options


if __name__ == '__main__':

    options = get_options(script='full_conf_pred')
    print 'save net path', options.save_net_path
    options.slices = range(options.slices_total)

    import os
    if not os.path.exists(options.save_net_path):
        os.mkdir(options.save_net_path)
    options.save_edges_path = options.save_net_path + '/edges'
    if not os.path.exists(options.save_edges_path):
        os.mkdir(options.save_edges_path)

    Predictor = GottaCatchemAllPredictor

    pred = Predictor(options)
    pred.predict()
