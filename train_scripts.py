import matplotlib
matplotlib.use('Agg')
# matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt
import os
from theano import tensor as T
import theano
import utils as u
import nets
import dataset_utils as du
import numpy as np
from theano.sandbox import cuda as c
import experience_replay as exp
import h5py
import sys
import configargparse
from copy import copy
import time
from trainer_config_parser import get_options

# TODO: add allowed slices?

class PokemonTrainer(object):
    def __init__(self, options):
        self.options = options
        self.prepare_paths()
        self.builder = nets.NetBuilder(options)
        self.define_loss()
        self.network_i_choose_you()
        # options.patch_len = 68
        self.init_BM()
        self.bm = self.BM(self.options)
        self.bm.init_batch()

        self.iterations = -1
        self.update_steps = 10
        self.free_voxel_empty = self.bm.get_num_free_voxel()
        self.free_voxel = self.free_voxel_empty
        self.observation_counter = self.options.save_counter
        self.update_history = []
        self.loss_history = []

    def network_i_choose_you(self):
        network = self.builder.get_net(self.options.net_arch)
        c.use(self.options.gpu)
        l_in, l_in_direction, l_out, l_out_direction,\
                 self.options.patch_len, l_eat = network()
        self.l_out = l_out
        self.options.network_channels = l_in.shape[1]
        target_t = T.ftensor4()
        self.loss_train_f, loss_valid_f, self.prediction_f = \
            self.loss(l_in, target_t, l_out, L1_weight=self.options.regularization)

        if self.options.load_net_b:
            np.random.seed(np.random.seed(int(time.time())))
            # change seed so different images for retrain
            print "loading network parameters from ", self.options.load_net_path
            u.load_network(self.options.load_net_path, l_out)

    def init_BM(self):
        self.BM = du.HoneyBatcherPath

    def define_loss(self):
        self.loss = self.builder.get_loss('updates_probs_v0')

    def prepare_paths(self):
        self.save_net_path = './../data/nets/' + self.options.net_name + '/'
        self.debug_path = self.save_net_path + "/debug"
        self.image_path = self.save_net_path + '/images/pretrain/'
        self.image_path_reset = self.save_net_path + '/images/reset/'
        self.net_param_path = self.save_net_path + '/nets/'
        code_path = self.save_net_path + '/code/'
        paths = [self.save_net_path, self.debug_path,
                 self.save_net_path + '/images/', self.image_path,
                 self.net_param_path, code_path, self.image_path_reset]

        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

        os.system('cp -rf *.py ' + code_path)
        os.system('cp -rf ./../data/config/*.conf ' + code_path)


    def update_BM(self):
        inputs, gt, seeds, ids = self.bm.get_batches()
        heights = self.prediction_f(inputs)
        self.bm.update_priority_queue(heights, seeds, ids)
        return inputs, heights, gt

    def train(self):
        # make 10 update steps
        for i in range(self.update_steps):
            self.iterations += 1
            self.free_voxel -= 1
            inputs, heights, gt = self.update_BM()

            if self.iterations % self.observation_counter == 0:
                trainer.draw_debug()

            if self.free_voxel == 0:
                self.reset_image()

        # update parameters once
        loss_train, individual_loss, _ = self.loss_train_f(inputs, gt)
        self.update_history.append(self.iterations)
        self.loss_history.append(loss_train)
        u.plot_train_val_errors(
            [self.loss_history],
            self.update_history,
            self.save_net_path + '/training.png',
            names=['loss'])
        return loss_train

    def reset_image(self, reset=False):
        self.bm.init_batch()
        self.free_voxel = self.free_voxel_empty
        trainer.draw_debug(reset=True)

    def converged(self):
        return self.iterations >= self.options.pre_train_iter

    def draw_debug(self, reset=False, image_path=None):
        if reset:
            image_path = self.image_path_reset
        elif image_path is None:
            image_path = self.image_path

        for b in range(1):
            self.bm.draw_debug_image(
                "train_b_%03i_i_%08i_f_%i" %
                (b, self.iterations, self.free_voxel),
                path=image_path, b=b)

    def save_net(self, path=None, name=None):
        if path is None:
            path = self.net_param_path
        if name is None:
            name = 'net_%i' % self.iterations
        u.save_network(path, self.l_out,
                   name, add=self.options._get_kwargs())

    def load_net(self, file_path=None):
        if file_path is None:
            file_path = self.options.load_net_path
        print "loading network parameters from ", self.options.load_net_path
        u.load_network(self.options.load_net_path, self.l_out)

class Membertrainer(PokemonTrainer):
    def init_BM(self):
        self.BM = du.HoneyBatcherPath
        self.Memento = exp.BatcherBatcherBatcher(
                    scale_height_factor=self.options.scale_height_factor,
                    max_mem_size=self.options.exp_mem_size,
                    pl=self.options.patch_len,
                    warmstart=self.options.exp_warmstart,
                    n_channels=n_channels,
                    accept_rate=self.options.exp_acceptance_rate,
                    use_loss=self.options.exp_loss,
                    weight_last=self.options.exp_wlast)
        if self.options.exp_load != "None":
            # np.random.seed(len(self.options.net_name))
            print "loading Memento from ", self.options.exp_load
            Memento.load(self.options.exp_load)


class FusionPokemonTrainer(PokemonTrainer):
    def init_BM(self):
        self.BM = du.HungryHoneyBatcher

    def define_loss(self):
        self.loss = self.builder.get_loss('updates_v7_EAT')

    def network_i_choose_you(self):
        network = self.builder.get_net(self.options.net_arch)
        c.use(self.options.gpu)
        l_in, l_in_direction, l_out, l_out_direction,\
                 self.options.patch_len, l_eat = network()
        self.options.network_channels = l_in.shape[1]
        target_t = T.ftensor4()
        target_eat = T.ftensor4()
        target_eat_factor = T.ftensor4()
        loss_train_f, loss_valid_f, self.prediction_f, loss_merge_f, eat_f = \
            loss(l_in, target_t, l_out, l_eat, target_eat, target_eat_factor,\
            L1_weight=self.options.regularization)

        if self.options.load_net_b:
            np.random.seed(np.random.seed(int(time.time())))
            # change seed so different images for retrain
            print "loading network parameters from ", self.options.load_net_path
            u.load_network(self.options.load_net_path, l_out)

    def update_BM(self):
        inputs, gt, seeds, ids, merging_gt, merging_factor, merging_ids =\
                self.bm.get_batches()
        heights = probs_f(inputs)
        self.bm.update_priority_queue(heights, seeds, ids)
        # check inf there are neighbours that could be merged
        if np.any(merging_factor>0):
            bm.update_merge(eat_f(inputs), merging_factor, merging_ids, ids)
        # TODO: move merge_gt to seperate function
        return inputs, heights, gt


class FinePokemonTrainer(PokemonTrainer):
    def init_BM(self):
        self.BM = du.HoneyBatcherPath

    def define_loss(self):
        self.loss = self.builder.get_loss('updates_hydra_v5')

    def network_i_choose_you(self):
        network = self.builder.get_net(self.options.net_arch)
        c.use(self.options.gpu)
        l_in, l_in_direction, self.l_out, l_out_direction,\
                 self.options.patch_len, _ = network()
        self.options.network_channels = l_in.shape[1]
        target_t = T.ftensor4()

        self.loss_train_fine_f, self.loss_valid_fine_f, self.prediction_f  = \
            self.loss(l_in, l_in_direction, l_out_direction, self.l_out,
                          L1_weight=self.options.regularization)

    def train(self):
        while (self.free_voxel > 0):
            self.iterations += 1
            self.free_voxel -= 1
            self.update_BM()
            if self.free_voxel == 0:
                self.bm.init_batch()
                self.free_voxel = self.free_voxel_empty
                trainer.draw_debug()

        self.bm.find_global_error_paths()
        if self.bm.count_new_path_errors() > 0:
            error_b_type1, error_b_type2, dir1, dir2 = \
                        self.bm.reconstruct_path_error_inputs()
            batch_ft = exp.stack_batch(error_b_type1, error_b_type2)
            batch_dir_ft = exp.stack_batch(dir1, dir2)

            batch_ft = exp.flatten_stack(batch_ft).astype(theano.config.floatX)
            batch_dir_ft = exp.flatten_stack(batch_dir_ft).astype(np.int32)

            ft_loss_train, individual_loss_fine = \
                    self.loss_train_fine_f(batch_ft, batch_dir_ft)

            self.update_history.append(self.iterations)
            self.loss_history.append(ft_loss_train)
            u.plot_train_val_errors(
                    [self.loss_history],
                    self.update_history,
                    self.save_net_path + '/training_finetuning.png',
                    names=['loss finetuning'])

            return ft_loss_train
        else:
            return 0

    def converged(self):
        return self.iterations >= self.options.max_iter


class FCFinePokemonTrainer(FinePokemonTrainer):
    def init_BM(self):
        self.BM = du.HoneyBatcherPatchFast
        self.images_counter = -1

    def define_loss(self):
        self.loss = self.builder.get_loss('updates_hydra_v8')

    def network_i_choose_you(self):
        network = self.builder.get_net(options.net_arch)
        c.use(self.options.gpu)
        layers, self.options.patch_len, _ = network()
        self.l_out = layers["l_out_cross"]
        self.options.network_channels = layers['l_in_claims'].shape[1]
        target_t = T.ftensor4()

        if self.options.load_net_b:
            np.random.seed(np.random.seed(int(time.time())))
            # change seed so different images for retrain
            print "loading network parameters from ", self.options.load_net_path
            u.load_network(self.options.load_net_path, self.l_out)

        self.prediction_f,  self.fc_prec_conv_body, self.loss_train_fine_f, \
            self.debug_f, self.debug_singe_out = \
             self.loss(layers, L1_weight=self.options.regularization)

    def update_BM_FC(self):
        inputs = self.bm.global_input_batch[:, :, :-1, :-1]
        return inputs

    def update_BM(self):
        claims, gt, seeds, ids = self.bm.get_batches()
        seeds = np.array(seeds, dtype=np.int)
        precomp_input = self.precomp_input[range(self.bm.bs), :,
                                           seeds[:, 0] - self.bm.pad,
                                           seeds[:, 1] - self.bm.pad]
        precomp_input = precomp_input[:, :, None, None]
        heights = self.prediction_f(claims, precomp_input)
        self.bm.update_priority_queue(heights, seeds, ids)
        return

    def train(self):
        self.bm.init_batch()
        inputs = self.update_BM_FC()
        # precompute fc part
        self.precomp_input = self.fc_prec_conv_body(inputs)
        self.images_counter += 1
        while (self.free_voxel > 0):
            self.iterations += 1
            self.free_voxel -= 1

            self.update_BM()

            if self.iterations % self.observation_counter == 0:
                self.draw_debug(reset=False)

        self.bm.find_global_error_paths()
        if self.bm.count_new_path_errors() > 0:
            error_b_type1, error_b_type2, dir1, dir2, e_pos, batches = \
                self.bm.reconstruct_path_error_inputs(fc=True)
            precomp_input = self.precomp_input[batches, :,
                                               e_pos[:, 0] - self.bm.pad,
                                               e_pos[:, 1] - self.bm.pad]

            precomp_input = precomp_input[:, :, None, None]

            batch_ft = exp.stack_batch(error_b_type1, error_b_type2)
            batch_dir_ft = exp.stack_batch(dir1, dir2)

            batch_ft = exp.flatten_stack(batch_ft).astype(np.float32)
            batch_dir_ft = exp.flatten_stack(batch_dir_ft).astype(np.int32)

            # heights = self.prediction_f(batch_ft, precomp_input, batch_dir_ft)
            # batch_ft[...] = 0.

            single_heights, cross_heighst = self.debug_singe_out(batch_ft,
                                                                 precomp_input,
                                                                 batch_dir_ft)
            ft_loss_train, individual_loss_fine, _, heights = \
                self.loss_train_fine_f(batch_ft, precomp_input, batch_dir_ft)


            print 'loss ft', ft_loss_train
            # print 'error type II ', self.bm.error_II_type

            zip(heights, self.bm.e1heights + self.bm.e2heights)
            bs = len(heights) / 2
            for err, heightpreve1, heightpreve2, heightrec1, heightrec2, \
                ind_loss, errt in \
                    zip(self.bm.all_errorsq, self.bm.e1heights,
                        self.bm.e2heights, heights[:bs], heights[bs:],
                        individual_loss_fine, self.bm.error_II_type):

                print 'error', err["batch"], 'e1 pos', err["e1_pos"], \
                    err['e2_pos'], 'loss', ind_loss[0, 0, 0],\
                    heightpreve1 - heightrec1[0, 0, 0], heightpreve2 - heightrec2[0, 0, 0], errt, \
                    'plateau', err["plateau"]
            # print 'cross heights', cross_heighst
            # print 'where', np.where(self.bm.glokbal_prediction_map_nq[:] == \
            #                         single_heights[-1, 0, 0, 0])

            self.update_history.append(self.iterations)
            self.loss_history.append(ft_loss_train)
            u.plot_train_val_errors(
                [self.loss_history],
                self.update_history,
                self.save_net_path + '/training_finetuning.png',
                names=['loss finetuning'], log_scale=False)

        if self.images_counter % 1 == 0:
            self.save_net()
            trainer.draw_debug(reset=True)

        if self.free_voxel == 0:
            print 'init batch'
            # self.bm.draw_error_reconst('err_rec%i' %self.iterations)
            self.free_voxel = self.free_voxel_empty

class Pokedex(PokemonTrainer):
    """
    prediction of images only
    """
    def init_BM(self):
        self.BM = du.HoneyBatcherPredict

class GottaCatchemAllTrainer(PokemonTrainer):
    """
    Fully conv pretrain
    """
    def __init__(self,  options):
        super(GottaCatchemAllTrainer, self).__init__(options)
        self.update_steps = 1
        self.observation_counter = 100
        self.loss_history = [[], []]


    def init_BM(self):
        self.BM = du.HoneyBatcherPath

    def update_BM(self):
        # self.bm.batch_data_provider.load_data(self.options)
        # self.bm.batch_data_provider = PolygonDataProvider(self.options)
        self.bm.init_batch()
        inputs = self.bm.global_input_batch[:, :, :-1, :-1]
        heights_gt = du.height_to_fc_height_gt(self.bm.global_height_gt_batch)
        return inputs, None, heights_gt


    def train(self):
        self.iterations += 1
        inputs, _, heights = self.update_BM()

        height_pred = self.prediction_f(inputs)
        # this is intensive surgery to the BM

        self.bm.global_prediction_map_FC = height_pred

        if self.iterations % self.observation_counter == 0:
            trainer.draw_debug(reset=True)

        loss_train, individual_loss, _ = self.loss_train_f(inputs, heights)
        loss_no_reg = np.mean(individual_loss)

        if self.iterations % 100 == 0:
            self.save_net()

        # update parameters once
        self.update_history.append(self.iterations)
        self.loss_history[0].append(loss_train)
        self.loss_history[1].append(loss_no_reg)
        u.plot_train_val_errors(
            [self.loss_history[0], self.loss_history[1]],
            self.update_history,
            self.save_net_path + '/training.png',
            names=['loss', 'loss no reg'])
        return loss_train

    def draw_debug(self, reset=False, image_path=None):
        if reset:
            image_path = self.image_path_reset
        elif image_path is None:
            image_path = self.image_path

        for b in range(1):
            self.bm.draw_debug_image(
                "train_b_%03i_i_%08i_f_%i" %
                (b, self.iterations, self.free_voxel),
                path=image_path, b=b, plot_height_pred=True)




if __name__ == '__main__':
    options = get_options()
    # pret
    if options.net_arch == 'v8_hydra_dilated':
        trainer = GottaCatchemAllTrainer(options)
        while not trainer.converged():
            print "\r pretrain %0.4f iteration %i free voxel %i" \
                  %(trainer.train(), trainer.iterations, trainer.free_voxel),
        trainer.save_net(path=trainer.net_param_path, name='pretrain_final.h5')

    elif options.net_arch == 'v8_hydra_dilated_ft':

        trainer = FCFinePokemonTrainer(options)
        while not trainer.converged():
            trainer.train()
        trainer.save_net(path=trainer.net_param_path, name='pretrain_final.h5')

    # finetrainer = FinePokemonTrainer(options)
    # finetrainer.load_net(trainer.net_param_path + '/pretrain_final.h5')
    # while not finetrainer.converged():
    #     print "finetune", finetrainer.train()
    #     finetrainer.draw_debug()

    #
    #
    #
    #

    # trainer = PokemonTrainer(options)
    # while not trainer.converged():
    #     print "prertrain", trainer.train()
    #
    # trainer.save_net(path=trainer.net_param_path, name='pretrain_final.h5')
    # finetrainer = FinePokemonTrainer(options)
    # finetrainer.load_net(trainer.net_param_path + '/pretrain_final.h5')
    # while not finetrainer.converged():
    #     print "finetune", finetrainer.train()
    #     finetrainer.draw_debug()


    # ft
    # finetrainer = FinePokemonTrainer(options)
    # finetrainer.load_net(trainer.net_param_path + '/pretrain_final.h5')
    #
    # while not finetrainer.converged():
    #     print "finetune",finetrainer.train()
    #     finetrainer.draw_debug()
    #
