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
import progressbar
from lasagne import layers as L

import validation_scripts as vs


# TODO: add allowed slices?

class PokemonTrainer(object):
    def __init__(self, options):
        self.options = options
        c.use(self.options.gpu)
        self.options.theano = theano
        self.prepare_paths()
        # options.patch_len = 68
        self.builder = nets.NetBuilder(options)
        self.define_loss()
        self.network_i_choose_you()

        self.options.patch_len = self.builder.fov

        self.init_BM()
        self.bm = self.BM(self.options)

        if "val_options" in self.options:
            self.options.val_options.patch_len = self.builder.fov
            self.val_bm = self.BM(self.options.val_options)
        else:
            self.val_bm = None

        self.iterations = -1
        self.update_steps = 10
        self.free_voxel_empty = self.bm.get_num_free_voxel()
        self.free_voxel = self.free_voxel_empty
        self.observation_counter = self.options.save_counter
        self.update_history = []
        self.loss_history = [[], []]
        self.val_loss_history = [[], []]

    def network_i_choose_you(self):
        network = self.builder.get_net(self.options.net_arch)
        layers, self.options.patch_len, l_eat = network()
        self.l_out = layers['l_out_cross']
        target_t = T.ftensor4()
        self.loss_train_f, loss_valid_f, self.prediction_f = self.loss(layers, target_t,
                                                                       L1_weight=self.options.regularization)
        if self.options.load_net_b:
            np.random.seed(np.random.seed(int(time.time())))    # change seed so different images for retrain
            print "loading network parameters from ", self.options.load_net_path
            u.load_network(self.options.load_net_path, self.l_out)

    def init_BM(self):
        self.BM = du.HoneyBatcherPath

    def define_loss(self):
        self.loss = self.builder.get_loss('updates_probs_v0')

    def prepare_paths(self):
        self.save_net_path = './../data/nets/' + self.options.net_name + '/'
        self.debug_path = self.save_net_path + "/debug"
        self.image_path = self.save_net_path + '/images/pretrain/'
        self.image_path_reset = self.save_net_path + '/images/reset/'
        self.image_path_validation = self.save_net_path + '/images/validation/'
        self.net_param_path = self.save_net_path + '/nets/'
        code_path = self.save_net_path + '/code/'
        paths = [self.save_net_path, self.debug_path,
                 self.save_net_path + '/images/', self.image_path,
                 self.net_param_path, code_path, self.image_path_reset,
                 self.image_path_validation]

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
        u.plot_train_val_errors([self.loss_history], self.update_history, self.save_net_path + '/training.png',
                                names=['loss'])
        return loss_train

    def reset_image(self, reset=False):
        self.bm.init_batch()
        self.free_voxel = self.free_voxel_empty
        trainer.draw_debug(reset=True)

    def converged(self):
        return self.iterations >= self.options.pre_train_iter

    def draw_debug(self, reset=False, image_path=None, image_name='train'):
        if reset:
            image_path = self.image_path_reset
        elif image_path is None:
            image_path = self.image_path

        for b in range(self.bm.bs):
            self.bm.draw_debug_image(
                "%s_b_%03i_i_%08i_f_%i" %
                (image_name, b, self.iterations, self.free_voxel),
                path=image_path, b=b)

    def save_net(self, path=None, name=None):
        if path is None:
            path = self.net_param_path
        if name is None:
            name = 'net_%i' % self.iterations

        u.save_network(path, self.l_out, name, options=self.options)

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
        layers, self.options.patch_len, l_eat = network()
        self.l_out = layers['l_out_cross']
        target_t = T.ftensor4()
        target_eat = T.ftensor4()
        target_eat_factor = T.ftensor4()
        loss_train_f, loss_valid_f, self.prediction_f, loss_merge_f, eat_f = \
            loss(l_in, target_t, self.l_out, l_eat, target_eat, target_eat_factor,\
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
        self.loss = self.builder.get_loss('updates_hydra_v8')

    def network_i_choose_you(self):
        network = self.builder.get_net(self.options.net_arch)
        c.use(self.options.gpu)
        layers, self.options.patch_len, _ = network()
        self.l_out = layers['l_out_cross']

        if self.options.load_net_b:
            np.random.seed(np.random.seed(int(time.time())))
            # change seed so different images for retrain
            print "loading network parameters from ", self.options.load_net_path
            u.load_network(self.options.load_net_path, layers["l_out_cross"])

        self.prediction_f,  self.fc_prec_conv_body, self.loss_train_fine_f, \
            self.debug_f, self.debug_singe_out = \
             self.loss(layers, L1_weight=self.options.regularization)

    def update_BM(self):
        inputs, gt, seeds, ids = self.bm.get_batches()
        heights = self.prediction_f(inputs[:, :2], inputs[:, 2:])
        self.bm.update_priority_queue(heights, seeds, ids)
        return inputs, heights, gt


    def train(self):
        self.bm.init_batch()
        self.free_voxel = self.free_voxel_empty
        bar = progressbar.ProgressBar(max_value=self.free_voxel_empty)
        while (self.free_voxel > 0):
            self.update_BM()
            # if self.iterations % 100 == 0:
            #     self.draw_debug()
            if self.free_voxel % 100 == 0:
                bar.update(self.free_voxel_empty - self.free_voxel)
            self.free_voxel -= 1
            self.iterations += 1

        ft_loss_train = 0

        self.bm.find_global_error_paths()
        if self.bm.count_new_path_errors() > 0:
            error_b_type1, error_b_type2, dir1, dir2 = \
                        self.bm.reconstruct_path_error_inputs()
            batch_ft = exp.stack_batch(error_b_type1, error_b_type2)
            batch_dir_ft = exp.stack_batch(dir1, dir2)

            batch_ft = exp.flatten_stack(batch_ft).astype(theano.config.floatX)
            batch_dir_ft = exp.flatten_stack(batch_dir_ft).astype(np.int32)

            ft_loss_train, individual_loss_fine, _, heights = \
                    self.loss_train_fine_f(batch_ft[:, :2], batch_ft[:, 2:],
                                           batch_dir_ft)

            print "ft_loss_train", ft_loss_train
            self.update_history.append(self.iterations)
            self.loss_history.append(ft_loss_train)
        u.plot_train_val_errors(
            [self.loss_history],
            self.update_history,
            self.save_net_path + '/training_finetuning.png',
            names=['loss finetuning'], log_scale=False)
        print ""

        self.save_net()
        trainer.draw_debug(reset=True)

    def converged(self):
        return self.iterations >= self.options.max_iter


class SpeedyPokemonTrainer(FinePokemonTrainer):
    def init_BM(self):
        self.BM = du.HoneyBatcherGonzales

    def define_loss(self):
        print "no loss defined yet"

    def network_i_choose_you(self):
        network = self.builder.get_net(self.options.net_arch)
        c.use(self.options.gpu)
        layers, self.options.patch_len, _ = network(l_image_in = self.bm.shared_input_batch,
                                                    l_claims_in = self.bm.shared_claims_batch)
        self.l_out = layers['l_out_cross']

        l_out_valid = L.get_output(self.l_out, deterministic=True)
        theano.config.exception_verbosity = 'high'
        coords, mes = self.bm.shared_input_coord_list
        # self.t1 = theano.function([coords], self.bm.shared_input_batch)

        l_out_claim = L.get_output(layers['l_in_claims'], deterministic=True)
        self.t2 = theano.function([coords, mes], l_out_claim)
        self.prediction_f = theano.function([coords, mes], l_out_valid)

    def update_BM(self):
        inputs, gt, seeds, ids = self.bm.get_batches()
        # # seeds[:] = 70
        heights = self.prediction_f(seeds, ids)
        self.bm.update_priority_queue(heights, seeds, ids)
        return inputs, heights, gt

    def train(self):
        print self.free_voxel_empty
        self.free_voxel = self.free_voxel_empty
        # while (self.free_voxel > 0):
        #     self.iterations += 1
        #     self.free_voxel -= 1
        #     print self.free_voxel,"\t",self.iterations
        #     self.update_BM()
        bar = progressbar.ProgressBar(max_value=self.free_voxel_empty)
        while (self.free_voxel > 0):
            self.update_BM()
            bar.update(self.free_voxel_empty - self.free_voxel)
            self.free_voxel -= 1
            self.iterations += 1

        trainer.draw_debug(reset=True)
        self.bm.init_batch()
        self.free_voxel = self.free_voxel_empty


class FCFinePokemonTrainer(FinePokemonTrainer):

    def init_BM(self):
        self.BM = du.HoneyBatcherPath
        self.images_counter = -1

    def define_loss(self):
        self.loss = self.builder.get_loss('updates_hydra_v8')

    def network_i_choose_you(self):
        network = self.builder.get_net(self.options.net_arch)
        c.use(self.options.gpu)
        layers, self.options.patch_len, _ = network()
        self.l_out = layers["l_out_cross"]

        # self.options.network_channels = layers['l_in_claims'].shape[1]

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
        heights = self.prediction_f(claims[:, :2], precomp_input)
        self.bm.update_priority_queue(heights, seeds, ids)
        return

    def train(self):
        self.bm.init_batch()
        bar = progressbar.ProgressBar(max_value=self.free_voxel_empty)
        inputs = self.update_BM_FC()
        # precompute fc part
        self.precomp_input = self.fc_prec_conv_body(inputs)
        self.images_counter += 1
        while (self.free_voxel > 0):
            self.iterations += 1
            self.free_voxel -= 1
            self.update_BM()
            # if self.iterations % self.observation_counter == 0:
            #     self.draw_debug(reset=False)

            if self.free_voxel % 100 == 0:
                bar.update(self.free_voxel_empty - self.free_voxel)

        self.bm.find_global_error_paths()
        if self.bm.count_new_path_errors() > 0:
            error_b_type1, error_b_type2, dir1, dir2 = \
                self.bm.reconstruct_path_error_inputs()

            batch_ft = exp.stack_batch(error_b_type1, error_b_type2)
            batch_dir_ft = exp.stack_batch(dir1, dir2)

            batch_ft = exp.flatten_stack(batch_ft).astype(np.float32)
            batch_dir_ft = exp.flatten_stack(batch_dir_ft).astype(np.int32)

            ft_loss_train, individual_loss_fine, _, heights = \
                    self.loss_train_fine_f(batch_ft[:, :2], batch_ft[:, 2:],
                                           batch_dir_ft)
            if np.any(individual_loss_fine <0):
                print 'any', min(individual_loss_fine)
            print 'loss ft', ft_loss_train
            # print 'error type II ', self.bm.error_II_type

            # zip(heights, self.bm.e1heights + self.bm.e2heights)
            # bs = len(heights) / 2
            # for err, heightpreve1, heightpreve2, heightrec1, heightrec2, \
            #     ind_loss, errt in \
            #         zip(self.bm.all_errorsq, self.bm.e1heights,
            #             self.bm.e2heights, heights[:bs], heights[bs:],
            #             individual_loss_fine, self.bm.error_II_type):

                # print 'error', err["batch"], 'e1 pos', err["e1_pos"], \
                #     err['e2_pos'], 'loss', ind_loss[0, 0, 0],\
                #     heightpreve1 - heightrec1[0, 0, 0], heightpreve2 - heightrec2[0, 0, 0], errt, \
                #     'plateau', err["plateau"]
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

        if self.images_counter % 5 == 0:
            self.save_net()
            trainer.draw_debug(reset=True)

        if self.free_voxel == 0:
            print 'init batch'
            # self.bm.draw_error_reconst('err_rec%i' %self.iterations)
            self.free_voxel = self.free_voxel_empty


class FCRecFinePokemonTrainer(FCFinePokemonTrainer):
    def init_BM(self):
        self.BM = du.HoneyBatcherRec
        self.images_counter = -1

    def update_BM(self, bm=None):
        if bm is None:
            bm = self.bm
        inputs, gt, centers, ids, hiddens = bm.get_batches()
        centers = np.array(centers, dtype=np.int)
        n_c_prec = self.precomp_input.shape[1]
        precomp_input_sliced = np.zeros((bm.bs, n_c_prec, 2, 2)).astype(np.float32)
        for b, seed in enumerate(centers):
            cross_x, cross_y, _ = bm.get_cross_coords(seed)
            # + 1 because we face the FOV for the BM + 2 because the cross is a inherent FC conv formulation
            precomp_input_sliced[b, :, :, :] = \
                            self.precomp_input[b, :, cross_x - bm.pad + 1,
                                                     cross_y - bm.pad + 1].swapaxes(0,1).reshape(n_c_prec, 2, 2)
        sequ_len = 1
        rnn_mask = np.ones((bm.bs*4, sequ_len), dtype=np.float32)
        hiddens = np.repeat(hiddens, 4, axis=0).astype(np.float32)

        # debug
        # hiddens[:] = 0
        # inputs[:, :2] = 0
        # inputs[:, 2:] = 0
        # precomp_input_sliced[:] = 0
        height_probs, hidden_out, _ = self.builder.probs_f_fc(inputs[:, :2], precomp_input_sliced, hiddens, rnn_mask, 1)
        ####
        # exit()
        # debug: is preprocessed input correctly reused
        # sum_hiddens = np.sum(np.abs(hiddens))
        # if sum_hiddens == 0:
        #     print 'hiddens is only 0'
        # d_height_probsd, d_hidden_outd, d_precomp_input_sliced = self.builder.probs_f(inputs[:, :2], inputs[:, 2:],
        #                                                                               hiddens, rnn_mask, 1)

        # hidden diff
        # print 'hidden d', hidden_out, 'd_height_probsd', d_hidden_outd
        # diff = np.abs(hidden_out - d_hidden_outd)
        # if np.max(diff) > 10**-4:
        #     coords = centers[int(np.where(diff == np.max(diff))[0][0]) / 4]
        #     # check for non boundary effects
        #     if self.bm.pad not in coords and self.bm.image_shape[-1] - self.bm.pad - 1 not in coords:
        #         print 'hiddens average deviation', np.mean(diff), 'max', np.max(diff)
        #
        # # height diff
        # diff = np.abs(d_height_probsd - height_probs)
        # if np.max(diff) > 10**-4:
        #     coords = centers[int(np.where(diff == np.max(diff))[0][0]) / 4]
        #     # check for non boundary effects
        #     if self.bm.pad not in coords and self.bm.image_shape[-1] - self.bm.pad - 1 not in coords:
        #         print 'av deviation', np.mean(diff), 'max dev', np.max(diff)
        #         print 'centers', coords


        hidden_new = hidden_out.reshape((bm.bs, 4, self.options.n_recurrent_hidden))
        height_probs = height_probs.reshape((bm.bs, 4))
        bm.update_priority_queue(height_probs, centers, ids, hidden_states=hidden_new)

    def validate(self):
        if self.options.val_options.quick_eval:
            self.val_bm.set_preselect_batches(range(self.val_bm.bs))
        self.val_bm.init_batch()
        inputs = self.val_bm.global_input_batch[:, :, :-1, :-1]
        self.precomp_input = self.builder.fc_prec_conv_body(inputs)

        total_free_voxel = self.val_bm.get_num_free_voxel()
        bar = progressbar.ProgressBar(max_value=total_free_voxel)
        for i in range(total_free_voxel):
            self.update_BM(bm=self.val_bm)
            bar.update(i)

        score, _ = vs.validate_segmentation(self.val_bm.global_claims[:,
                                            self.val_bm.pad:-self.val_bm.pad,
                                            self.val_bm.pad:-self.val_bm.pad],
                                            self.val_bm.global_label_batch)

        self.val_loss_history[0].append(1-score['Adapted Rand error'])
        self.val_loss_history[1].append(1-score['Adapted Rand error precision'])
        u.plot_train_val_errors([self.val_loss_history[0],
                                 self.val_loss_history[1]],
                                 range(len(self.val_loss_history[0])),
                                 self.save_net_path + '/validation.png',
                 names=['Adapted Rand error', 'Adapted Rand error precision'])
        for b in range(min(5, self.bm.bs)):
            self.val_bm.draw_debug_image("%i_validation_b_%03i_i_%08i_f_%i" % (b, 0, self.iterations, self.free_voxel),
                                        path=self.image_path_validation, b=b)
        return score

    def train(self):
        self.observation_counter = 500
        self.bm.init_batch()
        self.free_voxel = self.free_voxel_empty
        bar = progressbar.ProgressBar(max_value=self.free_voxel_empty)
        inputs = self.update_BM_FC()
        # precompute fc partf
        self.precomp_input = self.builder.fc_prec_conv_body(inputs)

        self.images_counter += 1
        while (self.free_voxel > 0):
            self.iterations += 1
            self.free_voxel -= 1
            self.update_BM()
            # if self.iterations % self.observation_counter == 0:
            #     self.draw_debug(reset=False)

            if self.free_voxel % 100 == 0:
                bar.update(self.free_voxel_empty - self.free_voxel)

        print 'hiddens mean', np.mean(self.bm.global_hidden_states), 'max', np.max(np.abs(self.bm.global_hidden_states))
        print
        self.bm.find_global_error_paths()
        if self.bm.count_new_path_errors() > 0:
            error_b_type1, error_b_type2, rnn_mask_e1, rnn_mask_e2, rnn_hiddens_e1, rnn_hiddens_e2 = \
                self.bm.reconstruct_path_error_inputs(backtrace_length=options.backtrace_length)

            batch_mask_ft = exp.flatten_stack(exp.stack_batch(rnn_mask_e1, rnn_mask_e2)).astype(np.float32)
            batch_inits = exp.flatten_stack(exp.stack_batch(rnn_hiddens_e1, rnn_hiddens_e2)).astype(np.float32)
            batch_ft = exp.flatten_stack(exp.stack_batch(error_b_type1, error_b_type2)).astype(np.float32)

            sequ_len = self.options.backtrace_length
            print 'sequ len', sequ_len, 'batch ft', batch_ft.shape, 'hidden', batch_inits.shape, \
                'rnn mask', batch_mask_ft.shape
            ft_loss_train, individual_loss_fine, heights, ft_loss_noreg, stat_conv, dyn_conv, hiddens_rec = \
                    self.builder.loss_train_fine_f(batch_ft[:, :2, :, :], batch_ft[:, 2:, :, :], batch_inits,
                                                   batch_mask_ft, options.backtrace_length)
            self.draw_loss(ft_loss_train, ft_loss_noreg)

            # self.debug_plots(heights, batch_mask_ft, hiddens_rec)

        if self.images_counter % 1 == 0:
            self.save_net()
            trainer.draw_debug(reset=True)
        # exit();

        if self.free_voxel == 0:
            self.free_voxel = self.free_voxel_empty

    def draw_loss(self, ft_loss_train, ft_loss_noreg):
        self.update_history.append(self.iterations)
        self.loss_history[0].append(ft_loss_train)
        self.loss_history[1].append(ft_loss_noreg)
        u.plot_train_val_errors([self.loss_history[0], self.loss_history[1]], self.update_history,
                                self.save_net_path + '/ft_training.png', names=['loss', 'loss no reg'],
                                log_scale=False)

    def debug_plots(self, heights, masks, hiddens):
        masks = np.array(masks.reshape((2, -1)), dtype=np.bool)
        heights = heights.reshape((2, -1))
        print 'shapes', heights.shape, masks.shape,  self.bm.error_selections.shape
        for i, (ei, heights, mask, err_selection) in enumerate(zip(['e1', 'e2'], heights, masks,
                                                                   self.bm.error_selections)):
            err_selection = np.array(err_selection)[mask]
            all_ei_pos = np.array([err[ei + '_pos'] for err in err_selection])
            all_b = [err['batch'] for err in err_selection]
            dirs = self.bm.global_directionmap_batch[all_b, all_ei_pos[:, 0] - self.bm.pad,
                                                            all_ei_pos[:, 1] - self.bm.pad]

            all_orig_h = self.bm.global_prediction_map_nq[all_b, all_ei_pos[:, 0] - self.bm.pad,
                                                                 all_ei_pos[:, 1] - self.bm.pad, dirs]
            all_rec_h = heights[mask]
            print 'height diff', ei
            print all_orig_h - all_rec_h
        exit()
        # self.bm.draw_batch(batch_ft, 'batch_tmp')
        # ts = self.options.backtrace_length
        n_err = len(self.bm.error_selections[0]) / ts
        all_e1 = np.array(self.bm.error_selections[0]).reshape(n_err, ts)
        all_e2 = np.array(self.bm.error_selections[1]).reshape(n_err, ts)
        all_h1 = heights[:n_err * ts].reshape(n_err, ts)
        all_h2 = heights[n_err * ts:].reshape(n_err, ts)
        # height diffs, hiddens, hiddens rec, stat_convs
        e1_pos = []

    def debug_plot_e_i(self):

        print
        k = -1
        for e_type, all_type_i_errs, all_type_i_hs in zip(['e1', 'e2'], [all_e1, all_e2], [all_h1, all_h2]):
            k += 1
            for i, (sequ_errs, sequ_h) in enumerate(zip(all_type_i_errs, all_type_i_hs)):
                for j, (err_t, h_t) in enumerate(zip(sequ_errs, sequ_h)):
                    pos = err_t[e_type + '_pos']
                    b = err_t['batch']
                    dir = self.bm.global_directionmap_batch[b, pos[0] - self.bm.pad, pos[1] - self.bm.pad]
                    if dir < 0:
                        continue

                    old_pos = self.bm.update_position(pos, dir)

                    # conv check
                    orig_conv = self.precomp_input[b, :, pos[0] - self.bm.pad + 1, pos[1] - self.bm.pad + 1]
                    diff = np.abs(orig_conv - stat_conv[i * ts + j + k * n_err * ts, :, 0, 0])
                    verbose = False
                    if np.max(diff) > 10 ** -4:
                        verbose = True
                        print 'conv comparison', np.max(diff), np.mean(diff), 'where', np.where(diff == np.max(diff))
                        print 'orig conv', orig_conv[:5], 'rconst conv', stat_conv[j, :5, 0, 0]

                    err_h = self.bm.global_prediction_map_nq[b, pos[0] - self.bm.pad, pos[1] - self.bm.pad, dir]
                    diff = h_t - err_h
                    if diff > 10 ** -4:
                        verbose = True
                        print 'height differences %.3f err h %.3f, diff %.3f hiddens max %.3f' \
                              % (h_t, err_h, (h_t - err_h), np.max(self.hiddens_rec))

                    # input check
                    old_stat = self.bm.crop_input(pos, b)[None, ...][0, :, 1:-1, 1:-1]
                    new_stat = batch_ft[i * ts + j + k * n_err * ts, 2:, :, :]
                    stat_inp_diff = np.mean((old_stat - new_stat).astype(np.float))
                    if stat_inp_diff > 10 ** -4:
                        verbose = True
                        print 'static inputs equals', np.mean(stat_inp_diff)
                    if verbose:
                        print 'type', e_type, 'pos', pos, 'b', b, 'orig pos', old_pos, 'dir', dir
                        if self.bm.pad in pos or self.bm.image_shape[-1] - self.bm.pad - 1 in pos:
                            print 'boundary case.............'
                        print


class FCERecFinePokemonTrainer(FCRecFinePokemonTrainer):
    def init_BM(self):
        self.BM = du.HoneyBatcherERec
        self.images_counter = -1


class StalinTrainer(FCFinePokemonTrainer):
    """
    Cold war training: Only increase height
    """
    def define_loss(self):
        self.loss = self.builder.get_loss('updates_hydra_coldwar')


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
        print 'Gotta Catchem All'
        super(GottaCatchemAllTrainer, self).__init__(options)
        self.update_steps = 1
        self.observation_counter = 200
        self.loss_history = [[], []]

    def init_BM(self):
        self.BM = du.HoneyBatcherPath

    def update_BM(self):
        self.bm.init_batch()
        inputs = self.bm.global_input_batch[:, :, :-1, :-1]
        heights_gt = self.bm.global_height_gt_batch[:, None, :, :]
        self.bm.edge_map_gt = heights_gt
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

        if self.iterations % 500 == 0:
            self.save_net()

        # update parameters once
        self.update_history.append(self.iterations)
        self.loss_history[0].append(loss_train)
        self.loss_history[1].append(loss_no_reg)
        u.plot_train_val_errors([self.loss_history[0], self.loss_history[1]], self.update_history,
                                self.save_net_path + '/training.png', names=['loss', 'loss no reg'])
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


class RecurrentTrainer(FCFinePokemonTrainer):

    def init_BM(self):
        self.BM = du.HoneyBatcherERec
        self.images_counter = -1

    def converged(self):
        return False

    def train(self):
        out_conv_f, out_rec_f = self.fcts
        print out_conv_f(self.input).shape
        print out_rec_f(self.input, self.hid_init).shape


if __name__ == '__main__':
    options = get_options()
    # pret
    if options.net_arch == 'net_v8_dilated':
        trainer = GottaCatchemAllTrainer(options)
        while not trainer.converged():
            print "\r pretrain %0.4f iteration %i free voxel %i" \
                  %(trainer.train(), trainer.iterations, trainer.free_voxel),
        trainer.save_net(path=trainer.net_param_path, name='pretrain_final.h5')

    elif options.net_arch == 'v8_hydra_dilated_ft_joint':
        options.fc_prec = True
        trainer = FCRecFinePokemonTrainer(options)
        trainer.bm.set_preselect_batches([0, 7])

        epoch = 0
        while not trainer.converged():
            trainer.train()
            trainer.save_net()
            if trainer.val_bm is not None and epoch % 10 == 0:
                trainer.validate()
            epoch += 1

        trainer.save_net(path=trainer.net_param_path, name='pretrain_final.h5')
    # elif options.net_arch == 'v8_hydra_dilated_ft_joint':
    #     # from pycallgraph import PyCallGraph
    #     # from pycallgraph.output import GraphvizOutput
    #     # with PyCallGraph(output=GraphvizOutput()):
    #     # trainer = SpeedyPokemonTrainer(options)
    #     trainer = FCFinePokemonTrainer(options)
    #     while not trainer.converged():
    #         trainer.train()
    #         trainer.save_net()
    print 'ende'
    # COLD WAR TEST 

    # trainer = StalinTrainer(options)
    # while not trainer.converged():
    #     trainer.train()
    #     trainer.save_net(path=trainer.net_param_path, name='pretrain_final.h5')
 


    # COLD WAR TEST 


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
