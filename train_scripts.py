import matplotlib
matplotlib.use('Agg')
import os
from theano import tensor as T
import theano
import lasagne
import utils as u
from matplotlib import pyplot as plt
import nets
import dataset_utils as du
import numpy as np
from theano.sandbox import cuda as c



def train_script_v1():
    print 'train script v1'
    # data params:
    # for each net a new folder is created. Here intermediate pred-
    # dictions and train, val... are saved
    save_net_b = True
    load_net_b = False

    net_name = 'trash_mc'
    label_path = './data/volumes/label_a.h5'
    label_path_val = './data/volumes/label_b.h5'
    height_gt_path = './data/volumes/height_a.h5'
    height_gt_key = 'height'
    height_gt_path_val = './data/volumes/height_b.h5'
    height_gt_key_val = 'height'
    raw_path = './data/volumes/raw_a.h5'
    raw_path_val = './data/volumes/raw_b.h5'
    membrane_path = './data/volumes/membranes_a.h5'
    membrane_path_val = './data/volumes/membranes_b.h5'
    save_net_path = './data/nets/' + net_name + '/'
    load_net_path = './data/nets/rough/net_2500000'      # if load true
    load_net_path = './data/nets/cnn_ID_2/net_300000'      # if load true
    load_net_path = './data/nets/cnn_path_v1_fine_tune/net_500000.h5'      # if load true

    tmp_path = '/media/liory/ladata/bla'        # debugging
    batch_size = 2         # > 4
    batch_size_ft = 2
    global_edge_len = 300
    gt_seeds_b = False
    find_errors = True
    fine_tune_b = find_errors


    # training parameter
    c.use('gpu0')
    pre_train_iter = 1000000
    max_iter = 10000000000
    save_counter = 100000        # save every n iterations
    # fine tune


    # choose your network from nets.py
    regularization = 10**-4
    network = nets.build_ID_v5_hydra
    loss = nets.loss_updates_probs_v0
    loss_fine = nets.loss_updates_hydra_v5

    # all params entered.......................

    # initialize the net
    print 'initializing network graph for net ', net_name
    target_t = T.ftensor4()

    l_in, l_in_direction, l_out, l_out_direction, patch_len = network()

    print 'compiling theano functions'
    loss_train_f, loss_valid_f, probs_f = \
        loss(l_in, target_t, l_out, L1_weight=regularization)

    if fine_tune_b:
        print 'compiling theano finetuningfunctions'
        loss_train_fine_f, loss_valid_fine_f, probs_fine_f = \
            loss_fine(l_in, l_in_direction, l_out_direction,
                          L1_weight=regularization)
        Memento1 = du.BatchMemento(batch_size_ft, 100)
        Memento2 = du.BatchMemento(batch_size_ft, 100)

    print 'Loading data and Priority queue init'
    bm = du.BatchManV0(membrane_path, label_path,
                       height_gt=height_gt_path,
                       height_gt_key=height_gt_key,
                       raw=raw_path,
                       batch_size=batch_size,
                       patch_len=patch_len, global_edge_len=global_edge_len,
                       padding_b=False,
                       find_errors=find_errors and fine_tune_b,
                       gt_seeds_b=gt_seeds_b)

    bm.init_train_path_batch()
    bm_val = du.BatchManV0(membrane_path_val, label_path_val,
                           height_gt=height_gt_path_val,
                           height_gt_key=height_gt_key_val,
                           raw=raw_path_val,
                           batch_size=batch_size,
                           patch_len=patch_len, global_edge_len=global_edge_len,
                           padding_b=False, gt_seeds_b=gt_seeds_b)

    bm_val.init_train_path_batch()  # Training

    # init a network folder where all images, models and hyper params are stored
    if save_net_b:
        if not os.path.exists(save_net_path):
            os.mkdir(save_net_path)
            os.mkdir(save_net_path + '/images')

    if load_net_b:
        print "loading network parameters from ",load_net_path
        u.load_network(load_net_path, l_out)

    # everything is initialized now train and predict every once in a while....
    converged = False       # placeholder, this is not yet implemented
    iteration = -1
    losses = [[], [], []]
    fine_tune_losses = [[], []]
    iterations = []
    ft_iteration = 0

    free_voxel_empty = (global_edge_len - patch_len)**2
    free_voxel = free_voxel_empty
    print 'training'
    while not converged and (iteration < max_iter):
        iteration += 1
        free_voxel -= 1

        if iteration % save_counter == 0 and save_net_b:
            u.save_network(save_net_path, l_out, 'net_%i' % iteration)

        # predict val
        membrane_val, gt, seeds_val, ids_val = bm_val.get_path_batches()
        probs_val = probs_f(membrane_val)
        bm_val.update_priority_path_queue(probs_val, seeds_val, ids_val)

        # predict train
        membrane, gt, seeds, ids = bm.get_path_batches()
        probs = probs_f(membrane)
        bm.update_priority_path_queue(probs, seeds, ids)

        # update height difference errors
        if iteration % 100 == 0:
            if (len(bm.global_error_dict) >= batch_size_ft or \
                            free_voxel < 101) and iteration > pre_train_iter \
                    and fine_tune_b:
                error_b_type1, error_b_type2, dir1, dir2 = \
                    bm.reconstruct_path_error_inputs()
                Memento1.add_to_memory(error_b_type1, dir1)
                Memento2.add_to_memory(error_b_type2, dir2)
                if save_net_b:
                    # plot train images
                    bm.draw_debug_image("train_iteration_iter_%08i_counter_%i" %
                                        (iteration, bm.counter),
                                        path=save_net_path + '/images/')
                    bm_val.draw_debug_image("val_iteration_%08i" % iteration,
                                            path=save_net_path + '/images/')
                bm.init_train_path_batch()
                bm_val.init_train_path_batch()
                free_voxel = free_voxel_empty

        if (free_voxel <= 100 or (free_voxel_empty / 4)  % (iteration + 1) == 0)\
                and free_voxel_empty - free_voxel > 10000 and not fine_tune_b:
            bm.draw_debug_image(
                "reset_train_iteration_%08i_counter_%i_freevoxel_%i" %
                (iteration, bm.counter, free_voxel),
                path=save_net_path + '/images/')
            bm_val.draw_debug_image(
                "reset_val_iteration_%08i_counter_%i_freevoxel_%i" %
                (iteration, bm.counter, free_voxel),
                path=save_net_path + '/images/')
            bm.init_train_path_batch()
            bm_val.init_train_path_batch()
            free_voxel = free_voxel_empty

        # clear memory, do fine-tuning and reset image
        if fine_tune_b and Memento1.is_ready():
            ft_iteration += 1
            print 'Finetuning...'
            batch_ft_t1, dir_t1 = Memento1.get_batch()
            batch_ft_t2, dir_t2 = Memento2.get_batch()
            batch_ft = np.concatenate((batch_ft_t1, batch_ft_t2), axis=0)
            batch_dir_ft = np.concatenate((dir_t1, dir_t2), axis=0)

            ft_loss_train_noreg = loss_valid_fine_f(batch_ft, batch_dir_ft)
            probs_fine = probs_fine_f(batch_ft, batch_dir_ft)
            ft_loss_train = loss_train_fine_f(batch_ft, batch_dir_ft)
            print "loss_train_fine %.4f" % ft_loss_train
            fine_tune_losses[0].append(ft_loss_train_noreg)
            fine_tune_losses[1].append(ft_loss_train)

            u.plot_train_val_errors(fine_tune_losses, range(ft_iteration),
                                    save_net_path + 'ft_training.png',
                                    ['ft loss no reg no dropout', 'ft loss'])

            Memento1.clear_memory()
            Memento2.clear_memory()
            bm_val.init_train_path_batch()
            bm.init_train_path_batch()
            free_voxel = free_voxel_empty

        if iteration % 10 == 0 and iteration < pre_train_iter:
            loss_train = float(loss_train_f(membrane, gt))

        # monitor training and plot loss
        if iteration % 1000 == 0 and (iteration < pre_train_iter or not
        fine_tune_b):
            loss_train_no_reg = float(loss_valid_f(membrane, gt))
            loss_valid = float(loss_valid_f(membrane_val, gt))
            print '\r loss train %.4f, loss train_noreg %.4f, ' \
                  'loss_validation %.4f, iteration %i' % \
                  (loss_train, loss_train_no_reg, loss_valid, iteration),
            if save_net_b:
                iterations.append(iteration)
                losses[0].append(loss_train)
                losses[1].append(loss_train_no_reg)
                losses[2].append(loss_valid)
                u.plot_train_val_errors(
                    losses,
                    iterations,
                    save_net_path + 'training.png',
                    names=['loss train', 'loss train no reg', 'loss valid'])

        # monitor growth on validation set tmp debug change train to val
        if iteration % 10000 == 0:
            bm.draw_debug_image("train_iteration_%08i_counter_%i_freevoxel_%i" %
                                (iteration, bm.counter, free_voxel),
                                path=save_net_path + '/images/')


if __name__ == '__main__':
    train_script_v1()
