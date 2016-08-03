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
    load_net_b = True

    net_name = 'cnn_path_v1_tune_trash'
    label_path = './data/volumes/label_a.h5'
    label_path_val = './data/volumes/label_b.h5'
    height_gt_path = './data/volumes/height_a.h5'
    height_gt_key = 'height'
    height_gt_path_val = './data/volumes/height_b.h5'
    height_gt_key_val = 'height'
    raw_path = './data/volumes/membranes_a.h5'
    raw_path_val = './data/volumes/membranes_b.h5'
    save_net_path = './data/nets/' + net_name + '/'
    load_net_path = './data/nets/rough/net_2500000'      # if load true
    load_net_path = './data/nets/cnn_ID_2/net_300000'      # if load true
    load_net_path = './data/net_2500000'      # if load true

    tmp_path = '/media/liory/ladata/bla'        # debugging
    batch_size = 16         # > 4
    batch_size_ft = 10
    global_edge_len = 300
    gt_seeds_b = False
    find_errors = True

    # training parameter
    c.use('gpu0')
    max_iter = 1000000000
    save_counter = 100000        # save every n iterations

    # choose your network from nets.py
    regularization = 10**-4
    network = nets.build_ID_v0_hydra
    loss = nets.loss_updates_probs_v0
    loss_fine = nets.loss_updates_hydra_v0

    # all params entered.......................

    # initialize the net
    print 'initializing network graph for net ', net_name
    target_t = T.ftensor4()

    l_in, l_in_direction, l_out, l_out_direction, patch_len = network()

    print 'compiling theano functions'
    loss_train_f, loss_valid_f, probs_f = \
        loss(l_in, target_t, l_out, L1_weight=regularization)

    print 'compiling theano finetuningfunctions'
    loss_train_fine_f, loss_valid_fine_f, probs_fine_f = \
        loss_fine(l_in, l_in_direction, l_out_direction, L1_weight=regularization)

    debug_f = theano.function([l_in.input_var, l_in_direction.input_var],
                [lasagne.layers.get_output(l_out, deterministic=True),
                lasagne.layers.get_output(l_out_direction, deterministic=True)])

    Memento1 = du.BatchMemento(batch_size_ft, 100)
    Memento2 = du.BatchMemento(batch_size_ft, 100)

    print 'Loading data and Priority queue init'
    bm = du.BatchManV0(raw_path, label_path,
                       height_gt=height_gt_path,
                       height_gt_key=height_gt_key,
                       batch_size=batch_size,
                       patch_len=patch_len, global_edge_len=global_edge_len,
                       padding_b=False,
                       find_errors=find_errors,
                       gt_seeds_b=gt_seeds_b)

    bm.init_train_path_batch()
    bm_val = du.BatchManV0(raw_path_val, label_path_val,
                           height_gt=height_gt_path_val,
                           height_gt_key=height_gt_key_val,
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
    iterations = []
    r1_shape_old = 0

    free_voxel_emtpy = (global_edge_len - patch_len)**2
    free_voxel = free_voxel_emtpy
    print 'training'
    while not converged and (iteration < max_iter):
        iteration += 1
        free_voxel -= 1
        # save image and update global field ground

        if iteration % save_counter == 0 and save_net_b:
            u.save_network(save_net_path, l_out, 'net_%i' % iteration)

        raw_val, gt, seeds_val, ids_val = bm_val.get_path_batches()
        probs_val = probs_f(raw_val)
        bm_val.update_priority_path_queue(probs_val, seeds_val, ids_val)

        # train da thing
        raw, gt, seeds, ids = bm.get_path_batches()
        probs = probs_f(raw)
        bm.update_priority_path_queue(probs, seeds, ids)

        if iteration % 100 == 0:
            if len(bm.global_error_dict) >= batch_size_ft or \
                            free_voxel < 101:
                error_b_type1, error_b_type2, dir1, dir2 = \
                    bm.reconstruct_path_error_inputs()
                print error_b_type2.shape
                print 'error shapes', error_b_type1.shape
                Memento1.add_to_memory(error_b_type1, dir1)
                Memento2.add_to_memory(error_b_type2, dir2)
                print "resetting",free_voxel,free_voxel_emtpy
                bm.init_train_path_batch()
                bm_val.init_train_path_batch()
                free_voxel = free_voxel_emtpy
                if save_net_b:
                    # plot train images
                    bm.draw_debug_image("train_iteration_" + str(iteration),
                                        path=save_net_path + '/images/')
                    bm_val.draw_debug_image("val_iteration_" + str(iteration),
                                            path=save_net_path + '/images/')

        if Memento1.is_ready():
            print 'Memento is ready ... finetuning...'
            batch_ft_t1, dir_t1 = Memento1.get_batch()
            batch_ft_t2, dir_t2 = Memento2.get_batch()
            batch_ft = np.concatenate((batch_ft_t1, batch_ft_t2), axis=0)
            batch_dir_ft = np.concatenate((dir_t1, dir_t2), axis=0)
            loss_train_fine = float(loss_train_fine_f(batch_ft, batch_dir_ft))
            print "loss_train_fine %.4f" % loss_train_fine
            Memento1.clear_memory()
            Memento2.clear_memory()
            bm_val.init_train_path_batch()
            bm.init_train_path_batch()
            free_voxel = free_voxel_emtpy

        # if iteration % 10 == 0 and iteration < 50000:
        #     loss_train = float(loss_train_f(raw, gt))

        # if iteration % 1000 == 0:
        #     loss_train_no_reg = float(loss_valid_f(raw, gt))
        #     loss_valid = float(loss_valid_f(raw_val, gt))
        #     print '\r loss train %.4f, loss train_noreg %.4f, ' \
        #           'loss_validation %.4f, iteration %i' % \
        #           (loss_train, loss_train_no_reg, loss_valid, iteration),
        #     if save_net_b:
        #         iterations.append(iteration)
        #         losses[0].append(loss_train)
        #         losses[1].append(loss_train_no_reg)
        #         losses[2].append(loss_valid)
        #         u.plot_train_val_errors(
        #             losses,
        #             iterations,
        #             save_net_path + 'training.png',
        #             names=['loss train', 'loss train no reg', 'loss valid'])
    #
        # monitor growing on validation set
        if iteration % 1000 == 0:
            # print "free_voxel ",free_voxel
            # print "errors",np.sum(bm.global_errormap)
            print "Memento size ",Memento1.counter
            bm.draw_debug_image("train_iteration_%i_freevoxel_%i" %
                                (iteration, free_voxel),
                                path=save_net_path + '/images/')



if __name__ == '__main__':
    train_script_v1()