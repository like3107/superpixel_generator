import matplotlib
matplotlib.use('Agg')
import os
from theano import tensor as T
import utils as u
from matplotlib import pyplot as plt
import nets
import dataset_utils as du
import numpy as np
from theano.sandbox import cuda as c


def train_script_v0():

    # data params:
    # for each net a new folder is created. Here intermediate pred-
    # dictions and train, val... are saved
    save_net_b = True
    load_net_b = False

    net_name = 'cnn_ID_2'
    label_path = './data/volumes/label_a.h5'
    label_path_val = './data/volumes/label_b.h5'
    raw_path = './data/volumes/membranes_a.h5'
    raw_path_val = './data/volumes/membranes_b.h5'
    save_net_path = './data/nets/' + net_name + '/'
    load_net_path = './data/nets/cnn_v1/net_10000'      # if load true
    tmp_path = '/media/liory/ladata/bla'        # debugging
    batch_size = 16         # > 4
    global_edge_len = 300

    # training parameter
    c.use('gpu0')
    max_iter = 1000000000
    save_counter = 100000        # save every n iterations
    # iterations until all pixels on image predicted before that stops early
    # grows linear until n_pixels of field starting at global field change
    global_field_change = 300
    iterations_to_max = 10000000

    # choose your network from nets.py
    network = nets.build_ID_v0

    # all params entered.......................

    # initialize the net
    print 'initializing network graph for net ', net_name
    target_t = T.ftensor4()
    l_in, l_out, patch_len = network()

    print 'compiling theano functions'
    loss_train_f, loss_valid_f, probs_f = \
        nets.loss_updates_probs_v0(l_in, target_t, l_out)

    print 'Loading data and Priority queue init'
    bm = du.BatchManV0(raw_path, label_path, batch_size=batch_size,
                       patch_len=patch_len, global_edge_len=global_edge_len,
                       padding_b=True)
    bm.init_train_batch()
    bm_val = du.BatchManV0(raw_path_val, label_path_val, batch_size=batch_size,
                           patch_len=patch_len, global_edge_len=global_edge_len,
                           padding_b=True)

    bm_val.init_train_batch()  # Training

    # init a network folder where all images, models and hyper params are stored
    if save_net_b:
        if not os.path.exists(save_net_path):
            os.mkdir(save_net_path)
            os.mkdir(save_net_path + '/images')

    if load_net_b:
        u.load_network(load_net_path, l_out)

    # everything is initialized now train and predict every once in a while....
    converged = False       # placeholder, this is not yet implemented
    global_field_counter = 0
    iteration = -1
    losses = [[], [], []]
    iterations = []

    while not converged and (iteration < max_iter):
        iteration += 1
        global_field_counter += 1

        # save image and update global field ground
        if global_field_counter % global_field_change == 0:
            if save_net_b:
                # plot train images
                u.save_3_images(
                    bm.global_claims[4, bm.pad:-bm.pad-1, bm.pad:-bm.pad-1],
                    bm.global_batch[4, bm.pad:-bm.pad-1, bm.pad:-bm.pad-1],
                    bm.global_label_batch[4, :, :],
                    save_net_path + '/images/',
                    iterations_per_image=global_field_counter, name='train',
                    iteration=iteration)
                # # # plot valid images
                u.save_3_images(
                    bm_val.global_claims[4, bm_val.pad:-bm_val.pad - 1,
                                         bm_val.pad:-bm_val.pad - 1],
                    bm_val.global_batch[4, bm_val.pad:-bm_val.pad - 1,
                                        bm_val.pad:-bm_val.pad - 1],
                    bm_val.global_label_batch[4, :, :],
                    save_net_path + '/images/',
                    iterations_per_image=global_field_counter, name='valid',
                    iteration=iteration)
                global_field_change = \
                    u.linear_growth(iteration,
                                    maximum=(global_edge_len - patch_len)**2-100,
                                    y_intercept=global_field_change,
                                    iterations_to_max=iterations_to_max)

                # print '\n global field change', global_field_change

            print '\r new global batch loaded', global_field_counter, \
                global_field_change,
            bm.init_train_batch()
            bm_val.init_train_batch()
            global_field_counter = 0

        if iteration % save_counter == 0 and save_net_b:
            u.save_network(save_net_path, l_out, 'net_%i' % iteration)

        # train da thing
        raw, gt, seeds, ids = bm.get_batches()
        probs = probs_f(raw)
        if iteration % 10 == 0:
            loss_train = float(loss_train_f(raw, gt))
        bm.update_priority_queue(probs, seeds, ids)

        # monitor growing on validation set
        raw_val, gt_val, seeds_val, ids_val = bm_val.get_batches()
        probs_val = probs_f(raw_val)
        bm_val.update_priority_queue(probs_val, seeds_val, ids_val)

        if iteration % 100 == 0:
            loss_valid = float(loss_valid_f(raw_val, gt_val))
            loss_train_no_reg = float(loss_valid_f(raw, gt))
            print '\r loss train %.4f, loss train_noreg %.4f, ' \
                  'loss_validation %.4f, iteration %i' % \
                  (loss_train, loss_train_no_reg, loss_valid, iteration),

            iterations.append(iteration)
            losses[0].append(loss_train)
            losses[1].append(loss_train_no_reg)
            losses[2].append(loss_valid)
            u.plot_train_val_errors(losses,
                                    iterations,
                                    save_net_path + 'training.png',
                                    names=['loss train', 'loss train no reg',
                                           'loss valid'])

            # debug
            # f, ax = plt.subplots(1, 3)
            # ax[0].imshow(bm.global_claims[4, bm.pad:-bm.pad, bm.pad:-bm.pad],
            #              interpolation='none', cmap=u.random_color_map())
            # ax[1].imshow(bm.global_batch[4, 0, bm.pad:-bm.pad, bm.pad:-bm.pad],
            #              cmap='gray')
            # print bm.global_label_batch.shape
            # ax[2].imshow(bm.global_label_batch[4, 0, bm.pad:-bm.pad, bm.pad:-bm.pad],
            #              interpolation='none', cmap=u.random_color_map())
            #
            # print 'gt', gt[4]
            # plt.savefig(tmp_path)
            # plt.close()
            #
            #
            # f, ax = plt.subplots(1, 3)
            # ax[0].imshow(raw[4, 0], cmap='gray')
            # ax[1].imshow(raw[4, 1], cmap=u.random_color_map(),
            #              interpolation='none')
            # ax[2].imshow(gt[4, :, :, 0], cmap=u.random_color_map(),
            #              interpolation='none')
            # plt.savefig(tmp_path + str(2))
            # plt.close()



def train_script_v1():
    print 'train script v1'
    # data params:
    # for each net a new folder is created. Here intermediate pred-
    # dictions and train, val... are saved
    save_net_b = True
    load_net_b = False

    net_name = 'cnn_ID2_trash'
    label_path = './data/volumes/label_a.h5'
    label_path_val = './data/volumes/label_b.h5'
    height_gt_path = './data/volumes/height_a.h5'
    height_gt_key = 'height'
    height_gt_path_val = './data/volumes/height_b.h5'
    height_gt_key_val = 'height'
    raw_path = './data/volumes/membranes_a.h5'
    raw_path_val = './data/volumes/membranes_b.h5'
    save_net_path = './data/nets/' + net_name + '/'
    load_net_path = './data/nets/cnn_ID_2/net_300000'      # if load true
    tmp_path = '/media/liory/ladata/bla'        # debugging
    batch_size = 16         # > 4
    global_edge_len = 300

    # training parameter
    c.use('gpu0')
    max_iter = 1000000000
    save_counter = 100000        # save every n iterations
    # iterations until all pixels on image predicted before that stops early
    # grows linear until n_pixels of field starting at global field change
    global_field_change = 300
    iterations_to_max = 100000

    # choose your network from nets.py
    network = nets.build_ID_v0

    # all params entered.......................

    # initialize the net
    print 'initializing network graph for net ', net_name
    target_t = T.ftensor4()
    l_in, l_out, patch_len = network()

    print 'compiling theano functions'
    loss_train_f, loss_valid_f, probs_f = \
        nets.loss_updates_probs_v0(l_in, target_t, l_out)

    print 'Loading data and Priority queue init'
    bm = du.BatchManV0(raw_path, label_path,
                       height_gt=height_gt_path,
                       height_gt_key=height_gt_key,
                       batch_size=batch_size,
                       patch_len=patch_len, global_edge_len=global_edge_len,
                       padding_b=False)
    bm.init_train_heightmap_batch()
    bm_val = du.BatchManV0(raw_path_val, label_path_val,
                           height_gt=height_gt_path_val,
                           height_gt_key=height_gt_key_val,
                           batch_size=batch_size,
                           patch_len=patch_len, global_edge_len=global_edge_len,
                           padding_b=False)

    bm_val.init_train_heightmap_batch()  # Training

    # init a network folder where all images, models and hyper params are stored
    if save_net_b:
        if not os.path.exists(save_net_path):
            os.mkdir(save_net_path)
            os.mkdir(save_net_path + '/images')

    if load_net_b:
        u.load_network(load_net_path, l_out)

    # everything is initialized now train and predict every once in a while....
    converged = False       # placeholder, this is not yet implemented
    global_field_counter = 0
    iteration = -1
    losses = [[], [], []]
    iterations = []
    print 'training'
    while not converged and (iteration < max_iter):
        iteration += 1
        global_field_counter += 1

        # save image and update global field ground
        if global_field_counter % global_field_change == 0:
            if save_net_b:
                # plot train images
                bms, names = [bm, bm_val], ['train', 'val']
                for b, name in zip(bms, names):


                    plot_images = []
                    plot_images.append({"title":"Claims",
                                        'cmap':"rand",
                                        'im':b.global_claims[4, b.pad:-b.pad-1, b.pad:-b.pad-1]})
                    plot_images.append({"title":"Raw Input",
                                        'im':b.global_batch[4, b.pad:-b.pad-1, b.pad:-b.pad-1]})
                    plot_images.append({"title":"Heightmap Prediciton",
                                        'im':b.global_heightmap_batch[4, b.pad:-b.pad-1, b.pad:-b.pad-1]})
                    plot_images.append({"title":"Heightmap Ground Truth",
                                        'im':b.global_height_gt_batch[4, b.pad:-b.pad-1, b.pad:-b.pad-1]})
                    plot_images.append({"title":"Direction Map",
                                        "cmap":"rand",
                                        'im':b.global_directionmap_batch[4, b.pad:-b.pad-1, b.pad:-b.pad-1]})
                    u.save_image_sub(plot_images,
                                  path=save_net_path + '/images/',
                                  name="iteration"+'_it%07d_im%07d' % (iteration, global_field_counter))

                global_field_change = \
                    u.linear_growth(iteration,
                                    maximum=(global_edge_len - patch_len)**2-100,
                                    y_intercept=global_field_change,
                                    iterations_to_max=iterations_to_max)

                # print '\n global field change', global_field_change

            print '\r new global batch loaded', global_field_counter, \
                global_field_change,
            bm.init_train_heightmap_batch()
            bm_val.init_train_heightmap_batch()
            global_field_counter = 0

        if iteration % save_counter == 0 and save_net_b:
            u.save_network(save_net_path, l_out, 'net_%i' % iteration)

        raw_val, gt_val, seeds_val, ids_val = bm_val.get_heightmap_batches()
        probs_val = probs_f(raw_val)
        bm_val.update_priority_queue(probs_val, seeds_val, ids_val)

        # train da thing
        raw, gt, seeds, ids = bm.get_heightmap_batches()
        probs = probs_f(raw)
        bm.update_priority_queue(probs, seeds, ids)
        if iteration % 10 == 0:
            loss_train = float(loss_train_f(raw, gt))

        # monitor growing on validation set
        if iteration % 400 == 0:
            loss_valid = float(loss_valid_f(raw_val, gt_val))
            loss_train_no_reg = float(loss_valid_f(raw, gt))
            print '\r loss train %.4f, loss train_noreg %.4f, ' \
                   'loss_validation %.4f, iteration %i' % \
                   (loss_train, loss_train_no_reg, loss_valid, iteration),

            if save_net_b:
                iterations.append(iteration)
                losses[0].append(loss_train)
                losses[1].append(loss_train_no_reg)
                losses[2].append(loss_valid)
                u.plot_train_val_errors(losses,
                                         iterations,
                                         save_net_path + 'training.png',
                                         names=['loss train', 'loss train no reg',
                                                'loss valid'])



            # f, ax = plt.subplots(1, 3)
            # ax[0].imshow(bm.global_claims[4, bm.pad:-bm.pad, bm.pad:-bm.pad],
            #              interpolation='none', cmap=u.random_color_map())
            # ax[1].imshow(bm.global_batch[4, 0, bm.pad:-bm.pad, bm.pad:-bm.pad],
            #              cmap='gray')
            # print bm.global_label_batch.shape
            # ax[2].imshow(bm.global_label_batch[4, 0, bm.pad:-bm.pad, bm.pad:-bm.pad],
            #              interpolation='none', cmap=u.random_color_map())
            #
            # print 'gt', gt[4]
            # plt.savefig(tmp_path)
            # plt.close()
            #
            #
            # f, ax = plt.subplots(1, 3)
            # ax[0].imshow(raw[4, 0], cmap='gray')
            # ax[1].imshow(raw[4, 1], cmap=u.random_color_map(),
            #              interpolation='none')
            # ax[2].imshow(gt[4, :, :, 0], cmap=u.random_color_map(),
            #              interpolation='none')
            # plt.savefig(tmp_path + str(2))
            # plt.close()


if __name__ == '__main__':
    train_script_v1()














