import lasagne as las
import theano
from theano import tensor as T
import numpy as np
import utils as u
from matplotlib import pyplot as plt
import nets
import dataset_utils as du

from theano.sandbox import cuda as c
c.use('gpu0')


def run_cnn_v1():
    # data params
    im_path = './data/images/'
    label_path = './data/labels_as.h5'
    raw_path = './data/raw_as.h5'
    batch_size = 20
    patch_len = 40
    global_edge_len = 100

    # initialize the net
    print 'initializing network graph'
    target_t = T.ftensor4()
    l_in, l_out = nets.build_net_v0()

    print 'compiling theano functions'
    loss_train_f, loss_valid_f, probs_f = \
        nets.loss_updates_probs_v0(l_in, target_t, l_out)

    print 'Loading data and Priority queue init'
    bm = du.BatchManV0(raw_path, label_path, batch_size=batch_size,
                       patch_len=patch_len, global_edge_len=global_edge_len)
    bm.init_train_batch()
    bm_val = du.BatchManV0(raw_path, label_path, batch_size=batch_size,
                       patch_len=patch_len, global_edge_len=global_edge_len,
                           remain_in_territory=False)
    bm_val.init_train_batch()  # Training


    converged = False
    max_iter = 10000000
    iteration = -1
    battle_field_change = 100
    battle_field_counter = 0
    while not converged and (iteration < max_iter):
        iteration += 1
        battle_field_counter += 1

        # save image and update battlefield ground
        if battle_field_counter % battle_field_change == 0:
            print
            # plot test images
            u.save_2_images(
                bm.global_claims[4, bm.pad:-bm.pad-1, bm.pad:-bm.pad-1],
                bm.global_batch[4, 0, bm.pad:-bm.pad-1, bm.pad:-bm.pad-1],
                im_path, iteration=battle_field_counter, name='train')

            u.save_2_images(
                bm_val.global_claims[4, bm_val.pad:-bm_val.pad - 1,
                                     bm_val.pad:-bm_val.pad - 1],
                bm.global_batch[4, 0, bm_val.pad:-bm.pad - 1,
                                      bm_val.pad:-bm_val.pad - 1],
                im_path, iteration=battle_field_counter, name='val')

            battle_field_change = \
                int((1.-1./(iteration + 2)**0.4) *
                    ((global_edge_len - bm.pl)**2 - 300))
            print 'new global batch loaded', battle_field_counter, battle_field_change
            bm.init_train_batch()
            battle_field_counter = 0

        # train da thing
        raw, gt, seeds, ids = bm.get_batches()
        probs = probs_f(raw)
        loss_train = loss_train_f(raw, gt)
        bm.update_priority_queue(probs, seeds, ids)

        raw, gt, seeds, ids = bm_val.get_batches()
        probs_val = probs_f(raw)
        bm_val.update_priority_queue(probs_val, seeds, ids)

        if iteration % 100 == 0:
            print '\r loss train %.4f, iteration %i' % (loss_train, iteration),

if __name__ == '__main__':

    run_cnn_v1()














