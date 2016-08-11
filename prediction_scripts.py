import matplotlib
matplotlib.use('Agg')    # Agg for GPU cluster has no display....
import utils as u
from matplotlib import pyplot as plt
import nets
import dataset_utils as du
import numpy as np
from theano.sandbox import cuda as c


def pred_script_v1():

    print 'pred script v1'
    # data params:
    # for each net a new folder is created. Here intermediate pred-
    # dictions and train, val... are saved

    net_name = 'trash_v5_1'
    membrane_path = './data/volumes/membranes_b.h5'
    raw_path = './data/volumes/raw_b.h5'
    label_path = None
    height_gt_path = None
    height_gt_key = None
    # label_path = './data/volumes/label_b.h5'
    # height_gt_path = './data/volumes/height_b.h5'
    # height_gt_key = 'height'

    load_net_path = './data/nets/' + net_name + '/net_3300000'
    save_path = './data/cnn_v5_ft.h5'
    BM = du.HoneyBatcherPath

    batch_size = 10  # > 4
    global_edge_len = 1290      # 1250  + patch_len for memb
    gt_seeds_b = False      # needs label path
    dummy_data_b = True

    # training parameter
    c.use('gpu0')

    # choose your network and train functions from nets.py
    network = nets.build_net_v5      # hydra only needs build_ID_v0
    # network = nets.build_ID_v1_multichannel
    # hydra only needs build_ID_v0
    # network = nets.build_ID_v1_hybrid
    #
    probs_funcs = nets.prob_funcs       # hydra, multichannel
    # probs_funcs = nets.prob_funcs_hybrid       # hybrid


    # all params entered.......................

    # initialize the net
    print 'initializing network graph for net ', net_name
    l_in, l_out, patch_len = network()
    probs_f = probs_funcs(l_in, l_out)
    print 'Loading data and Priority queue init'
    if dummy_data_b:
        raw_path, membrane_path, height_gt_path, label_path = \
            du.generate_dummy_data(batch_size, global_edge_len, patch_len)
    bm = BM(membrane_path, label=label_path,
                       batch_size=batch_size, raw=raw_path,
                       patch_len=patch_len, global_edge_len=global_edge_len,
                       padding_b=True)


    print load_net_path
    u.load_network(load_net_path, l_out)

    prediction = np.zeros((batch_size, bm.global_el - bm.pl,
                           bm.global_el - bm.pl),
                          dtype=np.uint64)
    for i in range(0, batch_size, batch_size):
        bm.init_batch()

        for j in range((bm.global_el - bm.pl) ** 2):
            print '\r remaining %.4f ' % (float(j) / (bm.global_el - bm.pl) ** 2),
            raw, centers, ids = bm.get_batches()
            probs = probs_f(raw)
            bm.update_priority_queue(probs, centers, ids)

            if j == 100 or j == 40000 or j % 100000 == 0:
                # test saving
                prediction[i * batch_size:(i + 1) * batch_size, :, :] = \
                    bm.global_claims[:, bm.pad:-bm.pad, bm.pad:-bm.pad]
                du.save_h5(save_path, 'pred', data=prediction, overwrite='w')

                fig, ax = plt.subplots(1, 2)
                bm.draw_debug_image('pred_%s_%i' % (net_name, j),
                                    path='./data/')

        prediction[i * batch_size:(i + 1) * batch_size, :, :] = \
            bm.global_claims[:, bm.pad:-bm.pad, bm.pad:-bm.pad]

    du.save_h5(save_path, 'pred', data=prediction, overwrite='w')

if __name__ == '__main__':
    pred_script_v1()


