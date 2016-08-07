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
    save_net_b = True
    load_net_b = False
    net_name = 'gt_seeds_2D'
    raw_path = './data/volumes/membranes_b.h5'
    label_path = './data/volumes/label_b.h5'
    height_gt_path = './data/volumes/height_b.h5'
    height_gt_key = 'height'

    load_net_path = './data/nets/cnn_path_v1_fine_tune2/net_100000'
    save_path = './data/membranes_real_seeds.h5'

    batch_size = 4  # > 4
    global_edge_len = 1290      # 1250  + patch_len for memb
    gt_seeds_b = True

    # training parameter
    c.use('gpu0')

    # choose your network from nets.py
    network = nets.build_ID_v0

    # all params entered.......................

    # initialize the net
    print 'initializing network graph for net ', net_name
    l_in, l_out, patch_len = network()
    probs_f = nets.prob_funcs(l_in, l_out)

    print 'Loading data and Priority queue init'
    bm = du.BatchManV0(raw_path, label=label_path,
                       batch_size=batch_size,
                       patch_len=patch_len, global_edge_len=global_edge_len,
                       padding_b=True, train_b=False, gt_seeds_b=gt_seeds_b,
                       height_gt=height_gt_path, height_gt_key=height_gt_key)
    # bm.init_train_path_batch()
    u.load_network(load_net_path, l_out)

    prediction = np.zeros((batch_size, bm.global_el - bm.pl,
                           bm.global_el - bm.pl),
                          dtype=np.uint64)
    for i in range(0, batch_size, batch_size):
        bm.init_prediction(i, i + batch_size)

        for j in range((bm.global_el - bm.pl) ** 2):
            print '\r remaining %.4f ' % (float(j) / (bm.global_el - bm.pl) ** 2),
            raw, centers, ids = bm.get_pred_batch()
            probs = probs_f(raw)
            bm.update_priority_path_queue_prediction(probs, centers, ids)

            if j == 100 or j == 40000:
                # test saving
                prediction[i * batch_size:(i + 1) * batch_size, :, :] = \
                    bm.global_claims[:, bm.pad:-bm.pad, bm.pad:-bm.pad]
                du.save_h5(save_path, 'pred', data=prediction, overwrite='w')

                fig, ax = plt.subplots(1, 2)
                bm.draw_debug_image('pred_%s_%i' % (net_name, j))
                ax[0].imshow(bm.global_claims[0, :, :],
                             cmap=u.random_color_map(), interpolation='none')
                ax[1].imshow(bm.global_batch[0, :, :],
                             cmap='gray', interpolation='none')
                # ax[2].imshow(bm.global_label_batch[0, :, :])
                plt.savefig('./data/' + 'trash' + str(j))

        prediction[i * batch_size:(i + 1) * batch_size, :, :] = \
            bm.global_claims[:, bm.pad:-bm.pad, bm.pad:-bm.pad]

    du.save_h5(save_path, 'pred', data=prediction, overwrite='w')

if __name__ == '__main__':
    pred_script_v1()


