import matplotlib
matplotlib.use('Agg')    # Agg for GPU cluster has no display....
import utils as u
from matplotlib import pyplot as plt
import nets
import dataset_utils as du
import numpy as np
from theano.sandbox import cuda as c


def pred_script_v0():

    # data params:
    # for each net a new folder is created. Here intermediate pred-
    # dictions and train, val... are saved
    load_net_b = True

    pred_name = 'predv1'
    label_path = './data/volumes/label_as.h5'
    raw_path = './data/volumes/membranes_as.h5'
    load_net_path = './data/nets/cnn_v0_test/net_50000'      # if load true
    tmp_path = '/media/liory/ladata/bla'        # debugging

    batch_size = 64         # > 4
    global_edge_len = 340

    # pred parameter
    c.use('gpu0')


    # choose your network from nets.py
    network = nets.build_net_v0

    # all params entered.......................

    # initialize the net

    print 'initializing network graph'
    l_in, l_out, patch_len = network()

    print 'compiling theano functions'
    probs_f = nets.prob_funcs(l_in, l_out)

    print 'Loading data and Priority queue init'
    bm = du.BatchManV0(raw_path, label_path, batch_size=batch_size,
                       patch_len=patch_len, global_edge_len=global_edge_len,
                       padding_b=True)

    prediction = np.zeros((batch_size, bm.global_el - bm.pl,
                           bm.global_el - bm.pl),
                          dtype=np.uint64)

    u.load_network(load_net_path, l_out)

    for i in range(0, batch_size, batch_size):
        bm.init_prediction(i, i+batch_size)

        for j in range((bm.global_el - bm.pl)**2):
            # print j
            print '\r remaining %.4f ' % (float(j) / (bm.global_el - bm.pl)**2),
            raw, gts, seeds, ids = bm.get_pred_batch()
            probs = probs_f(raw)
            bm.update_priority_queue(probs, seeds, ids)

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(bm.global_claims[0, :, :],
                   cmap=u.random_color_map(), interpolation='none')
        ax[1].imshow(bm.global_batch[0, :, :],
                   cmap='gray', interpolation='none')
        ax[2].imshow(bm.global_label_batch[0, :, :])
        plt.savefig(tmp_path + 'all' + str(j))

        prediction[i*batch_size:(i + 1) * batch_size, :, :] = \
            bm.global_claims[:, bm.pad:-bm.pad, bm.pad:-bm.pad]



    du.save_h5('./data/pred.h5', 'pred', data=prediction, overwrite='w')

if __name__ == '__main__':
    pred_script_v0()


