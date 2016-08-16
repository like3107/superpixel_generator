import matplotlib
matplotlib.use('Agg')    # Agg for GPU cluster has no display....
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Pool, Process, Queue
import time
import sys
import os


def pred_script_v2(
        q,
        # net to load
        net_name='trash_v5_debug_clip_quad_lowreg_freevoxel200_rawreg2',
        net_number='net_300000',

        # data
        membrane_path='./data/volumes/membranes_b.h5',
        raw_path='./data/volumes/raw_b.h5',
        save_path='./data/tmp/',

        global_edge_len=1290,  # 1250  + patch_len for memb

        # set below at wrapper!!!
        batch_size=None,  # do not change here!
        slices=None,      # do not change here
        n_slices=None       # do not change here
        ):
    # import within script du to multi-processing and GPU usage
    import utils as u
    import nets
    import dataset_utils as du
    import pickle
    from theano.sandbox import cuda as c
    c.use('gpu0')

    BM = du.HoneyBatcherPredict
    network = nets.build_net_v5     # hydra only needs build_ID_v0
    probs_funcs = nets.prob_funcs   # hydra, multichannel

    print 'pred script v2 start %i till %i' % (slices[0], slices[1])

    load_net_path = './data/nets/' + net_name + '/' + net_number
    assert (n_slices % batch_size == 0)
    print 'pred script v2'
    # data params:
    # for each net a new folder is created. Here intermediate pred-
    # dictions and train, val... are saved
    # choose your network and train functions from nets.py
    # network = nets.build_ID_v1_multichannel
    # hydra only needs build_ID_v0
    # network = nets.build_ID_v1_hybrid
    #
    # probs_funcs = nets.prob_funcs_hybrid       # hybrid



    # all params entered.......................

    # initialize the net
    print 'initializing network graph for net ', net_name
    l_in, l_out, patch_len = network()
    print 'network'
    probs_f = probs_funcs(l_in, l_out)
    print 'Loading data and Priority queue init'
    bm = BM(membrane_path,
            batch_size=batch_size, raw=raw_path,
            patch_len=patch_len, global_edge_len=global_edge_len,
            padding_b=True, slices=slices)

    print 'loading net', load_net_path
    u.load_network(load_net_path, l_out)

    prediction = np.zeros((batch_size, bm.global_el - bm.pl,
                           bm.global_el - bm.pl),
                          dtype=np.uint64)
    for i in range(0, batch_size, batch_size):
        bm.init_batch()

        for j in range((bm.global_el - bm.pl) ** 2):
            print '\r remaining %.4f ' % (
                    float(j) / (bm.global_el - bm.pl) ** 2),
            raw, centers, ids = bm.get_batches()
            probs = probs_f(raw)
            bm.update_priority_queue(probs, centers, ids)

            if j == 100 or j == 40000 or j % 100000 == 0:
                # test saving
                print 'saving %i' % j
                prediction[i * batch_size:(i + 1) * batch_size, :, :] = \
                    bm.global_claims[:, bm.pad:-bm.pad, bm.pad:-bm.pad]

                # du.save_h5(
                #     save_path + net_name + '_interediate_slice_%i_%i.h5' %
                #     (slices[0], j),
                #     'data',
                #     data=prediction, overwrite='w')
                bm.draw_debug_image('pred_%s_%i_slice_%i' %
                                    (net_name, j, slices[0]),
                                    path='./data/')
                # pickle.dump('./data/save_bm_%i_iter_%i' % (slices[0], j), bm)
        prediction[i * batch_size:(i + 1) * batch_size, :, :] = \
            bm.global_claims[:, bm.pad:-bm.pad, bm.pad:-bm.pad]
    du.save_h5(save_path + net_name + '_final_slice_%i.h5' % slices[0], 'data',
               data=prediction)

    print 'done slices_%i_till_%i' %(slices[0], slices[1])
    c.unuse()
    q.put((slices[0], prediction))


def pred_script_v2_wrapper(chunk_size=16, slices_total=64, n_worker=2,
                           save_path='./data/cnn_v5_test_final.h5',
                           debug=False):
    assert (slices_total % chunk_size == 0)

    processes = []
    q = Queue()
    for start in range(0, slices_total, chunk_size):
        print 'start slice %i till %i' %(start, start + chunk_size)
        time.sleep(1)
        processes.append(Process(target=pred_script_v2, args=(q,),
                                 kwargs=({'slices':[start,
                                                    start + chunk_size],
                                          'batch_size':chunk_size,
                                          'n_slices':slices_total}),
                                 ))
        # pred_script_v2(q, slices=[start, start+chunk_size])
        # exit()
    for p in processes:
        time.sleep(2)
        p.start()

    for p in processes:
        print 'joining'
        p.join()

    i = 0
    while not q.empty():
        print 'i'
        slic, pred = q.get()

        # prediction[slic:slic + chunk_size, :, :] = pred
        i += 1

    print prediction
    import dataset_utils as du
    du.save_h5(save_path, 'data', data=prediction, overwrite='w')


if __name__ == '__main__':
    # slice = sys.argv[1]

    prediction = pred_script_v2_wrapper()

    # pred_script_v2()


