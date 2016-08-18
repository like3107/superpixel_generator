import matplotlib
matplotlib.use('Agg')    # Agg for GPU cluster has no display....
# matplotlib.use('Qt4Agg')    # Agg for GPU cluster has no display....
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Process, Queue
import time
import os



def pred_script_v2_wrapper(
        chunk_size=16,
        slices_total=64,
        net_path='./data/nets/cnn_v5/',
        net_number='net_1300000',
        net_name='cnn_v5',
        global_edge_len=300,
        membrane_path='./data/volumes/membranes_first_repr.h5',
        raw_path='./data/volumes/raw_first_repr.h5',
        gt_path = './data/volumes/label_first_repr.h5'
        ):

    assert (slices_total % chunk_size == 0)
    assert (os.path.exists(net_path))
    assert (os.path.exists(membrane_path))
    assert (os.path.exists(raw_path))

    pred_save_folder = net_path + '/preds_many_seeds2/'

    if not os.path.exists(pred_save_folder):
        os.mkdir(pred_save_folder)
    else:
        print 'Warning: might overwrite old pred'

    processes = []
    q = Queue()
    for start in range(0, slices_total, chunk_size):
        print 'start slice %i till %i' % (start, start + chunk_size)
        time.sleep(1)
        processes.append(Process(
            target=pred_script_v2,
            args=(q,),
            kwargs=({'slices':range(start, start + chunk_size, 1),
                     'batch_size':chunk_size,
                     'n_slices':slices_total,
                     'net_name':net_name,
                     'net_number':net_number,
                     'raw_path':raw_path,
                     'membrane_path':membrane_path,
                     'global_edge_len':global_edge_len,
                     'pred_save_folder':pred_save_folder}),
             ))

    for p in processes:
        time.sleep(2)
        p.start()

    for p in processes:
        print 'joining'
        p.join()


    import dataset_utils as du
    import glob
    def concat_h5_in_folder(path_to_folder, slice_size, n_slices, edge_len,
                            save_name):
        files = sorted(glob.glob(path_to_folder + '/' + '*.h5'))
        le_final = np.zeros((n_slices, edge_len, edge_len), dtype=np.uint64)
        for start, file in zip(range(0, n_slices, slice_size), files):
            le_final[start:start + slice_size, :, :] = du.load_h5(file)[0]
        du.save_h5(path_to_folder + save_name, 'data', data=le_final,
                   overwrite='w')

    pred_save_name = 'pred_final_%s_%s.h5' % (net_name, net_number)
    concat_h5_in_folder(pred_save_folder, chunk_size, slices_total,
                          global_edge_len, pred_save_name)

    # du.save_h5(pred_save_folder + net_name + 'pred_final.h5',
    #            'data', data=prediction, overwrite='w')
    #
    import validation_scripts as vs
    vs.validate_segmentation(pred_path=pred_save_folder + pred_save_name,
                             gt_path=gt_path)


def pred_script_v2(
        q,
        # net to load
        net_name='',
        net_number='',
        # data
        membrane_path='',
        raw_path='',
        pred_save_folder='',
        global_edge_len=0,  # 1250  + patch_len for memb
        # set below at wrapper!!!
        batch_size=None,  # do not change here!
        slices=None,      # do not change here
        n_slices=None       # do not change here
        ):

    # import within script du to multi-processing and GPU usage
    import utils as u
    import nets
    import dataset_utils as du
    from theano.sandbox import cuda as c
    c.use('gpu0')

    BM = du.HoneyBatcherPredict
    network = nets.build_net_v5     # hydra only needs build_ID_v0
    probs_funcs = nets.prob_funcs   # hydra, multichannel

    print 'pred script v2 start %i till %i' % (slices[0], slices[-1])

    load_net_path = './data/nets/' + net_name + '/' + net_number
    assert (n_slices % batch_size == 0)

    # all params entered.......................

    # initialize the net
    print 'initializing network graph for net ', net_name
    l_in, l_out, patch_len = network()
    probs_f = probs_funcs(l_in, l_out)
    bm = BM(membrane_path,
            batch_size=batch_size, raw=raw_path,
            patch_len=patch_len, global_edge_len=global_edge_len,
            padding_b=True, slices=slices)

    u.load_network(load_net_path, l_out)

    bm.init_batch(start=0)

    for j in range((bm.global_el - bm.pl) ** 2):
        print '\r remaining %.4f ' % (float(j) / (bm.global_el - bm.pl) ** 2),
        raw, centers, ids = bm.get_batches()
        probs = probs_f(raw)
        bm.update_priority_queue(probs, centers, ids)
        if j % (global_edge_len ** 2/ 4) == 0:
            print 'saving %i' % j
            bm.draw_debug_image('b_0_pred_%s_%i_slice_%02i' %
                                (net_name, j, slices[0]),
                                path=pred_save_folder)

    prediction = bm.global_claims[:,
                                  bm.pad:-bm.pad,
                                  bm.pad:-bm.pad].astype(np.uint64)
    du.save_h5(pred_save_folder + net_name + '_final_slice_%03i.h5' % slices[0],
               'data',
               data=prediction)
    exit()


def pred_script_v2_wrapper(
        chunk_size=16,
        slices_total=64,
        net_path='./data/nets/cnn_v5/',
        net_number='net_1300000',
        net_name='cnn_v5',
        global_edge_len=300,
        membrane_path='./data/volumes/membranes_first_repr.h5',
        raw_path='./data/volumes/raw_first_repr.h5',
        ):

    assert (slices_total % chunk_size == 0)
    assert (os.path.exists(net_path))
    pred_save_folder = net_path + '/preds_0/'

    if not os.path.exists(pred_save_folder):
        os.mkdir(pred_save_folder)
    else:
        print 'Warning: might overwrite old pred'

    processes = []
    q = Queue()
    for start in range(0, slices_total, chunk_size):
        print 'start slice %i till %i' % (start, start + chunk_size)
        time.sleep(1)
        processes.append(Process(
            target=pred_script_v2,
            args=(q,),
            kwargs=({'slices':range(start, start + chunk_size, 1),
                     'batch_size':chunk_size,
                     'n_slices':slices_total,
                     'net_name':net_name,
                     'net_number':net_number,
                     'raw_path':raw_path,
                     'membrane_path':membrane_path,
                     'global_edge_len':global_edge_len,
                     'pred_save_folder':pred_save_folder}),
             ))

    for p in processes:
        time.sleep(2)
        p.start()

    for p in processes:
        print 'joining'
        p.join()


    import dataset_utils as du
    import glob
    def concat_h5_in_folder(path_to_folder, slice_size, n_slices, edge_len,
                            save_name):
        files = sorted(glob.glob(path_to_folder + '/' + '*.h5'))
        le_final = np.zeros((n_slices, edge_len, edge_len), dtype=np.uint64)
        for start, file in zip(range(0, n_slices, slice_size), files):
            le_final[start:start + slice_size, :, :] = du.load_h5(file)[0]
        du.save_h5(path_to_folder + save_name, 'data', data=le_final,
                   overwrite='w')

    concat_h5_in_folder(pred_save_folder, chunk_size, slices_total,
                          global_edge_len, 'pred_final_%s_%s' %
                          (net_name, net_number))

    du.save_h5(pred_save_folder + net_name + 'pred_final.h5',
               'data', data=prediction, overwrite='w')

    import validation_scripts


if __name__ == '__main__':
    # slice = sys.argv[1]
    prediction = pred_script_v2_wrapper()

