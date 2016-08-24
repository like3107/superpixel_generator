import matplotlib
matplotlib.use('Agg')    # Agg for GPU cluster has no display....
# matplotlib.use('Qt4Agg')    # Agg for GPU cluster has no display....
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Process, Queue
import time
import os


def pred_script_v2_wrapper(
        chunk_size=None,
        slices_total=None,
        net_number='',
        net_name='',
        global_edge_len=None,
        membrane_path='',
        raw_path='',
        gt_path = '',
        timos_seeds_b=None
        ):
    net_path = './data/nets/' + net_name + '/'

    assert (slices_total % chunk_size == 0)
    assert (os.path.exists(net_path))
    assert (os.path.exists(net_path + net_number))
    assert (os.path.exists(membrane_path))
    assert (os.path.exists(raw_path))

    pred_save_folder = net_path + '/preds'+net_number+'/'

    create_network_folder_structure(pred_save_folder)
    print 'net', net_name, net_number

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
                     'pred_save_folder':pred_save_folder,
                     'timos_seeds_b':timos_seeds_b}),
             ))

    for p in processes:
        time.sleep(8)
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
    return vs.validate_segmentation(pred_path=pred_save_folder + pred_save_name,
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
        n_slices=None,       # do not change here
        timos_seeds_b=None
        ):

    # import within script du to multi-processing and GPU usage
    import utils as u
    import nets
    import dataset_utils as du
    from theano.sandbox import cuda as c
    c.use('gpu0')

    BM = du.HoneyBatcherPredict
    netbuiler = nets.NetBuilder()
    network = netbuiler.build_net_v5_BN     # hydra only needs build_ID_v0
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
            padding_b=True, timos_seeds_b=timos_seeds_b, slices=slices,
            #tmp
            # label='./data/volumes/label_first_repr.h5',
            # height_gt='./data/volumes/height_first_repr.h5'

            )

    u.load_network(load_net_path, l_out)

    bm.init_batch(start=0)

    for j in range((bm.global_el - bm.pl) ** 2):
        print '\r remaining %.4f ' % (float(j) / (bm.global_el - bm.pl) ** 2),
        raw, centers, ids = bm.get_batches()
        # raw, _, centers, ids = bm.get_batches()
        probs = probs_f(raw)
        bm.update_priority_queue(probs, centers, ids)
        if j % (global_edge_len ** 2/ 4) == 0:
            print 'saving %i' % j
            bm.draw_debug_image('b_0_pred_%s_%i_slice_%02i' %
                                (net_name, j, slices[0]),
                                path=pred_save_folder)
    bm.draw_debug_image('b_0_pred_%s_%i_slice_%02i' %
                        (net_name, (bm.global_el - bm.pl) ** 2, slices[0]),
                        path=pred_save_folder)
    prediction = bm.global_claims[:,
                                  bm.pad:-bm.pad,
                                  bm.pad:-bm.pad].astype(np.uint64)
    du.save_h5(pred_save_folder + net_name + '_final_slice_%03i.h5' % slices[0],
               'data',
               data=prediction, overwrite='w')
    exit()


def create_network_folder_structure(save_pred_path, train_mode=True):
    if not os.path.exists(save_pred_path):
        os.mkdir(save_pred_path)
    else:
        print 'Warning: might overwrite old pred'

    code_save_folder = '/code_predict'
    if not os.path.exists(save_pred_path + code_save_folder):
        os.mkdir(save_pred_path + code_save_folder)
    os.system('cp -rf *.py ' + save_pred_path + code_save_folder)


if __name__ == '__main__':
    # slice = sys.argv[1]
    import configargparse

    p = configargparse.ArgParser(default_config_files=['./validation.conf'])
    p.add('-c', '--my-config', is_config_file=True, help='config file path')

    # multiprocessing params
    p.add('--chunk_size', default=16)
    p.add('--slices_total', default=64)     # number z slices
    # network params
    p.add('--net_number', default='net_336994',type=str)
    p.add('--net_name', default='V5_BN_bigreg_fixed_HEP', type=str)

    # data params
    p.add('--global_edge_len', default=300) # should be same as max(x)=max(y)
    p.add('--membrane_path', default='./data/volumes/membranes_first_repr.h5')
    p.add('--raw_path', default='./data/volumes/raw_first_repr.h5')
    p.add('--gt_path', default='./data/volumes/label_first_repr.h5')
    p.add('--timos_seeds_b', default=True)
    p.add('--save_validation', default="",type=str)
    options = p.parse_args()
    print options
    print
    prediction = pred_script_v2_wrapper(
                        chunk_size=options.chunk_size,
                        slices_total=options.slices_total,
                        net_number=options.net_number,
                        net_name=options.net_name,
                        global_edge_len=options.global_edge_len,
                        membrane_path=options.membrane_path,
                        raw_path=options.raw_path,
                        gt_path=options.gt_path,
                        timos_seeds_b=options.timos_seeds_b)


    if not options.save_validation != "":
        f = open(options.save_validation,'w')
        f.write(str(prediction['Variational information split']))
        f.write("+-")
        f.write(str(prediction['Variational information split_error']))
        f.write(",") 
        f.write(str(prediction['Variational information merge']))
        f.write("+-")
        f.write(str(prediction['Variational information merge_error']))
        f.write(",") 
        f.write(str(prediction['Adapted Rand error']))
        f.write("+-")
        f.write(str(prediction['Adapted Rand error_error']))
        f.write(",") 
        f.write(str(prediction['Adapted Rand error precision']))
        f.write("+-") 
        f.write(str(prediction['Adapted Rand error precision_error']))
        f.write(",")
        f.write(str(prediction['Adapted Rand error recall']))
        f.write("+-") 
        f.write(str(prediction['Adapted Rand error recall_error']))
        f.close()

