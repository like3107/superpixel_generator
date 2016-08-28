import matplotlib
matplotlib.use('Agg')    # Agg for GPU cluster has no display....
# matplotlib.use('Qt4Agg')    # Agg for GPU cluster has no display....
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Process, Queue
import time
import os
import h5py as h
from os import makedirs
from os.path import exists


def load_h5(path, h5_key=None, group=None, group2=None, slices=None):
    if not exists(path):
        error = 'path: %s does not exist, check' % path
        raise Exception(error)
    f = h.File(path, 'r')
    if group is not None:
        g = f[group]
        if group2 is not None:
            g = g[group2]
    else:   # no groups in file structure
        g = f
    if h5_key is None:     # no h5 key specified
        output = list()
        for key in g.keys():
            output.append(np.array(g[key], dtype='float32'))
    elif isinstance(h5_key, basestring):   # string
        output = [np.array(g[h5_key], dtype='float32')]
    elif isinstance(h5_key, list):          # list
        output = list()
        for key in h5_key:
            output.append(np.array(g[key], dtype='float32'))
    else:
        raise Exception('h5 key type is not supported')
    if slices is not None:
        output = [output[0][slices]]
    f.close()
    return output


def save_h5(path, h5_key, data, overwrite='w-'):
    f = h.File(path, overwrite)
    if isinstance(h5_key, str):
        f.create_dataset(h5_key, data=data)
    if isinstance(h5_key, list):
        for key, values in zip(h5_key, data):
            f.create_dataset(key, data=values)
    f.close()


def pred_script_v2_wrapper(
        chunk_size=None,
        slices_total=None,
        net_file='',
        pred_save_folder='',
        global_edge_len=None,
        membrane_path='',
        raw_path='',
        gt_path = '',
        timos_seeds_b=None
        ):

    assert (slices_total % chunk_size == 0)
    print membrane_path
    assert (os.path.exists(membrane_path))
    assert (os.path.exists(raw_path))
    assert (os.path.exists(net_file))
    create_network_folder_structure(pred_save_folder)
    print 'gt path', gt_path
    print 'net', net_file

    processes = []
    q = Queue()

    step_size = chunk_size
    if 'zstack' in raw_path:
        step_size *= 3

    for start in range(0, slices_total, step_size):
        print 'start slice %i till %i' % (start, start + step_size)
        time.sleep(1)
        processes.append(Process(
            target=pred_script_v2,
            args=(q,),
            kwargs=({'slices':range(start, start + step_size, 1),
                     'batch_size':chunk_size,
                     'n_slices':slices_total,
                     'net_file':net_file,
                     'raw_path':raw_path,
                     'membrane_path':membrane_path,
                     'global_edge_len':global_edge_len,
                     'pred_save_folder':pred_save_folder,
                     'timos_seeds_b':timos_seeds_b}),
             ))
        # pred_script_v2(q, slices=range(start, start + chunk_size, 1),
        #             batch_size=chunk_size,
        #             n_slices=slices_total,
        #             net_file=net_file,
        #             raw_path=raw_path,
        #             membrane_path=membrane_path,
        #             global_edge_len=global_edge_len,
        #             pred_save_folder=pred_save_folder,
        #             timos_seeds_b=timos_seeds_b)
    for p in processes:
        time.sleep(8)
        p.start()

    for p in processes:
        print 'joining'
        p.join()

    import glob
    def concat_h5_in_folder(path_to_folder, slice_size, n_slices, edge_len):
        files = sorted(glob.glob(path_to_folder + '/final_slice' + '*.h5'))
        le_final = np.zeros((n_slices, edge_len, edge_len), dtype=np.uint64)
        for start, file in zip(range(0, n_slices, slice_size), files):
            le_final[start:start + slice_size, :, :] = load_h5(file)[0]
        save_h5(path_to_folder + '/final.h5', 'data', data=le_final,
                   overwrite='w')

    concat_h5_in_folder(pred_save_folder, chunk_size, slices_total,
                        global_edge_len)

    # du.save_h5(pred_save_folder + net_name + 'pred_final.h5',
    #            'data', data=prediction, overwrite='w')
    #
    import validation_scripts as vs
    return vs.validate_segmentation(pred_path=pred_save_folder + '/final.h5',
                             gt_path=gt_path)


def pred_script_v2(
        q,
        # net to load
        net_file='',
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
    options = u.load_options(net_file)

    BM = du.HoneyBatcherPredict
    builder = nets.NetBuilder()
    network = builder.get_net(options['net_arch'])
    probs_funcs = nets.prob_funcs   # hydra, multichannel
    print 'pred script v2 start %i till %i' % (slices[0], slices[-1])

    assert (n_slices % batch_size == 0)
    print options['net_arch']
    # all params entered.......................

    # initialize the net
    print 'initializing network graph for net ', net_file
    l_in, _, l_out, _, patch_len = network()
    probs_f = probs_funcs(l_in, l_out)
    bm = BM(membrane_path,
            batch_size=batch_size, raw=raw_path,
            patch_len=patch_len, global_edge_len=global_edge_len,
            padding_b=True, timos_seeds_b=timos_seeds_b, slices=slices,
            z_stack=("zstack" in options['net_arch']),
            downsample = ("down" in options['net_arch'])
            #tmp
            # label='./data/volumes/label_first_repr.h5',
            # height_gt='./data/volumes/height_first_repr.h5'
            )
    print bm.rl, bm.global_el
    if "down" in options['net_arch']:
        assert (bm.rl == bm.global_el + bm.pl)
    else:
        assert (bm.rl == bm.global_el)
    u.load_network(net_file, l_out)

    sample_indices = u.get_stack_indices(raw_path,options['net_arch'])
    bm.init_batch(start=0, allowed_slices=sample_indices)

    for j in range((bm.global_el - bm.pl) ** 2):
        print '\r remaining %.4f ' % (float(j) / (bm.global_el - bm.pl) ** 2),
        raw, centers, ids = bm.get_batches()
        # raw, _, centers, ids = bm.get_batches()
        probs = probs_f(raw)
        bm.update_priority_queue(probs, centers, ids)
        # tmp 40 -> 4
        if j % (global_edge_len ** 2/ 4) == 0:
            print 'saving %i' % j
            bm.draw_debug_image('b_0_pred_%i_slice_%02i' %
                                (j, slices[0]),
                                path=pred_save_folder)
    bm.draw_debug_image('b_0_pred_%i_slice_%02i' %
                        ((bm.global_el - bm.pl) ** 2, slices[0]),
                        path=pred_save_folder)
    prediction = bm.global_claims[:,
                                  bm.pad:-bm.pad,
                                  bm.pad:-bm.pad].astype(np.uint64)
    du.save_h5(pred_save_folder + '/final_slice_%03i.h5' % slices[0],
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

    p = configargparse.ArgParser(default_config_files=['./data/config/validation.conf'])
    p.add('-c', '--my-config', is_config_file=True, help='config file path')

    # multiprocessing params
    p.add('--chunk_size', default=16, type=int)
    p.add('--slices_total', default=64, type=int)     # number z slices

    # network params
    p.add('--net_file', default='',type=str)
    p.add('--pred_save_folder', default='',type=str)
    p.add('--net_name', default='',type=str)
    p.add('--net_number', default='',type=str)

    # data params
    p.add('--data_version', default='', type=str)
    p.add('--global_edge_len', default=300, type=int) # has to be same as max(x)=max(y)
    p.add('--membrane_path', default='./data/volumes/membranes_first_repr.h5')
    p.add('--raw_path', default='./data/volumes/raw_first_repr.h5')
    p.add('--gt_path', default='./data/volumes/label_first_repr.h5')
    p.add('--timos_seeds_b', action='store_false')
    p.add('--save_validation', default="",type=str)
    options = p.parse_args()

    if options.net_file == '':
        options.net_file = './data/nets/' + options.net_name + '/' + options.net_number

    if options.pred_save_folder == '':
        if options.data_version == '':
            options.pred_save_folder = './data/nets/' + options.net_name + \
                                       '/preds_'+options.net_number+'/'
        else:
            options.pred_save_folder = './data/nets/' + options.net_name + \
                                       '/preds_' +options.data_version + \
                                       options.net_number+'/'

    if options.save_validation == "":
        options.save_validation = options.pred_save_folder + 'numbanumba.txt'

    if options.data_version != '':
        print 'here'
        options.raw_path = './data/volumes/raw_%s.h5' % options.data_version
        options.membrane_path =  './data/volumes/membranes_%s.h5' % options.data_version
        options.gt_path =  './data/volumes/label_%s.h5' % options.data_version

    prediction = pred_script_v2_wrapper(
                        chunk_size=options.chunk_size,
                        slices_total=options.slices_total,
                        net_file=options.net_file,
                        pred_save_folder=options.pred_save_folder,
                        global_edge_len=options.global_edge_len,
                        membrane_path=options.membrane_path,
                        raw_path=options.raw_path,
                        gt_path=options.gt_path,
                        timos_seeds_b=options.timos_seeds_b)

    if options.save_validation != "":
        if options.save_validation.endswith(".json"):
            import json
            with open(options.save_validation, 'w') as f:
                f.write(json.dumps(prediction))
        else:
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

