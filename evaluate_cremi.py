from multiprocessing import Process, Pool
from trainer_config_parser import get_options
import time
import os
import numpy as np
import re


def pred_wrapper(options, slices, gpu):
    import evaluate_net
    from data_provider import save_h5
    options.gpu = gpu
    options.slices = slices
    pred = evaluate_net.Predictor(options)
    pred.predict()

    save_slice_path = options.validation_save_path + "/slice_%04i_%04i.h5" % (slices[0], slices[-1])
    if not options.fully_conf_valildation_b:
        save_h5(save_slice_path, 'data',
                data=pred.bm.global_claims[:, pred.bm.pad:-pred.bm.pad, pred.bm.pad:-pred.bm.pad].astype(np.uint64),
                overwrite='w', compression='gzip')

    save_height_path = options.validation_save_path + "/height_%04i_%04i.h5" % (slices[0], slices[-1])
    save_h5(save_height_path, 'data', data=pred.bm.global_heightmap_batch.astype(np.float32), overwrite='w')
    if not options.fully_conf_valildation_b:

        save_pred_nq_path = options.validation_save_path + "/pred_nq_path_%04i_%04i.h5" % (slices[0], slices[-1])
        save_h5(save_pred_nq_path, 'data', data=pred.bm.global_prediction_map_nq.astype(np.float32), overwrite='w')

        save_slice_path = options.validation_save_path + "/baseline_%04i_%04i.h5" % (slices[0], slices[-1])
        save_h5(save_slice_path, 'data', data=pred.bm.get_ws_segmentation(), overwrite='w')


def concat_h5_in_folder(path_to_folder, slice_size, n_slices, base_file_name='slice', label_b=False):
    import glob
    from data_provider import load_h5, save_h5
    files = sorted(glob.glob(path_to_folder + '/' + base_file_name + '*.h5'))
    print 'files', files, (path_to_folder + '/' + base_file_name + '*.h5')
    if not os.path.exists(files[0]):
        print files[0]
    initial = load_h5(files[0])[0]
    fin_shape = list(initial.shape)
    fin_shape[0] = n_slices
    le_final = np.zeros(fin_shape, dtype=initial.dtype)
    for start, file in zip(range(0, n_slices, slice_size), files):
        le_final[start:start + slice_size, :, :] = load_h5(file)[0]
    save_h5(path_to_folder + '/'+base_file_name+'_concat.h5', 'data', data=le_final, overwrite='w',
            compression='gzip')


def evaluate_h5_files(prediction_path, gt_path, name, options):
    import validation_scripts as vs
    fov = 70
    print "############ Evaliation for ", name, "############"
    _, results = vs.validate_segmentation(pred_path=prediction_path, gt_path=gt_path,
                                          offset_xy=int(fov)/2, start_z=options.start_slice_z,
                                          n_z=options.slices_total,
                                          gel=options.global_edge_len,
                                          defect_slices=options.defect_slices_b)
    f = open(options.validation_save_path+'/'+name+'results.txt', 'w')
    f.write(results)
    f.close()
    print "####################################################"


if __name__ == '__main__':
    processes = []
    options = get_options(script='validation_valid')
    if options.global_edge_len > 0 and not options.quick_eval:
        for x in range(2):
            print "WARNING edge length is not set to 0. Are you sure ?"
            time.sleep(0.1)

    time.sleep(1)

    # loop over slices
    total_z_lenght = options.slices_total
    start_z = options.start_slice_z
    assert(total_z_lenght % options.batch_size == 0)
    if options.gpu != 'all':
        gpus = [options.gpu] * 4
    else:
        gpus = ["gpu%i"%i for i in range(4)]

    if not os.path.exists(options.save_net_path):
        os.makedirs(options.save_net_path)
    try:
        net_number = re.search(r'net_\d*', options.load_net_path).group()
        options.validation_save_path = options.save_net_path + '/validation_%s/' % net_number
    except:
        options.validation_save_path = options.save_net_path + '/validation_0/'

    if not os.path.exists(options.validation_save_path):
        os.mkdir(options.validation_save_path)
    if not os.path.exists(options.validation_save_path + '/images/'):
        os.mkdir(options.validation_save_path + '/images/')

    if options.max_processes > 1:
        pool = Pool(processes=options.max_processes)
        for i, start in enumerate(range(start_z, start_z+total_z_lenght, options.batch_size)):
            g = gpus[i % 4]
            pool.apply_async(pred_wrapper,  args=(options, range(start, start+options.batch_size), g))
            time.sleep(8)  # for GPU claim
        pool.close()
        pool.join()
    else:
        for i, start in enumerate(range(start_z, start_z+total_z_lenght, options.batch_size)):
            pred_wrapper(options, range(start, start + options.batch_size), options.gpu)

    if options.global_edge_len == 0:
        options.global_edge_len = 1250

    if not options.padding_b:
        options.global_edge_len -= 70

    if not options.fully_conf_valildation_b:
        label_types = [False, True, False, True]
        versions = ["height", "slice", "pred_nq_path", "baseline"]
    else:
        label_types = [False]
        versions = ["height"]

    for label_b, bn in zip(label_types, versions):

        concat_h5_in_folder(options.validation_save_path, options.batch_size, total_z_lenght, base_file_name=bn,
                            label_b=label_b)

    if not options.fully_conf_valildation_b:
        for name in ["slice", "baseline"]:
            prediction_path = options.validation_save_path + '/'+name+'_concat.h5'
            evaluate_h5_files(prediction_path, options.label_path, name, options)
