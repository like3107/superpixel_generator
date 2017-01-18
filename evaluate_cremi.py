from multiprocessing import Process, Pool
from trainer_config_parser import get_options
import time
import os
import numpy as np

def pred_wrapper(options, slices, gpu):
    import evaluate_net
    from data_provider import save_h5
    options.gpu = gpu
    options.slices = slices
    pred = evaluate_net.Predictor(options)
    pred.predict()
    save_slice_path = options.save_net_path \
                    + "/slice_%04i_%04i.h5"%(slices[0],slices[-1])
    save_h5(save_slice_path, 'data',
        data=pred.bm.global_claims[:, pred.bm.pad:-pred.bm.pad,
                                      pred.bm.pad:-pred.bm.pad],
        overwrite='w')

    save_height_path = options.save_net_path \
                    + "/height_%04i_%04i.h5"%(slices[0],slices[-1])
    save_h5(save_height_path, 'data',
        data=pred.bm.global_heightmap_batch,
        overwrite='w')


    save_pred_nq_path = options.save_net_path \
                    + "/pred_nq_path_%04i_%04i.h5"%(slices[0],slices[-1])
    save_h5(save_pred_nq_path, 'data',
            data=pred.bm.global_prediction_map_nq,
            overwrite='w')

    save_slice_path = options.save_net_path \
                    + "/baseline_%04i_%04i.h5"%(slices[0],slices[-1])
    save_h5(save_slice_path, 'data',
            data=pred.bm.get_ws_segmentation(),
            overwrite='w')

def concat_h5_in_folder(path_to_folder, slice_size, n_slices, edge_len,
                        base_file_name='slice'):
    import glob
    from data_provider import load_h5, save_h5
    files = sorted(glob.glob(path_to_folder + '/' + base_file_name + '*.h5'))
    initial = load_h5(files[0])[0]
    fin_shape = list(initial.shape)
    fin_shape[0] = n_slices
    le_final = np.zeros(fin_shape, dtype=initial.dtype)
    for start, file in zip(range(0, n_slices, slice_size), files):
        le_final[start:start + slice_size, :, :] = load_h5(file)[0]
    save_h5(path_to_folder + '/'+base_file_name+'_concat.h5', 'data', data=le_final,
               overwrite='w')


def evaluate_h5_files(prediction_path, gt_path, name, options):
    import validation_scripts as vs
    fov = 68
    print "############ Evaliation for ",name,"############"
    _, results = vs.validate_segmentation(pred_path=prediction_path, gt_path=gt_path,
                             offset_xy=int(fov)/2, start_z=options.start_slice_z,
                             n_z=options.slices_total,
                             gel=options.global_edge_len)
    f = open(options.save_net_path+'/'+name+'results.txt', 'w')
    f.write(results)
    f.close()
    print "####################################################"


if __name__ == '__main__':
    processes = []
    options = get_options(script='validation')
    pool = Pool(processes=options.max_processes)
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
    
    for i, start in enumerate(range(start_z,start_z+total_z_lenght, options.batch_size)):
        g = gpus[i%4]
        # processes.append(Process(
        #     target=pred_wrapper,
        #     args=(q, options, range(start,start+options.batch_size), g)))
        pool.apply_async(pred_wrapper,
            args=(options, range(start,start+options.batch_size), g))
        time.sleep(8)           # for GPU claim
        # debug
        # pred_wrapper(q, options, range(start, start + options.batch_size), g)

    # for p in processes:
    #     p.start()

    # for p in processes:
    #     print 'joining'
    #     p.join()
    pool.close()
    pool.join()


    if options.global_edge_len == 0:
        options.global_edge_len = 1250

    if not options.padding_b:
        options.global_edge_len -= 70

    for bn in ["slice", "height", "pred_nq_path", "baseline"]:
        concat_h5_in_folder(options.save_net_path,
                        options.batch_size,
                        total_z_lenght,
                        options.global_edge_len,
                        base_file_name=bn)

    
    for name in ["slice", "baseline"]:
        prediction_path = options.save_net_path + '/'+name+'_concat.h5'
        evaluate_h5_files(prediction_path, options.label_path, name, options)