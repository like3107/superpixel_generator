from multiprocessing import Process, Queue
from trainer_config_parser import get_options
import time
import os
import numpy as np

def pred_wrapper(q, options, slices, gpu):
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

def concat_h5_in_folder(path_to_folder, slice_size, n_slices, edge_len):
    import glob
    from data_provider import load_h5, save_h5
    files = sorted(glob.glob(path_to_folder + '/slice' + '*.h5'))
    le_final = np.zeros((n_slices, edge_len, edge_len), dtype=np.uint64)
    for start, file in zip(range(0, n_slices, slice_size), files):
        le_final[start:start + slice_size, :, :] = load_h5(file)[0]
    save_h5(path_to_folder + '/final.h5', 'data', data=le_final,
               overwrite='w')

if __name__ == '__main__':

    processes = []
    q = Queue()
    options = get_options()

    if options.global_edge_len > 0:
        for x in range(20):
            print "WARNING edge length is not set to 0. Are you sure ?"
            time.sleep(1)



    time.sleep(1)

    # loop over slices
    total_z_lenght = 150
    assert(total_z_lenght % options.batch_size == 0)
    gpus = ["gpu%i"%i for i in range(4)]
    if not os.path.exists(options.save_net_path):
        os.makedirs(options.save_net_path)
    
    for i, start in enumerate(range(0,total_z_lenght,options.batch_size)):
        g = gpus[i%4]
        processes.append(Process(
        target=pred_wrapper,
        args=(q, options, range(start,start+options.batch_size), g)))


    for p in processes:
        time.sleep(8)           # for GPU claim
        p.start()

    for p in processes:
        print 'joining'
        p.join()

    if options.global_edge_len == 0:
        options.global_edge_len = 1250

    concat_h5_in_folder(options.save_net_path,
                         options.batch_size,
                         total_z_lenght,
                         options.global_edge_len)