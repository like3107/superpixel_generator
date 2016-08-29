import dataset_utils as du
from ws_timo import wsDtSegmentation
import numpy as np

def double_ws(path_to_seg, path_to_gt, save_path):
    seg = du.load_h5(path_to_seg)[0]
    memb = np.zeros_like(seg)
    seg2 = np.zeros_like(seg)

    threshold_dist_trf = 0.3
    thres_memb_cc = 15
    thresh_seg_cc = 85
    sigma_dist_trf = 2
    somethingunimportant = 0
    two_dim = True
    groupSeeds = True
    print 'TImos Waterhshed'
    segmentation = np.zeros((64, 300, 300))

    for i, slice in enumerate(seg):
        print slice.shape
        print memb[i, :, :].shape
        memb[i, :, :], _ = du.segmenation_to_membrane_core(slice)
        seg2[i, :, :], seeds = \
                    wsDtSegmentation(memb[i, :, :],
                                          threshold_dist_trf, thres_memb_cc,
                                          thresh_seg_cc, sigma_dist_trf,
                                          somethingunimportant)
    du.save_h5(save_path, 'data', data=seg2, overwrite='w')






if __name__ == '__main__':
    # path_to_seg = './data/nets/sauron_the_blind/preds_first_repr_zstacknet_632000/final.h5'
    path_to_seg = './data/nets/sauron_the_blind/preds_first_repr_big_zstacknet_632000/final.h5'
    path_to_gt = './data/volumes/label_first_repr_zstack_cut.h5'
    save_path = './data/preds/preds_first_repr_big_zstacknet_632000.h5'
    double_ws(path_to_seg, path_to_gt, save_path)