import dataset_utils as du
from ws_timo import wsDtSegmentation
import numpy as np
import validation_scripts as v

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
        print '\r i', i,
        # print slice.shape
        # print memb[i, :, :].shape
        memb[i, :, :], _ = du.segmenation_to_membrane_core(slice)
        seg2[i, :, :], seeds = \
                    wsDtSegmentation(memb[i, :, :],
                                          threshold_dist_trf, thres_memb_cc,
                                          thresh_seg_cc, sigma_dist_trf,
                                          somethingunimportant)
    du.save_h5(save_path, 'data', data=seg2, overwrite='w')
    if path_to_gt is not None:
        gt = du.load_h5(path_to_gt)[0]
        v.validate_segmentation(seg2, gt)





if __name__ == '__main__':
    # path_to_seg = './data/nets/sauron_the_blind/preds_first_repr_zstacknet_632000/final.h5'
    base_path = '/media/liory/DAF6DBA2F6DB7D67/cremi/final/CREMI-pmaps-padded/'
    for version, stupid in zip(['B+'], ['stupid']):

        path_to_seg = base_path + '/seg_%s.h5' % version
        # path_to_gt = base_path + '/label_%s.h5' % stupid
        save_path = base_path + '/seg_%s_double.h5' % version
        double_ws(path_to_seg, None, save_path)