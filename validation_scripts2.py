import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append('/home/lschott_local/git/cremi_python/')
from cremi.evaluation import NeuronIds
from cremi.evaluation import voi
from cremi.evaluation import adapted_rand
import numpy as np
import data_provider as du



def validate_segmentation(pred=None, gt=None, gt_path=None, pred_path=None,
                          pred_key=None, gt_key=None, slice_by_slice=True,
                          offset_xy=0, gel=None, start_z=None, n_z=None,
                          defect_slices=None, resolution=4, border_thresh=25, verbose=True):
    assert (gt_path is not None or gt is not None)  # specify either gt path or gt as np array
    assert (pred_path is not None or pred is not None)    # specify either raw path or raw as np array

    if isinstance(gt_path, str):
        if gel is None:
            gel=1250
        if n_z is not None:
            print 'slicing gt start z: %i, end z: %i, start x: % end x %i' %\
                  (start_z, start_z+n_z, offset_xy, offset_xy+gel)
            gt = du.load_h5(gt_path, h5_key=gt_key)[0][start_z:start_z+n_z,
                                                       offset_xy:offset_xy+gel,
                                                       offset_xy:offset_xy+gel]

        else:
            gt = du.load_h5(gt_path, h5_key=gt_key)[0][:,
                  offset_xy:offset_xy + gel,
                  offset_xy:offset_xy + gel]


    if isinstance(pred_path, str):
        pred = du.load_h5(pred_path, h5_key=pred_key)[0]
    assert(gt.shape == pred.shape)

    if defect_slices:
        print 'removing defect slices'
        defect_slices_ind = [14, 15]
        pred = np.delete(pred, defect_slices_ind, axis=0)
        gt = np.delete(gt, defect_slices_ind, axis=0)

    if slice_by_slice:
        if verbose:
            print 'slice by slice evaluation'
        splits, merges, ares, precisions, recalls = [], [], [], [], []
        all_measures = [splits, merges, ares, precisions, recalls]
        for i in range(pred.shape[0]):
            groundtruth = CremiData(gt[i][None, :, :] + 1, resolution=resolution)
            segmentation = CremiData(pred[i][None, :, :].copy()+1, resolution=resolution)
            segmentation2 = CremiData(pred[i][None, :, :] + 1, resolution=resolution)
            ni = NeuronIds(groundtruth, border_threshold=border_thresh)

            if verbose:
                print '\r %.3f' % (float(i) / pred.shape[0]),
            split, merge = ni.voi(segmentation)
            are, prec, rec = ni.adapted_rand(segmentation2)

            vals = [split, merge, are, prec, rec]
            # print 'spolit merge', split, merge
            for val, meas in zip(vals, all_measures):
                meas.append(val)
        all_measures = np.array(all_measures)
        all_vars = np.var(all_measures, 1)
        all_means = np.mean(all_measures, 1)
        # all_means[2] = all_means[2]
        if verbose:
            print 'border thresh', border_thresh, 'resolution', resolution
            print 'Variational information split:, %.3f ,+- %.3f' % (all_means[0], all_vars[0])
            print 'Variational information merge:, %.3f ,+- %.3f' % (all_means[1], all_vars[1])
            print 'Adapted Rand error F1        :, %.3f ,+- %.3f' % (1-all_means[2], all_vars[2])
            print 'Adapted Rand error precision :, %.3f ,+- %.3f' % (all_means[3], all_vars[3])
            print 'Adapted Rand error recall    :, %.3f ,+- %.3f' % (all_means[4], all_vars[4])
            print 'cremi', np.sqrt((all_means[0] + all_means[1]) * all_means[2])

            # string for easy copy to google doc
            print ','.join(['%.3f,+-%.3f' % (all_means[i], all_vars[i]) for i in range(5)])
        text = ','.join(['%.3f+-%.3f' % (all_means[i], all_vars[i]) for i in range(5)])
        return np.sqrt((all_means[0] + all_means[1]) * all_means[2])
        # return make_val_dic(all_means, all_vars), text

    else:
        # variational information of split and merge error,  i.e., H(X|Y) and H(Y|X)
        split, merge = voi(pred.copy(), gt.copy())
        # are: adapted rand error, rand precision, rand recall
        are, precision, recall = adapted_rand(pred, gt, all_stats=True)
        print 'Variational information split:       ', split
        print 'Variational information merge:       ', merge
        print 'Adapted Rand error           :       ', are
        print 'Adapted Rand error precision :       ', precision
        print 'Adapted Rand error recall    :       ', recall



class CremiData(object):
    def __init__(self, gt, resolution=4):
        self.data = gt
        # print 'gt', gt.shape
        self.resolution = [10 * resolution, resolution, resolution]


if __name__ == '__main__':

    # seg = np.ones((10, 10))
    # seg[:5, 5:] = 2.        # right top 2
    # seg[5:, 5:] = 3.        # right bottom 3
    # seg[5:, :5] = 4.        # left bottom 1
    # gt = np.copy(seg)
    #
    # seg[4, 7:9] = 3.
    # seg[5, 1:3] = 1.
    # seg[7:9, 5] = 4.
    # # seg[1:3, 4] = 2.
    #
    # print seg


    # print 'RI %.3f PR %.3f REC %.3f' % adapted_rand(seg, gt, all_stats=True)


    # print
    pred_path='./../data/nets/ft_evol_rescale_noreg/validation_net_600/slice_concat.h5'
    # pred_path='./../data/nets/ft_evol_adam/validation_net_1400/slice_concat.h5'

    # # pred_path='./data/preds/timo_first_repr_zstack.h5'
    # # pred_path='./data/preds/tmp.h5'
    # # pred_path='./../data/nets/pred_C_big/final.h5'
    # pred_path='./../data/nets/pretrain_cremi_noz_ft_180_C10/final.h5'
    # # pred_path='./data/preds/random.h5'
    # # pred_path = '/home/liory/src/superpixel_generator/data/pred_10000.h5'
    gt_path = './../data/volumes/label_CREMI_noz_small_valid.h5'
    #
    # print 'gt path'
    # print pred_path
    #
    validate_segmentation(pred_path=pred_path, gt_path=gt_path, slice_by_slice=True,
                          start_z=0, n_z=30, gel=398, offset_xy=35, defect_slices=True)

    #
    # pred_path='./../data/nets/ft_bn_dowbd/validation_net_2000/baseline_concat.h5'
    pred_path='./../data/nets/ft_evol_adam/validation_net_1400/baseline_concat.h5'

    # # pred_path='./data/preds/timo_first_repr_zstack.h5'
    # # pred_path='./data/preds/tmp.h5'
    # # pred_path='./../data/nets/pred_C_big/final.h5'
    # pred_path='./../data/nets/pretrain_cremi_noz_ft_180_C10/final.h5'
    # # pred_path='./data/preds/random.h5'
    # # pred_path = '/home/liory/src/superpixel_generator/data/pred_10000.h5'
    gt_path = './../data/volumes/label_CREMI_noz_small_valid.h5'
    #
    # print 'gt path'
    # print pred_path
    #
    validate_segmentation(pred_path=pred_path, gt_path=gt_path, slice_by_slice=True,
                          start_z=0, n_z=30, gel=398, offset_xy=35, defect_slices=True)