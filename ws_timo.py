import numpy as np
import wsdt
import h5py as h
from os.path import exists

def load_h5(path, h5_key=None, group=None, group2=None):
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
            output.append(np.array(g[key]))
    elif isinstance(h5_key, basestring):   # string
        output = [np.array(g[h5_key])]
    elif isinstance(h5_key, list):          # list
        output = list()
        for key in h5_key:
            output.append(np.array(g[key], dtype=theano.config.floatX))
    else:
        raise Exception('h5 key type is not supported')
    return output

def save_h5(path, h5_key, data, overwrite='w-'):
    f = h.File(path, overwrite)
    if isinstance(h5_key, str):
        f.create_dataset(h5_key, data=data)
    if isinstance(h5_key, list):
        for key, values in zip(h5_key, data):
            f.create_dataset(key, data=values)
    f.close()


# wsdt.wsDtSegmentation(nparrayprobab, thresholddisttrf, thresh2dofCCofmemb15,
#                          CCofSeg85, sigmaofdisttrf2, 0)
# wsdt.wsDtSegmentation(nparrayprobab, thresholddisttrf, thresh3d:65)



if __name__ == '__main__':
    memb_path = ''
    memb_path = './data/volumes/membranes_a.h5'
    threshold_dist_trf = 0.3
    thres_memb_cc = 15
    thresh_seg_cc = 85
    sigma_dist_trf = 2
    somethingunimportant = 0
    two_dim = False
    memb_probs = load_h5(memb_path)[0]

    segmentation = np.zeros((125, 1250, 1250))
    if two_dim:
        for i in range(segmentation.shape[0]):
            segmentation[i, :, :] = \
            wsdt.wsDtSegmentation(memb_probs[i, :, :],
                                  threshold_dist_trf, thres_memb_cc,
                                  thresh_seg_cc, sigma_dist_trf,
                                  somethingunimportant)
    else:
        segmentation = wsdt.wsDtSegmentation(memb_probs,
                              threshold_dist_trf, thres_memb_cc,
                              thresh_seg_cc, sigma_dist_trf,
                              somethingunimportant)

    save_h5('./data/preds/ws_3D_timo_a.h5', 'pred', segmentation, 'w')