import numpy as np
import scipy.sparse as sparse
import dataset_utils as du


def validate_segmentation(pred=None, gt=None, gt_path=None, pred_path=None,
                          pred_key=None, gt_key=None, slice_by_slice=True):
    assert (gt_path != gt)  # specify either gt path or gt as np array
    assert (pred_path != pred)    # specify either raw path or raw as np array

    if isinstance(gt_path, str):
        gt = du.load_h5(gt_path, h5_key=gt_key)[0]
    if isinstance(pred_path, str):
        pred = du.load_h5(pred_path, h5_key=pred_key)[0]
    print gt.shape, pred.shape
    assert(gt.shape == pred.shape)

    if slice_by_slice:
        print 'slice by slice evaluation'
        splits, merges, ares, precisions, recalls = [], [], [], [], []
        all_measures = [splits, merges, ares, precisions, recalls]
        for i in range(pred.shape[0]):
            print '\r %.3f' % (float(i) / pred.shape[0]),
            split, merge = voi(pred[i].copy(), gt[i].copy())
            are, precision, recall = adapted_rand(pred[i][:, :, None],
                                                  gt[i][:, :, None],
                                                  all_stats=True)

            vals = [split, merge, are, precision, recall]
            for val, meas in zip(vals, all_measures):
                meas.append(val)
        all_measures = np.array(all_measures)
        all_vars = np.var(all_measures, 1)
        all_means = np.mean(all_measures, 1)
        all_means[2] = 1. - all_means[2]

        print 'Variational information split:, %.3f ,+- %.3f' % (all_means[0], all_vars[0])
        print 'Variational information merge:, %.3f ,+- %.3f' % (all_means[1], all_vars[1])
        print 'Adapted Rand error           :, %.3f ,+- %.3f' % (all_means[2], all_vars[2])
        print 'Adapted Rand error precision :, %.3f ,+- %.3f' % (all_means[3], all_vars[3])
        print 'Adapted Rand error recall    :, %.3f ,+- %.3f' % (all_means[4], all_vars[4])
        # string for easy copy to google doc
    	print ','.join(['%.3f+-%.3f' % (all_means[i], all_vars[i]) for i in range(5)])
    	return make_val_dic(all_means, all_vars)

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


# Evaluation code courtesy of Juan Nunez-Iglesias, taken from
# https://github.com/janelia-flyem/gala/blob/master/gala/evaluate.py
def adapted_rand(seg, gt, all_stats=False):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]
    Formula is given as 1 - the maximal F-score of the Rand index
    (excluding the zero component of the original labels). Adapted
    from the SNEMI3D MATLAB script, hence the strange style.
    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        whether to also return precision and recall as a 3-tuple with rand_error
    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)
    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """
    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)
    n = segA.size

    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1

    ones_data = np.ones(n)

    p_ij = sparse.csr_matrix((ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))

    a = p_ij[1:n_labels_A,:]
    b = p_ij[1:n_labels_A,1:n_labels_B]
    c = p_ij[1:n_labels_A,0].todense()
    d = np.array(b.todense()) ** 2

    a_i = np.array(a.sum(1))
    b_i = np.array(b.sum(0))

    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
    sumAB = np.sum(d) + (np.sum(c) / n)

    precision = sumAB / sumB
    recall = sumAB / sumA

    fScore = 2.0 * precision * recall / (precision + recall)
    are = 1.0 - fScore

    if all_stats:
        return (are, precision, recall)
    else:
        return are


# Evaluation code courtesy of Juan Nunez-Iglesias, taken from
# https://github.com/janelia-flyem/gala/blob/master/gala/evaluate.py
def voi(reconstruction, groundtruth, ignore_reconstruction=[],
        ignore_groundtruth=[0]):
    """Return the conditional entropies of the variation of information metric. [1]

    Let X be a reconstruction, and Y a ground truth labelling. The variation of
    information between the two is the sum of two conditional entropies:

        VI(X, Y) = H(X|Y) + H(Y|X).

    The first one, H(X|Y), is a measure of oversegmentation, the second one,
    H(Y|X), a measure of undersegmentation. These measures are referred to as
    the variation of information split or merge error, respectively.

    Parameters
    ----------
    seg : np.ndarray, int type, arbitrary shape
        A candidate segmentation.
    gt : np.ndarray, int type, same shape as `seg`
        The ground truth segmentation.
    ignore_seg, ignore_gt : list of int, optional
        Any points having a label in this list are ignored in the evaluation.
        By default, only the label 0 in the ground truth will be ignored.

    Returns
    -------
    (split, merge) : float
        The variation of information split and merge error, i.e., H(X|Y) and H(Y|X)

    References
    ----------
    [1] Meila, M. (2007). Comparing clusterings - an information based
    distance. Journal of Multivariate Analysis 98, 873-895.
    """
    (hyxg, hxgy) = split_vi(reconstruction, groundtruth, ignore_reconstruction, ignore_groundtruth)
    return (hxgy, hyxg)


def split_vi(x, y=None, ignore_x=[0], ignore_y=[0]):
    """Return the symmetric conditional entropies associated with the VI.

    The variation of information is defined as VI(X,Y) = H(X|Y) + H(Y|X).
    If Y is the ground-truth segmentation, then H(Y|X) can be interpreted
    as the amount of under-segmentation of Y and H(X|Y) is then the amount
    of over-segmentation.  In other words, a perfect over-segmentation
    will have H(Y|X)=0 and a perfect under-segmentation will have H(X|Y)=0.

    If y is None, x is assumed to be a contingency table.

    Parameters
    ----------
    x : np.ndarray
        Label field (int type) or contingency table (float). `x` is
        interpreted as a contingency table (summing to 1.0) if and only if `y`
        is not provided.
    y : np.ndarray of int, same shape as x, optional
        A label field to compare to `x`.
    ignore_x, ignore_y : list of int, optional
        Any points having a label in this list are ignored in the evaluation.
        Ignore 0-labeled points by default.

    Returns
    -------
    sv : np.ndarray of float, shape (2,)
        The conditional entropies of Y|X and X|Y.

    See Also
    --------
    vi
    """
    _, _, _ , hxgy, hygx, _, _ = vi_tables(x, y, ignore_x, ignore_y)
    # false merges, false splits
    return np.array([hygx.sum(), hxgy.sum()])


def vi_tables(x, y=None, ignore_x=[0], ignore_y=[0]):
    """Return probability tables used for calculating VI.

    If y is None, x is assumed to be a contingency table.

    Parameters
    ----------
    x, y : np.ndarray
        Either x and y are provided as equal-shaped np.ndarray label fields
        (int type), or y is not provided and x is a contingency table
        (sparse.csc_matrix) that may or may not sum to 1.
    ignore_x, ignore_y : list of int, optional
        Rows and columns (respectively) to ignore in the contingency table.
        These are labels that are not counted when evaluating VI.

    Returns
    -------
    pxy : sparse.csc_matrix of float
        The normalized contingency table.
    px, py, hxgy, hygx, lpygx, lpxgy : np.ndarray of float
        The proportions of each label in `x` and `y` (`px`, `py`), the
        per-segment conditional entropies of `x` given `y` and vice-versa, the
        per-segment conditional probability p log p.
    """
    if y is not None:
        pxy = contingency_table(x, y, ignore_x, ignore_y)
    else:
        cont = x
        total = float(cont.sum())
        # normalize, since it is an identity op if already done
        pxy = cont / total

    # Calculate probabilities
    px = np.array(pxy.sum(axis=1)).ravel()
    py = np.array(pxy.sum(axis=0)).ravel()
    # Remove zero rows/cols
    nzx = px.nonzero()[0]
    nzy = py.nonzero()[0]
    nzpx = px[nzx]
    nzpy = py[nzy]
    nzpxy = pxy[nzx, :][:, nzy]

    # Calculate log conditional probabilities and entropies
    lpygx = np.zeros(np.shape(px))
    lpygx[nzx] = xlogx(divide_rows(nzpxy, nzpx)).sum(axis=1)
                        # \sum_x{p_{y|x} \log{p_{y|x}}}
    hygx = -(px*lpygx) # \sum_x{p_x H(Y|X=x)} = H(Y|X)

    lpxgy = np.zeros(np.shape(py))
    lpxgy[nzy] = xlogx(divide_columns(nzpxy, nzpy)).sum(axis=0)
    hxgy = -(py*lpxgy)

    return [pxy] + list(map(np.asarray, [px, py, hxgy, hygx, lpygx, lpxgy]))


def contingency_table(seg, gt, ignore_seg=[0], ignore_gt=[0], norm=True):
    """Return the contingency table for all regions in matched segmentations.

    Parameters
    ----------
    seg : np.ndarray, int type, arbitrary shape
        A candidate segmentation.
    gt : np.ndarray, int type, same shape as `seg`
        The ground truth segmentation.
    ignore_seg : list of int, optional
        Values to ignore in `seg`. Voxels in `seg` having a value in this list
        will not contribute to the contingency table. (default: [0])
    ignore_gt : list of int, optional
        Values to ignore in `gt`. Voxels in `gt` having a value in this list
        will not contribute to the contingency table. (default: [0])
    norm : bool, optional
        Whether to normalize the table so that it sums to 1.

    Returns
    -------
    cont : scipy.sparse.csc_matrix
        A contingency table. `cont[i, j]` will equal the number of voxels
        labeled `i` in `seg` and `j` in `gt`. (Or the proportion of such voxels
        if `norm=True`.)
    """
    segr = seg.ravel()
    gtr = gt.ravel()
    ignored = np.zeros(segr.shape, np.bool)
    data = np.ones(len(gtr))
    for i in ignore_seg:
        ignored[segr == i] = True
    for j in ignore_gt:
        ignored[gtr == j] = True
    data[ignored] = 0
    cont = sparse.coo_matrix((data, (segr, gtr))).tocsc()
    if norm:
        cont /= float(cont.sum())
    return cont


def divide_columns(matrix, row, in_place=False):
    """Divide each column of `matrix` by the corresponding element in `row`.

    The result is as follows: out[i, j] = matrix[i, j] / row[j]

    Parameters
    ----------
    matrix : np.ndarray, scipy.sparse.csc_matrix or csr_matrix, shape (M, N)
        The input matrix.
    column : a 1D np.ndarray, shape (N,)
        The row dividing `matrix`.
    in_place : bool (optional, default False)
        Do the computation in-place.

    Returns
    -------
    out : same type as `matrix`
        The result of the row-wise division.
    """
    if in_place:
        out = matrix
    else:
        out = matrix.copy()
    if type(out) in [sparse.csc_matrix, sparse.csr_matrix]:
        if type(out) == sparse.csc_matrix:
            convert_to_csc = True
            out = out.tocsr()
        else:
            convert_to_csc = False
        row_repeated = np.take(row, out.indices)
        nz = out.data.nonzero()
        out.data[nz] /= row_repeated[nz]
        if convert_to_csc:
            out = out.tocsc()
    else:
        out /= row[np.newaxis, :]
    return out


def divide_rows(matrix, column, in_place=False):
    """Divide each row of `matrix` by the corresponding element in `column`.

    The result is as follows: out[i, j] = matrix[i, j] / column[i]

    Parameters
    ----------
    matrix : np.ndarray, scipy.sparse.csc_matrix or csr_matrix, shape (M, N)
        The input matrix.
    column : a 1D np.ndarray, shape (M,)
        The column dividing `matrix`.
    in_place : bool (optional, default False)
        Do the computation in-place.

    Returns
    -------
    out : same type as `matrix`
        The result of the row-wise division.
    """
    if in_place:
        out = matrix
    else:
        out = matrix.copy()
    if type(out) in [sparse.csc_matrix, sparse.csr_matrix]:
        if type(out) == sparse.csr_matrix:
            convert_to_csr = True
            out = out.tocsc()
        else:
            convert_to_csr = False
        column_repeated = np.take(column, out.indices)
        nz = out.data.nonzero()
        out.data[nz] /= column_repeated[nz]
        if convert_to_csr:
            out = out.tocsr()
    else:
        out /= column[:, np.newaxis]
    return out


def xlogx(x, out=None, in_place=False):
    """Compute x * log_2(x).

    We define 0 * log_2(0) = 0

    Parameters
    ----------
    x : np.ndarray or scipy.sparse.csc_matrix or csr_matrix
        The input array.
    out : same type as x (optional)
        If provided, use this array/matrix for the result.
    in_place : bool (optional, default False)
        Operate directly on x.

    Returns
    -------
    y : same type as x
        Result of x * log_2(x).
    """
    if in_place:
        y = x
    elif out is None:
        y = x.copy()
    else:
        y = out
    if type(y) in [sparse.csc_matrix, sparse.csr_matrix]:
        z = y.data
    else:
        z = y
    nz = z.nonzero()
    z[nz] *= np.log2(z[nz])
    return y


def make_val_dic(all_means, all_vars):
    val_dic = {}
    val_dic['Variational information split'] = all_means[0]
    val_dic['Variational information split_error'] = all_vars[0]
    val_dic['Variational information merge'] = all_means[1]
    val_dic['Variational information merge_error'] = all_vars[1]
    val_dic['Adapted Rand error'] = all_means[2]
    val_dic['Adapted Rand error_error'] = all_vars[2]
    val_dic['Adapted Rand error precision'] = all_means[3]
    val_dic['Adapted Rand error precision_error'] = all_vars[3]
    val_dic['Adapted Rand error recall'] = all_means[4]
    val_dic['Adapted Rand error recall_error'] = all_vars[4]
    return val_dic




if __name__ == '__main__':
    print   
    # pred_path='./data/preds/pred2_net_real_seeds_2D.h5'
    pred_path='./data/preds/ws_3D_timo_b.h5'
    # pred_path='./data/preds/random.h5'
    # pred_path = '/home/liory/src/superpixel_generator/data/pred_10000.h5'
    gt_path = './data/volumes/label_b.h5'


    print pred_path
    validate_segmentation(
        pred_path=pred_path,
        gt_path=gt_path,
        slice_by_slice=True)


