import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append('/home/lschott_local/git/cremi_python/')
# from cremi.evaluation import NeuronIds
# from cremi.evaluation import voi
# from cremi.evaluation import adapted_rand
import numpy as np
import data_provider as du
import h5py
import numpy as np
import scipy
import scipy.sparse as sparse



class NeuronIds:

    def __init__(self, groundtruth, border_threshold = None):
        """Create a new evaluation object for neuron ids against the provided ground truth.

        Parameters
        ----------

            groundtruth: Volume
                The ground truth volume containing neuron ids.

            border_threshold: None or float, in world units
                Pixels within `border_threshold` to a label border in the
                same section will be assigned to background and ignored during
                the evaluation.
        """

        assert groundtruth.resolution[1] == groundtruth.resolution[2], \
            "x and y resolutions of ground truth are not the same (%f != %f)" % \
            (groundtruth.resolution[1], groundtruth.resolution[2])

        self.groundtruth = groundtruth
        self.border_threshold = border_threshold

        if self.border_threshold:

            # print "Computing border mask..."

            self.gt = np.zeros(groundtruth.data.shape, dtype=np.uint64)
            create_border_mask(
                groundtruth.data,
                self.gt,
                float(border_threshold)/groundtruth.resolution[1],
                0)
        else:
            self.gt = np.array(self.groundtruth.data).copy()

        # current voi and rand implementations don't work with np.uint64(-1) as
        # background label, so we make it 0 here and bump all other labels
        # print 'gt',self.gt
        # self.gt += 1
        # print 'sum', np.sum(self.gt == 0)


    def voi(self, segmentation):

        assert list(segmentation.data.shape) == list(self.groundtruth.data.shape)
        assert list(segmentation.resolution) == list(self.groundtruth.resolution)

        # print "Computing VOI..."

        return voi(np.array(segmentation.data), self.gt, ignore_groundtruth = [0])

    def adapted_rand(self, segmentation):

        assert list(segmentation.data.shape) == list(self.groundtruth.data.shape)
        assert list(segmentation.resolution) == list(self.groundtruth.resolution)

        # print "Computing RAND..."

        return adapted_rand(np.array(segmentation.data), self.gt, all_stats=True)

class CremiData(object):
    def __init__(self, gt, resolution=4):
        self.data = gt
        # print 'gt', gt.shape
        self.resolution = [10 * resolution, resolution, resolution]

    # coding=utf-8

    import numpy as np
    import scipy.sparse as sparse

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
        if np.any(seg == 0):
            print 'waarning zeros in seg, treat as background'
        # if np.any(gt == 0):
        #     print 'waarning zeros in gt, 0 labels will be ignored'


        # boundaries = dp.segmenation_to_membrane_core(gt.squeeze())[0]
        # boundaries = binary_dilation(boundaries, iterations=1)
        # gt[boundaries, 0] = 0

        # print 'after gt', np.sum(gt == 0)

        # segA is truth, segB is query
        segA = np.ravel(gt)
        segB = np.ravel(seg)

        # mask to foreground in A
        mask = (segA > 0)
        segA = segA[mask]
        segB = segB[mask]
        n = segA.size  # number of nonzero pixels in original segA

        # print 'n', n
        n_labels_A = np.amax(segA) + 1
        n_labels_B = np.amax(segB) + 1

        ones_data = np.ones(n)

        p_ij = sparse.csr_matrix((ones_data, (segA.ravel(), segB.ravel())),
                                 shape=(n_labels_A, n_labels_B),
                                 dtype=np.uint64)

        # print 'pij'
        # print p_ij.todense()

        # In the paper where adapted rand is proposed, they treat each background
        # pixel in segB as a different value (i.e., unique label for each pixel).
        # To do this, we sum them differently than others

        B_nonzero = p_ij[:, 1:]  # ind (label_gt, label_seg), so ignore 0 seg labels
        B_zero = p_ij[:, 0]

        # this is a count
        num_B_zero = B_zero.sum()

        # sum of the joint distribution
        #   separate sum of B>0 and B=0 parts
        sum_p_ij = (B_nonzero).power(2).sum() + num_B_zero

        # print 'sum pij', sum_p_ij

        # these are marginal probabilities
        a_i = p_ij.sum(1)  # sum over all seg labels overlapping one gt label (except 0 labels)
        b_i = B_nonzero.sum(0)
        # print 'ai', a_i
        # print 'bi', b_i

        sum_a = np.power(a_i, 2).sum()
        sum_b = np.power(b_i, 2).sum() + num_B_zero

        precision = float(sum_p_ij) / sum_b
        recall = float(sum_p_ij) / sum_a

        fScore = 2.0 * precision * recall / (precision + recall)
        are = 1.0 - fScore

        if all_stats:
            return (are, precision, recall)
        else:
            return are


def voi(reconstruction, groundtruth, ignore_reconstruction=[], ignore_groundtruth=[0]):
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

# coding=utf-8

import numpy as np
import scipy.sparse as sparse

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
    if np.any(seg == 0):
        print 'waarning zeros in seg, treat as background'
    # if np.any(gt == 0):
    #     print 'waarning zeros in gt, 0 labels will be ignored'


    # boundaries = dp.segmenation_to_membrane_core(gt.squeeze())[0]
    # boundaries = binary_dilation(boundaries, iterations=1)
    # gt[boundaries, 0] = 0

    # print 'after gt', np.sum(gt == 0)

    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)

    # mask to foreground in A
    mask = (segA > 0)
    segA = segA[mask]
    segB = segB[mask]
    n = segA.size  # number of nonzero pixels in original segA

    # print 'n', n
    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1

    ones_data = np.ones(n)

    p_ij = sparse.csr_matrix((ones_data, (segA.ravel(), segB.ravel())),
                             shape=(n_labels_A, n_labels_B),
                             dtype=np.uint64)

    # print 'pij'
    # print p_ij.todense()

    # In the paper where adapted rand is proposed, they treat each background
    # pixel in segB as a different value (i.e., unique label for each pixel).
    # To do this, we sum them differently than others

    B_nonzero = p_ij[:, 1:]             # ind (label_gt, label_seg), so ignore 0 seg labels
    B_zero = p_ij[:, 0]

    # this is a count
    num_B_zero = B_zero.sum()

    # sum of the joint distribution
    #   separate sum of B>0 and B=0 parts
    sum_p_ij = (B_nonzero).power(2).sum() + num_B_zero

    # print 'sum pij', sum_p_ij

    # these are marginal probabilities
    a_i = p_ij.sum(1)           # sum over all seg labels overlapping one gt label (except 0 labels)
    b_i = B_nonzero.sum(0)
    # print 'ai', a_i
    # print 'bi', b_i

    sum_a = np.power(a_i, 2).sum()
    sum_b = np.power(b_i, 2).sum() + num_B_zero

    precision = float(sum_p_ij) / sum_b
    recall = float(sum_p_ij) / sum_a

    fScore = 2.0 * precision * recall / (precision + recall)
    are = 1.0 - fScore

    if all_stats:
        return (are, precision, recall)
    else:
        return are

def create_border_mask(input_data, target, max_dist, background_label, axis=0):
    """
    Overlay a border mask with background_label onto input data.
    A pixel is part of a border if one of its 4-neighbors has different label.

    Parameters
    ----------
    input_data : h5py.Dataset or numpy.ndarray - Input data containing neuron ids
    target : h5py.Datset or numpy.ndarray - Target which input data overlayed with border mask is written into.
    max_dist : int or float - Maximum distance from border for pixels to be included into the mask.
    background_label : int - Border mask will be overlayed using this label.
    axis : int - Axis of iteration (perpendicular to 2d images for which mask will be generated)
    """
    sl = [slice(None) for d in xrange(len(target.shape))]

    for z in xrange(target.shape[axis]):
        sl[axis] = z
        border = create_border_mask_2d(input_data[tuple(sl)], max_dist)
        target_slice = input_data[tuple(sl)] if isinstance(input_data, h5py.Dataset) else np.copy(input_data[tuple(sl)])
        target_slice[border] = background_label
        target[tuple(sl)] = target_slice


def create_and_write_masked_neuron_ids(in_file, out_file, max_dist, background_label, overwrite=False):
    """
    Overlay a border mask with background_label onto input data loaded from in_file and write into out_file.
    A pixel is part of a border if one of its 4-neighbors has different label.

    Parameters
    ----------
    in_file : CremiFile - Input file containing neuron ids
    out_file : CremiFile - Output file which input data overlayed with border mask is written into.
    max_dist : int or float - Maximum distance from border for pixels to be included into the mask.
    background_label : int - Border mask will be overlayed using this label.
    overwrite : bool - Overwrite existing data in out_file (True) or do nothing if data is present in out_file (False).
    """
    if (not in_file.has_neuron_ids()) or ((not overwrite) and out_file.has_neuron_ids()):
        return

    neuron_ids, resolution, offset, comment = in_file.read_neuron_ids()
    comment = ('' if comment is None else comment + ' ') + 'Border masked with max_dist=%f' % max_dist

    path = "/volumes/labels/neuron_ids"
    group_path = "/".join(path.split("/")[:-1])
    ds_name = path.split("/")[-1]
    if (out_file.has_neuron_ids()):
        del out_file.h5file[path]
    if (group_path not in out_file.h5file):
        out_file.h5file.create_group(group_path)

    group = out_file.h5file[group_path]
    target = group.create_dataset(ds_name, shape=neuron_ids.shape, dtype=neuron_ids.dtype)
    target.attrs["resolution"] = resolution
    target.attrs["comment"] = comment
    if offset != (0.0, 0.0, 0.0):
        target.attrs["offset"] = offset

    create_border_mask(neuron_ids, target, max_dist, background_label)


def create_border_mask_2d(image, max_dist):
    """
    Create binary border mask for image.
    A pixel is part of a border if one of its 4-neighbors has different label.

    Parameters
    ----------
    image : numpy.ndarray - Image containing integer labels.
    max_dist : int or float - Maximum distance from border for pixels to be included into the mask.

    Returns
    -------
    mask : numpy.ndarray - Binary mask of border pixels. Same shape as image.
    """
    max_dist = max(max_dist, 0)

    padded = np.pad(image, 1, mode='edge')

    border_pixels = np.logical_and(
        np.logical_and(image == padded[:-2, 1:-1], image == padded[2:, 1:-1]),
        np.logical_and(image == padded[1:-1, :-2], image == padded[1:-1, 2:])
    )

    distances = scipy.ndimage.distance_transform_edt(
        border_pixels,
        return_distances=True,
        return_indices=False
    )
    return distances <= max_dist


if __name__ == '__main__':
    None