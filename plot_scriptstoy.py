import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import data_provider as dp
import validation_scripts as vs
import dataset_utils as du
from skimage import measure
import ast
from scipy import ndimage
import utils as u

def get_color_maps():
    rand_nums = np.random.rand(256, 3)
    rand_nums[:, 0] /= 100    # remove bright red colors from rgb
    rcmap = matplotlib.colors.ListedColormap(rand_nums)

    red_nums = np.zeros((256, 3))
    red_nums[:, 0] = np.random.uniform(0.7, 1, size=256)
    redcmap = matplotlib.colors.ListedColormap(red_nums)

    green_nums = np.zeros((256, 3))
    green_nums[:, 1] = np.random.uniform(0.7, 1, size=256)
    greencmap = matplotlib.colors.ListedColormap(green_nums)
    return rcmap, redcmap, greencmap

def find_best_cherry(gt_seg, std_ws_seg, nn_ws_seg, criterium='diff'):
    # splits, merges, are, prec, rec
    print 'eval nnws'
    all_evals_nn_ws = vs.validate_segmentation(nn_ws_seg+1, gt_seg+1, verbose=False, return_all_vals=True)[2]
    print 'eval stdws'
    if 'best_nn_ws' not in criterium:
        all_evals_std_ws = vs.validate_segmentation(std_ws_seg+1, gt_seg+1, verbose=False, return_all_vals=True)[2]
    else:
        all_evals_std_ws = np.zeros_like(all_evals_nn_ws) - 1

    if criterium == 'diff':
        diff = all_evals_nn_ws - all_evals_std_ws
        print 'diff', diff
        worst_best_stdws_insd = np.argsort(diff)[:n_plots]
        print 'diff sorted', diff[worst_best_stdws_insd]
    elif criterium == 'worst_best':
        best_inds = np.argsort(all_evals_nn_ws, 0)[:n_plots]
        worst_best_stdws_insd = best_inds[np.argsort(all_evals_std_ws[best_inds])][::-1]
    elif criterium == 'best_nn_ws':
        print 'selecting best nn ws', all_evals_nn_ws
        worst_best_stdws_insd = np.argsort(all_evals_nn_ws)[:n_plots][::-1]
    print 'the indices you want', worst_best_stdws_insd

    return worst_best_stdws_insd, all_evals_std_ws[worst_best_stdws_insd], all_evals_nn_ws[worst_best_stdws_insd]


def fix_color_table(gt_seg, nn_ws_seg, std_ws_seg):
    sm = du.SeedMan()
    new_std_ws_seg = np.empty_like(gt_seg)
    new_nn_ws_seg = np.empty_like(gt_seg)
    for i, label_slice in enumerate(gt_seg):
        gt_seg[i] = measure.label(label_slice)

    all_seeds = []
    for z, (label_slice, stdws_slice, nnws_slice) in enumerate(zip(gt_seg, std_ws_seg, nn_ws_seg)):
        seeds = sm.get_seed_coords_gt(np.copy(label_slice))
        gt_ids = []
        all_seeds.append(seeds)
        for seed in seeds:
            gt_id = label_slice[seed[0], seed[1]]
            if gt_id in gt_ids:
                print 'warning double ids'
            gt_ids.append(gt_id)

            sdtws_id = stdws_slice[seed[0], seed[1]]
            nnws_id = nnws_slice[seed[0], seed[1]]

            new_std_ws_seg[z][stdws_slice == sdtws_id] = gt_id
            print 'seed ids', sdtws_id, 'gt', gt_id, np.sum([stdws_slice == sdtws_id])

            new_nn_ws_seg[z][nnws_slice == nnws_id] = gt_id

        #
        # f, ax = plt.subplots(3)
        # ax[0].imshow(gt_seg[z])
        # ax[1].imshow(new_std_ws_seg[z])
        # ax[2].imshow(new_nn_ws_seg[z])
        # print 'label', label_slice
        # print 'std ws', new_std_ws_seg[z]
        # print 'nnws', new_nn_ws_seg[z]
        # plt.show()
    return gt_seg, new_nn_ws_seg, new_std_ws_seg, all_seeds


def set_zero_except_boundaries(seg, n_dil=2):
    boundary = dp.segmenation_to_membrane_wrapper(seg)
    for i in range(len(boundary)):
        boundary[i] = ndimage.binary_dilation(boundary[i], iterations=n_dil)
        seg[boundary[i] != 1] = 0
    return seg


def make_toy_data_plot(gt_seg, std_ws_seg, nn_ws_seg, best_edges, nn_hm,
                       seeds=None,
                       slices=None,
                       criterum='best_nn_ws',      # also best_nn_ws, o diff
                       indices=None):

    print 'nn_ws seg', nn_ws_seg.shape, 'std_ws seg', std_ws_seg.shape, 'gt seg', gt_seg.shape

    bnn_wss, are_stdws, are_nnws = find_best_cherry(gt_seg, std_ws_seg, nn_ws_seg, criterium=criterum)
    if slices is None:
        slices = bnn_wss
        print 'fixing colors'
    else:
        bnn_wss = np.arange(len(slices))

    gt_seg = gt_seg[bnn_wss]
    std_ws_seg = std_ws_seg[bnn_wss]
    nn_ws_seg = nn_ws_seg[bnn_wss]
    best_edges = best_edges[bnn_wss]

    # gt_seg, std_ws_seg, nn_ws_seg = fix_color_table(gt_seg[bnn_wss], std_ws_seg[bnn_wss], nn_ws_seg[bnn_wss])
    static_data = dp.load_h5(static_data_p, slices=slices)[0][:, :, pad:-pad, pad:-pad]
    nn_hm = dp.load_h5(nn_hm_p, slices=slices)[0]

    print 'worst best std', bnn_wss

    tri_mask = np.zeros(static_data[0, 0, :, :].shape, dtype=np.bool)
    tri_l = np.mask_indices(gt_seg.shape[-1], np.tril)
    tri_mask[tri_l] = True
    tri_l_mask = tri_mask.copy()
    tri_mask = np.fliplr(tri_mask)

    # gs1 = gridspec.GridSpec(0.5, 0.5)
    # gs1.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.
    if seeds is not None:
        seeds = np.array(seeds[0])

    for j in range(10):
        for i in range(n_plots):
            rcmap, redcmap, greencmap = get_color_maps()


            fig, ax = plt.subplots(1, 4)
            fig.canvas.set_window_title('i_%i_real slice %i' % (i, bnn_wss[i]))
            fig.tight_layout()

            ax = u.make_axis_great_again(ax)

            # normalize so colors are always the same
            norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max((gt_seg[i], std_ws_seg[i], nn_ws_seg[i])))

            nn = nn_ws_seg[i]

            diag_image = static_data[i, 1, :, :]
            diag_image /= np.max(diag_image)
            diag_image[~tri_l_mask] = static_data[i, 0, :, :][~tri_l_mask]

            print 'best edges', best_edges.shape, 'static', static_data.shape

            # ax[0, 0].set_title('raw and edges')
            ax[0, 0].imshow(diag_image, interpolation='none', cmap='gray')
            ax[0, 0].axis('off')

            # plt.savefig('../data/tmp/cherries_%i.pdf' % i, bbox_inches='tight')
            # plt.show(block=False)



            gt = norm(gt_seg[i].copy())
            # ax[1, 0].set_title('gt')
            # ax[0, 1].imshow(diag_image*256, interpolation='none', cmap='gray')
            if seeds is not None:
                print 'adding seeds'

                ax[1, 0].scatter(seeds[:, 1], seeds[:, 0])
            ax[1, 0].imshow(gt, interpolation='none', alpha=1, cmap=rcmap)
            ax[1, 0].axis('off')

            ws = norm(std_ws_seg[i].copy())
            # errors = np.ma.masked_where(std_ws_seg[i] == gt_seg[i], ws)
            # errorsnn = np.ma.masked_where(nn_ws_seg[i] == gt_seg[i], nn)

            # diag_image = best_edges[i, :, :]
            # diag_image /= np.max(diag_image)
            # diag_image[tri_mask] = static_data[i, 0, :, :][tri_mask]

            ax[2, 0].imshow(best_edges[i, :, :], interpolation='none', cmap='gray')
            ax[2, 0].imshow(ws, interpolation='none', alpha=0.4, cmap=rcmap)
            # ax[1, 0].imshow(errorsnn, interpolation='none', cmap=greencmap, alpha=0.4)
            # ax[1, 0].imshow(errors, interpolation='none', cmap=redcmap, alpha=0.4)

            # ax[2, 0].set_title('std ws, Rand Error %.3f' % are_stdws[i])
            ax[2, 0].axis('off')

            ax[3, 0].imshow(nn_hm[i, :, :], interpolation='none', cmap='gray')
            ax[3, 0].imshow(nn, interpolation='none', alpha=0.4, cmap=rcmap)
            # ax[1, 1].imshow(errorsnn, interpolation='none', cmap=redcmap, alpha=0.4)
            # ax[3, 0].set_title('nn ws, Rand Error %.3f' % are_nnws[i])
            ax[3, 0].axis('off')
            print 'enter in style[[2, 726, 340], [2, 726, 340]]'
            plt.savefig('../data/tmp/cherries_toyi_color_%i.pdf' % j, bbox_inches='tight')
            plt.show()

    # i = raw_input()     # enter in style[[2, 726, 340], [2, 726, 340]]
    # i = str([[2, 460, 110], [2, 470, 305], [2, 685, 1124]])
    # i = str([[0, 470, 305]])      cherry for 51 comparison
    # i = str([[3, 66, 465]])
    # i = str([[0, 95, 900], [0, 440, 995]])
    i = str([[0, 476, 1032], [0, 122, 960], [0, 4, 1066], [0, 198, 1143], [0, 299, 1102], [0, 350, 1117],
              [0, 446, 1118], [0, 423, 1159], [0, 464, 815], [0, 244, 640]])
    if indices is None:
        indices = ast.literal_eval(i)
    indices = np.array(indices)


    gt_ind = indices[:, 0]
    x_ind = indices[:, 2]
    y_ind = indices[:, 1]
    static = static_data[gt_ind[0], 0, :, :]
    print 'stat', static.shape

    gt = gt_seg[gt_ind[0]]
    ws = std_ws_seg[gt_ind[0]]
    nn = nn_ws_seg[gt_ind[0]]

    new_gt = np.zeros_like(gt)
    new_ws = np.zeros_like(gt)
    new_nn = np.zeros_like(gt)

    for counter, (x, y) in enumerate(zip(x_ind, y_ind)):
        gt_Id = gt[x, y]
        ws_Id = ws[x, y]
        nn_Id = nn[x, y]
        new_gt[gt == gt_Id] = counter + 1
        new_ws[ws == ws_Id] = counter + 1
        new_nn[nn == nn_Id] = counter + 1

    # id_ind = [indices[2], indices[1]]
    # Id = gt_seg[gt_ind, id_ind[0], id_ind[1]]


    new_gt[new_gt != 0] = 1
    new_ws[new_ws != 0] = 1
    new_nn[new_nn != 0] = 1

    f, ax = plt.subplots(1)



    # ax[0].imshow(new_gt, interpolation='none')
    # ax[1].imshow(new_ws, interpolation='none')
    # ax[2].imshow(new_nn, interpolation='none')

    steffens_rgb = np.zeros((new_gt.shape[0], new_gt.shape[1], 3))


    print np.sum(new_gt)
    print np.sum(new_ws)
    print np.sum(new_ws)
    print new_gt.shape[0]**2

    for bg, g_op in zip([True, False], ([np.equal, np.not_equal])):
        for bw, ws_op in zip([True, False], ([np.equal, np.not_equal])):
            for bn, nn_op in zip([True, False], ([np.equal, np.not_equal])):

                bbb = g_op(new_gt, 1) & ws_op(new_ws, 1) & nn_op(new_nn, 1)
                print 'bbb', bg, bw, bn, np.sum(bbb)
                for i, b in enumerate([bg, bw, bn]):
                    if b:
                        steffens_rgb[:, :, i][bbb] = 1
    print np.sum(steffens_rgb)

    # for cropping
    sx, sy = [630, 0]
    ex, ey = [1176, 850]
    # [sx:ex, sy:ey]

    steffens_rgb = np.ma.masked_where(steffens_rgb != 0, steffens_rgb)
    ax.imshow(static[sx:ex, sy:ey], interpolation='none', cmap='gray')
    ax.imshow(steffens_rgb[sx:ex, sy:ey], interpolation='none', alpha=0.5)
    # ax.imshow(static, interpolation='none', cmap='gray')
    # ax.imshow(steffens_rgb, interpolation='none', alpha=0.5)
    ax.axis('off')

    # ax.imshow(steffens_rgb, interpolation='none')
    plt.savefig('../data/tmp/differences.pdf', bbox_inches='tight')

    # plt.show()

    # i = 3

    diag_image = static_data[i, 0, :, :]
    diag_image /= np.max(diag_image)
    diag_image[tri_mask] = static_data[i, 1, :, :][tri_mask] / 2

    f, ax = plt.subplots(1, 1)
    ax[0].set_title('a) Raw Image and Edge Detector')
    ax[0].imshow(diag_image, interpolation='none', cmap='gray')
    ax[0].axis('off')

    # ax[1].set_title('b) Groundtruth Segmentation')

    ax[1].imshow(gt_seg[i], interpolation='none')
    seeds = np.array(seeds[0])
    ax[1].axis('off')

    # ax[2].set_title('c) Watershed Segmentation')
    ax[2].imshow(std_ws_seg[i], interpolation='none')
    ax[2].axis('off')

    ax.imshow(nn_ws_seg[i], interpolation='none')
    ax.axis('off')
    plt.savefig('../data/tmp/figure.pdf', format='pdf', bbox_inches='tight')
    # plt.show()


def make_toy_data_graph():
    PRIM = [0.935, 0.851, 0.666]
    PRIM_sig = [0.8, 3.6, 1.7]

    PWS = [0.935, 0.851, 0.668]
    PWS_sig = [0.8, 1.7, 1.7]

    kruskal = [0.935, 0.818, 0.668]
    kruskal_sig = [0.8, 1.7, 1.7]

    nn_seg = [0.942, 0.875,	0.678]
    nn_seg_sig = [0.8, 1.7, 1.8]

    RAW = [0.86, 0.581, 0.450]
    RAW_sig = [0.016, 0.018]

    sigmas = [0.3, 0.6, 0.9]

    fig, ax = plt.subplots()
    PRIM, = ax.plot(sigmas, PRIM)
    PWS, = ax.plot(sigmas, PWS)
    kruskal, = ax.plot(sigmas, kruskal)
    nn_seg, = ax.plot(sigmas, nn_seg)
    RAW, = ax.plot(sigmas, RAW)

    # plt.legend([stwsl, nnsegl, justws_segl], ['Standard WS + Edge detector',
    #                                           'NN + Edge detector',
    #                                           'WS + Gauss'])
    plt.savefig('../data/tmp/plot_toy_eval.pdf', format='pdf', bbox_inches='tight')
    print 'saved'
    plt.show()



if __name__ == '__main__':

    # plots RI over sigmas
    # make_toy_data_graph()
    # exit()




    volume_path = '../data/volumes/'
    dataset = 'toy_nh0_sig6_valid'     # CREMI_noz_test, toy_nh0_sig6_valid
    network = 'dt_s6_np_st_cont'            # dt_s3_st_cont, best cremi
    net_file = 'net_600/'           # net_600
    add_input = 'all_'                  # normally '', for toy all_
    use_fixed = ''                # fixed_ if you already precomputed the fixed labels
    slices = None
    indices = None

    # data paths
    static_data_p = volume_path + add_input + 'input_' + dataset + '.h5'

    # segmenation
    nn_ws_seg_p = '../data/nets/%s/validation_%s/%sslice_concat.h5' % (network, net_file, use_fixed)
    nn_hm_p = '../data/nets/%s/validation_%s/height_concat.h5' % (network, net_file)
    std_ws_seg_p = '../data/tmp/best_seg_sig6.h5'
    best_edges = '../data/tmp/best_edges_sig6.h5'
    # std_ws_seg_p = '../data/nets/%s/validation_%s/%sbaseline_concat.h5' % (network, net_file, use_fixed)
    gt_seg_p = volume_path + '%slabel_' % use_fixed + dataset + '.h5'
    pad = 70 / 2
    n_plots = 50

    slices = [59]

    print 'using slices', slices

    # sx, sy = (0, 0)
    # ex, ey = (-1, -1)
    if slices is not None:
        n_plots = min(n_plots, len(slices))

    # gt_seg = dp.load_h5(gt_seg_p, slices=slices)[0]
    gt_seg = dp.load_h5(gt_seg_p, slices=slices)[0][:, pad:-pad, pad:-pad]
    nn_ws_seg = dp.load_h5(nn_ws_seg_p, slices=slices)[0]
    std_ws_seg = dp.load_h5(std_ws_seg_p, slices=slices)[0]
    best_edges = dp.load_h5(best_edges, slices=slices)[0]
    nn_hm = dp.load_h5(nn_hm_p, slices=slices)[0][:, pad:-pad, pad:-pad]

    print 'gt shape', gt_seg.shape, 'nn ws seg', nn_ws_seg.shape, 'std ws', std_ws_seg.shape, 'beedges', best_edges.shape
    gt_seg, nn_ws_seg, std_ws_seg, gt_seeds = fix_color_table(gt_seg, nn_ws_seg, std_ws_seg)

    make_toy_data_plot(gt_seg, std_ws_seg,
                       nn_ws_seg, best_edges,
                       nn_hm,
                       seeds=gt_seeds,
                       slices=slices,
                       criterum='diff',           # best_nn_ws, diff, worst_best
                       indices=indices)                # which processes (ids) to select from image

