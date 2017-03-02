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
import time

def gen_colormaps():
    # np.random.seed(339)
    np.random.seed(int(time.time())) #39
    # np.random.seed(439)
    rand_nums = np.random.rand(256, 3)
    # rand_nums[:, 0] /= 100    # remove bright red colors from rgb
    rand_nums[0, 0] = 0
    rcmap = matplotlib.colors.ListedColormap(rand_nums)

    red_nums = np.zeros((256, 3))
    red_nums[:, 0] = np.random.uniform(0.7, 1, size=256)
    redcmap = matplotlib.colors.ListedColormap(red_nums)

    green_nums = np.zeros((256, 3))
    green_nums[:, 1] = np.random.uniform(0.7, 1, size=256)
    greencmap = matplotlib.colors.ListedColormap(green_nums)
    return  rcmap


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


def fix_color_table(gt_seg, std_ws_seg, nn_ws_seg):
    sm = du.SeedMan()
    new_std_ws_seg = np.empty_like(gt_seg)
    new_nn_ws_seg = np.empty_like(gt_seg)
    for i, label_slice in enumerate(gt_seg):
        gt_seg[i] = measure.label(label_slice)

    for z, (label_slice, stdws_slice, nnws_slice) in enumerate(zip(gt_seg, std_ws_seg, nn_ws_seg)):
        seeds = sm.get_seed_coords_gt(np.copy(label_slice), minsize=5)
        gt_ids = []
        for seed in seeds:
            gt_id = label_slice[seed[0], seed[1]]
            if gt_id in gt_ids:
                print 'warning double ids'
            gt_ids.append(gt_id)

            sdtws_id = stdws_slice[seed[0], seed[1]]
            nnws_id = nnws_slice[seed[0], seed[1]]

            new_std_ws_seg[z][stdws_slice == sdtws_id] = gt_id

            new_nn_ws_seg[z][nnws_slice == nnws_id] = gt_id
        # f, ax = plt.subplots(3)
        # ax[0].imshow(gt_seg[z])
        # ax[1].imshow(new_std_ws_seg[z])
        # ax[2].imshow(new_nn_ws_seg[z])
        # print label_slice
        # print new_std_ws_seg[z]
        # print new_nn_ws_seg[z]
        # plt.show()
    return gt_seg, new_std_ws_seg, new_nn_ws_seg


def set_zero_except_boundaries(seg, n_dil=2):
    boundary = dp.segmenation_to_membrane_wrapper(seg)
    for i in range(len(boundary)):
        boundary[i] = ndimage.binary_dilation(boundary[i], iterations=n_dil)
        seg[boundary[i] != 1] = 0
    return seg


def make_cremi_data_plot(gt_seg, std_ws_seg, nn_ws_seg, nn_hm,
                       slices=None,
                       criterum='best_nn_ws',      # also best_nn_ws, o diff
                        seeds=None,
                       indices=None,
                    single_plots=True,
                         coords=None):

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

    print "slices = ",slices

    # gt_seg, std_ws_seg, nn_ws_seg = fix_color_table(gt_seg[bnn_wss], std_ws_seg[bnn_wss], nn_ws_seg[bnn_wss])
    static_data = dp.load_h5(static_data_p, slices=slices)[0][:, :, pad:-pad, pad:-pad]
    #nn_hm = dp.load_h5(nn_hm_p, slices=slices)[0]
    nn_hm = np.min(dp.load_h5(nn_hm_p, slices=slices)[0],axis=3)


    print 'worst best std', bnn_wss

    tri_mask = np.zeros(static_data[0, 0, :, :].shape, dtype=np.bool)
    tri_l = np.mask_indices(gt_seg.shape[-1], np.tril)
    tri_mask[tri_l] = True
    tri_l_mask = tri_mask.copy()
    tri_mask = np.fliplr(tri_mask)


    # gs1 = gridspec.GridSpec(0.5, 0.5)
    # gs1.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.

    for i in range(n_plots):
        for j in range(25):
            rcmap = gen_colormaps()
            fig, ax = plt.subplots(2, 2)
            fig.canvas.set_window_title('i_%i_real slice%i' %(i, bnn_wss[i]))
            fig.tight_layout()

            ax = u.make_axis_great_again(ax)

            # normalize so colors are always the same
            norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max((gt_seg[i], std_ws_seg[i], nn_ws_seg[i])))

            diag_image = static_data[i, 0, :, :].copy()
            raw = static_data[i, 0, :, :].copy()
            probs = static_data[i, 1, :, :].copy()
            diag_image /= np.max(diag_image)
            nn_hm[nn_hm == -np.inf] = 0
            nn_hm[i, :, :] /= np.max(nn_hm[i, :, :])

            diag_image[tri_mask] = probs[:, :][tri_mask]

            # top left
            nn_seg = norm(nn_ws_seg[i].copy())
            nn = norm(static_data[i, 0, :, :]).copy()

            diag_seg = np.zeros_like(nn)
            diag_seg[~tri_mask] = nn[~tri_mask]
            diag_seg = np.ma.masked_where(tri_mask, diag_seg)

            if coords is not None:
                xlimits = [coords[i]['x']-coords[i]['xel'], coords[i]['x']+coords[i]['xel']]
                ylimits = [coords[i]['y']-coords[i]['yel'], coords[i]['y']+coords[i]['yel']]

            # edges only
            # nn_eroded = set_zero_except_boundaries(nn.copy(), n_dil=2)
            # nn_eroded = np.ma.masked_where(nn_eroded == 0, nn_eroded)       # creates invisible part

            if single_plots:
                fig, axs = plt.subplots()
                # fig.canvas.set_window_title('i_%i_real slice%i_%i' %(i, bnn_wss[i]))
                fig.tight_layout()
            else:
                axs = ax[0, 0]

            x,y = 0, 0

            # axs = u.make_axis_great_again(ax)
            # ax[0, 0].set_title('raw and edges')
            # ax[0, 0].imshow(diag_image, interpolation='none', cmap='gray')
            axs.imshow(raw, interpolation='none', cmap='gray')
            # ax[0, 0].imshow(diag_seg, interpolation='none', cmap=rcmap, alpha=0.3)
            axs.axis('off')
            axs.set_xlim(xlimits[0], xlimits[1])
            axs.set_ylim(ylimits[0], ylimits[1])
            if single_plots:
                fig.savefig('../data/tmp/cremi_png/picked_cherries_raw_%i_%i.png'%(slices[i], j))
                fig.savefig('../data/tmp/cremi_pdf/picked_cherries_raw_%i_%i.pdf' %(slices[i], j))
                print "saving to ",'../data/tmp/cremi_pdf/picked_cherries_raw_%i_%i.pdf'%(slices[i], j)
                fig, axs = plt.subplots()
                fig.tight_layout()
            else:
                axs = ax[0,1]
            # plt.show(block=False)


            gt = norm(gt_seg[i].copy())
            # axs.set_title('gt')
            axs.imshow(raw, interpolation='none', cmap='gray')
            axs.imshow(gt, interpolation='none', cmap=rcmap, alpha=0.5)
            axs.axis('off')
            axs.set_xlim(xlimits[0], xlimits[1])
            axs.set_ylim(ylimits[0], ylimits[1])
            if seeds is not None:
                print 'adding seeds'
                si = np.array(seeds[i])
                axs.scatter(si[:, 1], si[:, 0], marker='.', c='w', s=1)      # move one away from boundary for visibility

            if single_plots:
                fig.savefig('../data/tmp/cremi_png/picked_cherries_gt_%i_%i.png'%(slices[i], j))
                fig.savefig('../data/tmp/cremi_pdf/picked_cherries_gt_%i_%i.pdf' %(slices[i], j))
                fig, axs = plt.subplots()
                fig.tight_layout()
            else:
                axs = ax[1,0]

            ws = norm(std_ws_seg[i].copy())
            errors = np.ma.masked_where(std_ws_seg[i] == gt_seg[i], ws)
            errorsnn = np.ma.masked_where(nn_ws_seg[i] == gt_seg[i], nn)

            axs.imshow(probs, interpolation='none', cmap='gray')
            # ax[1, 0].imshow(ws, interpolation='none', cmap=rcmap, alpha=0.4)
            axs.imshow(ws, interpolation='none', cmap=rcmap, alpha=0.8)
            # ax[1, 0].imshow(errors, interpolation='none', cmap=redcmap, alpha=0.4)

            # axs.set_title('std ws, Rand Error %.3f' % are_stdws[i])
            axs.set_xlim(xlimits[0], xlimits[1])
            axs.set_ylim(ylimits[0], ylimits[1])
            axs.axis('off')

            if single_plots:
                fig.savefig('../data/tmp/cremi_png/picked_cherries_distws_%i_%i.png'%(slices[i], j))
                fig.savefig('../data/tmp/cremi_pdf/picked_cherries_distws_%i_%i.pdf' %(slices[i], j))
                fig, axs = plt.subplots()
                fig.tight_layout()
            else:
                axs = ax[1, 1]

            axs.imshow(nn_hm[i], interpolation='none', cmap='gray')
            axs.imshow(nn_seg, interpolation='none', cmap=rcmap, alpha=0.8)
            # ax[1, 1].imshow(errorsnn, interpolation='none', cmap=redcmap, alpha=0.4)
            # axs.set_title('nn ws, Rand Error %.3f' % are_nnws[i])
            axs.set_xlim(xlimits[0], xlimits[1])
            axs.set_ylim(ylimits[0], ylimits[1])
            axs.axis('off')

            if single_plots:
                fig.savefig('../data/tmp/cremi_png/picked_cherries_learned_ws_%i_%i.png'%(slices[i], j))
                fig.savefig('../data/tmp/cremi_pdf/picked_cherries_learned_ws_%i_%i.pdf' %(slices[i], j))

            # ax[2, 0].imshow(probs, interpolation='none', cmap='gray')
            # # ax[2, 0].imshow(nn_seg, interpolation='none', cmap=rcmap, alpha=0.5)
            # ax[2, 0].imshow(errors, interpolation='none', cmap=redcmap, alpha=0.5)
            # ax[2, 0].set_title('std ws error')
            # ax[2, 0].axis('off')
            #
            # ax[2, 1].imshow(nn_hm[i], interpolation='none', cmap='gray')
            # # ax[2, 0].imshow(errorsnn, interpolation='none', cmap=rcmap, alpha=0.5)
            # ax[2, 1].imshow(errorsnn, interpolation='none', cmap=redcmap, alpha=0.5)
            # ax[2, 1].set_title('std ws error')
            # ax[2, 1].axis('off')

            if not single_plots:
                plt.savefig('../data/tmp/cremi_pdf/picked_cherries_%i_%i.pdf'%(slices[i], j), bbox_inches='tight')
                plt.savefig('../data/tmp/cremi_png/picked_cherries_%i_%i.png'%(slices[i], j), bbox_inches='tight')
                # plt.show()

    print "early_finish"
    return
    # i = raw_input()     # enter in style[[2, 726, 340], [2, 726, 340]]
    # i = str([[2, 460, 110], [2, 470, 305], [2, 685, 1124]])
    # i = str([[0, 470, 305]])      cherry for 51 comparison
    # i = str([[3, 66, 465]])
    # i = str([[0, 95, 900], [0, 440, 995]])
    # i = str([[0, 476, 1032], [0, 122, 960], [0, 4, 1066], [0, 198, 1143], [0, 299, 1102], [0, 350, 1117],
    #           [0, 446, 1118], [0, 423, 1159], [0, 464, 815], [0, 244, 640]])
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

    plt.show()

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
    ax[1].axis('off')

    # ax[2].set_title('c) Watershed Segmentation')
    ax[2].imshow(std_ws_seg[i], interpolation='none')
    ax[2].axis('off')

    ax.imshow(nn_ws_seg[i], interpolation='none')
    ax.axis('off')
    plt.savefig('../data/tmp/figure.pdf', format='pdf', bbox_inches='tight')
    # plt.show()


def make_network_plot(gt_seg, nn_ws_seg, std_ws_seg, nn_hm_p, slices, thresh=50):
    print 'making network plot'
    static_data = dp.load_h5(static_data_p, slices=slices)[0][:, :, pad:-pad, pad:-pad]
    nn_hm = dp.load_h5(nn_hm_p, slices=slices)[0]


    nn_hm[nn_hm == -np.inf] = 0
    nn_hm /= np.max(nn_hm)

    nn_ws_seg[nn_hm > thresh / 100.] = 0
    ones = np.ones_like(nn_hm) / 2
    nn_hm_mask = np.ma.masked_where(nn_hm <= thresh / 100., ones)

    nn_hm[nn_hm > thresh / 100.] = 00.00000

    # me them
    Id = nn_ws_seg[0, 290, 228]
    me = np.zeros_like(gt_seg[0])
    me[nn_ws_seg[0] == Id] = 1
    them = np.zeros_like(gt_seg[0])
    them[(nn_ws_seg[0] != Id) & (nn_ws_seg[0] != 0)] = 1
    nobody = np.zeros_like(gt_seg[0])
    nobody[(them != 1) & (me != 1)] = 1

    # plt.hist((np.log(nn_hm + 1)).flatten(), 100)
    # plt.show()

    i = 0

    fig, ax = plt.subplots(1, 1)
    ax = u.make_axis_great_again(ax)
    fig.canvas.set_window_title('i' + str(slices[0]))
    ax[0, 0].imshow(static_data[0, 0, :, :], interpolation='none', cmap='gray')
    ax[0, 0].axis('off')
    fig.tight_layout()
    plt.savefig('../data/tmp/network_plot_%i' % i)
    i += 1


    fig, ax = plt.subplots(1, 1)
    ax = u.make_axis_great_again(ax)
    fig.canvas.set_window_title('i' + str(slices[0]))
    ax[0, 0].imshow(static_data[0, 1, :, :], interpolation='none', cmap='gray')
    ax[0, 0].axis('off')
    fig.tight_layout()
    plt.savefig('../data/tmp/network_plot_%i' % i)
    i += 1

    nn_hm[nn_hm > thresh / 100.] = 0.00000001

    fig, ax = plt.subplots(1, 1)
    ax = u.make_axis_great_again(ax)
    fig.canvas.set_window_title('i' + str(slices[0]))
    nn_hm = np.log(nn_hm)
    ax[0, 0].imshow(nn_hm[0], interpolation='none', cmap='gray')
    # ax[0, 0].axis('off')
    # plt.savefig('../data/tmp/network_plot_%i' % i)
    # i += 1
    # plt.close()

    # fig, ax = plt.subplots(1, 1)
    # ax = u.make_axis_great_again(ax)
    # fig.canvas.set_window_title('i' + str(slices[0]))
    nn_hm_mask[0, 0, 0] = 0 # hack
    ax[0, 0].imshow(nn_hm_mask[0], interpolation='none', cmap='Greens')
    ax[0, 0].axis('off')
    fig.tight_layout()
    # plt.show()
    plt.savefig('../data/tmp/network_plot_%i' % i)
    plt.show()
    i += 1
    plt.close()

    fig, ax = plt.subplots(1, 1)
    ax = u.make_axis_great_again(ax)
    fig.canvas.set_window_title('i' + str(slices[0]))
    ax[0, 0].imshow(me, interpolation='none', cmap='gray')
    ax[0, 0].axis('off')
    fig.tight_layout()
    plt.savefig('../data/tmp/network_plot_%i' % i)
    i += 1
    plt.close()

    fig, ax = plt.subplots(1, 1)
    ax = u.make_axis_great_again(ax)
    fig.canvas.set_window_title('i' + str(slices[0]))
    ax[0, 0].imshow(them, interpolation='none', cmap='gray')
    ax[0, 0].axis('off')
    fig.tight_layout()
    plt.savefig('../data/tmp/network_plot_%i' % i)
    i += 1
    plt.close()

    fig, ax = plt.subplots(1, 1)
    ax = u.make_axis_great_again(ax)
    fig.canvas.set_window_title('i' + str(slices[0]))
    ax[0, 0].imshow(nobody, interpolation='none', cmap='gray')
    ax[0, 0].axis('off')
    fig.tight_layout()
    plt.savefig('../data/tmp/network_plot_%i' % i)
    i += 1
    plt.close()



    # ax[3, 0].imshow(nn_ws_seg[0, :, :], interpolation='none', cmap=rcmap)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':

    volume_path = '../data/volumes/'
    dataset = 'CREMI_noz_test'     # CREMI_noz_test, label_nh0_sig3_valid
    network = 'best_cremi'            # dt_s3_st_cont, best cremi
    net_file = '0'           # net_600
    add_input = ''                  # normally '', for toy all_

    slices = None
    indices = None

    # data paths
    static_data_p = volume_path + add_input + 'input_' + dataset + '.h5'

    # segmenation

    # nn_ws_seg_p = '../data/nets/%s/validation_%s/slice_concat.h5' % (network, net_file)
    # std_ws_seg_p = '../data/nets/%s/validation_%s/baseline_concat.h5' % (network, net_file)

    nn_ws_seg_p = '../data/nets/%s/validation_%s/fixed_slice_concat.h5' % (network, net_file)
    std_ws_seg_p = '../data/nets/%s/validation_%s/fixed_baseline_concat.h5' % (network, net_file)
    nn_hm_p = '../data/nets/%s/validation_%s/height_concat.h5' % (network, net_file)
    nn_hm_p = '../data/nets/%s/validation_%s/pred_nq_path_concat.h5' % (network, net_file)
    gt_seg_p = volume_path + 'fixed_label_' + dataset + '.h5'
    pad = 70 / 2
    n_plots = 150

    slices = [39, 116, 51]#[39, 116, 51]#[20,39,116,121,126]#[116]#[20,39,116,121,126]#range(150)         # for network file
    xel = 75
    yel = 250
    x, y = 210, 210

    # cdict = [{'x': 210, 'y': 210, 'xel': 75, 'yel': 250}] #20
    #cdict = [{'x': 210, 'y': 210, 'xel': 75, 'yel': 250}] #39
    # cdict = [{'x': 450, 'y': 800, 'xel': 250, 'yel': 75}] #116
    # cdict = [{'x': 800, 'y': 800, 'xel': 250, 'yel': 125}]   #39
    # cdict = [{'x': 800, 'y': 260, 'xel': 150, 'yel': 150}]   #51
    # cdict = [{'x': 300, 'y': 500, 'xel': 150, 'yel': 150}] #13
    cdict = [{'x': 210, 'y': 210, 'xel': 75, 'yel': 250},
             {'x': 450, 'y': 800, 'xel': 250, 'yel': 75},
             {'x': 800, 'y': 260, 'xel': 150, 'yel': 150}]



    # slices = [138, 96, 51, 50, 52]      # cherry slices for comparison, 51 current favorite
    # slices = [24, 82, 83, 81, 93, 97]     # our best nns, 83 steffens favourite




    # slices = [126]
    # indices = [[0, 476, 1032], [0, 122, 960], [0, 4, 1066], [0, 198, 1143], [0, 299, 1102], [0, 350, 1117],
    #          [0, 446, 1118], [0, 423, 1159], [0, 464, 815], [0, 244, 640]]
    # comparison cherries 1+ :[126], center 890, 460

    # slices = [20]
    # indices = [[0, 212, 228]]   # timo has two dumb mistakes
    # comparison cherries 1+ :[17], bbox: [0:500, 0:500]]

    # others 2+: 44, 2: 106,  2+: 116, 121, 1-: 1, 2:70, 1: 19
    # print 'using slices', slices

    # sx, sy = (0, 0)
    # ex, ey = (-1, -1)
    if slices is not None:
        n_plots = min(n_plots, len(slices))

    gt_seg = dp.load_h5(gt_seg_p, slices=slices)[0]
    # gt_seg = dp.load_h5(gt_seg_p, slices=slices)[0][:, sx+pad:-pad, sx+pad:-pad]
    nn_ws_seg = dp.load_h5(nn_ws_seg_p, slices=slices)[0]
    std_ws_seg = dp.load_h5(std_ws_seg_p, slices=slices)[0]

    # fix color table only once
    # gt_seg, nn_ws_seg, std_ws_seg = fix_color_table(gt_seg, nn_ws_seg, std_ws_seg)
    # dp.save_h5('../data/nets/%s/validation_%s/fixed_slice_concat.h5' % (network, net_file), 'data', data=nn_ws_seg,
    #            overwrite='w')
    # dp.save_h5('../data/nets/%s/validation_%s/fixed_baseline_concat.h5' % (network, net_file), 'data',
    #            data=std_ws_seg, overwrite='w')
    # dp.save_h5(volume_path + 'fixed_label_' + dataset + '.h5', 'data', data=gt_seg, overwrite='w')

    # exit()

    sm = du.SeedMan()
    all_seeds = []
    for z, label_slice in enumerate(gt_seg):
        seeds = sm.get_seed_coords_gt(np.copy(label_slice), minsize=5)
        all_seeds.append(seeds)

    make_cremi_data_plot(gt_seg, std_ws_seg,
                       nn_ws_seg, None,
                       slices=slices,
                       criterum='diff',           # best_nn_ws, diff, worst_best
                        seeds=all_seeds,
                       indices=indices,
                        coords=cdict)                # which processes (ids) to select from image



    # make_network_plot(gt_seg, nn_ws_seg, std_ws_seg, nn_hm_p,
    #                   slices=slices,
    #                   # thresh=16.1) # percent of total height
    #                   thresh=3.2) # percent of total height
