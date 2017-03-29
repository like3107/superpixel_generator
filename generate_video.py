import matplotlib
matplotlib.use('Qt4Agg')
import h5py as h
import numpy as np
import utils as u
# from matplotlib i1mport pyplot as plt
import pylab as plt
from matplotlib import collections  as mc
import progressbar
import json
from scipy import ndimage
from plot_scripts import *
from plot_scriptstoy import *
from draw_MSF import *
from subprocess import call
from multiprocessing import Pool
from mayavi import mlab
from tvtk.common import configure_input_data, configure_source_data, is_old_pipeline
from tvtk.api import tvtk
import moviepy.editor as editor



coordinate_offset = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.int)
fov = 35
sl = 1



def msf_plot_i(args):
    mlab.clf()
    print 'args', args
    i, zoom, time, n_times = args
    BM_FILE = '../data/nets/full_validation_path_01/validation_0/bm_116_116.h5'
    # BM_FILE = '../data/nets/fig1_draw_29_without_min/pretrain/serial_00447999'

    zoom = 1

    ROI = get_new_roi(zoom)
    # ROI = ((0,40),(0,140))
    b = 0

    DDD_OUTPUT_PNG = "img/DDD_%08i.png"
    HEIGHT_PNG = "img/height_%08i.png"
    COMBINED_PNG = "img/comb_%08i.png"

    CM = u.random_color_map()

    ZOOM_STEPS = 0.1

    FINAL_ZOOM = 6

    DDD_PLOT = True
    MSF_PLOT = False
    DEBUG = True

    with h.File(BM_FILE,'r') as h5f:
        times = sorted(h5f['global_timemap'][b, fov+ROI[0][0]+1:ROI[0][1]+fov-1, ROI[1][0]+fov+1:ROI[1][1]+fov-1].flatten())
        print 'current step i %i out of %i in percent %f' % (i, len(times), float(i) / n_times * 100)

        # debug
        MAX_TIME = np.max(h5f['global_timemap'][b])

        # START_ZOOM_TIME /= 5

        if DEBUG:
            times = [100, 1000, 10000, 1000000]

        claims = np.array(h5f['global_claims'])
        D_raw = np.array(h5f['global_input'][b, 0, fov+ROI_DDD[0][0]:ROI_DDD[0][1]+fov, ROI_DDD[1][0]+fov:ROI_DDD[1][1]+fov])
        Z = np.min(h5f['global_prediction_map_nq'][b, ROI_DDD[0][0]:ROI_DDD[0][1], ROI_DDD[1][0]:ROI_DDD[1][1]], axis=2)*10
        Z_total = np.min(h5f['global_prediction_map_nq'][b], axis=2)


        D_timemap = h5f['global_timemap'][b, fov+ROI_DDD[0][0]:ROI_DDD[0][1]+fov, ROI_DDD[1][0]+fov:ROI_DDD[1][1]+fov]
        D_claims = claims[b, fov+ROI_DDD[0][0]:ROI_DDD[0][1]+fov, ROI_DDD[1][0]+fov:ROI_DDD[1][1]+fov].astype(int)
        xlim, ylim = Z.shape
        X, Y = np.mgrid[0:xlim+2, 0:ylim+2]


        font = {'family': 'serif',
                'color':  'darkred',
                'weight': 'normal',
                'size': 16,
                        }

        max_label = np.max(claims[b])
        if MSF_PLOT:
            if i > START_ZOOM_TIME and zoom < FINAL_ZOOM:
                # zoom = 7 * float(i-START_ZOOM_TIME) / len(times) + 1
                zoom += ZOOM_STEPS
                if zoom > FINAL_ZOOM:
                    zoom = FINAL_ZOOM
                ROI = get_new_roi(zoom)

            fig, ax = plt.subplots()
            fig.tight_layout()
            ax.autoscale()
            ax.margins(0.1)
            # print fov+ROI[0][0], ROI[0][1]+fov, ROI[1][0]+fov, ROI[1][1]+fov
            raw = np.array(h5f['global_input'][b, 0, fov + ROI[0][0]:ROI[0][1] + fov, ROI[1][0] + fov:ROI[1][1] + fov])
            mimshow = ax.imshow(raw, interpolation=None, cmap='gray', extent=[0, raw.shape[0], 0, raw.shape[1]])
            # glob_time_map = h5f['global_timemap'][b, fov+ROI[0][0]:ROI[0][1]+fov, ROI[1][0]+fov:ROI[1][1]+fov]
            # print np.sum(glob_time_map < time)
            # ax.imshow(glob_time_map, interpolation=None, cmap='gray')

            if zoom < 2:
                lc = get_masked_MSF(h5f, ROI, time, CM, max_label)
                ax.add_collection(lc)
            elif zoom >= 2 and zoom < 3:
                lc = get_masked_MSF(h5f, ROI, time, CM, max_label)
                ax.add_collection(lc)
                masked_claims = get_masked_claims(h5f, ROI, time, claims)
                print "alpha", 0.8 * ((zoom - 2.))
                ax.imshow(masked_claims, interpolation='none', cmap=CM, alpha=0.8 * ((zoom - 2.) / 2.5),
                          clim=(0, max_label))
            else:
                masked_claims = get_masked_claims(h5f, ROI, time, claims)
                ax.imshow(masked_claims, interpolation='none', cmap=CM, alpha=0.8, clim=(0, max_label))

            # ax.text(0.1, 0.9,'learned watershed on CREMI dataset C', ha='left', va='center', transform=ax.transAxes, fontdict=font)

            ax.axis('off')
            fig.savefig(OUTPUT_PNG % i, bbox_inches='tight', dpi=200)
            # call(["mogrify", "-crop", "1308x760+0x0", OUTPUT_PNG%i])
            call(["mogrify", OUTPUT_PNG % i, "-trim", OUTPUT_PNG % i])

            # make height image
            fig, ax = plt.subplots(facecolor='red')
            fig.tight_layout()

            ax.text(0.004, 0.95, 'dynamic altitude prediction', ha='left', va='center', transform=ax.transAxes,
                    fontdict=font)

            # if zoom == 1.:
            #     ax.imshow(raw, interpolation=None, cmap='gray')
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.axis('off')

            z_zoom = Z_total[ROI[0][0]:ROI[0][1], ROI[1][0]:ROI[1][1]]
            pred = get_masked_height(h5f, ROI, time, z_zoom, b=0)
            ax.imshow(pred, interpolation=None, cmap='gray', clim=(0, max_height))
            ax.axis('off')

            # ax.axis('off')

            fig.savefig(HEIGHT_PNG % i, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())
            call(["mogrify", HEIGHT_PNG % i, "-trim", HEIGHT_PNG % i])

            # fuse images together
            call(["convert", OUTPUT_PNG % i, HEIGHT_PNG % i, "-append", COMBINED_PNG % i])

        # bar.update(i)

        if DDD_PLOT:
            D_raw = np.pad(D_raw, 1, 'constant')
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            Z_masked = np.array(Z)
            mask = ~ndimage.binary_dilation(D_timemap <= time, iterations=1)
            Z_masked[mask] = -0.501


            mimshow = mlab.imshow(D_raw, colormap='gray', extent=[0, D_raw.shape[0], 0, D_raw.shape[1], -0.5, -0.1],
                                  figure=FIGURE)


            Z_masked = np.log(Z_masked + 1.00001) * 10
            Z_masked = np.pad(Z_masked, 1, 'constant', )
            Z_masked[Z_masked <= 0] = - 0.5001
            surf = mlab.surf(X, Y, Z_masked, colormap='gray', color=(0.8,0.8,0.8), opacity=0.7, figure=FIGURE)


            array_2d = CM(D_claims.astype(np.int))         # uint 8 255, 3
            img = image_from_array(array_2d)

            texture_img = tvtk.Texture(interpolate=1)
            configure_input_data(texture_img, img)
            texture_img.input = img

            surf.actor.enable_texture = True
            surf.actor.tcoord_generator_mode = 'plane'
            surf.actor.texture = texture_img

            return mlab.screenshot(antialiased=True)

def image_from_array(ary):
    """ Create a VTK image object that references the data in ary.
        The array is either 2D or 3D with.  The last dimension
        is always the number of channels.  It is only tested
        with 3 (RGB) or 4 (RGBA) channel images.
        Note: This works no matter what the ary type is (accept
        probably complex...).  uint8 gives results that make since
        to me.  Int32 and Float types give colors that I am not
        so sure about.  Need to look into this...
    """

    sz = ary.shape
    dims = len(sz)
    # create the vtk image data
    img = tvtk.ImageData()

    if dims == 2:
        # 1D array of pixels.
        img.whole_extent = (0, sz[0]-1, 0, 0, 0, 0)
        img.dimensions = sz[0], 1, 1
        img.point_data.scalars = ary

    elif dims == 3:
        # 2D array of pixels.
        if is_old_pipeline():
            img.whole_extent = (0, sz[0]-1, 0, sz[1]-1, 0, 0)
        else:
            img.extent = (0, sz[0]-1, 0, sz[1]-1, 0, 0)
        img.dimensions = sz[0], sz[1], 1

        # create a 2d view of the array
        ary_2d = ary[:]
        ary_2d.shape = sz[0]*sz[1],sz[2]
        img.point_data.scalars = ary_2d

    else:
        raise ValueError("ary must be 3 dimensional.")

    return img

def get_masked_MSF(h5, roi, time, cm, max_label):
    # lines = [[(0, 1), (100, 100)], [(20, 30), (30, 30)], [(10, 20), (10, 30)]]
    # c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])
    lines = []
    c = []
    Z_total = np.log(np.min(h5['global_prediction_map_nq'][b], axis=2))

    direction_map = h5['global_directionmap_batch'][b, roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
    labels = np.array(h5['global_claims'])[b, fov+roi[0][0]:roi[0][1]+fov, roi[1][0]+fov:roi[1][1]+fov]
    timemap = h5['global_timemap'][b, fov+roi[0][0]:roi[0][1]+fov, roi[1][0]+fov:roi[1][1]+fov]
    for x in range(+1,roi[0][1]-roi[0][0]-1):
        for y in range(+1,roi[1][1]-roi[1][0]-1):
            direction = direction_map[x,y]
            if timemap[x,y] <= time:
                if direction >= 0:# and labels[b,x+fov,y+fov] in [big_id,small_id]:
                    xx, yy = [x,y] - coordinate_offset[direction]
                    lines.append([(y,x),(yy,xx)])
                    c.append(cm(labels[x,y]/max_label))
                    # c.append(tuple(custom[labels[b,x+fov,y+fov]]))
            if direction == -1:

                lines.append([(y-sl,x-sl),(y+sl,x+sl)])
                lines.append([(y-sl,x+sl),(y+sl,x-sl)])
                c.append(cm(labels[x,y]/max_label))
                c.append(cm(labels[x,y]/max_label))

    return mc.LineCollection(lines, colors=np.array(c), linewidths=1.)

def get_new_roi(zoom, poi=(670,460), ratio=(16,9)):
    dx,dy = (5*ratio[0]*zoom),(5*ratio[1]*zoom)
    # return ((int(poi[1]-dy),int(poi[1]+dy)),(int(poi[0]-dx),int(poi[0]+dx)))
    return ((int(poi[1]-dy),int(poi[1]+dy)),(int(poi[0]-dy),int(poi[0]+dy)))

def get_full_roi():
    return ((int(0),int(-1)),(int(0),int(-1)))


def get_masked_claims(h5, roi, time, claims, b=0, crop_claim=None):
    # # edges only
    if crop_claim is None:
        crop_claim = claims[b, fov+roi[0][0]:roi[0][1]+fov, roi[1][0]+fov:roi[1][1]+fov]
    timemap = h5['global_timemap'][b, fov+roi[0][0]:roi[0][1]+fov, roi[1][0]+fov:roi[1][1]+fov]
    return np.ma.masked_where(timemap > time, crop_claim)

def get_masked_height(h5, roi, time, Z, b=0):
    # # edges only
    timemap = h5['global_timemap'][b, fov+roi[0][0]:roi[0][1]+fov, roi[1][0]+fov:roi[1][1]+fov]
    mask = ndimage.binary_dilation(timemap <= max(time,100), iterations=1)
    return np.ma.masked_where(~mask, Z)


if __name__ == '__main__':

    BM_FILE = '../data/nets/full_validation_path_01/validation_0/bm_116_116.h5'
    # BM_FILE = '../data/nets/fig1_draw_29_without_min/pretrain/serial_00447999'

    zoom = 1

    ROI = get_new_roi(zoom)
    # ROI = ((0,40),(0,140))
    b = 0

    OUTPUT_PNG = "img/test_%08i.png"
    HEIGHT_PNG = "img/height_%08i.png"      # double
    COMBINED_PNG = "img/comb_%08i.png"      # double


    STOP_TIME = 100
    DDD_OUTPUT_PNG = "img/DDD_%08i.png"     # double
    DDD_PLOT = True                         # dobule

    START_ZOOM_TIME = 120
    FPS = 24
    MSF_PLOT = False

    ROI_DDD = get_new_roi(6)

    with h.File(BM_FILE,'r') as h5f:
        times = sorted(h5f['global_timemap'][b, fov+ROI_DDD[0][0]+1:ROI_DDD[0][1]+fov-1, ROI_DDD[1][0]+fov+1:ROI_DDD[1][1]+fov-1].flatten())
        # Z_total = np.log(np.min(h5f['global_prediction_map_nq'][b], axis=2))
        print 'times total', len(times)
        # exit()
        times = times[::30000] + 20 * [np.max(h5f['global_timemap'][b])]
        # times = times[-30:]

    # max_height = np.max(Z_total)
    # times = times[0:START_ZOOM_TIME:1] + times[START_ZOOM_TIME:1000:10] + times[1000::30]
    # times = times[100:1000:10]


    pool = Pool(5)

    with progressbar.ProgressBar(max_value=len(times)+STOP_TIME) as bar:
        # for i, time in enumerate(times):

        n_times = len(times)
        args = [(i, zoom, time, n_t) for zoom, i, time, n_t in zip(n_times * [zoom],
                                                         range(n_times),
                                                         times,
                                                         n_times * [n_times])]
        print 'list', args
        print 'len times', len(times), np.max(times)
        #debug


        # msf_plot_i(args[0])
        FIGURE = mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
        animation = editor.VideoClip(lambda  t: msf_plot_i(args[int(t*len(times))]), duration=len(times)/FPS)
        animation.write_videofile('crayc.mp4', fps=FPS)
        # animation.write_gif('crayc.gif', fps=24, opt='nq')
        print 'before pool'
        mlab.show()

        # pool.map(msf_plot_i, args)

        # if MSF_PLOT:
        #     for ii in range(len(times)-1,len(times)+STOP_TIME):
        #         call(["cp", OUTPUT_PNG%(len(times)-1), OUTPUT_PNG%ii])
        #         call(["cp", HEIGHT_PNG%(len(times)-1), HEIGHT_PNG%ii])
        #         call(["cp", COMBINED_PNG%(len(times)-1), COMBINED_PNG%ii])
        #         bar.update(ii)
        #
        # if DDD_PLOT:
        #     for ii in range(len(times)-1,len(times)+100):
        #         call(["cp", DDD_OUTPUT_PNG%(len(times)-1), DDD_OUTPUT_PNG%ii])


    if MSF_PLOT:
        call(["ffmpeg","-y","-i","img/test_%08d.png","img/MSF.mp4"])
        call(["ffmpeg","-y","-i","img/comb_%08d.png","img/COMB.mp4"])

