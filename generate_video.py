import h5py as h
import numpy as np
import utils as u
import matplotlib
matplotlib.use('Agg')
# from matplotlib import pyplot as plt
import pylab as plt
from matplotlib import collections  as mc
import progressbar
import json

from plot_scripts import *
from plot_scriptstoy import *
from draw_MSF import *
from subprocess import call



coordinate_offset = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.int)
fov = 35
sl = 1

def get_masked_MSF(h5, roi, time, cm, max_label):
    # lines = [[(0, 1), (100, 100)], [(20, 30), (30, 30)], [(10, 20), (10, 30)]]
    # c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])
    lines = []
    c = []

    direction_map = h5['global_directionmap_batch'][b, roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
    labels = np.array(h5f['global_claims'])[b, fov+roi[0][0]:roi[0][1]+fov, roi[1][0]+fov:roi[1][1]+fov]
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
    return ((int(poi[1]-dy),int(poi[1]+dy)),(int(poi[0]-dx),int(poi[0]+dx)))


def get_masked_claims(h5, roi, time, claims, b=0):
    # # edges only
    crop_claim = claims[b, fov+roi[0][0]:roi[0][1]+fov, roi[1][0]+fov:roi[1][1]+fov]
    timemap = h5['global_timemap'][b, fov+roi[0][0]:roi[0][1]+fov, roi[1][0]+fov:roi[1][1]+fov]
    return np.ma.masked_where(timemap > time, crop_claim)

def get_masked_height(h5, Z, roi, time, claims, b=0):
    # # edges only
    crop_Z = Z[b, fov+roi[0][0]:roi[0][1]+fov, roi[1][0]+fov:roi[1][1]+fov]
    timemap = h5['global_timemap'][b, fov+roi[0][0]:roi[0][1]+fov, roi[1][0]+fov:roi[1][1]+fov]
    return np.ma.masked_where(timemap > time, crop_Z)


if __name__ == '__main__':

    BM_FILE = '../data/nets/full_validation_path_01/validation_0/bm_116_116.h5'
    # BM_FILE = '../data/nets/fig1_draw_29_without_min/pretrain/serial_00447999'

    zoom = 1

    ROI = get_new_roi(zoom)
    # ROI = ((0,40),(0,140))
    b = 0

    OUTPUT_PNG = "img/test_%08i.png"
    DDD_OUTPUT_PNG = "img/DDD_%08i.png"
    CM = u.random_color_map()

    START_ZOOM_TIME = 200
    ZOOM_STEPS = 0.2
    STOP_TIME = 100

    FINAL_ZOOM = 7


    DDD_PLOT = False
    MSF_PLOT = True

    ROI_DDD = get_new_roi(4)


    with h.File(BM_FILE,'r') as h5f:
        times = sorted(h5f['global_timemap'][b, fov+ROI[0][0]+1:ROI[0][1]+fov-1, ROI[1][0]+fov+1:ROI[1][1]+fov-1].flatten())
        times = times[:START_ZOOM_TIME] + times[START_ZOOM_TIME::25] + [np.max(h5f['global_timemap'][b])]

        claims = np.array(h5f['global_claims'])
        D_raw = np.array(h5f['global_input'][b, 0, fov+ROI_DDD[0][0]:ROI_DDD[0][1]+fov, ROI_DDD[1][0]+fov:ROI_DDD[1][1]+fov])
        Z = np.min(h5f['global_prediction_map_nq'][b, ROI_DDD[0][0]:ROI_DDD[0][1], ROI_DDD[1][0]:ROI_DDD[1][1]], axis=2)
        D_timemap = h5f['global_timemap'][b, fov+ROI_DDD[0][0]:ROI_DDD[0][1]+fov, ROI_DDD[1][0]+fov:ROI_DDD[1][1]+fov]
        xlim, ylim = Z.shape
        X, Y = np.mgrid[0:xlim, 0:ylim]
        x, y = np.mgrid[0:D_raw.shape[0], 0:D_raw.shape[1]]

        max_label =  np.max(claims[b])
        # times = times[0:100:1] + times[100:1000:10] + times[1000::100]

        with progressbar.ProgressBar(max_value=len(times)+STOP_TIME) as bar:
            for i, time in enumerate(times):

                if MSF_PLOT:
                    if i > START_ZOOM_TIME and zoom < FINAL_ZOOM:
                        # zoom = 7 * float(i-START_ZOOM_TIME) / len(times) + 1
                        zoom += ZOOM_STEPS
                        if zoom > FINAL_ZOOM:
                            zoom = FINAL_ZOOM
                        ROI = get_new_roi(zoom)

                    fig, ax = plt.subplots()
                    fig.tight_layout()
                    # print fov+ROI[0][0], ROI[0][1]+fov, ROI[1][0]+fov, ROI[1][1]+fov
                    raw = np.array(h5f['global_input'][b, 0, fov+ROI[0][0]:ROI[0][1]+fov, ROI[1][0]+fov:ROI[1][1]+fov])
                    ax.imshow(raw, interpolation=None, cmap='gray')



                    # glob_time_map = h5f['global_timemap'][b, fov+ROI[0][0]:ROI[0][1]+fov, ROI[1][0]+fov:ROI[1][1]+fov]
                    # print np.sum(glob_time_map < time)
                    # ax.imshow(glob_time_map, interpolation=None, cmap='gray')

                    if zoom < 2:
                        lc = get_masked_MSF(h5f, ROI, time, CM, max_label)
                        ax.add_collection(lc)
                        ax.autoscale()
                        ax.margins(0.1)
                    elif zoom >= 2 and zoom < 3:
                        lc = get_masked_MSF(h5f, ROI, time, CM, max_label)
                        ax.add_collection(lc)
                        ax.autoscale()
                        ax.margins(0.1)
                        masked_claims = get_masked_claims(h5f, ROI, time, claims)
                        print "alpha",0.8*((zoom-2.))
                        ax.imshow(masked_claims, interpolation='none', cmap=CM, alpha=0.8*((zoom-2.)/2.5), clim=(0,max_label))
                    else:
                        masked_claims = get_masked_claims(h5f, ROI, time, claims)
                        ax.imshow(masked_claims, interpolation='none', cmap=CM, alpha=0.8, clim=(0,max_label))

                    ax.axis('off')
                    fig.savefig(OUTPUT_PNG%i, bbox_inches='tight', dpi=200)
                    # call(["mogrify", "-crop", "1308x760+0x0", OUTPUT_PNG%i])
                    call(["mogrify", OUTPUT_PNG%i, "-trim", OUTPUT_PNG%i])
                    
                bar.update(i)

                if DDD_PLOT:
                    fig = plt.figure()
                    ax = fig.gca(projection='3d')
                    Z[D_timemap>time] = np.nan
                    masked_claims = get_masked_claims(h5f, ROI_DDD, time, claims)
                    # ax.plot_surface(x, y, 0, rstride=1, cstride=1, facecolors=plt.cm.gray(D_raw))
                    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.3, facecolors=CM(masked_claims), vmax=max_label)
                    fig.savefig(DDD_OUTPUT_PNG%i, bbox_inches='tight', dpi=200)

            for ii in range(len(times)-1,len(times)+STOP_TIME):
                call(["cp", OUTPUT_PNG%(len(times)-1), OUTPUT_PNG%ii])
                bar.update(ii)
    if MSF_PLOT:
        call(["ffmpeg","-y","-i","img/test_%08d.png","img/out.mp4"])
      