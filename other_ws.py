import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt
import numpy as np
import data_provider as dp
from skimage import measure
import cv2
import os
import subprocess
from dataset_utils import SeedMan
from scipy import ndimage
from validation_scripts2 import validate_segmentation
import progressbar
import time
import data_provider as dp

class Watershednator(object):
    def __init__(self):
        self.label_image = None
        self.image = None
        self.SeedMan = SeedMan()
        self.global_seeds = None
        self.tmp_path = './../data/tmp/'
        self.seg = None

    def set_image(self, image, label_image=None):
        self.image = image
        self.label_image = label_image

    def get_seeds(self):
        self.global_seeds = self.SeedMan.get_seed_coords_gt(self.label_image)

    def do_ws(self, image, label_image=None):
        self.seg = None
        self.set_image(image, label_image=label_image)
        self.get_seeds()


class PWS(Watershednator):
    def __init__(self, ws_type='PWS'):
        super(PWS, self).__init__()
        availible_ws = ['MSF_Kruskal', 'PWS', 'MSF_Prim']

        if ws_type not in availible_ws:
            raise Exception('ws type not availibe choose from: ' + str(availible_ws))
        self.ws = np.argwhere(ws_type == availible_ws)
        self.exception_counter = 0

    def convert_data_to_pgm(self):
        seed_image = np.zeros_like(self.image)
        for i, seed in enumerate(self.global_seeds):
            seed_image[seed[0], seed[1]] = i + 1
        cv2.imwrite(self.tmp_path + 'img.pgm', self.image)
        cv2.imwrite(self.tmp_path + 'seeds_img.pgm', seed_image)

    def load_from_pgm(self):
        try:
            seg = np.array(cv2.imread(self.tmp_path + 'seg.ppm'), dtype=np.uint64)
            self.exception_counter = 0
        except:
            if self.exception_counter < 10:
                self.load_from_pgm()
            else:
                exit()

        new_seg = np.zeros_like(seg)
        for i, ind in enumerate(np.unique(seg)):
            new_seg[seg == ind] = i + 1
        self.seg = new_seg

    def execute_PWS(self):
        path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
        img_path = '/data/tmp/'
        execute_pws = 'cd ./../data/tmp/ && ~/git/PW_1.0.1/powerwatsegm.exe '
        algo = '-a 3 '
        # 1:  Maximum Spanning Forest computed by Kruskal algorithm
        # 2:  Powerwatersheds (p=infinite, q=2) : Maximum Spanning Forest computed by Kruskal algorithm and Random walker on plateaus
        # 3:  Maximum Spanning Forest computed by Prim algorithm using Red and black trees
        load_image = '-i ' + path + img_path + 'img.pgm '
        load_seed_image = '-m ' + path + img_path + '/seeds_img.pgm '
        cmd = execute_pws + algo + load_image + load_seed_image + ' -o seg.ppm '
        os.system(cmd)

    def do_ws(self, image, label_image=None):
        super(PWS, self).do_ws(image, label_image=label_image)
        self.convert_data_to_pgm()
        self.execute_PWS()
        self.load_from_pgm()
        return self.seg


class EvaluateWSs(object):
    def __init__(self, raw, label):
        self.WSs = None
        self.raw = raw
        self.label = label
        self.edges = None
        self.init_WSs()
        # self.sigmas = [0., 0.1, 0.5, 1, 1.5, 2, 4, 8]
        self.sigmas = [0.]

    def init_WSs(self):
        self.WSs = [PWS('MSF_Kruskal')]
        # self.WSs = [PWS('MSF_Kruskal'), PWS('MSF_Prim'), PWS('PWS')]

    def preprocess_data(self, sigma=2):
        if sigma != 0:
            self.edges = np.empty_like(raw, dtype=np.uint8)
            for i, raw_slice in enumerate(self.raw):
                self.edges[i] = (ndimage.gaussian_gradient_magnitude(raw_slice, sigma=sigma) * 254).astype(np.uint8)
        else:
            self.edges = self.raw

    def evaluate_ws(self):
        n_z = (1 if len(self.raw.shape) == 2 else self.raw.shape[0])
        bests_edges = [None] * len(self.WSs)
        bests_segs = [None] * len(self.WSs)
        bar = progressbar.ProgressBar(max_value=len(self.WSs) * len(self.sigmas))
        i = 0
        best_scores = np.empty(len(self.WSs))
        best_scores.fill(np.inf)
        for sigma in self.sigmas:
            self.preprocess_data(sigma=sigma)
            for num, ws in enumerate(self.WSs):
                i += 1
                bar.update(i)
                segs = []
                for z in range(n_z):
                    seg = ws.do_ws(self.edges[z], self.label[z])
                    segs.append(seg[:, :, 0])
                score = validate_segmentation(np.array(segs), self.label, resolution=1, border_thresh=2,
                                              verbose=False)
                if score < best_scores[num]:
                    bests_edges[num] = self.edges
                    best_scores[num] = score
                    bests_segs[num] = segs
                    print 'best score', score

        print 'best scores', best_scores
        print 'label', self.label.shape
        gt_edges, _ = dp.segmenation_to_membrane_core(self.label[0])
        print 'gt edges', gt_edges.shape
        bests_segs[0][0][gt_edges == 1] = 0


        fig, ax = plt.subplots(3, 3)
        print 'raw', self.raw.shape
        ax[0, 0].imshow(self.raw[0], interpolation='none', cmap='gray')
        ax[1, 0].imshow(self.label[0], interpolation='none')

        ax[0, 1].imshow(bests_edges[0][2], interpolation='none', cmap='gray')
        ax[0, 2].imshow(bests_segs[0][2], interpolation='none')

        # ax[1, 1].imshow(bests_edges[1][0], interpolation='none', cmap='gray')
        # ax[1, 2].imshow(bests_segs[1][0], interpolation='none')
        #
        # ax[2, 1].imshow(bests_edges[2][0], interpolation='none', cmap='gray')
        # ax[2, 2].imshow(bests_segs[1][0], interpolation='none')
        plt.show()


if __name__ == '__main__':
    label = np.ones((2, 100, 100))
    label[:, 50:100] = 2
    raw = np.zeros((2, 100, 100))
    raw[:, 50:52, :] = 1.

    raw = dp.load_h5('./../data/nets/toy_nh10_sig9_valid/edges/edges.h5', 'data')[0][:, 0]
    label = dp.load_h5('./../data/volumes/label_toy_nh10_sig9_valid.h5', 'data')[0]
    print label.shape
    # exit()

    # pws = PWS('PWS')
    # pws.do_ws(raw, label)
    ews = EvaluateWSs(raw, label)
    ews.evaluate_ws()

