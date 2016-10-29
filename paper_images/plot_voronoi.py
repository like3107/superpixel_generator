import h5py as h
import numpy as np
import random
from os import makedirs
import utils as u
from os.path import exists
import cairo
import math
from scipy import ndimage
from scipy import stats
from scipy.ndimage import convolve, gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion
from skimage.feature import peak_local_max
from skimage.morphology import label, watershed
from scipy.spatial import Voronoi as voronoi
from voronoi_polygon import voronoi_finite_polygons_2d
import png
from data_provider import *
from dataset_utils import *
from trainer_config_parser import get_options
import train_scripts

if __name__ == '__main__':

    options = get_options()
    
    predicter = train_scripts.Pokedex(options)
    # monkey patch test data loader !

    hbp = HoneyBatcherPredict(options)
    hbp.init_batch()
    hbp.draw_debug_image("test",path="paper_images/")

