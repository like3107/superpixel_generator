from trainer_config_parser import get_options
import h5py
import utils as u
import data_provider as dp
import numpy as np
from scipy.ndimage import convolve
import progressbar

class indexgetter():
    def __init__(self, z_length):
        self.maxoffset = 1
        self.dex = [0]*self.maxoffset+list(range(z_lenght))+[-1]*self.maxoffset

    def __call__(self, z,offset):
        return self.dex[self.maxoffset+z+offset]


if __name__ == '__main__':
    """
    creates the CREMI data as input format in options.input_data_path
    as {label,height,input_}CREMI_train.h5
    as {label,height,input_}CREMI_test.h5
    
    please replace a,b,c with %s in  options.{raw/label/height}_path
    """

    options = get_options()
    datasets = ["a","b","c"]
    # define slices here if you want to reduce the size of the dataset
    # None means all slices
    dic_slices = {"test":{"a":slice(50),"b":slice(50),"c":slice(50)},
                  "train":{"a":slice(50,125),"b":slice(50,125),"c":slice(50,125)}}

    for name,sl in dic_slices.items():

        total_z_lenght = 0 
        for ds in datasets:
            with h5py.File(options.label_path%ds,"r") as f:
                z_lenght = f[f.keys()[0]].shape[0]
                y_lenght = f[f.keys()[0]].shape[1]
                x_lenght = f[f.keys()[0]].shape[2]
                total_z_lenght += len(range(z_lenght)[sl[ds]])

        print "creating dataset for ",total_z_lenght,"zslices"

        i = 0
        input_data = np.empty((total_z_lenght,2,y_lenght,x_lenght),
                                dtype=np.float32)
        label_data = np.empty((total_z_lenght,y_lenght,x_lenght),
                                dtype=np.uint64)
        height_data  = np.empty((total_z_lenght,y_lenght,x_lenght),
                                dtype=np.float32)
        boundary_data  = np.empty((total_z_lenght,y_lenght,x_lenght),
                                dtype=np.float32)

        print "processing ",name," sets ",sl.keys()
        bar = progressbar.ProgressBar(max_value=total_z_lenght)

        for ds in datasets:

            rp = options.raw_path%ds
            mp = options.membrane_path%ds
            lp = options.label_path%ds
            hp = options.height_gt_path%ds

            membrane = dp.load_h5(mp)[0]
            raw = dp.load_h5(rp)[0]
            raw /= 256. - 0.5
            height = dp.load_h5(hp, h5_key='height')[0]
            label = dp.load_h5(lp)[0]
            z_lenght_in = raw.shape[0]
     
            gz = indexgetter(z_lenght)
            for z in range(z_lenght)[sl[ds]]:
                for j in [0]:
                    input_data[i,0+j] = raw[gz(z,j)]
                    input_data[i,1+j] = membrane[gz(z,j)]

                label_data[i] = label[gz(z,0)]

                gx = convolve(label_data[i] + 1, np.array([-1., 0., 1.]).reshape(1, 3))
                gy = convolve(label_data[i] + 1, np.array([-1., 0., 1.]).reshape(3, 1))
                boundary_data[i] = np.float32((gx ** 2 + gy ** 2) > 0)

                height_data[i] = height[gz(z,0)]
                i += 1
                bar.update(i)

        print "writing data to files"

        dp.save_h5(options.input_data_path+"label_CREMI_noz_%s.h5"%name,'data',
                         data=label_data, overwrite='w')

        dp.save_h5(options.input_data_path+"input_CREMI_noz_%s.h5"%name,'data',
                         data=input_data, overwrite='w')

        dp.save_h5(options.input_data_path+"height_CREMI_noz_%s.h5"%name,'data',
                         data=height_data, overwrite='w')

        dp.save_h5(options.input_data_path+"boundary_CREMI_noz_%s.h5"%name,'data',
                         data=boundary_data, overwrite='w')