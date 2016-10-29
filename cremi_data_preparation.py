from trainer_config_parser import get_options
import h5py
import utils as u
import data_provider as dp
import numpy as np

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
    datasets = ["a","b"]
    # define slices here if you want to reduce the size of the dataset
    # None means all slices
    dic_slices = {"test":{"a":slice(50),"b":slice(50),"c":slice(50)},
                  "train":{"a":slice(50,125),"b":slice(50,125),"c":slice(50,125)}}

    for name,sl in dic_slices.items():

        total_z_lenght = 0 
        for ds in datasets:
            with h5py.File(options.label_path%ds,"r") as f:
                print f.keys()[0]
                z_lenght = f[f.keys()[0]].shape[0]
                y_lenght = f[f.keys()[0]].shape[1]
                x_lenght = f[f.keys()[0]].shape[2]
                total_z_lenght += len(range(z_lenght)[sl[ds]])

        print "creating dataset for ",total_z_lenght,"zslices"

        i = 0
        input_data = np.empty((total_z_lenght,6,y_lenght,x_lenght),
                                dtype=np.float32)
        label_data = np.empty((total_z_lenght,y_lenght,x_lenght),
                                dtype=np.uint64)
        height_data  = np.empty((total_z_lenght,y_lenght,x_lenght),
                                dtype=np.float32)

        for ds in datasets:

            rp = options.raw_path%ds
            mp = options.membrane_path%ds
            lp = options.label_path%ds
            hp = options.height_gt_path%ds


            print "loading datasets"
            membrane = dp.load_h5(mp)[0]
            raw = dp.load_h5(rp)[0]
            raw /= 256. - 0.5
            height = dp.load_h5(hp)[0]
            label = dp.load_h5(lp)[0]
            z_lenght_in = raw.shape[0]
     
            # make a list of padded indices
            # (which cover the boundary cases for the later slicing)


            gz = indexgetter(z_lenght)
            for z in range(z_lenght)[sl[ds]]:
                print "\rz",z,"i",i
                for j in [-1,0,1]:
                    print raw.shape,input_data.shape, gz(z,j)
                    input_data[i,1+j] = raw[gz(z,j)]
                    input_data[i,4+j] = membrane[gz(z,j)]

                    label_data[i] = label[gz(z,j)]
                    height_data[i] = height[gz(z,j)]
                i += 1


        dp.save_h5(options.input_data_path+"label_CREMI_%s.h5"%name,'data',
                         data=label, overwrite='w')

        dp.save_h5(options.input_data_path+"input_CREMI_%s.h5"%name,'data',
                         data=input_data, overwrite='w')

        dp.save_h5(options.input_data_path+"height_CREMI_%s.h5"%name,'data',
                         data=height_data, overwrite='w')