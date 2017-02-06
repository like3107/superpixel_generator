import copy
from os import system


a = open('./../data/config/validation_validtmp.conf', 'r').read()

sigmas = [3, 6, 9]
holes = [0, 10]
versions = ['train', 'test', 'valid']
start_slice = [0, 0, 0]
slices_total = [10, 10, 100]
c = copy.copy(a.splitlines())


for sigma in sigmas:
    for n_holes in holes:
        for version, start, total in zip(versions, start_slice, slices_total):
            config = './../data/config/validation_toy_nh%i_sig%i_%s.conf' % (n_holes, sigma, version)
            b = open(config, 'w')
            for i, line in enumerate(c):
                if 'net_name' in line:
                    c[i] = 'train_version = toy_nh%i_sig%i_%s' %(n_holes, sigma, version)
                if 'slices_total' in line:
                    c[i] = 'start_slice_z         =            %i' % start
                    c[i] = 'slices_total          =            %i' % total
                if 'gpu' in line:
                    c[i] = 'gpu = gpu0'
                if 'load_net_path' in line:
                    c[i] = 'load_net_path =./../data/nets/toy_nh%i_sig%i/nets/net_2000' % (n_holes, sigma)
                if 'train_version' in line:
                    c[i] = 'train_version = toy_nh%i_sig%i_%s'  % (n_holes, sigma, version)

            b.write("\n".join(c))
            b.close()

            mypython = '/home/lschott_local/anaconda2/envs/ws_timo/bin/python '

            cmd = mypython + 'evaluate_cremi.py -c ' + config
            print cmd
            system(cmd)

