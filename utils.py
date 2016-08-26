import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import lasagne as las
import dataset_utils as du
import os
import sys
import h5py

np.random.seed(1234)
fixed_rand = np.random.rand(256, 3)
import multiprocessing

# A random colormap for matplotlib, https://gist.github.com/jgomezdans/402500
def random_color_map():
    fixed_rand[0, :] = 0
    cmap = matplotlib.colors.ListedColormap(fixed_rand)
    return cmap


def save_2_images(im_x, im_y, path, name='iteration', iteration=0,
                  iterations_per_image=0):
    f, ax = plt.subplots(ncols=2)
    ax[0].imshow(im_x, interpolation='none', cmap=random_color_map())
    ax[1].imshow(im_y, cmap='gray')
    f.savefig(path + name + '_it%07d_im%07d' % (iteration, iterations_per_image))
    plt.close(f)


def save_3_images(im_x, im_y, im_z, path, name='iteration', iteration=0,
                  iterations_per_image=0):
    f, ax1 = plt.subplots(ncols=3)
    ax1[0].imshow(im_x, interpolation='none', cmap=random_color_map())
    ax1[1].imshow(im_y, cmap='gray')
    im_z[0,0] = 0
    ax1[2].imshow(im_z, interpolation='none', cmap='gray')

    f.savefig(path + name + '_it%07d_im%07d' % (iteration, iterations_per_image))
    plt.close()


def save_4_images(im_x, im_y, im_z, im_zz, path, name='iteration', iteration=0,
                  iterations_per_image=0):
    f, ax = plt.subplots(ncols=4)
    ax[0].imshow(im_x, interpolation='none', cmap=random_color_map())
    ax[1].imshow(im_y, cmap='gray', interpolation='none')
    im_z[0,0] = 0
    ax[2].imshow(im_z, interpolation='none', cmap='gray')
    ax[3].imshow(im_zz, interpolation='none', cmap='gray')
    f.savefig(path + name + '_it%07d_im%07d' % (iteration, iterations_per_image))
    plt.close()


def draw_image(image_info, target):
    interp = 'none'
    if "interpolation" in image_info:
        interp = image_info["interpolation"]
    color_map = 'gray'
    if "cmap" in image_info:
        if image_info["cmap"] == "rand":
            color_map = random_color_map()
    if "title" in image_info:
        target.set_title(image_info["title"])
    if 'scatter' in image_info:
        if len(image_info['scatter']) > 0:
            centers = np.array(image_info['scatter'])
            target.scatter(centers[:, 1], centers[:, 0], s=1, marker='.', color='r')
    target.imshow(image_info["im"], interpolation=interp, cmap=color_map)
    target.axis('off')

def save_images(image_dicts, path, name, terminate=False, column_size=3):
    if len(image_dicts) == 0:
        return
    f, ax = plt.subplots(ncols=column_size, nrows=((len(image_dicts)-1) / column_size) + 1)
    if ((len(image_dicts)-1) / column_size) + 1 > 1:
        for i, image_info in enumerate(image_dicts):
            draw_image(image_info, ax[i / column_size, i % column_size])
    else:
        for i, image_info in enumerate(image_dicts):
            draw_image(image_info, ax[i])
    f.savefig(path + name, dpi=400)
    plt.close()
    if terminate:
        exit()


def save_images_sub(image_dicts,path,name):
    p = multiprocessing.Process(target=save_images, args=(image_dicts,path,name,True))
    p.daemon = True
    p.start()


def show_image(image_dicts):
    f, ax = plt.subplots(ncols=3, nrows=(len(image_dicts) / 3) + 1)
    for i, image_info in enumerate(image_dicts):
        draw_image(image_info, ax[i / 3, i % 3])
    plt.show()


def decay_func(iteration, edge_len, safty_margin=300, decay_factor=0.4):
    return int((1. - 1. / (iteration + 2) ** decay_factor) *
               (edge_len ** 2 - safty_margin))


def linear_growth(iteration, maximum=60**2-50, y_intercept=50,
                  iterations_to_max=10000):
    if iteration >= iterations_to_max:
        return maximum
    else:
        m = (maximum - y_intercept) / float(iterations_to_max)
        return int(m * iteration + y_intercept)


def make_bash_executable(base_path, add_option=''):
    script_file_name = base_path+'/code_train/run_training.sh'
    f = open(script_file_name, 'w')
    f.write("#!/bin/bash\n")
    f.write("cd "+os.getcwd()+"\n")
    f.write("python "+" ".join(sys.argv))
    f.write(" "+add_option)
    f.close()


def save_network(save_path, l_last, net_name, poolings=None, filter_sizes=None,
                 n_filter=None,add=[]):

    h5_values = []
    h5_keys = []

    # load network parameter
    all_params = las.layers.get_all_params(l_last)
    i = -1
    for param in all_params:
        i += 1
        h5_keys.append(str(param) + str(i))
        h5_values.append(param.get_value())

    if poolings is not None:
        h5_keys.append('poolings')
        h5_values.append(poolings)

    if filter_sizes is not None:
        h5_keys.append('filter_sizes')
        h5_values.append(filter_sizes)

    if n_filter is not None:
        h5_keys.append('n_filter')
        h5_values.append(n_filter)

    du.save_h5(save_path + net_name, h5_keys, h5_values, overwrite='w')
    save_options(save_path + net_name,add)



def create_network_folder_structure(save_net_path,
                                    save_net_path_pre='',
                                    save_net_path_ft='',
                                    save_net_path_reset='',
                                    train_mode=True):
    if not os.path.exists(save_net_path):
        os.mkdir(save_net_path)
    if not os.path.exists(save_net_path + '/images'):
        os.mkdir(save_net_path + '/images')
    if not os.path.exists(save_net_path_reset):
        os.mkdir(save_net_path_reset)
    if not os.path.exists(save_net_path_ft):
        os.mkdir(save_net_path_ft)
    if not os.path.exists(save_net_path_pre):
            os.mkdir(save_net_path_pre)
    if not os.path.exists(save_net_path + '/exp'):
        os.mkdir(save_net_path + '/exp')
    if train_mode:
        code_save_folder = '/code_train'
    else:
        code_save_folder = '/code_predict'
    if not os.path.exists(save_net_path + code_save_folder):
        os.mkdir(save_net_path + code_save_folder)
    os.system('cp -rf *.py ' + save_net_path + code_save_folder)
    os.system('cp -rf *./data/config/*.conf ' + save_net_path + code_save_folder)


def load_network(load_path, l_last):
    h5_keys = []
    all_params = las.layers.get_all_params(l_last)
    i = -1
    for param in all_params:
        i += 1
        h5_keys.append(str(param) + str(i))

    all_param_values = du.load_h5(load_path, h5_keys)
    las.layers.set_all_param_values(l_last, all_param_values)

def load_options(load_path, options={}):
    with h5py.File(load_path, 'r') as net_file:
        for op_key, op_val in [(k, net_file['options/'+k].value)\
                                for k in net_file['options'].keys()]:
            if isinstance(options,dict):
                options[op_key] = op_val
            else:
                options.__setattr__(op_key, op_val)
    return options

def save_options(load_path, options):
    if len(options) > 0:
        with h5py.File(load_path, 'r+') as net_h5:
            for op_key, op_val in options:
                if "options/"+op_key in net_h5:
                    f.__delitem__("options/"+op_key)
                net_h5.create_dataset("options/"+op_key,data=op_val)

def print_options_for_net(options):
    to_print = str([options.net_name, options.load_net_b, options.load_net_path,
                options.train_version, options.valid_version, options.val_b,
                options.global_edge_len, options.fast_reset,
                options.clip_method, options.pre_train_iter,
                options.regularization, options.batch_size,
                options.augment_pretraining, options.scale_height_factor,
                options.batch_size_ft, options.margin, options.augment_ft,
                options.exp_bs, options.exp_ft_bs, options.exp_warmstart,
                options.exp_height, options.exp_save, options.exp_load, options.net_arch])
    print to_print[1:-1]
    print

def plot_train_val_errors(all_y_values, x_values, save_path, names):
    fig = plt.figure()
    plots = []
    for y_values in all_y_values:
        # print 'yvalues', y_values
        plot, = plt.plot(x_values, y_values)
        # if np.any(all_y_values < 0):
        plt.yscale('log')

        plots.append(plot)
    fig.legend(plots, names)
    fig.savefig(save_path)
    plt.close(fig)
    return

if __name__ == '__main__':
    path = './data/nets/cnn_v5/preds_0'
    concat_h5_in_folder(path, 8, 64, 300)
    # print random_color_map()
    # a = np.random.randint(0, 2, size=(100, 100))
    # print a.shape
    #
    # plt.imshow(a, interpolation='none', cmap=random_color_map())
    # plt.show()