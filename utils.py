import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import lasagne as las
import dataset_utils as du
np.random.seed(1234)
fixed_rand = np.random.rand(256, 3)



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
    ax1[2].imshow(im_z, interpolation='none', cmap=random_color_map())

    f.savefig(path + name + '_it%07d_im%07d' % (iteration, iterations_per_image))
    plt.close()


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


def save_network(save_path, l_last, net_name, poolings=None, filter_sizes=None,
                 n_filter=None):

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


def load_network(load_path, l_last):
    h5_keys = []
    all_params = las.layers.get_all_params(l_last)
    i = -1
    for param in all_params:
        i += 1
        h5_keys.append(str(param) + str(i))

    all_param_values = du.load_h5(load_path, h5_keys)
    las.layers.set_all_param_values(l_last, all_param_values)
    return


def plot_train_val_errors(all_y_values, x_values, save_path, names):
    fig = plt.figure()
    plots = []
    for y_values in all_y_values:
        plot, = plt.plot(x_values, y_values)
        plt.yscale('log')
        plots.append(plot)
    fig.legend(plots, names)
    fig.savefig(save_path)
    plt.close(fig)
    return

if __name__ == '__main__':
    print random_color_map()
    a = np.random.randint(0, 2, size=(100, 100))
    print a.shape

    plt.imshow(a, interpolation='none', cmap=random_color_map())
    plt.show()