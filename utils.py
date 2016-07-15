import matplotlib
from matplotlib import pyplot as plt
import numpy as np
fixed_rand = np.random.rand(256, 3)



# A random colormap for matplotlib, https://gist.github.com/jgomezdans/402500
def random_color_map():
    cmap = matplotlib.colors.ListedColormap(fixed_rand)
    return cmap


def save_2_images(im_x, im_y, path, name='iteration', iteration=''):
    f, ax = plt.subplots(ncols=2)
    ax[0].imshow(im_x, interpolation='none',
                 cmap=random_color_map())
    ax[1].imshow(im_y, cmap='gray')

    plt.savefig(path + name + '_%s' % str(iteration))
    plt.close()


if __name__ == '__main__':
    print random_color_map()
