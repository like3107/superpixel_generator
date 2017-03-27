import h5py as h
import numpy as np
import utils as u
import matplotlib
matplotlib.use('Agg')
# from matplotlib import pyplot as plt
import pylab as plt
from matplotlib import collections  as mc

import json

from plot_scripts import *
from plot_scriptstoy import *

b = 0

first_name =  "test_gt.png"
e_name =  "test_e.svg"
name = "test.png"
path = './'
fov = 35

h_name = "test_h.png"

f1 = "serial_00319199"
f2 = "error_"+f1

rcmap = u.random_color_map()

green = np.zeros((256, 3))
green[:,1] = 1
gcmap = matplotlib.colors.ListedColormap(green)

red = np.zeros((256, 3))
red[:,0] = 1
redcmap = matplotlib.colors.ListedColormap(red)

custom = np.ones((256, 4), dtype=int)
custom[2] = [0,1,0,1]
custom[3] = [0,0,1,1]
custom[1] = [1,0,0,1]
custom[4] = [1,1,0,1]
custom[5] = [1,0,1,1]
custom[6] = [0,1,1,1]
custcmap = matplotlib.colors.ListedColormap(custom)
coordinate_offset = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.int)

big_id = 2
small_id = 1

def update_position(pos, direction):
        """
        update position by following the minimal spanning tree backwards
        for this reason: subtract direction for direction offset
        """
        assert(direction >= 0)
        offsets = coordinate_offset[int(direction)]
        new_pos = [pos[0] - offsets[0], pos[1] - offsets[1]]
        return new_pos

def get_path_to_root(start_position, direction_map):

        current_position = start_position
        current_direction = direction_map[current_position[0]-fov,
                                                          current_position[1]-fov]
        yield start_position, current_direction
        while current_direction != -1:
            current_position = update_position(current_position, current_direction)
            current_direction = direction_map[current_position[0]-fov,
                                                                      current_position[1]-fov]
            yield current_position, current_direction


def line_collection_from_direction(direction_map, labels):
    print direction_map.shape
    # lines = [[(0, 1), (100, 100)], [(20, 30), (30, 30)], [(10, 20), (10, 30)]]
    # c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])
    lines = []
    c = []
    max_label = np.max(labels[b])
    print np.bincount(labels[b].flatten().astype(int)+1)

    for x in range(direction_map.shape[0]):
        for y in range(direction_map.shape[1]):
            direction = direction_map[x,y]
            if direction >= 0:# and labels[b,x+fov,y+fov] in [big_id,small_id]:
                xx, yy = [x,y] - coordinate_offset[direction]
                lines.append([(y,x),(yy,xx)])
                # c.append(custcmap(labels[b,x+fov,y+fov]/max_label))
                c.append(tuple(custom[labels[b,x+fov,y+fov]]))
    return mc.LineCollection(lines, colors=np.array(c), linewidths=0.5)

def get_path(h5f, e_h5f, claims):


    # find maximal e1 error
    max_unique_e1 = -1
    max_weight_e1 = 0
    for e in e_h5f['error']:
        ed = json.loads(str(e_h5f['error'][e].value))
        if ed['weight'] > max_weight_e1 and ed['type'] == 'e1' and ed['id'] == big_id:
            max_unique_e1 = ed['unique_id']
            max_weight_e1 = ed['weight']

    c = []
    path = []

    height = []
    prediction = []

    height_map = h5f['global_heightmap_batch'][b]
    prediction_map = h5f['global_prediction_map_nq'][b]
    direction_map = h5f['global_directionmap_batch'][b]

    for e in e_h5f['error']:
        ed = json.loads(str(e_h5f['error'][e].value))
        if ed['id'] in [big_id,small_id] and ed["unique_id"] == max_unique_e1:
            for pos in get_path_to_root(np.array(ed['pos']),h5f['global_directionmap_batch'][b]):
                print pos
                p = np.array(pos[0]) - fov

                path.append(np.array([p[0],p[1]])-fov)
                # c.append('orange')
                c.append(tuple(custom[claims[b,p[1]-fov-1,p[0]-fov-1]]))

                height.append(height_map[p[0], p[1]])
                prediction.append(np.min(prediction_map[p[0], p[1]]))
            
            height.append(0)
            prediction.append(0)

            path = list(reversed(path))
            height = list(reversed(height))

            prediction = list(reversed(prediction))

            for pos in get_path_to_root(np.array(ed['source_pos']),h5f['global_directionmap_batch'][b]):
                p = np.array(pos[0]) - fov
                path.append(np.array([p[0],p[1]])-fov)
                # c.append('blue')
                c.append(tuple(custom[claims[b,p[1]-fov-1,p[0]-fov-1]]))

                height.append(height_map[p[0], p[1]])
                prediction.append(np.min(prediction_map[p[0], p[1]]))
    height.append(0)
    prediction.append(0)

    return path,c, height, prediction

def line_collection_from_path(path, c):
    lines = []
    for p1,p2 in zip(path,path[1:]):
        lines.append([p1,p2])
    return mc.LineCollection(lines, colors=np.array(c), linewidths=3)


# def height_from_path(path, file):
    # height = []
    # prediction = []

    # height_map = file['global_heightmap_batch'][b]
    # prediction_map = file['global_prediction_map_nq'][b]
    # direction_map = file['global_directionmap_batch'][b]
    # for p in path:
    #     height.append(height_map[p[1], p[0]])
    #     print p[1], p[0]
    #     direction = direction_map[p[1], p[0]]
    #     prediction.append(prediction_map[p[1], p[0], direction])
    # return height, prediction


def draw_fig1(name, file_1, file_2):


    with h.File(file_1,'r') as h5f:
        with h.File(file_2,'r') as e_h5f:
            # draw raw + gt boundaries
            fig, ax = plt.subplots()
            raw = np.array(h5f['global_input'][b, 0, fov:-fov, fov:-fov])
            print raw.shape
            ax.imshow(raw, interpolation=None, cmap='gray')

            labels = np.array(h5f['global_label_batch'][:])
            print labels.shape
            bc = np.bincount(labels.flatten())
            print np.where(bc>0)
            print bc[bc>0]

            # edges only
            labelseroded = set_zero_except_boundaries(labels.copy(), n_dil=1)
            labelseroded = np.ma.masked_where(labelseroded == 0, labelseroded)

            ax.imshow(labelseroded[b], interpolation='none', cmap=gcmap, alpha=0.4)

            ax.axis('off')
            if first_name.endswith('pdf'):
                fig.savefig(path + first_name, format='pdf')
            else:
                fig.savefig(path + first_name, dpi=400)

            claims = np.array(h5f['global_claims'])


            fig, ax = plt.subplots()
            pred = np.sqrt(np.array(np.min(h5f['global_prediction_map_nq'][b],axis=2)))
            ax.imshow(pred, interpolation=None, cmap='gray')


            # edges only
            claimseroded = set_zero_except_boundaries(claims[:, fov:-fov, fov:-fov].copy(), n_dil=1)
            claimseroded = np.ma.masked_where(claimseroded == 0, claimseroded)

            ax.imshow(claimseroded[b], interpolation='none', cmap=redcmap, alpha=0.4)
            
            # edges only
            labelseroded = set_zero_except_boundaries(labels.copy(), n_dil=1)
            labelseroded = np.ma.masked_where(labelseroded == 0, labelseroded)

            ax.imshow(labelseroded[b], interpolation='none', cmap=gcmap, alpha=0.4)

            lc = line_collection_from_direction(h5f['global_directionmap_batch'][b], claims)
            ax.add_collection(lc)
            ax.autoscale()
            ax.margins(0.1)




            # # # find maximal e1 error
            # max_unique_e1 = -1
            # max_weight_e1 = 0
            # for e in e_h5f['error']:
            #     ed = json.loads(str(e_h5f['error'][e].value))
            #     if ed['weight'] > max_weight_e1 and ed['type'] == 'e1' and ed['id'] == 2:
            #         max_unique_e1 = ed['unique_id']
            #         max_weight_e1 = ed['weight']

            ms_path, c, height, prediction = get_path(h5f, e_h5f, claims)

            lc_path = line_collection_from_path(ms_path, c)

            ax.add_collection(lc_path)
            ax.autoscale()
            ax.margins(0.1)

            x = []
            y = []
            c = []

            # find path 

            # path = []

            # for e in e_h5f['error']:
            #     ed = json.loads(str(e_h5f['error'][e].value))
            #     if ed['id'] in [big_id,small_id] and ed["unique_id"] == max_unique_e1:


            #         for pos in get_path_to_root(np.array(ed['pos']),h5f['global_directionmap_batch'][b]):

            #             path.append(pos[0]-fov)

            #             x.append(pos[0][1]-fov)
            #             y.append(pos[0][0]-fov)

            #             c.append('g')

            #         for pos in get_path_to_root(np.array(ed['pos']),e_h5f['global_directionmap_batch'][b]):
            #             path.append(pos[0]-fov)
                        
            #             x.append(pos[0][1]-fov)
            #             y.append(pos[0][0]-fov)
            #             c.append('r')


            # ax.scatter(x, y,alpha=0.5, zorder=2)

            # errors
            # errors = []
            # x = []
            # y = []
            # c = []


            # for e in e_h5f['error']:
            #     ed = json.loads(str(e_h5f['error'][e].value))
            #     if ed['id'] in [big_id,small_id] and ed["unique_id"] == max_unique_e1:
            #         edge_pos = np.array(ed['source_pos']) + 0.5*(np.array(ed['pos']) - np.array(ed['source_pos'])) - fov
            #         x.append(edge_pos[1])
            #         y.append(edge_pos[0])

            #         if ed['type'] == 'e1':
            #             c.append('r')
            #         else:
            #             c.append('g')

            ax.scatter(x, y, c=c,alpha=0.5, zorder=2)





            # plt.plot([10],[10], [100],[100], 's')

            ax.axis('off')
            if name.endswith('pdf'):
                fig.savefig(path + name, format='pdf')
            else:
                fig.savefig(path + name, dpi=400)
            # create empty image

            fig, ax = plt.subplots()
            print ms_path, height, prediction

            plt.plot(range(len(height)),height)
            # plt.plot(range(len(prediction)),prediction)

            # ax.axis('off')
            if name.endswith('pdf'):
                fig.savefig(path + "h_"+name, format='pdf')
            else:
                fig.savefig(path + "h_"+name, dpi=400)
            # create empty image


# draw_fig1(name, f1, f2)
# draw_fig1(e_name, f2, f2)