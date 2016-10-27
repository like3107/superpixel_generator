from theano import tensor as T
import dataset_utils as du
from train_scripts import get_options
import nets
from theano.sandbox import cuda as s
import utils as u
import numpy as np
import data_provider as dp

s.use('gpu0')

options = get_options()

print 'data'
options.patch_len = 68
options.network_channels = 1
fov = int(options.patch_len)

options.raw_path = './../data/volumes/raw_%s.h5' % options.train_version
options.membrane_path = './../data/volumes/membranes_%s.h5' % options.train_version
options.label_path = './../data/volumes/label_%s.h5' % options.train_version
options.height_gt_path = './../data/volumes/height_%s.h5' % options.train_version


# BM = du.HoneyBatcherPath
# bm = BM(options)
# bm.prepare_global_batch()
# print bm.global_input_batch.shape
# print 'ende'
# print 'shape', bm.batch_shape
# print 'compiling'

target_t = T.ftensor4()
builder = nets.NetBuilder(n_channels = options.network_channels)
loss = builder.get_loss('updates_probs_v0')
network = builder.get_net(options.net_arch)
l_in, l_in_direction, l_out, l_out_direction, options.patch_len, l_eat =\
    network()
loss_train_f, loss_valid_f, probs_f = \
    loss(l_in, target_t, l_out, L1_weight=options.regularization)


# input = bm.global_input_batch
# print 'input', input.shape
# probs = probs_f(input)
# print 'output', probs.shape
# print du.height_to_fc_height_gt(bm.global_heightmap_batch)

for j in range(10000):

    input, _, out, _ = dp.generate_dummy_data3(1, edge_len=168, nz=8)
    input = input[:, None, :, :]
    out = out[:, fov / 2:-fov/2, fov / 2:-fov/2] / 30.
    out = du.height_to_fc_height_gt(out)
    probs = probs_f(input)
    loss_train, loss_individual_batch, l_out_train = \
        loss_train_f(input, out)
    if j % 10 == 0:
        p = []
        for i in range(4):
            p.append({"title": "raw_%i" % i,
                    "cmap": "grey",
                    'im': input[0, 0, fov / 2:-fov/2, fov / 2:-fov/2],
                    'interpolation': 'none'})
            p.append({"title": "gt_%i" % i,
                    "cmap": "grey",
                    'im': out[0, i],
                    'interpolation': 'none'})
            p.append({"title": "pred_%i" % i,
                      "cmap": "grey",
                      'im': probs[0, i],
                      'interpolation': 'none'})
        u.save_images(p, './../data/debug/', 'reg_dumdum%i.png' %j)
    print '\r trining', j, 'iter', loss_train,
    if j % 100 == 0:
        u.save_network('./../data/debug/', l_out, 'net_save_%i.net' %j)


def train(self):
    # make 10 update steps
    for i in range(10):
        self.iterations += 1
        self.free_voxel -= 1
        print 'here'
        inputs, heights, gt = self.update_BM()

        if self.iterations % self.observation_counter == 0:
            trainer.draw_debug()

        if self.free_voxel == 0:
            self.reset_image()
    # update parameters once
    loss_train, individual_loss, _ = self.loss_train_f(inputs, gt)
    self.update_history.append(self.iterations)
    self.loss_history.append(loss_train)
    u.plot_train_val_errors(
        [self.loss_history],
        self.update_history,
        self.save_net_path + '/training.png',
        names=['loss'])
    return loss_train
