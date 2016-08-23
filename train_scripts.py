import matplotlib
matplotlib.use('Agg')
# matplotlib.use('Qt4Agg')
import os
from theano import tensor as T
import theano
import utils as u
import nets
import dataset_utils as du
import numpy as np
from theano.sandbox import cuda as c
import experience_replay as exp
import h5py
import sys
import configargparse

def train_script_v1(options):
    print 'train script v1'
    # data params:
    # for each net a new folder is created. Here intermediate pred-
    # dictions and train, val... are saved

    BM = du.HoneyBatcherPath

    save_net_path = './data/nets/' + options.net_name + '/'

    debug_path = save_net_path + "/batches"
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)

    builder = nets.NetBuilder()
    network = builder.get_net(options.net_arch)
    loss = builder.get_loss('updates_probs_v0')
    loss_fine = builder.get_loss('updates_hydra_v5')

    # all params entered.......................

    # initialize the net
    # init a network folder where all images, models and hyper params are stored
    if options.save_net_b:
        u.create_network_folder_structure(save_net_path)
        if not options.no_bash_backup:
            u.make_bash_executable(save_net_path, add_option="--no_bash_backup")
    c.use('gpu0')

    print 'initializing network graph for net ', options.net_name
    target_t = T.ftensor4()

    l_in, l_in_direction, l_out, l_out_direction, patch_len = network()
    patch_len = 40
    if options.dummy_data_b:
        options.raw_path, options.membrane_path, options.height_gt_path, options.label_path = \
            du.generate_dummy_data(options.batch_size, options.global_edge_len, patch_len)
        options.raw_path_val, options.membrane_path_val, options.height_gt_path_val, options.label_path_val = \
            du.generate_dummy_data(options.batch_size, options.global_edge_len, patch_len)


    print 'compiling theano functions'
    loss_train_f, loss_valid_f, probs_f = \
        loss(l_in, target_t, l_out, L1_weight=options.regularization)

    # debug_f = theano.function([l_in.input_var, l_in_direction.input_var],
    #                 [lasagne.layers.get_output(l_out, deterministic=True),
    #                 lasagne.layers.get_output(l_out_direction, deterministic=True)],
    #                           allow_input_downcast=True)

    Memento = exp.BatchMemento()

    if options.exp_load != "None":
        Memento.load(options.exp_load)

    if options.fine_tune_b:
        print 'compiling theano finetuningfunctions'
        loss_train_fine_f, loss_valid_fine_f, probs_fine_f = \
            loss_fine(l_in, l_in_direction, l_out_direction,
                          L1_weight=options.regularization, margin=options.margin)
        Memento_ft = exp.BatchMemento()

    print 'Loading data and Priority queue init'
    bm = BM(options.membrane_path, label=options.label_path,
           height_gt=options.height_gt_path,
           raw=options.raw_path,
           batch_size=options.batch_size,
           patch_len=patch_len, global_edge_len=options.global_edge_len,
           padding_b=False,
           find_errors_b=options.fine_tune_b,
           clip_method=options.clip_method, timos_seeds_b=options.timos_seeds_b,
           scale_height_factor=options.scale_height_factor,
           perfect_play=options.perfect_play)
    bm.init_batch()

    if options.val_b:
        bm_val = BM(options.membrane_path_val, label=options.label_path_val,
                    height_gt=options.height_gt_path_val,
                    raw=options.raw_path_val,
                    batch_size=options.batch_size,
                    find_errors_b=options.fine_tune_b,
                    patch_len=patch_len, global_edge_len=options.global_edge_len,
                    padding_b=False,
                    clip_method=options.clip_method,
                    timos_seeds_b=options.timos_seeds_b,
                    scale_height_factor=options.scale_height_factor)
        bm_val.init_batch()

    if options.load_net_b:
        np.random.seed(651)     # change seed so different images for retrain
        print "loading network parameters from ", options.load_net_path
        u.load_network(options.load_net_path, l_out)

    # everything is initialized now train and predict every once in a while....
    converged = False       # placeholder, this is not yet implemented
    iteration = -1
    losses = [[], [], []]
    fine_tune_losses = [[], []]
    iterations = []
    ft_iteration = 0
    free_voxel_empty = (options.global_edge_len - patch_len)**2
    free_voxel = free_voxel_empty
    print 'training'
    while not converged and (iteration < options.max_iter):
        iteration += 1
        free_voxel -= 1

        if (iteration % options.save_counter == 0 or free_voxel <= 201)\
                and options.save_net_b:
            u.save_network(save_net_path, l_out, 'net_%i' % iteration)

            if options.exp_save:
                Memento.save(save_net_path +'/exp/exp_%i.h5' % iteration)
                if options.fine_tune_b:
                    Memento_ft.save(save_net_path +'/exp/exp_ft_%i.h5' % iteration)

        # predict val
        if options.val_b:
            membrane_val, gt_val, seeds_val, ids_val = bm_val.get_batches()
            probs_val = probs_f(membrane_val)
            bm_val.update_priority_queue(probs_val, seeds_val, ids_val)

        # predict train
        membrane, gt, seeds, ids = bm.get_batches()
        probs = probs_f(membrane)
        bm.update_priority_queue(probs, seeds, ids)

        # debug
        if False and iteration % 100 == 0:
            with h5py.File(debug_path+"/batch_%08i_counter_%i"%(iteration, bm.counter), 'w') as out_h5:
                out_h5.create_dataset("membrane",data=membrane ,compression="gzip")
                out_h5.create_dataset("gt",data=gt ,compression="gzip")
                out_h5.create_dataset("seeds",data=seeds ,compression="gzip")
                out_h5.create_dataset("ids",data=ids ,compression="gzip")
                out_h5.create_dataset("probs",data=probs ,compression="gzip")

        # fine-tuning: update height difference errors
        if iteration % 100 == 0:
            if options.fine_tune_b \
                    and (Memento_ft.count_new() + len(bm.global_error_dict) >= options.batch_size_ft)\
                    and iteration > options.pre_train_iter:
                error_b_type1, error_b_type2, dir1, dir2 = \
                    bm.reconstruct_path_error_inputs()

                print "mem size ", len(Memento_ft), Memento_ft.count_new(), len(bm.global_error_dict)

                Memento_ft.add_to_memory(
                    exp.stack_batch(error_b_type1, error_b_type2),
                    exp.stack_batch(dir1, dir2))

                # print debug_f(error_b_type1, dir1)
                # print debug_f(error_b_type2, dir2)
                bm.serialize_to_h5("finetunging_ser_batch_%08i_counter_%i" %
                        (iteration, bm.counter))
                bm.draw_batch(error_b_type1, "finetunging_error1_batch_%08i_counter_%i" %
                        (iteration, bm.counter),path=save_net_path + '/images/')
                bm.draw_batch(error_b_type2,
                              "finetunging_error2_batch_%08i_counter_%i" %
                        (iteration, bm.counter),path=save_net_path + '/images/')
                bm.draw_error_paths("finetuning_path_error_iter_%08i_counter_%i" %
                        (iteration, bm.counter),
                        path=save_net_path + '/images/')
                for b in range(bm.bs):
                    bm.draw_debug_image(
                        "finetunging_pic_iteration_%08i_counter_%i_b_%i" %
                        (iteration, bm.counter, b),
                        path=save_net_path + '/images/', b=b)

                if options.save_net_b:
                    # plot train images
                    bm.draw_debug_image("train_iteration_iter_%08i_counter_%i" %
                                        (iteration, bm.counter),
                                        path=save_net_path + '/images/')
                    bm.draw_error_paths("train_path_error_iter_%08i_counter_%i" %
                                        (iteration, bm.counter),
                                        path=save_net_path + '/images/')
                    if options.val_b:
                        bm_val.draw_debug_image("val_iteration_%08i" % iteration,
                                                path=save_net_path + '/images/')

                # fine tuning
                print 'Finetuning...'
                ft_iteration += 1
                batch_ft, dir_ft, mem_choice_ft = Memento_ft.get_batch(options.batch_size_ft+options.exp_ft_bs)

                if options.augment_ft:
                    batch_ft_t1, dir_t1 = du.augment_batch(batch_ft[:,0], direction=dir_ft[:,0])
                    batch_ft_t2, dir_t2 = du.augment_batch(batch_ft[:,1], direction=dir_ft[:,1])
                    batch_ft = np.concatenate((batch_ft_t1, batch_ft_t2), axis=0)
                    batch_dir_ft = np.concatenate((dir_t1, dir_t2), axis=0)
                else:
                    batch_ft = exp.flatten_stack(batch_ft).astype(theano.config.floatX)
                    batch_dir_ft = exp.flatten_stack(dir_ft).astype(np.int32)

                if options.val_b:
                    ft_loss_train_noreg = loss_valid_fine_f(batch_ft, batch_dir_ft)
                probs_fine = probs_fine_f(batch_ft, batch_dir_ft)

                ft_loss_train, individual_loss_fine = loss_train_fine_f(batch_ft, batch_dir_ft)

                if options.augment_ft:
                    individual_loss_fine = du.average_ouput(individual_loss_fine, ft=True)

                Memento_ft.update_loss(individual_loss_fine, mem_choice_ft)

                with h5py.File(debug_path+"/ft_batch_%08i_counter_%i"%(iteration, bm.counter), 'w') as out_h5:
                    out_h5.create_dataset("batch_ft",data=batch_ft, compression="gzip")
                    out_h5.create_dataset("batch_dir_ft",data=batch_dir_ft, compression="gzip")
                    out_h5.create_dataset("probs_fine",data=probs_fine, compression="gzip")
                    out_h5.create_dataset("ft_loss_train",data=ft_loss_train)

                print "loss_train_fine %.4f" % ft_loss_train
                # fine_tune_losses[0].append(ft_loss_train_noreg)
                fine_tune_losses[0].append(1)
                fine_tune_losses[1].append(ft_loss_train)

                u.plot_train_val_errors(fine_tune_losses, range(ft_iteration),
                                        save_net_path + 'ft_training.png',
                                        ['ft loss no reg no dropout', 'ft loss'])

                if options.val_b:
                    bm_val.init_batch()

                if options.reset_after_fine_tune:
                    bm.init_batch()
                    if options.val_b:
                        bm_val.init_batch()
                    free_voxel = free_voxel_empty

        # pretraining
        if iteration % 10 == 0 and iteration < options.pre_train_iter:

            if options.add_height_penalty and \
                    np.any(bm.error_indicator_pass > 0):
                gt = (gt.transpose()+bm.error_indicator_pass).transpose()


            if options.exp_bs > 0:
                Memento.add_to_memory(membrane, gt, [{"height":g.mean()} for g in gt])
                # start using exp replay only after #options.exp_warmstart iterations
                if options.exp_warmstart < iteration:
                    if options.exp_height:
                        membrane, gt, mem_choice = Memento.get_evenheight_batch(options.batch_size+options.exp_bs)
                    else:
                        membrane, gt, mem_choice = Memento.get_batch(options.batch_size+options.exp_bs)


            if options.augment_pretraining:
                a_membrane, a_gt = du.augment_batch(membrane, gt=gt)
                loss_train, individual_loss = loss_train_f(a_membrane, a_gt)
                individual_loss = du.average_ouput(individual_loss)
            else:
                loss_train, individual_loss = loss_train_f(membrane, gt)

            if options.exp_bs > 0 and options.exp_warmstart < iteration:
                Memento.update_loss(individual_loss, mem_choice)

                if iteration % 1000 == 0:
                    Memento.forget()

        # reset bms
        if free_voxel <= 201 \
            or (options.fast_reset \
                and (free_voxel_empty / 4)  % (iteration + 1) == 0 \
                and free_voxel_empty - free_voxel > 1000):

            if (len(bm.global_error_dict) > 0)\
                        and iteration > options.pre_train_iter \
                        and options.fine_tune_b:
                error_b_type1, error_b_type2, dir1, dir2 = \
                    bm.reconstruct_path_error_inputs()

                Memento_ft.add_to_memory(
                    exp.stack_batch(error_b_type1, error_b_type2),
                    exp.stack_batch(dir1, dir2))

            bm.draw_debug_image(
                "reset_train_iteration_%08i_counter_%i_freevoxel_%i" %
                (iteration, bm.counter, free_voxel),
                path=save_net_path + '/images/')
            if options.val_b:
                bm_val.draw_debug_image(
                    "reset_val_iteration_%08i_counter_%i_freevoxel_%i" %
                    (iteration, bm.counter, free_voxel),
                    path=save_net_path + '/images/')
                bm_val.init_batch()
            bm.init_batch()
            free_voxel = free_voxel_empty

        # monitor training and plot loss
        if iteration % 1000 == 0 and (iteration < options.pre_train_iter or not
                options.fine_tune_b):
            if options.augment_pretraining:
                loss_train_no_reg = float(loss_valid_f(a_membrane, a_gt))
            else:
                loss_train_no_reg = float(loss_valid_f(membrane, gt))
            if options.val_b:
                loss_valid = float(loss_valid_f(membrane_val, gt_val))
            else:
                loss_valid = 100.
            print '\x1b[2K\r loss train %.4f, loss train_noreg %.4f, ' \
                  'loss_validation %.4f, iteration %i' % \
                  (loss_train, loss_train_no_reg, loss_valid, iteration),
            sys.stdout.flush()

            if options.save_net_b:
                iterations.append(iteration)
                losses[0].append(loss_train)
                losses[1].append(loss_train_no_reg)
                losses[2].append(loss_valid)
                u.plot_train_val_errors(
                    losses,
                    iterations,
                    save_net_path + 'training.png',
                    names=['loss train', 'loss train no reg', 'loss valid'])

        # monitor growth on validation set tmp debug change train to val
        if iteration % options.save_counter == 0:
            for b in range(0, options.batch_size, 8):
                bm.draw_debug_image(
                    "train_iteration_%08i_counter_%i_freevoxel_%i_b_%i" %
                    (iteration, bm.counter, free_voxel, b),
                    path=save_net_path + '/images/', b=b)
            # bm.draw_batch(membrane,
            #               path=save_net_path+ '/images/',
            #               image_name='bat_in_%i_b' % (iteration),
            #               gt=gt,
            #               probs=probs)

        # if iteration % 100000 == 0:
        #     bm.serialize_to_h5("finetunging_ser_batch_%08i_counter_%i" %
        #         (iteration, bm.counter))
            # bm.draw_debug_image("train_iteration_%08i_counter_%i_freevoxel_%i" %
            #                     (iteration, bm.counter, free_voxel),
            #                     path=save_net_path + '/images/')


if __name__ == '__main__':
    p = configargparse.ArgParser(default_config_files=['./training.conf'])

    # where to save the net
    def_net_name = 'V5_BN_times100_ft'
    p.add('--net_name', default=def_net_name)
    p.add('--net_arch', default="ID_v5_hydra_BN")
    p.add('--no-save_net', dest='save_net_b', action='store_false')

    # reload existing net
    p.add('--load_net', dest='load_net_b', action='store_true')
    p.add('--load_net_path', default='./data/nets/V5_BN_times100/net_60000')

    # train data paths
    def_train_version = 'second'
    p.add('--train_version', default=def_train_version)
    p.add('--raw_path', default='./data/volumes/raw_%s.h5' % def_train_version)
    p.add('--membrane_path',
          default='./data/volumes/membranes_%s.h5' % def_train_version)
    p.add('--label_path',
          default='./data/volumes/label_%s.h5' % def_train_version)
    p.add('--height_gt_path',
          default='./data/volumes/height_%s.h5' % def_train_version)
    p.add('--timos_seeds_b', default=True, type=bool)

    # valid data paths
    def_valid_version = 'first_repr'
    p.add('--valid_version', default=def_valid_version)
    p.add('--raw_path_val',
          default='./data/volumes/raw_%s.h5' % def_valid_version)
    p.add('--membrane_path_val',
          default='./data/volumes/membranes_%s.h5' % def_valid_version)
    p.add('--label_path_val',
          default='./data/volumes/label_%s.h5' % def_valid_version)
    p.add('--height_gt_path_val',
          default='./data/volumes/height_%s.h5' % def_valid_version)

    # training general
    p.add('--val_b', default=True)
    p.add('--save_counter', default=10000, type=int)
    p.add('--dummy_data', dest='dummy_data_b', action='store_true')
    p.add('--global_edge_len', default=300, type=int)
    p.add('--fast_reset', action='store_true')
    p.add('--clip_method', default='clip')
    p.add('--perfect_play', action='store_true')

    # pre-training
    p.add('--pre_train_iter', default=600000, type=int)
    p.add('--regularization', default=10. ** 1, type=float)
    p.add('--batch_size', default=16, type=int)
    p.add('--no-augment_pretraining', dest='augment_pretraining',
                                      action='store_false')
    p.add('--scale_height_factor', default=100,type=float)
    p.add('--ahp', dest='add_height_penalty', action='store_true')
    

    # fine-tuning
    p.add('--batch_size_ft', default=4, type=int)
    p.add('--reset_after_fine_tune', action='store_true')
    p.add('--no-ft', dest='fine_tune_b', action='store_false')
    p.add('--margin', default=0.5, type=float)
    p.add('--no-aug-ft', dest='augment_ft', action='store_false')
    # experience replay
    # clip_method="exp20"
    p.add('--exp_bs', default=16, type=int)
    p.add('--exp_ft_bs', default=8, type=int)
    p.add('--exp_warmstart', default=1000, type=int)
    p.add('--no-exp_height', dest='exp_height', action='store_false')
    p.add('--no-exp_save', dest='exp_save', action='store_false')
    p.add('--exp_load', default="None", type=str)

    p.add('--max_iter', default=10000000000000, type=int)
    p.add('--no_bash_backup', action='store_true')

    options = p.parse_args()

    u.print_options_for_net(options)
    train_script_v1(options)
