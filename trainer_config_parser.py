import configargparse

def get_options():
    """
    config parser wrapper. Used to generate options object that can be 
    propagated throughout all member classes of the trainer class
    :param options:
    """
    p = configargparse.ArgParser(default_config_files=['./../data/config/training.conf'])

    # where to save the net
    def_net_name = 'V5_BN_times100_ft'
    p.add('-c', '--my-config', is_config_file=True)
    p.add('--net_name', default=def_net_name)
    p.add('--net_arch', default="ID_v5_hydra_BN")
    p.add('--no-save_net', dest='save_net_b', action='store_false')

    # reload existing net
    p.add('--load_net', dest='load_net_b', action='store_true')
    p.add('--load_net_path', default='./data/nets/V5_BN_times100/net_60000')

    p.add('--gpu', default='gpu0')

    # train data paths
    def_train_version = 'second_repr'       # def change me
    p.add('--dataset', default="Polygon")

    p.add('--train_version', default=def_train_version)
    p.add('--seed_method', type=str, default="timo",
          help='available metods: gt, timo, grid',
          dest='seed_method')
    p.add('--input_data_path',type=str, default="../data/volumes/raw_honeycomb.h5")

    # valid data paths
    def_valid_version = 'first_repr'
    p.add('--valid_version', default=def_valid_version)

    # training general
    p.add('--no-val', dest='val_b', action='store_false')
    p.add('--export_quick_eval', action='store_true')
    p.add('--save_counter', default=1000, type=int)
    p.add('--dummy_data', dest='dummy_data_b', action='store_true')
    p.add('--global_edge_len', default=300, type=int)
    p.add('--fast_reset', action='store_true')
    p.add('--clip_method', default='clip')
    p.add('--padding_b', action='store_true')
    p.add('--merge_seeds', dest='merge_seeds', action='store_true')
    p.add('--train_merge', dest='train_merge', action='store_true')

    # pre-training
    p.add('--pre_train_iter', default=600000, type=int)
    p.add('--regularization', default=10. ** 1, type=float)
    p.add('--network_channels', default=0, type=int)
    p.add('--batch_size', default=16, type=int)
    p.add('--no-augment_pretraining', dest='augment_pretraining',
                                      action='store_false')
    p.add('--create_holes', action='store_true', default=False)

    p.add('--scale_height_factor', default=100,type=float)
    p.add('--ahp', dest='add_height_penalty', action='store_true')
    p.add('--max_penalty_pixel', default=3, type=float)

    # fine-tuning
    p.add('--batch_size_ft', default=4, type=int)
    p.add('--reset_after_fine_tune', action='store_true')
    p.add('--no-ft', dest='fine_tune_b', action='store_false')
    p.add('--reset-ft', dest='rs_ft', action='store_true')
    p.add('--reset_pretraining', dest='reset_pretraining', action='store_true')
    p.add('--margin', default=0.5, type=float)
    p.add('--no-aug-ft', dest='augment_ft', action='store_false')
    # experience replay
    # clip_method="exp20"
    p.add('--exp_bs', default=16, type=int)
    p.add('--exp_ft_bs', default=8, type=int)
    p.add('--exp_warmstart', default=1000, type=int)
    p.add('--exp_acceptance_rate', default=0.1, type=float)
    p.add('--no-exp_height', dest='exp_height', action='store_false')
    p.add('--no-exp_save', dest='exp_save', action='store_false')
    p.add('--exp_mem_size', default=20000, type=int)
    p.add('--exp_load', default="None", type=str)
    p.add('--no-exp_loss', dest='exp_loss', action='store_false')
    p.add('--exp_wlast', default=1., type=float)

    p.add('--max_iter', default=10000000000000, type=int)
    p.add('--no_bash_backup', action='store_true')
    p.add('--lowercomplete_e', default=0., type=float)

    p.add('--raw_path', default="None", type=str)
    p.add('--membrane_path', default="None", type=str)
    p.add('--label_path', default="None", type=str)
    p.add('--height_gt_path', default="None", type=str)

    options = p.parse_args()

    if options.raw_path == "None":
        options.raw_path ='./../data/volumes/input_%s.h5' % options.train_version

    if options.raw_path == "None":
        options.raw_path ='./../data/volumes/raw_%s.h5' % options.train_version
    if options.membrane_path == "None":
        options.membrane_path ='./../data/volumes/membranes_%s.h5' % options.train_version
    if options.label_path == "None":
        options.label_path ='./../data/volumes/label_%s.h5' % options.train_version
    if options.height_gt_path == "None":
        options.height_gt_path ='./../data/volumes/height_%s.h5' % options.train_version
    print 'saving files to ', options.net_name
    return options


if __name__ == '__main__':

    options = get_options()
    print options