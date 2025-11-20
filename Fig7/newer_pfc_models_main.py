'''
This is used to train PFC1Clipping and PFC1ClippingNoFG on heavy_occ (vr: 0.4-0.9) or light_occ (vr: 0.25-0.30)
and for various pose conditions (gt, pred, and original poses). For pred pose we need a baseline and posenet.

The save and result directories are structured such that we first classify by pose type, then by data type, and then by model type.

As of 07/03/18, We need to train models for :
gt_pose: both heavy and light_occ AND both PFC and PFCNoFG (2*2 = 4 model to train).
pred_pose: light_occ AND PFC (1 model to train).
original_pose: light_occ AND PFC (1 model to train).

Also, we want to train each model 5 times with different random initializations.
'''

import argparse, os
from newer_pfc_models_exp import PFCModelsExp
import math

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--train_all', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--n_trials', nargs='+', type=int, default=[])

    parser.add_argument('--data_name', type=str, default='light_occ')
    parser.add_argument('--pose_type', type=str, default='pred')

    parser.add_argument('--baseline_type', type=str, default='Baseline')
    parser.add_argument('--baseline_save_dir', type=str, default='new_pose_net_models')

    parser.add_argument('--posenet_type', type=str, default='PoseNetBasicDetached')
    parser.add_argument('--posenet_save_dir', type=str, default='new_pose_net_models')

    parser.add_argument('--model_type', type=str, default='PFCNoFG')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=100)

    parser.add_argument('--gan_model_type', type=str, default='GAN')
    parser.add_argument('--gan_save_dir', type=str, default='search_space_canon_gans_models')

    parser.add_argument('--reduced_data_dir', type=str, default='reduced_fashion_mnist/0289')

    parser.add_argument('--save_dir', type=str, default='newer_pfc_models_models')
    parser.add_argument('--result_dir', type=str, default='newer_pfc_models_results')
    
    parser.add_argument('--n_vis', type=int, default=50)

    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--gpu_idx', type=int, default=1)

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')


    if args.debug:
        args.batch_size = 2
        args.test_batch_size = 2

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()


    if args.train:
        exp = PFCModelsExp(args)
        exp.train()

    if args.train_all:
        models_to_train = [('gt', 'heavy_occ', 'PFC'), ('gt', 'heavy_occ', 'PFCNoFG'), ('gt', 'light_occ', 'PFC'), ('gt', 'light_occ', 'PFCNoFG'), 
                            ('pred', 'light_occ', 'PFC'), 
                            ('original', 'light_occ', 'PFC')]

        for pose_type, data_name, model_type in models_to_train:
            for i in args.n_trials:
                print('')
                print('pose_type: {} / data_name: {} / model_type: {} / trial_idx: {}'.format(pose_type, data_name, model_type, i+1))
                print('')
                args.pose_type = pose_type
                args.data_name = data_name
                args.model_type = model_type
                exp = PFCModelsExp(args)
                exp.train(trial_idx=i)

if __name__ == '__main__':
    main()