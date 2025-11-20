import argparse, os
from new_pose_net_exp import PoseNetExp
import math

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--train_baseline', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--train_refiner', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--n_trials', nargs='+', type=int, default=[])

    parser.add_argument('--data_names', nargs='+', type=str, default=[])
    parser.add_argument('--data_name', type=str, default='light_occ')

    parser.add_argument('--baseline_type', type=str, default='Baseline')
    parser.add_argument('--baseline_lr', type=float, default=1e-3)
    parser.add_argument('--baseline_batch_size', type=int, default=64)
    parser.add_argument('--baseline_test_batch_size', type=int, default=100)
    parser.add_argument('--baseline_epoch', type=int, default=100)

    parser.add_argument('--model_type', type=str, default='PoseNetBasicDetached')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=100)

    parser.add_argument('--refiner_type', type=str, default='PoseNetRefiner')
    parser.add_argument('--refiner_lr', type=float, default=2e-5)
    parser.add_argument('--refiner_batch_size', type=int, default=64)
    parser.add_argument('--refiner_test_batch_size', type=int, default=100)
    parser.add_argument('--refiner_epoch', type=int, default=100)

    parser.add_argument('--save_dir', type=str, default='new_pose_net_models')
    parser.add_argument('--result_dir', type=str, default='new_pose_net_results')
    
    parser.add_argument('--n_vis', type=int, default=20)

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
        args.batch_size = 3
        args.test_batch_size = 2

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.train_baseline:
        for data_name in args.data_names:
            args.data_name = data_name
            for i in args.n_trials:
                print('')
                print('data_name: {} / trial_idx: {}'.format(data_name, i+1))
                print('')
                exp = PoseNetExp(args)
                exp.train_baseline(trial_idx=i)

    if args.train:
        for data_name in args.data_names:
            args.data_name = data_name
            for i in args.n_trials:
                print('')
                print('data_name: {} / trial_idx: {}'.format(data_name, i+1))
                print('')
                exp = PoseNetExp(args)
                exp.train(trial_idx=i)
    
    if args.test:
        exp = PoseNetExp(args)
        exp.test()

    if args.train_refiner:
        exp = PoseNetExp(args)
        exp.train_refiner()

if __name__ == '__main__':
    main()