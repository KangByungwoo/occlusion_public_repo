import argparse, os, math
from unoccluded_multi_object_classifier_exp import UnoccludedMultiObjectClassifierExp
import torch
import numpy as np

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()
    # As of 03/27/18, we distinguish different settings, including both training and architectural hps, of the same underlying model as different numbers.
    parser.add_argument('--model_type', type=str, default='FD')
    parser.add_argument('--hp', action='store_true', default=False)
    parser.add_argument('--hp_model_types', nargs='+', type=str, default=[])
    parser.add_argument('--n_hp_trials', type=int, default=11)

    parser.add_argument('--best_hp_test', action='store_true', default=False)
    parser.add_argument('--best_hp_test_model_types', nargs='+', type=str, default=[])
    parser.add_argument('--n_best_hp_test', type=int, default=5)

    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument('--data_type', type=str, default='fashion_mnist')
    parser.add_argument('--data_rotate', action='store_true', default=True)
    parser.add_argument('--valid_classes', type=str, default='0289')
    parser.add_argument('--angle_min', type=float, default=-math.pi)
    parser.add_argument('--angle_max', type=float, default=math.pi)
    parser.add_argument('--trans_min', type=float, default=0.1)
    parser.add_argument('--trans_max', type=float, default=0.15)
    parser.add_argument('--n_objs', type=int, default=2)
    parser.add_argument('--in_size', type=int, default=28)
    parser.add_argument('--out_size', type=int, default=50)
    parser.add_argument('--vr_min', type=float, default=0.25)    
    parser.add_argument('--vr_max', type=float, default=0.50)    
    parser.add_argument('--vr_bin_size', type=float, default=0.05)    
    parser.add_argument('--eps', type=float, default=1e-2)
    parser.add_argument('--eps2', type=float, default=1e-1)

    parser.add_argument('--test_data_type', type=str, default='fashion_mnist')
    parser.add_argument('--test_data_rotate', action='store_true', default=True)
    parser.add_argument('--test_valid_classes', type=str, default='0289')
    parser.add_argument('--test_angle_min', type=float, default=-math.pi)
    parser.add_argument('--test_angle_max', type=float, default=math.pi)
    parser.add_argument('--test_trans_min', type=float, default=0.1)
    parser.add_argument('--test_trans_max', type=float, default=0.15)
    parser.add_argument('--test_n_objs', type=int, default=2)
    parser.add_argument('--test_in_size', type=int, default=28)
    parser.add_argument('--test_out_size', type=int, default=50)
    parser.add_argument('--test_vr_min', type=float, default=0.25)    
    parser.add_argument('--test_vr_max', type=float, default=0.50)    
    parser.add_argument('--test_vr_bin_size', type=float, default=0.05)    
    parser.add_argument('--test_eps', type=float, default=1e-2)
    parser.add_argument('--test_eps2', type=float, default=1e-1)

    parser.add_argument('--vis_wrong_cls2', action='store_true', default=False)
    parser.add_argument('--n_vis', type=int, default=100)

    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=32, help='The size of batch')
    parser.add_argument('--dropout_p', type=float, default=0.5)
    
    parser.add_argument('--test_batch_size', type=int, default=100, help='The size of test batch')
    parser.add_argument('--save_dir', type=str, default='unoccluded_multi_object_classifier_models')
    parser.add_argument('--result_dir', type=str, default='unoccluded_multi_object_classifier_results')
    parser.add_argument('--log_dir', type=str, default='unoccluded_multi_object_classifier_logs')
    parser.add_argument('--hp_dir', type=str, default='unoccluded_multi_object_classifier_hp_searchs')
    parser.add_argument('--best_hp_test_dir', type=str, default='unoccluded_multi_object_classifier_best_hp_test')

    parser.add_argument('--gpu_mode', action='store_true', default=True)
    parser.add_argument('--gpu_idx', type=int, default=0)

    return check_args(parser.parse_args())

def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --log_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if not os.path.exists(args.hp_dir):
        os.makedirs(args.hp_dir)

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

    # costruct args.test_data_dir
    args.test_valid_classes_flag = args.test_valid_classes

    if args.test:
        if args.test_data_rotate:
            test_data_rotate_flag = '_rotated_{:.0f}_{:.0f}_trans_{:.2f}_{:.2f}'.format(args.test_angle_min*180/math.pi, args.test_angle_max*180/math.pi, args.test_trans_min, args.test_trans_max)
        else:
            test_data_rotate_flag = ''
        args.test_data_dir = os.path.join('extensive_info_multi_' + args.test_data_type + test_data_rotate_flag + '_' + args.test_valid_classes_flag, 
                                        '{}_{}_{}_{:.2f}_{:.2f}_{:.0e}_{:.0e}'.format(args.test_n_objs, args.test_in_size, args.test_out_size, args.test_vr_min, args.test_vr_max, args.test_eps, args.test_eps2))
    else:
        args.test_data_dir = ''

    # --gpu_mode:
    if args.gpu_mode:
        try:
            assert torch.cuda.is_available()
        except:
            print('cuda is not available.')

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.best_hp_test:
        best_hps = {}
        best_hps['FD'] = {'lr': 6.3*10**(-4), 'weight_decay': 2.3*10**(-4), 'batch_size': 64, 'dropout_p': 0.52}
        best_hps['FDFat'] = {'lr': 6.2*10**(-4), 'weight_decay': 7.0*10**(-5), 'batch_size': 64, 'dropout_p': 0.56}
        best_hps['RC'] = {'lr': 2.0*10**(-3), 'weight_decay': 2.2*10**(-6), 'batch_size': 64, 'dropout_p': 0.41}
        best_hps['RCControl'] = {'lr': 6.9*10**(-4), 'weight_decay': 6.0*10**(-6), 'batch_size': 64, 'dropout_p': 0.32}
        best_hps['RCLast'] = {'lr': 1.3*10**(-3), 'weight_decay': 1.4*10**(-11), 'batch_size': 32, 'dropout_p': 0.46}
        best_hps['RC5'] = {'lr': 1.0*10**(-3), 'weight_decay': 4.1*10**(-9), 'batch_size': 32, 'dropout_p': 0.68}
        best_hps['RCLast5'] = {'lr': 1.5*10**(-3), 'weight_decay': 2.2*10**(-5), 'batch_size': 32, 'dropout_p': 0.47}
        best_hps['RCLast5Uniform'] = {'lr': 1.5*10**(-3), 'weight_decay': 2.2*10**(-5), 'batch_size': 32, 'dropout_p': 0.47}

        best_hps['FDFatterTaller2'] = {'lr': 2.2*10**(-4), 'weight_decay': 2.4*10**(-6), 'batch_size': 64, 'dropout_p': 0.65, 'val_acc': 0.915}
        best_hps['WeightSharedFDFatterTaller2'] = {'lr': 2.1*10**(-4), 'weight_decay': 3.1*10**(-10), 'batch_size': 32, 'dropout_p': 0.34, 'val_acc': 0.915}
        best_hps['TDRCTypeC'] = {'lr': 1.5*10**(-3), 'weight_decay': 1.9*10**(-5), 'batch_size': 64, 'dropout_p': 0.30}
        best_hps['TDRCTypeC5'] = {'lr': 4.3*10**(-4), 'weight_decay': 4.7*10**(-8), 'batch_size': 32, 'dropout_p': 0.60}
        best_hps['TDRCTypeD'] = {'lr': 3.5*10**(-3), 'weight_decay': 3.9*10**(-5), 'batch_size': 64, 'dropout_p': 0.35, 'val_acc': 0.923}
        best_hps['TDRCTypeD5'] = {'lr': 6.2*10**(-4), 'weight_decay': 3.4*10**(-5), 'batch_size': 32, 'dropout_p': 0.50, 'val_acc': 0.935}


        for model_type, hps in best_hps.items():
            if len(args.best_hp_test_model_types) != 0 and model_type not in args.best_hp_test_model_types:
                continue
            args.model_type = model_type
            args.lr = hps['lr']
            args.weight_decay = hps['weight_decay']
            args.batch_size = hps['batch_size']
            args.dropout_p = hps['dropout_p']
            if 'val_acc' in hps:
                hp_search_val_total_acc = hps['val_acc']
            else:
                # I only added 'val_acc' to the models I need to re-run due to bad best_hp_test trials. For the rest of the models, I just set hp_search_val_acc to zero, so that it doesn't have any effect.
                hp_search_val_total_acc = 0


            total_accs = []
            ind_cls_acc1s = []
            ind_cls_acc2s = []

            trial_idx = 0
            good_trial_count = 0

            while True:
                print('[{}-th Trial / {} Good Trials so far] model_type: {}, lr: {:.2e}, weight_decay: {:.2e}, batch_size: {}, dropout_p: {:.2f}'.format((trial_idx+1), (good_trial_count), args.model_type, args.lr, args.weight_decay, args.batch_size, args.dropout_p))
                # Train the model, save the best val acc model. Then, load it and evaluate its accuracy on the test set.
                exp = UnoccludedMultiObjectClassifierExp(args)
                total_acc, ind_cls_acc1, ind_cls_acc2, val_total_acc = exp.best_hp_test(trial_idx=(trial_idx+1))
                if val_total_acc > hp_search_val_total_acc - 0.05:
                    total_accs.append(total_acc)
                    ind_cls_acc1s.append(ind_cls_acc1)
                    ind_cls_acc2s.append(ind_cls_acc2)
                    good_trial_count += 1
                trial_idx += 1
                if good_trial_count == args.n_best_hp_test:
                    print('{} Good Trials acheived!'.format(good_trial_count))
                    break

            total_accs = np.array(total_accs)
            total_accs_mean = np.mean(total_accs)
            total_accs_std = np.std(total_accs)

            ind_cls_acc1s = np.array(ind_cls_acc1s)
            ind_cls_acc1s_mean = np.mean(ind_cls_acc1s)
            ind_cls_acc1s_std = np.std(ind_cls_acc1s)

            ind_cls_acc2s = np.array(ind_cls_acc2s)
            ind_cls_acc2s_mean = np.mean(ind_cls_acc2s)
            ind_cls_acc2s_std = np.std(ind_cls_acc2s)

            with open(os.path.join(exp.best_hp_test_dir, 'average_test_performance.txt'), 'w') as f:
                f.write('total_accs_mean: {:.4f}\n'.format(total_accs_mean))
                f.write('total_accs_std: {:.4f}\n'.format(total_accs_std))
                f.write('ind_cls_acc1s_mean: {:.4f}\n'.format(ind_cls_acc1s_mean))
                f.write('ind_cls_acc1s_std: {:.4f}\n'.format(ind_cls_acc1s_std))
                f.write('ind_cls_acc2s_mean: {:.4f}\n'.format(ind_cls_acc2s_mean))
                f.write('ind_cls_acc2s_std: {:.4f}\n'.format(ind_cls_acc2s_std))


    if args.hp:
        model_types = ['FD', 'FDFat', 'RC', 'RCReversed', 'RCLast', 'RCControl', 'RC5', 'RCReversed5', 'RCLast5', 'FDFatterTaller2', 'WeightSharedFDFatterTaller2', 'TDRCTypeC', 'TDRCTypeC5', 'TDRCTypeD', 'TDRCTypeD5']
        for model_type in model_types:
            if len(args.hp_model_types) != 0 and model_type not in args.hp_model_types:
                continue
            args.model_type = model_type
            for i in range(args.n_hp_trials):
                args.lr = 10**np.random.uniform(-4, math.log(5.0*10**(-3), 10))
                args.weight_decay = np.random.choice([0.0, 10**np.random.uniform(-12, -3)], p=[0.15, 0.85])
                args.batch_size = np.random.choice([32, 64, 128])
                args.dropout_p = np.random.uniform(0.2, 0.7)
                print('[{}-th Search] lr: {:.2e}, weight_decay: {:.2e}, batch_size: {}, dropout_p: {:.2f}'.format((i+1), args.lr, args.weight_decay, args.batch_size, args.dropout_p))
                exp = UnoccludedMultiObjectClassifierExp(args)
                current_val_acc = exp.hp_search()

                if not os.path.exists(os.path.join(exp.hp_dir, 'best_val_acc_hp.txt')):
                    with open(os.path.join(exp.hp_dir, 'best_val_acc_hp.txt'), 'w') as summary:
                        summary.write('acc_{:.3f}_lr_{:.1e}_wd_{:.1e}_bs_{}_dp_{:.2f}'.format(current_val_acc, exp.lr, exp.weight_decay, exp.batch_size, exp.dropout_p))
                else:
                    with open(os.path.join(exp.hp_dir, 'best_val_acc_hp.txt'), 'r') as summary:
                        best_val_acc = float(summary.read(9)[4:])
                    if current_val_acc > best_val_acc:
                        with open(os.path.join(exp.hp_dir, 'best_val_acc_hp.txt'), 'w') as summary:
                            summary.write('acc_{:.3f}_lr_{:.1e}_wd_{:.1e}_bs_{}_dp_{:.2f}'.format(current_val_acc, exp.lr, exp.weight_decay, exp.batch_size, exp.dropout_p))

    if args.train:
        exp = UnoccludedMultiObjectClassifierExp(args)
        exp.train()

    if args.test:
        exp = UnoccludedMultiObjectClassifierExp(args)
        exp.test(args.test_data_dir)

if __name__ == '__main__':
    main()