import argparse, os
from occ_linear_regression_exp import OccLinearRegExp
import math
import numpy as np

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--train_FD', action='store_true', default=False)
    parser.add_argument('--train_RC', action='store_true', default=False)
    parser.add_argument('--train_TDRC', action='store_true', default=False)

    parser.add_argument('--readout', action='store_true', default=False)
    parser.add_argument('--n_iter_min', type=int, default=1)
    parser.add_argument('--n_iter_max', type=int, default=100)

    parser.add_argument('--test_FD', action='store_true', default=False)
    parser.add_argument('--test_RC', action='store_true', default=False)
    parser.add_argument('--test_TDRC', action='store_true', default=False)

    parser.add_argument('--test_FD_barplot', action='store_true', default=False)
    parser.add_argument('--test_RC_barplot', action='store_true', default=False)

    parser.add_argument('--pose_type', type=str, default='gt')
    parser.add_argument('--no_occ_data', action='store_true', default=False)
    parser.add_argument('--n_model_instances', type=int, default=5)
    parser.add_argument('--total_var_thr', type=float, default=0)

    parser.add_argument('--test_accs_FD', action='store_true', default=False)
    parser.add_argument('--test_accs_RC', action='store_true', default=False)
    parser.add_argument('--test_accs_TDRC', action='store_true', default=False)

    parser.add_argument('--test_FD_diff_by_layer', action='store_true', default=False)
    parser.add_argument('--test_FD_diff_readout_only', action='store_true', default=False)
    parser.add_argument('--test_RC_time_course', action='store_true', default=False)
    parser.add_argument('--test_RC_diff_time_course', action='store_true', default=False)

    parser.add_argument('--test_TDRC_diff_time_course', action='store_true', default=False)

    parser.add_argument('--data_type', type=str, default='fashion_mnist')
    parser.add_argument('--data_rotate', action='store_true', default=True)
    parser.add_argument('--valid_classes', type=str, default='0289')
    parser.add_argument('--scale_min', type=float, default=2.2) # By definition, scale_min, max are assumed to be positive.
    parser.add_argument('--scale_max', type=float, default=2.2)
    parser.add_argument('--angle_min', type=float, default=-math.pi)
    parser.add_argument('--angle_max', type=float, default=math.pi)
    parser.add_argument('--trans_min', type=float, default=-0.55) # To ensure that Spatial Transformer covers the entire object, trans_min, max has magnitude scale/sqrt(2)-1.
    parser.add_argument('--trans_max', type=float, default=0.55)
    parser.add_argument('--n_objs', type=int, default=2)
    parser.add_argument('--in_size', type=int, default=28)
    parser.add_argument('--out_size', type=int, default=50)
    parser.add_argument('--vr_min', type=float, default=0.40)    
    parser.add_argument('--vr_max', type=float, default=0.90)    
    parser.add_argument('--vr_bin_size', type=float, default=0.10)    
    parser.add_argument('--eps', type=float, default=1e-2)
    parser.add_argument('--eps2', type=float, default=1e-1)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=20)

    parser.add_argument('--add_dir_prefix', '--adp', action='store_true', default=False)
    parser.add_argument('--dir_prefix', type=str, default='/mnt/fs6/byungwookang/Occlusion')

    parser.add_argument('--model_dir', type=str, default='occ_linear_regression_models')
    parser.add_argument('--result_dir', type=str, default='occ_linear_regression_results')
    parser.add_argument('--sample_unit_dir', type=str, default='occ_linear_regression_sample_units')
    parser.add_argument('--test_result_dir', type=str, default='occ_linear_regression_test_results')

    parser.add_argument('--classifier_dir', type=str, default='no_occ_gen_models')

    parser.add_argument('--no_gpu_mode', action='store_true', default=False)
    parser.add_argument('--gpu_idx', type=int, default=0)

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    if args.add_dir_prefix:
        args.model_dir = os.path.join(args.dir_prefix, args.model_dir)
        args.result_dir = os.path.join(args.dir_prefix, args.result_dir)
        args.sample_unit_dir = os.path.join(args.dir_prefix, args.sample_unit_dir)
        args.test_result_dir = os.path.join(args.dir_prefix, args.test_result_dir)
        args.classifier_dir = os.path.join(args.dir_prefix, args.classifier_dir)
        
    # --save_dir
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --epoch
    try:
        assert args.epochs >= 1
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
        args.epochs = 1
        args.n_model_instances = 2

    args.gpu_mode = not args.no_gpu_mode 

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    exp = OccLinearRegExp(args)
    if args.train_FD:
        n_iter = 0
        readout = args.readout
        pose_type = args.pose_type
        no_occ_data = args.no_occ_data
        n_model_instances = args.n_model_instances
        n_iter_min = args.n_iter_min
        n_iter_max = args.n_iter_max

        # Determine sample units.
        print('Determine sample units.')
        for target_model_name in ['FDClassifier', 'NoOccFDClassifier']:
            for target_idx in range(n_model_instances):
                print('target_model_name: {}'.format(target_model_name))
                print('target_idx: {}'.format(target_idx+1))
                print('')
                exp.determine_sample_units(pose_type, target_model_name, target_idx)


        for source_model_name in ['FDClassifier', 'NoOccFDClassifier']:
            for target_model_name in ['FDClassifier', 'NoOccFDClassifier']:
                for source_idx in range(n_model_instances):
                    for target_idx in range(n_model_instances):
                        print('')
                        print('n_iter: {}'.format(n_iter+1))
                        n_iter += 1
                        if n_iter >= n_iter_min and n_iter <= n_iter_max:
                            print('no_occ_data: {}'.format(no_occ_data))
                            print('pose_type: {}'.format(pose_type))
                            print('source_model_name: {}'.format(source_model_name))
                            print('target_model_name: {}'.format(target_model_name))
                            print('source_idx: {}'.format(source_idx+1))                        
                            print('target_idx: {}'.format(target_idx+1))
                            print('')
                            exp.train_linear_regressor(pose_type, no_occ_data, source_model_name, target_model_name, source_idx, target_idx, readout)

    if args.train_RC:
        n_iter = 0
        readout = args.readout
        pose_type = args.pose_type
        no_occ_data = args.no_occ_data
        n_model_instances = args.n_model_instances
        n_iter_min = args.n_iter_min
        n_iter_max = args.n_iter_max

        # Determine sample units.
        print('Determine sample units.')
        for target_model_name in ['RCClassifier', 'NoOccRCClassifier']:
            for target_idx in range(n_model_instances):
                print('target_model_name: {}'.format(target_model_name))
                print('target_idx: {}'.format(target_idx+1))
                print('')
                exp.determine_sample_units(pose_type, target_model_name, target_idx)


        for source_model_name in ['RCClassifier', 'NoOccRCClassifier']:
            for target_model_name in ['RCClassifier', 'NoOccRCClassifier']:                    
                for source_idx in range(n_model_instances):
                    for target_idx in range(n_model_instances):
                        print('')
                        print('n_iter: {}'.format(n_iter+1))
                        n_iter += 1
                        if n_iter >= n_iter_min and n_iter <= n_iter_max:
                            print('no_occ_data: {}'.format(no_occ_data))
                            print('pose_type: {}'.format(pose_type))
                            print('source_model_name: {}'.format(source_model_name))
                            print('target_model_name: {}'.format(target_model_name))
                            print('source_idx: {}'.format(source_idx+1))                        
                            print('target_idx: {}'.format(target_idx+1))
                            print('')
                            exp.train_linear_regressor(pose_type, no_occ_data, source_model_name, target_model_name, source_idx, target_idx, readout)



    if args.train_TDRC:
        n_iter = 0
        pose_type = args.pose_type
        no_occ_data = args.no_occ_data
        n_model_instances = args.n_model_instances
        n_iter_min = args.n_iter_min
        n_iter_max = args.n_iter_max


        # Determine sample units.
        print('Determine sample units.')
        for target_model_name in ['TDRCClassifier', 'NoOccTDRCClassifier']:
            for target_idx in range(n_model_instances):
                print('target_model_name: {}'.format(target_model_name))
                print('target_idx: {}'.format(target_idx+1))
                print('')
                exp.determine_sample_units(pose_type, target_model_name, target_idx)


        for source_model_name in ['TDRCClassifier', 'NoOccTDRCClassifier']:
            for target_model_name in ['TDRCClassifier', 'NoOccTDRCClassifier']:                    
                for source_idx in range(n_model_instances):
                    for target_idx in range(n_model_instances):
                        print('')
                        print('n_iter: {}'.format(n_iter+1))
                        n_iter += 1
                        if n_iter >= n_iter_min and n_iter <= n_iter_max:
                            print('no_occ_data: {}'.format(no_occ_data))
                            print('pose_type: {}'.format(pose_type))
                            print('source_model_name: {}'.format(source_model_name))
                            print('target_model_name: {}'.format(target_model_name))
                            print('source_idx: {}'.format(source_idx+1))                        
                            print('target_idx: {}'.format(target_idx+1))
                            print('')
                            # all layers instead of doing readout and the other layers separately.
                            exp.train_linear_regressor_all_layers(pose_type, no_occ_data, source_model_name, target_model_name, source_idx, target_idx) 


    if args.test_FD:
        n_iter = 0
        readout = args.readout
        pose_type = args.pose_type
        n_model_instances = args.n_model_instances
        total_var_thr = args.total_var_thr
        for no_occ_data in [True, False]:
            for source_model_name in ['FDClassifier', 'NoOccFDClassifier']:
                for target_model_name in ['FDClassifier', 'NoOccFDClassifier']:
                    print('')
                    print('n_iter: {}'.format(n_iter+1))
                    n_iter += 1
                    print('no_occ_data: {}'.format(no_occ_data))
                    print('pose_type: {}'.format(pose_type))
                    print('source_model_name: {}'.format(source_model_name))
                    print('target_model_name: {}'.format(target_model_name))
                    print('')
                    exp.test_linear_regressor_aggregate_over_model_instances(pose_type, no_occ_data, source_model_name, target_model_name, readout, n_model_instances, total_var_thr=total_var_thr)

    if args.test_FD_barplot:
        readout = args.readout
        pose_type = args.pose_type
        total_var_thr = args.total_var_thr
        model_type = 'FD'
        for measure_type in ['median_r2', 'median_pcc']:
            exp.test_results_bar_graphs(measure_type, pose_type, model_type, readout, total_var_thr)


    if args.test_RC:
        n_iter = 0
        readout = args.readout
        pose_type = args.pose_type
        n_model_instances = args.n_model_instances
        total_var_thr = args.total_var_thr
        for no_occ_data in [True, False]:
            for source_model_name in ['RCClassifier', 'NoOccRCClassifier']:
                for target_model_name in ['RCClassifier', 'NoOccRCClassifier']:
                    print('')
                    print('n_iter: {}'.format(n_iter+1))
                    n_iter += 1
                    print('no_occ_data: {}'.format(no_occ_data))
                    print('pose_type: {}'.format(pose_type))
                    print('source_model_name: {}'.format(source_model_name))
                    print('target_model_name: {}'.format(target_model_name))
                    print('')
                    exp.test_linear_regressor_aggregate_over_model_instances(pose_type, no_occ_data, source_model_name, target_model_name, readout, n_model_instances, total_var_thr=total_var_thr)

    if args.test_RC_barplot:
        readout = args.readout
        pose_type = args.pose_type
        total_var_thr = args.total_var_thr
        model_type = 'RC'
        for measure_type in ['median_r2', 'median_pcc']:
            exp.test_results_bar_graphs(measure_type, pose_type, model_type, readout, total_var_thr)



    if args.test_TDRC:
        n_iter = 0
        pose_type = args.pose_type
        n_model_instances = args.n_model_instances
        total_var_thr = args.total_var_thr
        for no_occ_data in [True, False]:
            for source_model_name in ['TDRCClassifier', 'NoOccTDRCClassifier']:
                for target_model_name in ['TDRCClassifier', 'NoOccTDRCClassifier']:
                    print('')
                    print('n_iter: {}'.format(n_iter+1))
                    n_iter += 1
                    print('no_occ_data: {}'.format(no_occ_data))
                    print('pose_type: {}'.format(pose_type))
                    print('source_model_name: {}'.format(source_model_name))
                    print('target_model_name: {}'.format(target_model_name))
                    print('')
                    # all layers instead of doing readout and the other layers separately.
                    exp.test_linear_regressor_aggregate_over_model_instances_all_layers(pose_type, no_occ_data, source_model_name, target_model_name, n_model_instances, total_var_thr=total_var_thr)



    if args.test_accs_FD:
        pose_type = args.pose_type
        n_model_instances = args.n_model_instances
        for no_occ_data in [True, False]:
            for model_name in ['FDClassifier', 'NoOccFDClassifier']:
                exp.test_accuracy(pose_type, no_occ_data, model_name, n_model_instances)

        exp.test_accuracy_bar_graphs(pose_type, 'FD')

    if args.test_accs_RC:
        pose_type = args.pose_type
        n_model_instances = args.n_model_instances
        for no_occ_data in [True, False]:
            for model_name in ['RCClassifier', 'NoOccRCClassifier']:
                exp.test_accuracy(pose_type, no_occ_data, model_name, n_model_instances)

        exp.test_accuracy_bar_graphs(pose_type, 'RC')


    if args.test_accs_TDRC:
        pose_type = args.pose_type
        n_model_instances = args.n_model_instances
        for no_occ_data in [True, False]:
            for model_name in ['TDRCClassifier', 'NoOccTDRCClassifier']:
                exp.test_accuracy(pose_type, no_occ_data, model_name, n_model_instances)

        exp.test_accuracy_bar_graphs(pose_type, 'TDRC')


    if args.test_FD_diff_by_layer:
        pose_type = args.pose_type
        total_var_thr = args.total_var_thr
        separate = True
        for target_model_name in ['FDClassifier', 'NoOccFDClassifier']:
            for measure_type in ['median_pcc']:
                print('')
                print('pose_type: {}'.format(pose_type))
                print('target_model_name: {}'.format(target_model_name))
                print('measure_type: {}'.format(measure_type))
                print('')
                exp.test_results_FD_distinguishability_by_layer_bar_graph(measure_type, target_model_name, pose_type, total_var_thr=total_var_thr, separate=separate)


    if args.test_FD_diff_readout_only:
        pose_type = args.pose_type
        total_var_thr = args.total_var_thr
        for target_model_name in ['FDClassifier', 'NoOccFDClassifier']:
            for measure_type in ['median_pcc']:
                print('')
                print('pose_type: {}'.format(pose_type))
                print('target_model_name: {}'.format(target_model_name))
                print('measure_type: {}'.format(measure_type))
                print('')
                exp.test_results_FD_distinguishability_readout_layer_bar_graph(measure_type, target_model_name, pose_type, total_var_thr=total_var_thr)


    if args.test_RC_time_course:
        pose_type = args.pose_type
        readout = args.readout
        total_var_thr = args.total_var_thr
        for source_model_name in ['RCClassifier', 'NoOccRCClassifier']:
            for target_model_name in ['RCClassifier', 'NoOccRCClassifier']:
                for measure_type in ['median_r2', 'median_pcc']:
                    print('')
                    print('pose_type: {}'.format(pose_type))
                    print('source_model_name: {}'.format(source_model_name))
                    print('target_model_name: {}'.format(target_model_name))
                    print('measure_type: {}'.format(measure_type))
                    print('')
                    exp.test_results_RC_time_course_graph(measure_type, source_model_name, target_model_name, pose_type, readout, total_var_thr=total_var_thr)


    if args.test_RC_diff_time_course:
        pose_type = args.pose_type
        readout = args.readout
        total_var_thr = args.total_var_thr
        for target_model_name in ['RCClassifier', 'NoOccRCClassifier']:
            for measure_type in ['median_pcc']:
                print('')
                print('pose_type: {}'.format(pose_type))
                print('target_model_name: {}'.format(target_model_name))
                print('measure_type: {}'.format(measure_type))
                print('')
                exp.test_results_RC_distinguishability_time_course_graph(measure_type, target_model_name, pose_type, readout, total_var_thr=total_var_thr)


    if args.test_TDRC_diff_time_course:
        pose_type = args.pose_type
        total_var_thr = args.total_var_thr
        for target_model_name in ['TDRCClassifier', 'NoOccTDRCClassifier']:
            for measure_type in ['median_pcc']:
                print('')
                print('pose_type: {}'.format(pose_type))
                print('target_model_name: {}'.format(target_model_name))
                print('measure_type: {}'.format(measure_type))
                print('')
                # all layers instead of doing readout and the other layers separately.
                exp.test_results_TDRC_distinguishability_time_course_graph_all_layers(measure_type, target_model_name, pose_type, total_var_thr=total_var_thr)



if __name__ == '__main__':
    main()
