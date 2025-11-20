
import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pfc_classifiers import *

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

'''
TO DO LIST (as of 09/10/18 10pm)
2. Implement a method that shows the all trial-to-trial R2 values.
'''

class OccLinearRegExp(object):
    def __init__(self, args):
        self.debug = args.debug

        self.data_type = args.data_type
        self.data_rotate = args.data_rotate
        self.valid_classes = args.valid_classes
        self.scale_min = args.scale_min
        self.scale_max = args.scale_max
        self.angle_min = args.angle_min
        self.angle_max = args.angle_max
        self.trans_min = args.trans_min
        self.trans_max = args.trans_max
        self.n_objs = args.n_objs
        self.in_size = args.in_size
        self.out_size = args.out_size 
        self.vr_min = args.vr_min
        self.vr_max = args.vr_max
        self.vr_bin_size = args.vr_bin_size
        self.eps = args.eps
        self.eps2 = args.eps2

        self.lr = args.lr
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        
        self.model_dir = args.model_dir
        self.result_dir = args.result_dir
        self.sample_unit_dir = args.sample_unit_dir
        self.test_result_dir = args.test_result_dir

        self.classifier_dir = args.classifier_dir # no_occ_gen_models
        
        self.gpu_mode = args.gpu_mode
        self.gpu_idx = args.gpu_idx
        # set gpu device
        if self.gpu_mode:
            torch.cuda.set_device(self.gpu_idx)


        if self.data_rotate:
            self.data_rotate_flag = '_vr_{:.2f}_{:.2f}_scale_{:.1f}_{:.1f}_angle_{:.0f}_{:.0f}_trans_{:.2f}_{:.2f}'.format(self.vr_min, self.vr_max, self.scale_min, self.scale_max, 
                self.angle_min*180/math.pi, self.angle_max*180/math.pi, self.trans_min, self.trans_max)
        else:
            self.data_rotate_flag = ''
        
        if self.debug:
            self.data_dir = os.path.join('debugging_gen3_multi_' + self.data_type + self.data_rotate_flag + '_' + self.valid_classes,
                                            '{}_{}_{}_{:.0e}_{:.0e}'.format(self.n_objs, self.in_size, self.out_size, self.eps, self.eps2))
        else:
            self.data_dir = os.path.join('gen3_multi_' + self.data_type + self.data_rotate_flag + '_' + self.valid_classes,
                                        '{}_{}_{}_{:.0e}_{:.0e}'.format(self.n_objs, self.in_size, self.out_size, self.eps, self.eps2))

        self.init_dataset()
        self.init_dataset(test_set=True)

    def generate_sample_indices(self, C, H, W, n_units, n_unit_samplings):
        indices = None
        for i in range(n_unit_samplings):
            if indices is None:
                indices = torch.randperm(C*H*W)[:n_units].unsqueeze(0)
            else:
                indices = torch.cat([indices, torch.randperm(C*H*W)[:n_units].unsqueeze(0)], 0)

        if self.gpu_mode:
            indices = indices.cuda()

        return indices

    def sample_units(self, activations, indices):
        N = activations.size(0)
        if not isinstance(indices, Variable):
            indices = Variable(indices)
        return activations.view(N, -1)[:, indices]

    def flatten(self, activations):
        N = activations.size(0)
        return activations.view(N, -1)

    def init_dataset(self, data_dir=None, batch_size=None, test_batch_size=None, test_set=False):

        if data_dir is None:
            data_dir = self.data_dir

        if batch_size is None:
            batch_size = self.batch_size
        if test_batch_size is None:
            test_batch_size = self.test_batch_size
            
        kwargs = {'num_workers': 8, 'pin_memory': True} if self.gpu_mode else {}
        if not test_set:
            self.train_loader = DataLoader(utils.Gen3MultiObjectDataset(data_dir, data_type='train', requires_ind_objs=True), shuffle=True, batch_size=batch_size, 
                drop_last=True, **kwargs)
            self.val_loader = DataLoader(utils.Gen3MultiObjectDataset(data_dir, data_type='val', requires_ind_objs=True), shuffle=True, batch_size=test_batch_size, 
                drop_last=False, **kwargs)
        else:
            self.test_loader = DataLoader(utils.Gen3MultiObjectDataset(data_dir, data_type='test', requires_ind_objs=True), shuffle=False, batch_size=test_batch_size, 
                drop_last=False, **kwargs)

    def test_accuracy_bar_graphs(self, pose_type, model_type):
        bar_graphs_save_path = os.path.join(self.test_result_dir, self.data_dir, pose_type)
        if not os.path.exists(bar_graphs_save_path):
            os.makedirs(bar_graphs_save_path, exist_ok=True)
        bar_graphs_save_name = 'test_accuracies_{}.eps'.format(model_type)

        def draw_barplot(x_pos, bar_width, means1, errors1, means2, errors2, label1, label2, title, savepath, savename, x_tick_labels, y_label, limit_y_range):
            # Build the plot
            fig, ax = plt.subplots()
            barlist1 = ax.bar(x_pos, means1, bar_width, yerr=errors1, alpha=1.0, color='r', ecolor='black', capsize=2, error_kw={'elinewidth':0.2, 'capthick': 0.2}, label=label1)
            barlist2 = ax.bar(x_pos + bar_width, means2, bar_width, yerr=errors2, alpha=1.0, color='b', ecolor='black', capsize=2, error_kw={'elinewidth':0.2, 'capthick': 0.2}, label=label2)

            if limit_y_range:
                ax.set_ylim([0,100])
            ax.set_ylabel(y_label)
            ax.set_xticks(x_pos+bar_width/2.0)
            ax.set_xticklabels(x_tick_labels, fontsize=5)
            ax.set_title(title)
            ax.legend(fancybox=True, framealpha=0.5)
            # ax.yaxis.grid(True)

            # Save the figure and show
            plt.tight_layout()
            plt.savefig(os.path.join(savepath, savename))
            plt.close()

        model_names = ['NoOcc{}Classifier'.format(model_type), '{}Classifier'.format(model_type)]
        no_occ_acc_means = []
        no_occ_acc_stds = []
        occ_acc_means = []
        occ_acc_stds = []

        for model_name in model_names:
            for no_occ_data_name in ['NoOcc', 'Occ']:
                test_acc_save_path = os.path.join(self.test_result_dir, self.data_dir, pose_type, no_occ_data_name, model_name)
                loaded = np.load(os.path.join(test_acc_save_path, 'test_accs.npy'))*100
                if no_occ_data_name == 'NoOcc':
                    no_occ_acc_means.append(loaded.mean())
                    no_occ_acc_stds.append(loaded.std())
                else:
                    occ_acc_means.append(loaded.mean())
                    occ_acc_stds.append(loaded.std())

        x_pos = 0.1*np.arange(len(model_names))
        bar_width=0.02

        if pose_type == 'gt':
            pose_type_name = 'canonical pose'
        elif pose_type == 'original':
            pose_type_name = 'original pose'

        print('')
        print('Drawing barplots for test accuracies..')
        print('')
        draw_barplot(x_pos, bar_width, no_occ_acc_means, no_occ_acc_stds, occ_acc_means, occ_acc_stds, 'no_occ', 'occ', 
                    'Test accuracies for pose type: {}'.format(pose_type_name), bar_graphs_save_path,
                    bar_graphs_save_name, model_names, 'Test accuracies (%)', True)


    def test_accuracy(self, pose_type, no_occ_data, model_name, n_model_instances):
        if no_occ_data:
            no_occ_data_name = 'NoOcc'
        else:
            no_occ_data_name = 'Occ'
        test_acc_save_path = os.path.join(self.test_result_dir, self.data_dir, pose_type, no_occ_data_name, model_name)
        if not os.path.exists(test_acc_save_path):
            os.makedirs(test_acc_save_path, exist_ok=True)

        test_accs = []
        for model_idx in range(n_model_instances):
            test_accs.append(self.test_accuracy_helper(pose_type, no_occ_data, model_name, model_idx))
        test_accs = np.array(test_accs)
        np.save(os.path.join(test_acc_save_path, 'test_accs.npy'), test_accs)


    def test_accuracy_helper(self, pose_type, no_occ_data, model_name, model_idx):
        if pose_type =='gt':
            input_size = 28
        elif pose_type == 'original':
            input_size = 50

        if model_name in ['FDClassifier', 'NoOccFDClassifier']:
            model = FDClassifier(input_size=input_size)
        elif model_name in ['RCClassifier', 'NoOccRCClassifier']:
            model = RCClassifier(input_size=input_size)
        elif model_name in ['TDRCClassifier', 'NoOccTDRCClassifier']:
            model = TDRCClassifier(input_size=input_size)

        model_save_path = os.path.join(os.path.join(self.classifier_dir, pose_type, model_name), 
                                    'trial_{}'.format(model_idx+1), model_name+'_best_val_acc.pth.tar')

        model.load_state_dict(torch.load(model_save_path, map_location=lambda storage, loc: storage))
        model.eval()
        if self.gpu_mode:
            model.cuda()

        iter_start_time = time.time()
        test_acc = 0
        for iter, (data, ind_objs, ind_ps, labels) in enumerate(self.test_loader):
            N, C, _, _ = data.size()
            data = Variable(data, volatile=True)
            ind_objs = Variable(ind_objs, volatile=True)
            ind_ps = Variable(ind_ps, volatile=True)
            labels = Variable(labels, volatile=True)
            if self.gpu_mode:
                data = data.cuda()
                ind_objs = ind_objs.cuda()
                ind_ps = ind_ps.cuda()
                labels = labels.cuda()

            if pose_type == 'gt':
                aff_ps = self.affine_form(ind_ps[:,1])
                grid = F.affine_grid(aff_ps, torch.Size([N, C, self.in_size, self.in_size]))

            if pose_type == 'gt':                
                if no_occ_data:
                    x = F.grid_sample(ind_objs[:,1], grid)
                else:
                    x = F.grid_sample(data, grid)
            elif pose_type == 'original':
                if no_occ_data:
                    x = ind_objs[:,1]
                else:
                    x = data

            scores = model(x)
            pred = scores.data.max(1)[1]
            test_acc += (pred == labels.data[:,1]).float().mean()
        test_acc /= len(self.test_loader)
        print('pose_type: {} / no_occ_data: {} / model_name: {} / model_idx: {} / test_acc: {:.3f}'.format(pose_type, no_occ_data, model_name, model_idx+1, test_acc))

        return test_acc



    def test_results_FD_distinguishability_readout_layer_bar_graph(self, measure_type, target_model_name, pose_type, total_var_thr=0, include_self_maps=False, image_format='.png'):
        assert target_model_name in ['NoOccFDClassifier', 'FDClassifier']

        bar_graphs_save_path = os.path.join(self.test_result_dir, self.data_dir, pose_type)
        if not os.path.exists(bar_graphs_save_path):
            os.makedirs(bar_graphs_save_path, exist_ok=True)
        bar_graphs_save_name = '{}s_distinguishability_bar_plot_{}_{}_thr{}_readout_layer'.format(measure_type, target_model_name[:-10], pose_type, total_var_thr)

        def draw_readout_distinguishability_barplot(x_pos, self_pccs, self_pccs_stds, diff_pccs, diff_pccs_stds, title, savepath, savename, x_tick_labels, y_label, limit_y_range):

            # Build the plot
            fig, ax = plt.subplots(1)
            barlist1 = ax.bar(x_pos, self_pccs, 0.2, yerr=self_pccs_stds, alpha=0.3, 
                ecolor='black', capsize=2, error_kw={'elinewidth':0.2, 'capthick': 0.2}, label='self pcc')

            barlist2 = ax.bar(x_pos, diff_pccs, 0.2, yerr=diff_pccs_stds, alpha=1,
                ecolor='black', capsize=2, error_kw={'elinewidth':0.2, 'capthick': 0.2}, label='diff pcc')

            barlist1[0].set_color('r')
            barlist1[1].set_color('b')

            barlist2[0].set_color('r')
            barlist2[1].set_color('b')

            ax.set_xlim([-0.5, 1.5])
            if limit_y_range:
                ax.set_ylim([0, 1])
            ax.set_ylabel(y_label)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_tick_labels)
            ax.set_title(title)
            ax.yaxis.grid(True)

            # Save the figure and show
            plt.tight_layout()
            plt.savefig(os.path.join(savepath, savename))
            plt.close()

        def exclude_self_maps(results):
            def flatten_first_two_dims(results):
                if len(results.shape) == 2:
                    results = results.reshape(-1)
                elif len(results.shape) == 3:
                    results = results.reshape(-1, results.shape[2])
                elif len(results.shape) == 4:
                    results = results.reshape(-1, results.shape[2], results.shape[3])

                return results

            n_model_instances = results.shape[0]
            results = flatten_first_two_dims(results)
            masks = np.ones((n_model_instances**2), dtype=bool)
            for i in range(n_model_instances):
                masks[i*n_model_instances + i] = False
            masked_results = results[masks]

            return masked_results


        self_pccs = []
        self_pccs_stds = []
        diff_pccs = []
        diff_pccs_stds = []

        for source_model_name in ['NoOccFDClassifier', 'FDClassifier']:
            for no_occ_data_name in ['NoOcc', 'Occ']:
                test_result_save_path = os.path.join(self.test_result_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name)
                results = np.load(os.path.join(test_result_save_path, 'readout_{}s_thr{:.1e}.npz'.format(measure_type, total_var_thr)))
                result = results['readout']
                if source_model_name == target_model_name and not include_self_maps:
                    result = exclude_self_maps(result)
                if source_model_name == target_model_name:
                    self_pccs.append(result.mean())
                    self_pccs_stds.append(result.std())
                else:
                    diff_pccs.append(result.mean())
                    diff_pccs_stds.append(result.std())

        if pose_type == 'gt':
            pose_type_name = 'canonical pose'
        elif pose_type == 'original':
            pose_type_name = 'original pose'

        print('')
        print('Drawing distinguishability barplots readout layer for pose type: {} target model: {} measure type {}...'.format(pose_type, target_model_name, measure_type))
        print('')
        if 'pcc' in measure_type:
            limit_y_range = True
        else:
            limit_y_range = False
        if 'pcc' in measure_type:
            y_label = 'median pccs'
        else:
            y_label = 'median r2s'

        x_pos = np.arange(2)

        x_tick_labels = ['no occ', 'occ']
        draw_readout_distinguishability_barplot(x_pos, self_pccs, self_pccs_stds, diff_pccs, diff_pccs_stds, 'Distinguishability of target model {} at readout layer: {}'.format(target_model_name[:-10], pose_type_name), 
            bar_graphs_save_path, bar_graphs_save_name + image_format, x_tick_labels, y_label, limit_y_range)



    def test_results_FD_distinguishability_by_layer_bar_graph(self, measure_type, target_model_name, pose_type, total_var_thr=0, include_self_maps=False, image_format='.png', separate=False):
        assert target_model_name in ['NoOccFDClassifier', 'FDClassifier']

        bar_graphs_save_path = os.path.join(self.test_result_dir, self.data_dir, pose_type)
        if not os.path.exists(bar_graphs_save_path):
            os.makedirs(bar_graphs_save_path, exist_ok=True)
        if separate:
            bar_graphs_save_name = '{}s_distinguishability_bar_plot_{}_{}_thr{}_separate'.format(measure_type, target_model_name[:-10], pose_type, total_var_thr)
        else:
            bar_graphs_save_name = '{}s_distinguishability_bar_plot_{}_{}_thr{}'.format(measure_type, target_model_name[:-10], pose_type, total_var_thr)

        def draw_distinguishability_barplot(x_pos, bar_width, no_occ_self_pccs, no_occ_self_pccs_stds, no_occ_diff_pccs, no_occ_diff_pccs_stds, occ_self_pccs, occ_self_pccs_stds, 
            occ_diff_pccs, occ_diff_pccs_stds, label1, label2, title, savepath, savename, x_tick_labels, y_label, limit_y_range, separate):
            if separate:
                fig = plt.figure(figsize=(12.8, 4.8))
                axes = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
                axes[0].bar(x_pos, no_occ_self_pccs, bar_width, yerr=no_occ_self_pccs_stds, alpha=0.3, color='r', 
                    ecolor='black', capsize=2, error_kw={'elinewidth':0.2, 'capthick': 0.2}, label=label1 + ' self pcc')
                axes[0].bar(x_pos, no_occ_diff_pccs, bar_width, yerr=no_occ_diff_pccs_stds, alpha=1.0, color='r', 
                    ecolor='black', capsize=2, error_kw={'elinewidth':0.2, 'capthick': 0.2}, label=label1 + ' diff pcc')
                axes[1].bar(x_pos, occ_self_pccs, bar_width, yerr=occ_self_pccs_stds, alpha=0.3, color='b', 
                    ecolor='black', capsize=2, error_kw={'elinewidth':0.2, 'capthick': 0.2}, label=label2 + ' self pcc')
                axes[1].bar(x_pos, occ_diff_pccs, bar_width, yerr=occ_diff_pccs_stds, alpha=1.0, color='b', 
                    ecolor='black', capsize=2, error_kw={'elinewidth':0.2, 'capthick': 0.2}, label=label2 + ' diff pcc')
                for ax in axes:
                    if limit_y_range:
                        ax.set_ylim([0, 1])
                    ax.set_ylabel(y_label)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(x_tick_labels, fontsize=10)
                    ax.set_title(title)
                    ax.legend(fancybox=True, framealpha=0.5, loc='lower left')
                    ax.yaxis.grid(True)

                # Save the figure and show
                plt.tight_layout()
                plt.savefig(os.path.join(savepath, savename))
                plt.close()
            else:
                # Build the plot
                fig, ax = plt.subplots(1)
                ax.bar(x_pos, no_occ_self_pccs, bar_width, yerr=no_occ_self_pccs_stds, alpha=0.3, color='r', 
                    ecolor='black', capsize=2, error_kw={'elinewidth':0.2, 'capthick': 0.2}, label=label1 + ' self pcc')
                ax.bar(x_pos, no_occ_diff_pccs, bar_width, yerr=no_occ_diff_pccs_stds, alpha=1.0, color='r', 
                    ecolor='black', capsize=2, error_kw={'elinewidth':0.2, 'capthick': 0.2}, label=label1 + ' diff pcc')

                ax.bar(x_pos + bar_width, occ_self_pccs, bar_width, yerr=occ_self_pccs_stds, alpha=0.3, color='b', 
                    ecolor='black', capsize=2, error_kw={'elinewidth':0.2, 'capthick': 0.2}, label=label2 + ' self pcc')
                ax.bar(x_pos + bar_width, occ_diff_pccs, bar_width, yerr=occ_diff_pccs_stds, alpha=1.0, color='b', 
                    ecolor='black', capsize=2, error_kw={'elinewidth':0.2, 'capthick': 0.2}, label=label2 + ' diff pcc')


                if limit_y_range:
                    ax.set_ylim([0, 1])
                ax.set_ylabel(y_label)
                ax.set_xticks(x_pos+bar_width/2.0)
                ax.set_xticklabels(x_tick_labels, fontsize=10)
                ax.set_title(title)
                ax.legend(fancybox=True, framealpha=0.5, loc='lower left')
                ax.yaxis.grid(True)

                # Save the figure and show
                plt.tight_layout()
                plt.savefig(os.path.join(savepath, savename))
                plt.close()

        def exclude_self_maps(results):
            def flatten_first_two_dims(results):
                if len(results.shape) == 2:
                    results = results.reshape(-1)
                elif len(results.shape) == 3:
                    results = results.reshape(-1, results.shape[2])
                elif len(results.shape) == 4:
                    results = results.reshape(-1, results.shape[2], results.shape[3])

                return results

            n_model_instances = results.shape[0]
            results = flatten_first_two_dims(results)
            masks = np.ones((n_model_instances**2), dtype=bool)
            for i in range(n_model_instances):
                masks[i*n_model_instances + i] = False
            masked_results = results[masks]

            return masked_results


        no_occ_self_pccs = []
        no_occ_self_pccs_stds = []
        no_occ_diff_pccs = []
        no_occ_diff_pccs_stds = []
        occ_self_pccs =[]
        occ_self_pccs_stds = []
        occ_diff_pccs = []
        occ_diff_pccs_stds = []


        for source_model_name in ['NoOccFDClassifier', 'FDClassifier']:
            for no_occ_data_name in ['NoOcc', 'Occ']:
                test_result_save_path = os.path.join(self.test_result_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name)
                if no_occ_data_name == 'NoOcc':
                    results = np.load(os.path.join(test_result_save_path, '{}s_thr{:.1e}.npz'.format(measure_type, total_var_thr)))
                    for i in range(4):
                        result = results['hidden{}'.format(i+1)]
                        if source_model_name == target_model_name and not include_self_maps:
                            result = exclude_self_maps(result)
                        if source_model_name == target_model_name:
                            no_occ_self_pccs.append(result.mean())
                            no_occ_self_pccs_stds.append(result.std())
                        else:
                            no_occ_diff_pccs.append(result.mean())
                            no_occ_diff_pccs_stds.append(result.std())

                    results = np.load(os.path.join(test_result_save_path, 'readout_{}s_thr{:.1e}.npz'.format(measure_type, total_var_thr)))
                    result = results['readout']
                    if source_model_name == target_model_name and not include_self_maps:
                        result = exclude_self_maps(result)
                    if source_model_name == target_model_name:
                        no_occ_self_pccs.append(result.mean())
                        no_occ_self_pccs_stds.append(result.std())
                    else:
                        no_occ_diff_pccs.append(result.mean())
                        no_occ_diff_pccs_stds.append(result.std())
                elif no_occ_data_name == 'Occ':
                    results = np.load(os.path.join(test_result_save_path, '{}s_thr{:.1e}.npz'.format(measure_type, total_var_thr)))
                    for i in range(4):
                        result = results['hidden{}'.format(i+1)]
                        if source_model_name == target_model_name and not include_self_maps:
                            result = exclude_self_maps(result)
                        if source_model_name == target_model_name:
                            occ_self_pccs.append(result.mean())
                            occ_self_pccs_stds.append(result.std())
                        else:
                            occ_diff_pccs.append(result.mean())
                            occ_diff_pccs_stds.append(result.std())

                    results = np.load(os.path.join(test_result_save_path, 'readout_{}s_thr{:.1e}.npz'.format(measure_type, total_var_thr)))
                    result = results['readout']
                    if source_model_name == target_model_name and not include_self_maps:
                        result = exclude_self_maps(result)
                    if source_model_name == target_model_name:
                        occ_self_pccs.append(result.mean())
                        occ_self_pccs_stds.append(result.std())
                    else:
                        occ_diff_pccs.append(result.mean())
                        occ_diff_pccs_stds.append(result.std())

        layer_names = ['hidden{}'.format(i+1) for i in range(4)] + ['readout']

        if pose_type == 'gt':
            pose_type_name = 'canonical pose'
        elif pose_type == 'original':
            pose_type_name = 'original pose'

        print('')
        print('Drawing distinguishability barplots for pose type: {} target model: {} measure type {}...'.format(pose_type, target_model_name, measure_type))
        print('')
        if 'pcc' in measure_type:
            limit_y_range = True
        else:
            limit_y_range = False
        if 'pcc' in measure_type:
            y_label = 'median pccs'
        else:
            y_label = 'median r2s'
        if separate:
            x_pos = np.arange(len(layer_names))
            bar_width = 0.3
        else:
            x_pos = 2*np.arange(len(layer_names))
            bar_width=0.3

        draw_distinguishability_barplot(x_pos, bar_width, no_occ_self_pccs, no_occ_self_pccs_stds, no_occ_diff_pccs, no_occ_diff_pccs_stds, occ_self_pccs, occ_self_pccs_stds, 
        occ_diff_pccs, occ_diff_pccs_stds, 'no_occ', 'occ', 'Distinguishability of target model {}: {}'.format(target_model_name[:-10], pose_type_name), bar_graphs_save_path,
                    bar_graphs_save_name + image_format, layer_names, y_label, limit_y_range, separate)




    def test_results_RC_distinguishability_time_course_graph(self, measure_type, target_model_name, pose_type, readout, total_var_thr=0, include_self_maps=False, image_format='.png'):
        assert target_model_name in ['NoOccRCClassifier', 'RCClassifier']

        graphs_save_path = os.path.join(self.test_result_dir, self.data_dir, pose_type)
        if not os.path.exists(graphs_save_path):
            os.makedirs(graphs_save_path, exist_ok=True)
        graphs_save_name = '{}s_distinguishability_time_course_{}_{}_thr{}'.format(measure_type, target_model_name[:-10], pose_type, total_var_thr)


        def draw_distinguishability_time_course_plot(no_occ_self_pccs, no_occ_self_pccs_stds, no_occ_diff_pccs, no_occ_diff_pccs_stds, occ_self_pccs, occ_self_pccs_stds, 
            occ_diff_pccs, occ_diff_pccs_stds, label1, label2, title, savepath, savename, y_label, limit_y_range):
            no_occ_self_pccs = np.array(no_occ_self_pccs)
            no_occ_diff_pccs = np.array(no_occ_diff_pccs)
            occ_self_pccs = np.array(occ_self_pccs)
            occ_diff_pccs = np.array(occ_diff_pccs)

            fig, ax = plt.subplots(1)
            timesteps = np.arange(1,len(no_occ_self_pccs)+1)

            if limit_y_range:
                ax.set_ylim([0, 1])
            ax.plot(timesteps, no_occ_self_pccs, label=label1 + ' self pcc', color='r', ls='--')
            ax.plot(timesteps, no_occ_diff_pccs, label=label1 + ' diff pcc', color='r')
            ax.plot(timesteps, occ_self_pccs, label=label2 + ' self pcc', color='b', ls='--')
            ax.plot(timesteps, occ_diff_pccs, label=label2 + ' diff pcc', color='b')

            ax.fill_between(timesteps, no_occ_diff_pccs-no_occ_diff_pccs_stds, no_occ_diff_pccs+no_occ_diff_pccs_stds, facecolor='r', alpha=0.5)            
            ax.fill_between(timesteps, occ_diff_pccs-occ_diff_pccs_stds, occ_diff_pccs+occ_diff_pccs_stds, facecolor='b', alpha=0.5)            
            ax.fill_between(timesteps, no_occ_self_pccs-no_occ_self_pccs_stds, no_occ_self_pccs+no_occ_self_pccs_stds, facecolor='r', alpha=0.1)            
            ax.fill_between(timesteps, occ_self_pccs-occ_self_pccs_stds, occ_self_pccs+occ_self_pccs_stds, facecolor='b', alpha=0.1)            


            ax.set_xlabel('timesteps')
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.legend(fancybox=True, framealpha=0.5, loc='lower left')
            ax.yaxis.grid(True)

            # Save the figure and show
            plt.tight_layout()
            plt.savefig(os.path.join(savepath, savename))
            plt.close()

        def exclude_self_maps(results):
            def flatten_first_two_dims(results):
                if len(results.shape) == 2:
                    results = results.reshape(-1)
                elif len(results.shape) == 3:
                    results = results.reshape(-1, results.shape[2])
                elif len(results.shape) == 4:
                    results = results.reshape(-1, results.shape[2], results.shape[3])

                return results

            n_model_instances = results.shape[0]
            results = flatten_first_two_dims(results)
            masks = np.ones((n_model_instances**2), dtype=bool)
            for i in range(n_model_instances):
                masks[i*n_model_instances + i] = False
            masked_results = results[masks]

            return masked_results

        no_occ_self_pccs = {}
        no_occ_self_pccs_stds = {}
        no_occ_diff_pccs = {}
        no_occ_diff_pccs_stds = {}
        occ_self_pccs ={}
        occ_self_pccs_stds = {}
        occ_diff_pccs = {}
        occ_diff_pccs_stds = {}

        if readout:
            no_occ_self_pccs['readout'] = None
            no_occ_self_pccs_stds['readout'] = None
            no_occ_diff_pccs['readout'] = None
            no_occ_diff_pccs_stds['readout'] = None
            occ_self_pccs['readout'] = None
            occ_self_pccs_stds['readout'] = None
            occ_diff_pccs['readout'] = None
            occ_diff_pccs_stds['readout'] = None
        else:            
            for i in range(4):
                no_occ_self_pccs['hidden{}'.format(i+1)] = None
                no_occ_self_pccs_stds['hidden{}'.format(i+1)] = None
                no_occ_diff_pccs['hidden{}'.format(i+1)] = None
                no_occ_diff_pccs_stds['hidden{}'.format(i+1)] = None
                occ_self_pccs['hidden{}'.format(i+1)] = None
                occ_self_pccs_stds['hidden{}'.format(i+1)] = None
                occ_diff_pccs['hidden{}'.format(i+1)] = None
                occ_diff_pccs_stds['hidden{}'.format(i+1)] = None


        for source_model_name in ['NoOccRCClassifier', 'RCClassifier']:
            for no_occ_data_name in ['NoOcc', 'Occ']:
                test_result_save_path = os.path.join(self.test_result_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name)
                if readout:
                    results = np.load(os.path.join(test_result_save_path, 'readout_{}s_thr{:.1e}.npz'.format(measure_type, total_var_thr)))                            
                else:
                    results = np.load(os.path.join(test_result_save_path, '{}s_thr{:.1e}.npz'.format(measure_type, total_var_thr)))

                if no_occ_data_name == 'NoOcc':
                    if readout:
                        result = results['readout']
                        if source_model_name == target_model_name and not include_self_maps:
                            result = exclude_self_maps(result)
                        n_iter = result.shape[-1]
                        result = result.reshape(-1, n_iter)
                        if source_model_name == target_model_name:
                            no_occ_self_pccs['readout'] = result.mean(0)
                            no_occ_self_pccs_stds['readout'] = result.std(0)
                        else:
                            no_occ_diff_pccs['readout'] = result.mean(0)
                            no_occ_diff_pccs_stds['readout'] = result.std(0)                            
                    else:
                        for i in range(4):
                            result = results['hidden{}'.format(i+1)]
                            if source_model_name == target_model_name and not include_self_maps:
                                result = exclude_self_maps(result)
                            n_iter = result.shape[-1]
                            result = result.reshape(-1, n_iter)
                            if source_model_name == target_model_name:
                                no_occ_self_pccs['hidden{}'.format(i+1)] = result.mean(0)
                                no_occ_self_pccs_stds['hidden{}'.format(i+1)] = result.std(0)
                            else:
                                no_occ_diff_pccs['hidden{}'.format(i+1)] = result.mean(0)
                                no_occ_diff_pccs_stds['hidden{}'.format(i+1)] = result.std(0)
                elif no_occ_data_name == 'Occ':
                    if readout:
                        result = results['readout']
                        if source_model_name == target_model_name and not include_self_maps:
                            result = exclude_self_maps(result)
                        n_iter = result.shape[-1]
                        result = result.reshape(-1, n_iter)
                        if source_model_name == target_model_name:
                            occ_self_pccs['readout'] = result.mean(0)
                            occ_self_pccs_stds['readout'] =result.std(0)
                        else:
                            occ_diff_pccs['readout'] = result.mean(0)
                            occ_diff_pccs_stds['readout'] = result.std(0)
                    else:
                        for i in range(4):
                            result = results['hidden{}'.format(i+1)]
                            if source_model_name == target_model_name and not include_self_maps:
                                result = exclude_self_maps(result)
                            n_iter = result.shape[-1]
                            result = result.reshape(-1, n_iter)
                            if source_model_name == target_model_name:
                                occ_self_pccs['hidden{}'.format(i+1)] = result.mean(0)
                                occ_self_pccs_stds['hidden{}'.format(i+1)] = result.std(0)
                            else:
                                occ_diff_pccs['hidden{}'.format(i+1)] = result.mean(0)
                                occ_diff_pccs_stds['hidden{}'.format(i+1)] = result.std(0)


        if pose_type == 'gt':
            pose_type_name = 'canonical pose'
        elif pose_type == 'original':
            pose_type_name = 'original pose'

        print('')
        print('Drawing distinguishability time course plot for pose type: {} target model: {} measure_type: {}...'.format(pose_type, target_model_name, measure_type))
        print('')
        if 'pcc' in measure_type:
            limit_y_range = True
        else:
            limit_y_range = False
        if 'pcc' in measure_type:
            y_label = 'median pccs'
        else:
            y_label = 'median r2s'
        if readout:
            draw_distinguishability_time_course_plot(no_occ_self_pccs['readout'], no_occ_self_pccs_stds['readout'], no_occ_diff_pccs['readout'], no_occ_diff_pccs_stds['readout'], 
                occ_self_pccs['readout'], occ_self_pccs_stds['readout'], occ_diff_pccs['readout'], occ_diff_pccs_stds['readout'], 'no_occ', 'occ', 
                        'Distinguishability time course of target model {} in readout: {}'.format(target_model_name[:-10], pose_type_name), 
                        graphs_save_path, graphs_save_name + '_readout' + image_format, y_label, limit_y_range)
        else:
            for i in range(4):
                draw_distinguishability_time_course_plot(no_occ_self_pccs['hidden{}'.format(i+1)], no_occ_self_pccs_stds['hidden{}'.format(i+1)], no_occ_diff_pccs['hidden{}'.format(i+1)], no_occ_diff_pccs_stds['hidden{}'.format(i+1)], 
                    occ_self_pccs['hidden{}'.format(i+1)], occ_self_pccs_stds['hidden{}'.format(i+1)], occ_diff_pccs['hidden{}'.format(i+1)], occ_diff_pccs_stds['hidden{}'.format(i+1)], 'no_occ', 'occ', 
                            'Distinguishability time course of target model {} in layer {}: {}'.format(target_model_name[:-10], i+1, pose_type_name), 
                            graphs_save_path, graphs_save_name + '_hidden{}'.format(i+1) + image_format , y_label, limit_y_range)




    def test_results_TDRC_distinguishability_time_course_graph_all_layers(self, measure_type, target_model_name, pose_type, total_var_thr=0, include_self_maps=False, image_format='.png'):
        assert target_model_name in ['NoOccTDRCClassifier', 'TDRCClassifier']

        graphs_save_path = os.path.join(self.test_result_dir, self.data_dir, pose_type)
        if not os.path.exists(graphs_save_path):
            os.makedirs(graphs_save_path, exist_ok=True)
        graphs_save_name = '{}s_distinguishability_time_course_{}_{}_thr{}'.format(measure_type, target_model_name[:-10], pose_type, total_var_thr)


        def draw_distinguishability_time_course_plot(no_occ_self_pccs, no_occ_self_pccs_stds, no_occ_diff_pccs, no_occ_diff_pccs_stds, occ_self_pccs, occ_self_pccs_stds, 
            occ_diff_pccs, occ_diff_pccs_stds, label1, label2, title, savepath, savename, y_label, limit_y_range):
            no_occ_self_pccs = np.array(no_occ_self_pccs)
            no_occ_diff_pccs = np.array(no_occ_diff_pccs)
            occ_self_pccs = np.array(occ_self_pccs)
            occ_diff_pccs = np.array(occ_diff_pccs)

            fig, ax = plt.subplots(1)
            timesteps = np.arange(1,len(no_occ_self_pccs)+1)

            if limit_y_range:
                ax.set_ylim([0, 1])
            ax.plot(timesteps, no_occ_self_pccs, label=label1 + ' self pcc', color='r', ls='--')
            ax.plot(timesteps, no_occ_diff_pccs, label=label1 + ' diff pcc', color='r')
            ax.plot(timesteps, occ_self_pccs, label=label2 + ' self pcc', color='b', ls='--')
            ax.plot(timesteps, occ_diff_pccs, label=label2 + ' diff pcc', color='b')

            ax.fill_between(timesteps, no_occ_diff_pccs-no_occ_diff_pccs_stds, no_occ_diff_pccs+no_occ_diff_pccs_stds, facecolor='r', alpha=0.5)            
            ax.fill_between(timesteps, occ_diff_pccs-occ_diff_pccs_stds, occ_diff_pccs+occ_diff_pccs_stds, facecolor='b', alpha=0.5)            
            ax.fill_between(timesteps, no_occ_self_pccs-no_occ_self_pccs_stds, no_occ_self_pccs+no_occ_self_pccs_stds, facecolor='r', alpha=0.1)            
            ax.fill_between(timesteps, occ_self_pccs-occ_self_pccs_stds, occ_self_pccs+occ_self_pccs_stds, facecolor='b', alpha=0.1)            


            ax.set_xlabel('timesteps')
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.legend(fancybox=True, framealpha=0.5, loc='lower left')
            ax.yaxis.grid(True)

            # Save the figure and show
            plt.tight_layout()
            plt.savefig(os.path.join(savepath, savename))
            plt.close()

        def exclude_self_maps(results):
            def flatten_first_two_dims(results):
                if len(results.shape) == 2:
                    results = results.reshape(-1)
                elif len(results.shape) == 3:
                    results = results.reshape(-1, results.shape[2])
                elif len(results.shape) == 4:
                    results = results.reshape(-1, results.shape[2], results.shape[3])

                return results

            n_model_instances = results.shape[0]
            results = flatten_first_two_dims(results)
            masks = np.ones((n_model_instances**2), dtype=bool)
            for i in range(n_model_instances):
                masks[i*n_model_instances + i] = False
            masked_results = results[masks]

            return masked_results

        no_occ_self_pccs = {}
        no_occ_self_pccs_stds = {}
        no_occ_diff_pccs = {}
        no_occ_diff_pccs_stds = {}
        occ_self_pccs ={}
        occ_self_pccs_stds = {}
        occ_diff_pccs = {}
        occ_diff_pccs_stds = {}

        no_occ_self_pccs['readout'] = None
        no_occ_self_pccs_stds['readout'] = None
        no_occ_diff_pccs['readout'] = None
        no_occ_diff_pccs_stds['readout'] = None
        occ_self_pccs['readout'] = None
        occ_self_pccs_stds['readout'] = None
        occ_diff_pccs['readout'] = None
        occ_diff_pccs_stds['readout'] = None

        for i in range(4):
            no_occ_self_pccs['hidden{}'.format(i+1)] = None
            no_occ_self_pccs_stds['hidden{}'.format(i+1)] = None
            no_occ_diff_pccs['hidden{}'.format(i+1)] = None
            no_occ_diff_pccs_stds['hidden{}'.format(i+1)] = None
            occ_self_pccs['hidden{}'.format(i+1)] = None
            occ_self_pccs_stds['hidden{}'.format(i+1)] = None
            occ_diff_pccs['hidden{}'.format(i+1)] = None
            occ_diff_pccs_stds['hidden{}'.format(i+1)] = None


        for source_model_name in ['NoOccTDRCClassifier', 'TDRCClassifier']:
            for no_occ_data_name in ['NoOcc', 'Occ']:
                test_result_save_path = os.path.join(self.test_result_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name)
                results = np.load(os.path.join(test_result_save_path, '{}s_thr{:.1e}.npz'.format(measure_type, total_var_thr)))

                if no_occ_data_name == 'NoOcc':

                    result = results['readout']
                    if source_model_name == target_model_name and not include_self_maps:
                        result = exclude_self_maps(result)
                    n_iter = result.shape[-1]
                    result = result.reshape(-1, n_iter)
                    if source_model_name == target_model_name:
                        no_occ_self_pccs['readout'] = result.mean(0)
                        no_occ_self_pccs_stds['readout'] = result.std(0)
                    else:
                        no_occ_diff_pccs['readout'] = result.mean(0)
                        no_occ_diff_pccs_stds['readout'] = result.std(0)                            

                    for i in range(4):
                        result = results['hidden{}'.format(i+1)]
                        if source_model_name == target_model_name and not include_self_maps:
                            result = exclude_self_maps(result)
                        n_iter = result.shape[-1]
                        result = result.reshape(-1, n_iter)
                        if source_model_name == target_model_name:
                            no_occ_self_pccs['hidden{}'.format(i+1)] = result.mean(0)
                            no_occ_self_pccs_stds['hidden{}'.format(i+1)] = result.std(0)
                        else:
                            no_occ_diff_pccs['hidden{}'.format(i+1)] = result.mean(0)
                            no_occ_diff_pccs_stds['hidden{}'.format(i+1)] = result.std(0)

                elif no_occ_data_name == 'Occ':

                    result = results['readout']
                    if source_model_name == target_model_name and not include_self_maps:
                        result = exclude_self_maps(result)
                    n_iter = result.shape[-1]
                    result = result.reshape(-1, n_iter)
                    if source_model_name == target_model_name:
                        occ_self_pccs['readout'] = result.mean(0)
                        occ_self_pccs_stds['readout'] =result.std(0)
                    else:
                        occ_diff_pccs['readout'] = result.mean(0)
                        occ_diff_pccs_stds['readout'] = result.std(0)

                    for i in range(4):
                        result = results['hidden{}'.format(i+1)]
                        if source_model_name == target_model_name and not include_self_maps:
                            result = exclude_self_maps(result)
                        n_iter = result.shape[-1]
                        result = result.reshape(-1, n_iter)
                        if source_model_name == target_model_name:
                            occ_self_pccs['hidden{}'.format(i+1)] = result.mean(0)
                            occ_self_pccs_stds['hidden{}'.format(i+1)] = result.std(0)
                        else:
                            occ_diff_pccs['hidden{}'.format(i+1)] = result.mean(0)
                            occ_diff_pccs_stds['hidden{}'.format(i+1)] = result.std(0)


        if pose_type == 'gt':
            pose_type_name = 'canonical pose'
        elif pose_type == 'original':
            pose_type_name = 'original pose'

        print('')
        print('Drawing distinguishability time course plot for pose type: {} target model: {} measure_type: {}...'.format(pose_type, target_model_name, measure_type))
        print('')
        if 'pcc' in measure_type:
            limit_y_range = True
        else:
            limit_y_range = False
        if 'pcc' in measure_type:
            y_label = 'median pccs'
        else:
            y_label = 'median r2s'

        draw_distinguishability_time_course_plot(no_occ_self_pccs['readout'], no_occ_self_pccs_stds['readout'], no_occ_diff_pccs['readout'], no_occ_diff_pccs_stds['readout'], 
            occ_self_pccs['readout'], occ_self_pccs_stds['readout'], occ_diff_pccs['readout'], occ_diff_pccs_stds['readout'], 'no_occ', 'occ', 
                    'Distinguishability time course of target model {} in readout: {}'.format(target_model_name[:-10], pose_type_name), 
                    graphs_save_path, graphs_save_name + '_readout' + image_format, y_label, limit_y_range)

        for i in range(4):
            draw_distinguishability_time_course_plot(no_occ_self_pccs['hidden{}'.format(i+1)], no_occ_self_pccs_stds['hidden{}'.format(i+1)], no_occ_diff_pccs['hidden{}'.format(i+1)], no_occ_diff_pccs_stds['hidden{}'.format(i+1)], 
                occ_self_pccs['hidden{}'.format(i+1)], occ_self_pccs_stds['hidden{}'.format(i+1)], occ_diff_pccs['hidden{}'.format(i+1)], occ_diff_pccs_stds['hidden{}'.format(i+1)], 'no_occ', 'occ', 
                        'Distinguishability time course of target model {} in layer {}: {}'.format(target_model_name[:-10], i+1, pose_type_name), 
                        graphs_save_path, graphs_save_name + '_hidden{}'.format(i+1) + image_format , y_label, limit_y_range)




    def test_results_RC_time_course_graph(self, measure_type, source_model_name, target_model_name, pose_type, readout, total_var_thr=0, include_self_maps=False, image_format='.png'):
        assert source_model_name in ['NoOccRCClassifier', 'RCClassifier']
        assert target_model_name in ['NoOccRCClassifier', 'RCClassifier']

        graphs_save_path = os.path.join(self.test_result_dir, self.data_dir, pose_type)
        if not os.path.exists(graphs_save_path):
            os.makedirs(graphs_save_path, exist_ok=True)
        graphs_save_name = '{}s_time_course_{}_{}_{}_thr{}'.format(measure_type, source_model_name[:-10], target_model_name[:-10], pose_type, total_var_thr)

        def draw_time_course_plot(means1, errors1, means2, errors2, label1, label2, title, savepath, savename, y_label, limit_y_range):
            means1 = np.array(means1)
            errors1 = np.array(errors1)
            means2 = np.array(means2)
            errors2 = np.array(errors2)

            fig, ax = plt.subplots(1)
            timesteps = np.arange(1,len(means1)+1)

            if limit_y_range:
                ax.set_ylim([0, 1])
            ax.plot(timesteps, means1, label=label1, color='r')
            ax.plot(timesteps, means2, label=label2, color='b')

            ax.fill_between(timesteps, means1-errors1, means1+errors1, facecolor='r', alpha=0.5)            
            ax.fill_between(timesteps, means2-errors2, means2+errors2, facecolor='b', alpha=0.5)            

            ax.set_xlabel('timesteps')
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.legend(fancybox=True, framealpha=0.5)
            ax.yaxis.grid(True)

            # Save the figure and show
            plt.tight_layout()
            plt.savefig(os.path.join(savepath, savename))
            plt.close()

        def exclude_self_maps(results):
            def flatten_first_two_dims(results):
                if len(results.shape) == 2:
                    results = results.reshape(-1)
                elif len(results.shape) == 3:
                    results = results.reshape(-1, results.shape[2])
                elif len(results.shape) == 4:
                    results = results.reshape(-1, results.shape[2], results.shape[3])

                return results

            n_model_instances = results.shape[0]
            results = flatten_first_two_dims(results)
            masks = np.ones((n_model_instances**2), dtype=bool)
            for i in range(n_model_instances):
                masks[i*n_model_instances + i] = False
            masked_results = results[masks]

            return masked_results

        no_occ_means = {}
        no_occ_stds = {}
        occ_means ={}
        occ_stds = {}
        if readout:
            no_occ_means['readout'] = None
            no_occ_stds['readout'] = None
            occ_means['readout'] = None
            occ_stds['readout'] = None
        else:            
            for i in range(4):
                no_occ_means['hidden{}'.format(i+1)] = None
                no_occ_stds['hidden{}'.format(i+1)] = None
                occ_means['hidden{}'.format(i+1)] = None
                occ_stds['hidden{}'.format(i+1)] = None

        for no_occ_data_name in ['NoOcc', 'Occ']:
            test_result_save_path = os.path.join(self.test_result_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name)
            if measure_type in ['median_r2', 'median_pcc']:
                if readout:
                    results = np.load(os.path.join(test_result_save_path, 'readout_{}s_thr{:.1e}.npz'.format(measure_type, total_var_thr)))                            
                else:
                    results = np.load(os.path.join(test_result_save_path, '{}s_thr{:.1e}.npz'.format(measure_type, total_var_thr)))
            else:
                if readout:
                    results = np.load(os.path.join(test_result_save_path, 'readout_{}s.npz'.format(measure_type)))
                else:
                    results = np.load(os.path.join(test_result_save_path, '{}s.npz'.format(measure_type)))

            if no_occ_data_name == 'NoOcc':
                if readout:
                    result = results['readout']
                    if source_model_name == target_model_name and not include_self_maps:
                        result = exclude_self_maps(result)
                    n_iter = result.shape[-1]
                    result = result.reshape(-1, n_iter)
                    no_occ_means['readout'] = result.mean(0)
                    no_occ_stds['readout'] = result.std(0)
                else:
                    for i in range(4):
                        result = results['hidden{}'.format(i+1)]
                        if source_model_name == target_model_name and not include_self_maps:
                            result = exclude_self_maps(result)
                        n_iter = result.shape[-1]
                        result = result.reshape(-1, n_iter)
                        no_occ_means['hidden{}'.format(i+1)] = result.mean(0)
                        no_occ_stds['hidden{}'.format(i+1)] = result.std(0)
            elif no_occ_data_name == 'Occ':
                if readout:
                    result = results['readout']
                    if source_model_name == target_model_name and not include_self_maps:
                        result = exclude_self_maps(result)
                    n_iter = result.shape[-1]
                    result = result.reshape(-1, n_iter)
                    occ_means['readout'] = result.mean(0)
                    occ_stds['readout'] = result.std(0)
                else:
                    for i in range(4):
                        result = results['hidden{}'.format(i+1)]
                        if source_model_name == target_model_name and not include_self_maps:
                            result = exclude_self_maps(result)
                        n_iter = result.shape[-1]
                        result = result.reshape(-1, n_iter)
                        occ_means['hidden{}'.format(i+1)] = result.mean(0)
                        occ_stds['hidden{}'.format(i+1)] = result.std(0)


        if pose_type == 'gt':
            pose_type_name = 'canonical pose'
        elif pose_type == 'original':
            pose_type_name = 'original pose'

        print('')
        print('Drawing time course plot for pose type: {} source model: {} target model: {} measure type {}...'.format(pose_type, source_model_name, target_model_name, measure_type))
        print('')
        if 'pcc' in measure_type:
            limit_y_range = True
        else:
            limit_y_range = False
        if readout:
            draw_time_course_plot(no_occ_means['readout'], no_occ_stds['readout'], occ_means['readout'], occ_stds['readout'], 'no_occ', 'occ', 
                        'Time course of {} values of linear regressions between {} and {} in readout: {}'.format(measure_type, source_model_name[:-10], target_model_name[:-10], pose_type_name), 
                        graphs_save_path, graphs_save_name + '_readout' + image_format, measure_type, limit_y_range)
        else:
            for i in range(4):
                draw_time_course_plot(no_occ_means['hidden{}'.format(i+1)], no_occ_stds['hidden{}'.format(i+1)], occ_means['hidden{}'.format(i+1)], occ_stds['hidden{}'.format(i+1)], 'no_occ', 'occ', 
                            'Time course of {} values of linear regressions between {} and {} in layer {}: {}'.format(measure_type, source_model_name[:-10], target_model_name[:-10], i+1, pose_type_name), 
                            graphs_save_path, graphs_save_name + '_hidden{}'.format(i+1) + image_format , measure_type, limit_y_range)




    def test_results_bar_graphs(self, measure_type, pose_type, model_type, readout, total_var_thr=0, include_self_maps=False):
        bar_graphs_save_path = os.path.join(self.test_result_dir, self.data_dir, pose_type)
        if not os.path.exists(bar_graphs_save_path):
            os.makedirs(bar_graphs_save_path, exist_ok=True)
        bar_graphs_save_name = '{}s_bar_plot_{}_{}_thr{}'.format(measure_type, model_type, pose_type, total_var_thr)

        def draw_barplot(x_pos, bar_width, means1, errors1, means2, errors2, label1, label2, title, savepath, savename, x_tick_labels, y_label, limit_y_range):
            # Build the plot
            fig, ax = plt.subplots()
            barlist1 = ax.bar(x_pos, means1, bar_width, yerr=errors1, alpha=1.0, color='r', ecolor='black', capsize=2, error_kw={'elinewidth':0.2, 'capthick': 0.2}, label=label1)
            barlist2 = ax.bar(x_pos + bar_width, means2, bar_width, yerr=errors2, alpha=1.0, color='b', ecolor='black', capsize=2, error_kw={'elinewidth':0.2, 'capthick': 0.2}, label=label2)

            if limit_y_range:
                ax.set_ylim([0, 1])
            ax.set_ylabel(y_label)
            ax.set_xticks(x_pos+bar_width/2.0)
            ax.set_xticklabels(x_tick_labels, fontsize=5)
            ax.set_title(title)
            ax.legend(fancybox=True, framealpha=0.5)
            ax.yaxis.grid(True)

            # Save the figure and show
            plt.tight_layout()
            plt.savefig(os.path.join(savepath, savename))
            plt.close()

        def exclude_self_maps(results):
            def flatten_first_two_dims(results):
                if len(results.shape) == 2:
                    results = results.reshape(-1)
                elif len(results.shape) == 3:
                    results = results.reshape(-1, results.shape[2])
                elif len(results.shape) == 4:
                    results = results.reshape(-1, results.shape[2], results.shape[3])

                return results

            n_model_instances = results.shape[0]
            results = flatten_first_two_dims(results)
            masks = np.ones((n_model_instances**2), dtype=bool)
            for i in range(n_model_instances):
                masks[i*n_model_instances + i] = False
            masked_results = results[masks]

            return masked_results


        source_target_model_names = []
        no_occ_means = {}
        no_occ_stds = {}
        occ_means ={}
        occ_stds = {}
        if readout:
            no_occ_means['readout'] = []
            no_occ_stds['readout'] = []
            occ_means['readout'] = []
            occ_stds['readout'] = []
        else:            
            for i in range(4):
                no_occ_means['hidden{}'.format(i+1)] = []
                no_occ_stds['hidden{}'.format(i+1)] = []
                occ_means['hidden{}'.format(i+1)] = []
                occ_stds['hidden{}'.format(i+1)] = []

        for source_model_name in ['{}Classifier'.format(model_type), 'NoOcc{}Classifier'.format(model_type)]:
            for target_model_name in ['{}Classifier'.format(model_type), 'NoOcc{}Classifier'.format(model_type)]:
                source_target_model_names.append(source_model_name[:-10] + ' > ' + target_model_name[:-10])
                for no_occ_data_name in ['NoOcc', 'Occ']:
                    test_result_save_path = os.path.join(self.test_result_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name)
                    if measure_type in ['median_r2', 'median_pcc']:
                        if readout:
                            results = np.load(os.path.join(test_result_save_path, 'readout_{}s_thr{:.1e}.npz'.format(measure_type, total_var_thr)))                            
                        else:
                            results = np.load(os.path.join(test_result_save_path, '{}s_thr{:.1e}.npz'.format(measure_type, total_var_thr)))
                    else:
                        if readout:
                            results = np.load(os.path.join(test_result_save_path, 'readout_{}s.npz'.format(measure_type)))
                        else:
                            results = np.load(os.path.join(test_result_save_path, '{}s.npz'.format(measure_type)))

                    if no_occ_data_name == 'NoOcc':
                        if readout:
                            result = results['readout']
                            if source_model_name == target_model_name and not include_self_maps:
                                result = exclude_self_maps(result)
                            no_occ_means['readout'].append(result.mean())
                            no_occ_stds['readout'].append(result.std())                            
                        else:
                            for i in range(4):
                                result = results['hidden{}'.format(i+1)]
                                if source_model_name == target_model_name and not include_self_maps:
                                    result = exclude_self_maps(result)
                                no_occ_means['hidden{}'.format(i+1)].append(result.mean())
                                no_occ_stds['hidden{}'.format(i+1)].append(result.std())
                    elif no_occ_data_name == 'Occ':
                        if readout:
                            result = results['readout']
                            if source_model_name == target_model_name and not include_self_maps:
                                result = exclude_self_maps(result)
                            occ_means['readout'].append(result.mean())
                            occ_stds['readout'].append(result.std())
                        else:
                            for i in range(4):
                                result = results['hidden{}'.format(i+1)]
                                if source_model_name == target_model_name and not include_self_maps:
                                    result = exclude_self_maps(result)
                                occ_means['hidden{}'.format(i+1)].append(result.mean())
                                occ_stds['hidden{}'.format(i+1)].append(result.std())


        x_pos = 2*np.arange(len(source_target_model_names))
        bar_width=0.3

        if pose_type == 'gt':
            pose_type_name = 'canonical pose'
        elif pose_type == 'original':
            pose_type_name = 'original pose'

        print('')
        print('Drawing barplots for pose type: {} model type: {} measure type {}...'.format(pose_type, model_type, measure_type))
        print('')
        if 'pcc' in measure_type:
            limit_y_range = True
        else:
            limit_y_range = False
        if readout:
            draw_barplot(x_pos, bar_width, no_occ_means['readout'], no_occ_stds['readout'], occ_means['readout'], occ_stds['readout'], 'no_occ', 'occ', 
                        '{} values of linear regressions between different models in readout: {}'.format(measure_type, pose_type_name), bar_graphs_save_path,
                        bar_graphs_save_name + '_readout.eps', source_target_model_names, measure_type, limit_y_range)
        else:
            for i in range(4):
                draw_barplot(x_pos, bar_width, no_occ_means['hidden{}'.format(i+1)], no_occ_stds['hidden{}'.format(i+1)], occ_means['hidden{}'.format(i+1)], occ_stds['hidden{}'.format(i+1)], 'no_occ', 'occ', 
                            '{} values of linear regressions between different models in layer {}: {}'.format(measure_type, i+1, pose_type_name), bar_graphs_save_path,
                            bar_graphs_save_name + '_hidden{}.eps'.format(i+1), source_target_model_names, measure_type, limit_y_range)



    def test_linear_regressor_aggregate_over_model_instances_all_layers(self, pose_type, no_occ_data, source_model_name, target_model_name, n_model_instances=5, n_units=256, n_unit_samplings=3, n_iter=5, total_var_thr=0):
        if no_occ_data:
            no_occ_data_name = 'NoOcc'
        else:
            no_occ_data_name = 'Occ'
        test_result_save_path = os.path.join(self.test_result_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name)
        if not os.path.exists(test_result_save_path):
            os.makedirs(test_result_save_path, exist_ok=True)

        mses = {}
        if 'FD' in source_model_name:
            mses['readout'] = np.zeros((n_model_instances, n_model_instances))
        elif 'RC' in source_model_name:
            mses['readout'] = np.zeros((n_model_instances, n_model_instances, n_iter))

        for i in range(4):
            if 'FD' in source_model_name:
                if (i+1) == 4:
                    mses['hidden4'] = np.zeros((n_model_instances, n_model_instances))
                else:
                    mses['hidden{}'.format(i+1)] = np.zeros((n_model_instances, n_model_instances, n_unit_samplings))
            elif 'RC' in source_model_name:
                if (i+1) == 4:
                    mses['hidden4'] = np.zeros((n_model_instances, n_model_instances, n_iter))
                else:
                    mses['hidden{}'.format(i+1)] = np.zeros((n_model_instances, n_model_instances, n_unit_samplings, n_iter))

        median_r2s = {}
        if 'FD' in source_model_name:
            median_r2s['readout'] = np.zeros((n_model_instances, n_model_instances))
        elif 'RC' in source_model_name:
            median_r2s['readout'] = np.zeros((n_model_instances, n_model_instances, n_iter))

        for i in range(4):
            if 'FD' in source_model_name:
                if (i+1) == 4:
                    median_r2s['hidden4'] = np.zeros((n_model_instances, n_model_instances))
                else:
                    median_r2s['hidden{}'.format(i+1)] = np.zeros((n_model_instances, n_model_instances, n_unit_samplings))
            elif 'RC' in source_model_name:
                if (i+1) == 4:
                    median_r2s['hidden4'] = np.zeros((n_model_instances, n_model_instances, n_iter))
                else:
                    median_r2s['hidden{}'.format(i+1)] = np.zeros((n_model_instances, n_model_instances, n_unit_samplings, n_iter))

        median_pccs = {}
        if 'FD' in source_model_name:
            median_pccs['readout'] = np.zeros((n_model_instances, n_model_instances))
        elif 'RC' in source_model_name:
            median_pccs['readout'] = np.zeros((n_model_instances, n_model_instances, n_iter))

        for i in range(4):
            if 'FD' in source_model_name:
                if (i+1) == 4:
                    median_pccs['hidden4'] = np.zeros((n_model_instances, n_model_instances))
                else:
                    median_pccs['hidden{}'.format(i+1)] = np.zeros((n_model_instances, n_model_instances, n_unit_samplings))
            elif 'RC' in source_model_name:
                if (i+1) == 4:
                    median_pccs['hidden4'] = np.zeros((n_model_instances, n_model_instances, n_iter))
                else:
                    median_pccs['hidden{}'.format(i+1)] = np.zeros((n_model_instances, n_model_instances, n_unit_samplings, n_iter))

        live_unit_fracs = {}
        if 'FD' in source_model_name:
            live_unit_fracs['readout'] = np.zeros((n_model_instances, n_model_instances))
        elif 'RC' in source_model_name:
            live_unit_fracs['readout'] = np.zeros((n_model_instances, n_model_instances, n_iter))

        for i in range(4):
            if 'FD' in source_model_name:
                if (i+1) == 4:
                    live_unit_fracs['hidden4'] = np.zeros((n_model_instances, n_model_instances))
                else:
                    live_unit_fracs['hidden{}'.format(i+1)] = np.zeros((n_model_instances, n_model_instances, n_unit_samplings))
            elif 'RC' in source_model_name:
                if (i+1) == 4:
                    live_unit_fracs['hidden4'] = np.zeros((n_model_instances, n_model_instances, n_iter))
                else:
                    live_unit_fracs['hidden{}'.format(i+1)] = np.zeros((n_model_instances, n_model_instances, n_unit_samplings, n_iter))


        for source_idx in range(n_model_instances):
            for target_idx in range(n_model_instances):
                print('')
                print('source_idx: {}'.format(source_idx+1))                        
                print('target_idx: {}'.format(target_idx+1))
                print('')
                test_mses, test_median_r2s, test_median_pccs, test_live_unit_fracs = self.test_linear_regressor_all_layers(pose_type, no_occ_data, source_model_name, target_model_name, 
                                                                        source_idx, target_idx, n_units, n_unit_samplings, total_var_thr)
                mses['readout'][source_idx, target_idx] = test_mses['readout']
                median_r2s['readout'][source_idx, target_idx] = test_median_r2s['readout']
                median_pccs['readout'][source_idx, target_idx] = test_median_pccs['readout']
                live_unit_fracs['readout'][source_idx, target_idx] = test_live_unit_fracs['readout']

                for i in range(4):
                    mses['hidden{}'.format(i+1)][source_idx, target_idx] = test_mses['hidden{}'.format(i+1)]
                    median_r2s['hidden{}'.format(i+1)][source_idx, target_idx] = test_median_r2s['hidden{}'.format(i+1)]
                    median_pccs['hidden{}'.format(i+1)][source_idx, target_idx] = test_median_pccs['hidden{}'.format(i+1)]
                    live_unit_fracs['hidden{}'.format(i+1)][source_idx, target_idx] = test_live_unit_fracs['hidden{}'.format(i+1)]

        np.savez_compressed(os.path.join(test_result_save_path, 'mses'),**mses)
        np.savez_compressed(os.path.join(test_result_save_path, 'median_r2s_thr{:.1e}'.format(total_var_thr)),**median_r2s)
        np.savez_compressed(os.path.join(test_result_save_path, 'median_pccs_thr{:.1e}'.format(total_var_thr)),**median_pccs)
        np.savez_compressed(os.path.join(test_result_save_path, 'live_unit_fracs'),**live_unit_fracs)


    def test_linear_regressor_all_layers(self, pose_type, no_occ_data, source_model_name, target_model_name, source_idx, target_idx, n_units, n_unit_samplings, total_var_thr):
        if pose_type =='gt':
            input_size = 28
        elif pose_type == 'original':
            input_size = 50

        if source_model_name in ['FDClassifier', 'NoOccFDClassifier']:
            source_model = FDClassifier(input_size=input_size)
        elif source_model_name in ['RCClassifier', 'NoOccRCClassifier']:
            source_model = RCClassifier(input_size=input_size, record_activations=True, record_readouts=True)
        elif source_model_name in ['TDRCClassifier', 'NoOccTDRCClassifier']:
            source_model = TDRCClassifier(input_size=input_size, record_activations=True, record_readouts=True)
                
        if target_model_name in ['FDClassifier', 'NoOccFDClassifier']:
            target_model = FDClassifier(input_size=input_size)
        elif target_model_name in ['RCClassifier', 'NoOccRCClassifier']:
            target_model = RCClassifier(input_size=input_size, record_activations=True, record_readouts=True)
        elif target_model_name in ['TDRCClassifier', 'NoOccTDRCClassifier']:
            target_model = TDRCClassifier(input_size=input_size, record_activations=True, record_readouts=True)


        source_model_save_path = os.path.join(os.path.join(self.classifier_dir, pose_type, source_model_name), 
                                    'trial_{}'.format(source_idx+1), source_model_name+'_best_val_acc.pth.tar')
        target_model_save_path = os.path.join(os.path.join(self.classifier_dir, pose_type, target_model_name), 
                                    'trial_{}'.format(target_idx+1), target_model_name+'_best_val_acc.pth.tar')


        source_model.load_state_dict(torch.load(source_model_save_path, map_location=lambda storage, loc: storage))
        target_model.load_state_dict(torch.load(target_model_save_path, map_location=lambda storage, loc: storage))

        if source_model_name in ['FDClassifier', 'NoOccFDClassifier']:
            test_mses, test_median_r2s, test_median_pccs, test_live_unit_fracs = self.FD_test_linear_regressor_helper_all_layers(pose_type, no_occ_data, source_model, target_model, source_model_name, target_model_name, 
                                                source_idx, target_idx, n_units, n_unit_samplings, total_var_thr)
        elif source_model_name in ['RCClassifier', 'NoOccRCClassifier', 'TDRCClassifier', 'NoOccTDRCClassifier']:
            test_mses, test_median_r2s, test_median_pccs, test_live_unit_fracs = self.RC_test_linear_regressor_helper_all_layers(pose_type, no_occ_data, source_model, target_model, source_model_name, target_model_name, 
                                                source_idx, target_idx, n_units, n_unit_samplings, total_var_thr)

        return test_mses, test_median_r2s, test_median_pccs, test_live_unit_fracs



    def RC_test_linear_regressor_helper_all_layers(self, pose_type, no_occ_data, source_model, target_model, source_model_name, target_model_name, source_idx, target_idx, n_units, n_unit_samplings, total_var_thr):
        '''
        Return: 

        test_r2s
        test_mses
        test_pccs
        test_median_pccs
        '''

        source_model.eval()
        target_model.eval()

        if no_occ_data:
            no_occ_data_name = 'NoOcc'
        else:
            no_occ_data_name = 'Occ'

        model_save_path = os.path.join(self.model_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name, 'source_idx{}_target_idx{}'.format(source_idx+1, target_idx+1))

        # Readout
        regressors = {}
        regressors['readout'] = nn.Linear(source_model.n_classes, target_model.n_classes)
        regressors['readout'].load_state_dict(torch.load(os.path.join(model_save_path, 'readout_best_val_loss.pth.tar'),
                                                         map_location=lambda storage, loc: storage))

        if self.gpu_mode:
            regressors['readout'].cuda()

        # Non-readout layers
        # Determine units to sample.
        sample_unit_path = os.path.join(self.sample_unit_dir, self.data_dir, pose_type, target_model_name, 'target_idx{}'.format(target_idx+1))
        if not os.path.exists(sample_unit_path):
            os.makedirs(sample_unit_path, exist_ok=True)

        if os.path.isfile(os.path.join(sample_unit_path, 'sample_indices.pt')):
            sample_indices = torch.load(os.path.join(sample_unit_path, 'sample_indices.pt'),
                                map_location=lambda storage, loc: storage)
        else:
            C = source_model.hidden1
            H = source_model.input_size//2
            W = H
            sample_indices_hidden1 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

            C = source_model.hidden2
            H = source_model.input_size//4
            W = H
            sample_indices_hidden2 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

            C = source_model.hidden3
            H = source_model.input_size//8
            W = H
            sample_indices_hidden3 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

            sample_indices = torch.stack([sample_indices_hidden1, sample_indices_hidden2, sample_indices_hidden3], 0) # (n_layer, n_unit_samplings, n_units)
            torch.save(sample_indices, os.path.join(sample_unit_path, 'sample_indices.pt'))

        if self.gpu_mode:
            sample_indices = sample_indices.cuda()

        regressors['hidden1'] = []
        regressors['hidden2'] = []
        regressors['hidden3'] = []

        for i in range(n_unit_samplings):
            regressors['hidden1'].append(nn.Linear(source_model.hidden1*(source_model.input_size//2)**2, n_units))
            regressors['hidden2'].append(nn.Linear(source_model.hidden2*(source_model.input_size//4)**2, n_units))
            regressors['hidden3'].append(nn.Linear(source_model.hidden3*(source_model.input_size//8)**2, n_units))

            regressors['hidden1'][i].load_state_dict(torch.load(os.path.join(model_save_path, 'hidden1_sample{}_best_val_loss.pth.tar'.format(i+1)),
                                                         map_location=lambda storage, loc: storage))
            regressors['hidden2'][i].load_state_dict(torch.load(os.path.join(model_save_path, 'hidden2_sample{}_best_val_loss.pth.tar'.format(i+1)),
                                                         map_location=lambda storage, loc: storage))
            regressors['hidden3'][i].load_state_dict(torch.load(os.path.join(model_save_path, 'hidden3_sample{}_best_val_loss.pth.tar'.format(i+1)),
                                                         map_location=lambda storage, loc: storage))

            if self.gpu_mode:
                regressors['hidden1'][i].cuda()
                regressors['hidden2'][i].cuda()
                regressors['hidden3'][i].cuda()

        regressors['hidden4'] = nn.Linear(source_model.hidden4, target_model.hidden4)
        regressors['hidden4'].load_state_dict(torch.load(os.path.join(model_save_path, 'hidden4_best_val_loss.pth.tar'),
                                                         map_location=lambda storage, loc: storage))

        if self.gpu_mode:
            regressors['hidden4'].cuda()


        if self.gpu_mode:
            source_model.cuda()
            target_model.cuda()

        # Validation of the main model and z_classifier.
        test_mses = {}
        test_median_r2s = {}
        test_median_pccs = {}
        test_live_unit_fracs ={}

        test_mses['readout'] = np.zeros(source_model.n_iter)
        test_median_r2s['readout'] = np.zeros(source_model.n_iter)
        test_median_pccs['readout'] = np.zeros(source_model.n_iter)
        test_live_unit_fracs['readout'] = np.zeros(source_model.n_iter)            

        for i in range(4):
            if (i+1) == 4:
                test_mses['hidden4'] = np.zeros(source_model.n_iter)
                test_median_r2s['hidden4'] = np.zeros(source_model.n_iter)
                test_median_pccs['hidden4'] = np.zeros(source_model.n_iter)
                test_live_unit_fracs['hidden4'] = np.zeros(source_model.n_iter)
            else:
                test_mses['hidden{}'.format(i+1)] = np.zeros((n_unit_samplings, source_model.n_iter))
                test_median_r2s['hidden{}'.format(i+1)] = np.zeros((n_unit_samplings, source_model.n_iter))
                test_median_pccs['hidden{}'.format(i+1)] = np.zeros((n_unit_samplings, source_model.n_iter))
                test_live_unit_fracs['hidden{}'.format(i+1)] = np.zeros((n_unit_samplings, source_model.n_iter))

        test_pred_activations = {}

        test_pred_activations['readout'] = torch.zeros(len(self.test_loader.dataset), source_model.n_iter, target_model.n_classes)
        if self.gpu_mode:
            test_pred_activations['readout'] = test_pred_activations['readout'].cuda()

        for i in range(4):
            if (i+1) == 4:
                test_pred_activations['hidden4'] = torch.zeros(len(self.test_loader.dataset), source_model.n_iter, target_model.hidden4)
            else:
                test_pred_activations['hidden{}'.format(i+1)] = torch.zeros(len(self.test_loader.dataset), n_unit_samplings, source_model.n_iter, n_units)
            if self.gpu_mode:
                test_pred_activations['hidden{}'.format(i+1)] = test_pred_activations['hidden{}'.format(i+1)].cuda()


        test_target_activations = {}

        test_target_activations['readout'] = torch.zeros(len(self.test_loader.dataset), source_model.n_iter, target_model.n_classes)
        if self.gpu_mode:
            test_target_activations['readout'] = test_target_activations['readout'].cuda()

        for i in range(4):
            if (i+1) == 4:
                test_target_activations['hidden4'] = torch.zeros(len(self.test_loader.dataset), source_model.n_iter, target_model.hidden4)
            else:
                test_target_activations['hidden{}'.format(i+1)] = torch.zeros(len(self.test_loader.dataset), n_unit_samplings, source_model.n_iter, n_units)
            if self.gpu_mode:
                test_target_activations['hidden{}'.format(i+1)] = test_target_activations['hidden{}'.format(i+1)].cuda()


        iter_start_time = time.time()
        for iter, (data, ind_objs, ind_ps, _) in enumerate(self.test_loader):
            N, C, _, _ = data.size()
            data = Variable(data, volatile=True)
            ind_objs = Variable(ind_objs, volatile=True)
            ind_ps = Variable(ind_ps, volatile=True)
            if self.gpu_mode:
                data = data.cuda()
                ind_objs = ind_objs.cuda()
                ind_ps = ind_ps.cuda()

            if pose_type == 'gt':
                aff_ps = self.affine_form(ind_ps[:,1])
                grid = F.affine_grid(aff_ps, torch.Size([N, C, self.in_size, self.in_size]))

            if pose_type == 'gt':                
                if no_occ_data:
                    x = F.grid_sample(ind_objs[:,1], grid)
                else:
                    x = F.grid_sample(data, grid)
            elif pose_type == 'original':
                if no_occ_data:
                    x = ind_objs[:,1]
                else:
                    x = data

            _ = source_model(x)
            _ = target_model(x)

            regressor = regressors['readout']
            for t in range(source_model.n_iter):
                pred = regressor(source_model.readouts[t])
                test_pred_activations['readout'][iter*N:(iter+1)*N, t] = pred.data
                test_target_activations['readout'][iter*N:(iter+1)*N, t] = target_model.readouts[t].data                

            for i in range(4):
                regressor = regressors['hidden{}'.format(i+1)]
                if (i+1) == 4:
                    for t in range(source_model.n_iter):
                        pred = regressor(source_model.activations_hidden4[t])
                        test_pred_activations['hidden4'][iter*N:(iter+1)*N, t] = pred.data
                        test_target_activations['hidden4'][iter*N:(iter+1)*N, t] = target_model.activations_hidden4[t].data
                else:
                    for j in range(n_unit_samplings):
                        for t in range(source_model.n_iter):
                            sampled_target_activations = self.sample_units(getattr(target_model, 'activations_hidden{}'.format(i+1))[t], sample_indices[i,j])
                            pred = regressor[j](self.flatten(getattr(source_model, 'activations_hidden{}'.format(i+1))[t]))

                            test_pred_activations['hidden{}'.format(i+1)][iter*N:(iter+1)*N, j, t] = pred.data
                            test_target_activations['hidden{}'.format(i+1)][iter*N:(iter+1)*N, j, t] = sampled_target_activations.data


        # identify live units.
        # live_units = {}
        # eps = 1e-4
        # for i in range(4):
        #     if (i+1) == 4:
        #         live_units['hidden4'] = torch.nonzero((test_target_activations['hidden4'].abs().max(0)[0].min(0)[0] > eps)).squeeze()
        #         test_live_unit_fracs['hidden4'] = len(live_units['hidden4'])/target_model.hidden4
        #     else:
        #         live_units['hidden{}'.format(i+1)] = []
        #         for j in range(n_unit_samplings):
        #             live_units['hidden{}'.format(i+1)].append(torch.nonzero(test_target_activations['hidden{}'.format(i+1)][:,j].abs().max(0)[0].min(0)[0] > eps).squeeze())
        #             test_live_unit_fracs['hidden{}'.format(i+1)][j] = len(live_units['hidden{}'.format(i+1)][j])/n_units
        #     print(test_live_unit_fracs['hidden{}'.format(i+1)])

        live_units = {}

        target = test_target_activations['readout']
        mean = target.mean(0, True)
        total_var = (target - mean).pow(2).mean(0)
        live_units['readout'] = torch.nonzero((total_var.min(0)[0] > total_var_thr)).squeeze()
        test_live_unit_fracs['readout'] = len(live_units['readout'])/target_model.n_classes

        for i in range(4):
            if (i+1) == 4:
                target = test_target_activations['hidden4']
                mean = target.mean(0, True)
                total_var = (target - mean).pow(2).mean(0)
                live_units['hidden4'] = torch.nonzero((total_var.min(0)[0] > total_var_thr)).squeeze()
                test_live_unit_fracs['hidden4'] = len(live_units['hidden4'])/target_model.hidden4
            else:
                live_units['hidden{}'.format(i+1)] = []
                for j in range(n_unit_samplings):
                    target = test_target_activations['hidden{}'.format(i+1)][:,j]
                    mean = target.mean(0, True)
                    total_var = (target - mean).pow(2).mean(0)
                    live_units['hidden{}'.format(i+1)].append(torch.nonzero(total_var.min(0)[0] > total_var_thr).squeeze())
                    test_live_unit_fracs['hidden{}'.format(i+1)][j] = len(live_units['hidden{}'.format(i+1)][j])/n_units

        # mse, r2
        pred = test_pred_activations['readout'][:,:,live_units['readout']]
        target = test_target_activations['readout'][:,:,live_units['readout']]
        mean = target.mean(0, True)
        total_var = (target - mean).pow(2).mean(0)
        mse = (target - pred).pow(2).mean(0)
        test_median_r2s['readout'] = (1-(mse/total_var)).median(1)[0].cpu().numpy()
        test_mses['readout'] = mse.mean(1).cpu().numpy()

        for i in range(4):
            if (i+1) == 4:
                pred = test_pred_activations['hidden4'][:,:,live_units['hidden4']]
                target = test_target_activations['hidden4'][:,:,live_units['hidden4']]
                mean = target.mean(0, True)
                total_var = (target - mean).pow(2).mean(0)
                mse = (target - pred).pow(2).mean(0)
                test_median_r2s['hidden4'] = (1-(mse/total_var)).median(1)[0].cpu().numpy()
                test_mses['hidden4'] = mse.mean(1).cpu().numpy()
            else:
                for j in range(n_unit_samplings):
                    pred = test_pred_activations['hidden{}'.format(i+1)][:,j][:,:,live_units['hidden{}'.format(i+1)][j]]
                    target = test_target_activations['hidden{}'.format(i+1)][:,j][:,:,live_units['hidden{}'.format(i+1)][j]]
                    mean = target.mean(0, True)
                    total_var = (target - mean).pow(2).mean(0)
                    mse = (target - pred).pow(2).mean(0)
                    test_median_r2s['hidden{}'.format(i+1)][j] = (1-(mse/total_var)).median(1)[0].cpu().numpy()
                    test_mses['hidden{}'.format(i+1)][j] = mse.mean(1).cpu().numpy()


        # pcc, median_pccs
        pred = test_pred_activations['readout'][:,:,live_units['readout']]
        target = test_target_activations['readout'][:,:,live_units['readout']]
        pred_mean = pred.mean(0, True)
        target_mean = target.mean(0, True)
        pred_centered = pred - pred_mean
        target_centered = target - target_mean
        cov = (pred_centered*target_centered).mean(0)
        std_pred = pred_centered.pow(2).mean(0).pow(0.5)
        std_target = target_centered.pow(2).mean(0).pow(0.5)
        pccs = cov/(std_pred*std_target)
        test_median_pccs['readout'] = pccs.median(1)[0].cpu().numpy()            

        for i in range(4):
            if (i+1) == 4:
                pred = test_pred_activations['hidden4'][:,:,live_units['hidden4']]
                target = test_target_activations['hidden4'][:,:,live_units['hidden4']]
                pred_mean = pred.mean(0, True)
                target_mean = target.mean(0, True)
                pred_centered = pred - pred_mean
                target_centered = target - target_mean
                cov = (pred_centered*target_centered).mean(0)
                std_pred = pred_centered.pow(2).mean(0).pow(0.5)
                std_target = target_centered.pow(2).mean(0).pow(0.5)
                pccs = cov/(std_pred*std_target)
                test_median_pccs['hidden4'] = pccs.median(1)[0].cpu().numpy()
            else:
                for j in range(n_unit_samplings):
                    pred = test_pred_activations['hidden{}'.format(i+1)][:,j][:,:,live_units['hidden{}'.format(i+1)][j]]
                    target = test_target_activations['hidden{}'.format(i+1)][:,j][:,:,live_units['hidden{}'.format(i+1)][j]]
                    pred_mean = pred.mean(0, True)
                    target_mean = target.mean(0, True)
                    pred_centered = pred - pred_mean
                    target_centered = target - target_mean
                    cov = (pred_centered*target_centered).mean(0)
                    std_pred = pred_centered.pow(2).mean(0).pow(0.5)
                    std_target = target_centered.pow(2).mean(0).pow(0.5)
                    pccs = cov/(std_pred*std_target)
                    test_median_pccs['hidden{}'.format(i+1)][j] = pccs.median(1)[0].cpu().numpy()

        return test_mses, test_median_r2s, test_median_pccs, test_live_unit_fracs



    def FD_test_linear_regressor_helper_all_layers(self, pose_type, no_occ_data, source_model, target_model, source_model_name, target_model_name, source_idx, target_idx, n_units, n_unit_samplings, total_var_thr):
        '''
        Return: 

        test_r2s
        test_mses
        test_pccs
        test_median_pccs
        '''
        activations = {}
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook

        source_model.model[14].register_forward_hook(get_activation('source_readout'))
        target_model.model[14].register_forward_hook(get_activation('target_readout'))

        source_model.model[2].register_forward_hook(get_activation('source_hidden1'))
        target_model.model[2].register_forward_hook(get_activation('target_hidden1'))

        source_model.model[5].register_forward_hook(get_activation('source_hidden2'))
        target_model.model[5].register_forward_hook(get_activation('target_hidden2'))

        source_model.model[8].register_forward_hook(get_activation('source_hidden3'))
        target_model.model[8].register_forward_hook(get_activation('target_hidden3'))

        source_model.model[12].register_forward_hook(get_activation('source_hidden4'))
        target_model.model[12].register_forward_hook(get_activation('target_hidden4'))

        source_model.eval()
        target_model.eval()

        if no_occ_data:
            no_occ_data_name = 'NoOcc'
        else:
            no_occ_data_name = 'Occ'

        model_save_path = os.path.join(self.model_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name, 'source_idx{}_target_idx{}'.format(source_idx+1, target_idx+1))

        # Readout
        regressors = {}
        regressors['readout'] = nn.Linear(source_model.n_classes, target_model.n_classes)
        regressors['readout'].load_state_dict(torch.load(os.path.join(model_save_path, 'readout_best_val_loss.pth.tar'),
                                                         map_location=lambda storage, loc: storage))

        if self.gpu_mode:
            regressors['readout'].cuda()

        # Non-readout layers
        # Determine units to sample.
        sample_unit_path = os.path.join(self.sample_unit_dir, self.data_dir, pose_type, target_model_name, 'target_idx{}'.format(target_idx+1))
        if not os.path.exists(sample_unit_path):
            os.makedirs(sample_unit_path, exist_ok=True)

        if os.path.isfile(os.path.join(sample_unit_path, 'sample_indices.pt')):
            sample_indices = torch.load(os.path.join(sample_unit_path, 'sample_indices.pt'),
                                map_location=lambda storage, loc: storage)
        else:
            C = source_model.hidden1
            H = source_model.input_size//2
            W = H
            sample_indices_hidden1 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

            C = source_model.hidden2
            H = source_model.input_size//4
            W = H
            sample_indices_hidden2 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

            C = source_model.hidden3
            H = source_model.input_size//8
            W = H
            sample_indices_hidden3 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

            sample_indices = torch.stack([sample_indices_hidden1, sample_indices_hidden2, sample_indices_hidden3], 0) # (n_layer, n_unit_samplings, n_units)
            torch.save(sample_indices, os.path.join(sample_unit_path, 'sample_indices.pt'))

        if self.gpu_mode:
            sample_indices = sample_indices.cuda()
        regressors['hidden1'] = []
        regressors['hidden2'] = []
        regressors['hidden3'] = []

        for i in range(n_unit_samplings):
            regressors['hidden1'].append(nn.Linear(source_model.hidden1*(source_model.input_size//2)**2, n_units))
            regressors['hidden2'].append(nn.Linear(source_model.hidden2*(source_model.input_size//4)**2, n_units))
            regressors['hidden3'].append(nn.Linear(source_model.hidden3*(source_model.input_size//8)**2, n_units))


            regressors['hidden1'][i].load_state_dict(torch.load(os.path.join(model_save_path, 'hidden1_sample{}_best_val_loss.pth.tar'.format(i+1)),
                                                         map_location=lambda storage, loc: storage))
            regressors['hidden2'][i].load_state_dict(torch.load(os.path.join(model_save_path, 'hidden2_sample{}_best_val_loss.pth.tar'.format(i+1)),
                                                         map_location=lambda storage, loc: storage))
            regressors['hidden3'][i].load_state_dict(torch.load(os.path.join(model_save_path, 'hidden3_sample{}_best_val_loss.pth.tar'.format(i+1)),
                                                         map_location=lambda storage, loc: storage))


            if self.gpu_mode:
                regressors['hidden1'][i].cuda()
                regressors['hidden2'][i].cuda()
                regressors['hidden3'][i].cuda()

        regressors['hidden4'] = nn.Linear(source_model.hidden4, target_model.hidden4)
        regressors['hidden4'].load_state_dict(torch.load(os.path.join(model_save_path, 'hidden4_best_val_loss.pth.tar'),
                                                         map_location=lambda storage, loc: storage))

        if self.gpu_mode:
            regressors['hidden4'].cuda()

        if self.gpu_mode:
            source_model.cuda()
            target_model.cuda()

        test_mses = {}
        test_median_r2s = {}
        test_median_pccs = {}
        test_live_unit_fracs ={}

        test_mses['readout'] = 0.0
        test_median_r2s['readout'] = 0.0
        test_median_pccs['readout'] = 0.0
        test_live_unit_fracs['readout'] = 0.0

        for i in range(4):
            if (i+1) == 4:
                test_mses['hidden4'] = 0.0
                test_median_r2s['hidden4'] = 0.0
                test_median_pccs['hidden4'] = 0.0
                test_live_unit_fracs['hidden4'] = 0.0
            else:
                test_mses['hidden{}'.format(i+1)] = np.zeros((n_unit_samplings,))
                test_median_r2s['hidden{}'.format(i+1)] = np.zeros((n_unit_samplings,))
                test_median_pccs['hidden{}'.format(i+1)] = np.zeros((n_unit_samplings,))
                test_live_unit_fracs['hidden{}'.format(i+1)] = np.zeros((n_unit_samplings,))


        test_pred_activations = {}

        test_pred_activations['readout'] = torch.zeros(len(self.test_loader.dataset), target_model.n_classes)
        if self.gpu_mode:
            test_pred_activations['readout'] = test_pred_activations['readout'].cuda()

        for i in range(4):
            if (i+1) == 4:
                test_pred_activations['hidden4'] = torch.zeros(len(self.test_loader.dataset), target_model.hidden4)
            else:
                test_pred_activations['hidden{}'.format(i+1)] = torch.zeros(len(self.test_loader.dataset), n_unit_samplings, n_units)
            if self.gpu_mode:
                test_pred_activations['hidden{}'.format(i+1)] = test_pred_activations['hidden{}'.format(i+1)].cuda()


        test_target_activations = {}

        test_target_activations['readout'] = torch.zeros(len(self.test_loader.dataset), target_model.n_classes)            
        if self.gpu_mode:
            test_target_activations['readout'] = test_target_activations['readout'].cuda()

        for i in range(4):
            if (i+1) == 4:
                test_target_activations['hidden4'] = torch.zeros(len(self.test_loader.dataset), target_model.hidden4)
            else:
                test_target_activations['hidden{}'.format(i+1)] = torch.zeros(len(self.test_loader.dataset), n_unit_samplings, n_units)
            if self.gpu_mode:
                test_target_activations['hidden{}'.format(i+1)] = test_target_activations['hidden{}'.format(i+1)].cuda()


        iter_start_time = time.time()
        for iter, (data, ind_objs, ind_ps, _) in enumerate(self.test_loader):
            N, C, _, _ = data.size()
            data = Variable(data, volatile=True)
            ind_objs = Variable(ind_objs, volatile=True)
            ind_ps = Variable(ind_ps, volatile=True)
            if self.gpu_mode:
                data = data.cuda()
                ind_objs = ind_objs.cuda()
                ind_ps = ind_ps.cuda()

            if pose_type == 'gt':
                aff_ps = self.affine_form(ind_ps[:,1])
                grid = F.affine_grid(aff_ps, torch.Size([N, C, self.in_size, self.in_size]))

            if pose_type == 'gt':                
                if no_occ_data:
                    x = F.grid_sample(ind_objs[:,1], grid)
                else:
                    x = F.grid_sample(data, grid)
            elif pose_type == 'original':
                if no_occ_data:
                    x = ind_objs[:,1]
                else:
                    x = data

            _ = source_model(x)
            _ = target_model(x)

            regressor = regressors['readout']
            pred = regressor(activations['source_readout'])
            test_pred_activations['readout'][iter*N:(iter+1)*N] = pred.data
            test_target_activations['readout'][iter*N:(iter+1)*N] = activations['target_readout'].data                

            for i in range(4):
                regressor = regressors['hidden{}'.format(i+1)]
                if (i+1) == 4:
                    pred = regressor(activations['source_hidden4'])
                    test_pred_activations['hidden4'][iter*N:(iter+1)*N] = pred.data
                    test_target_activations['hidden4'][iter*N:(iter+1)*N] = activations['target_hidden4'].data
                else:
                    for j in range(n_unit_samplings):
                        sampled_target_activations = self.sample_units(activations['target_hidden{}'.format(i+1)], sample_indices[i,j])
                        pred = regressor[j](self.flatten(activations['source_hidden{}'.format(i+1)]))
                        test_pred_activations['hidden{}'.format(i+1)][iter*N:(iter+1)*N, j] = pred.data
                        test_target_activations['hidden{}'.format(i+1)][iter*N:(iter+1)*N, j] = sampled_target_activations.data


        # identify live units.
        # live_units = {}
        # for i in range(4):
        #     if (i+1) == 4:
        #         live_units['hidden4'] = torch.nonzero((test_target_activations['hidden4'].abs().sum(0) != 0)).squeeze()
        #         test_live_unit_fracs['hidden4'] = len(live_units['hidden4'])/target_model.hidden4
        #     else:
        #         live_units['hidden{}'.format(i+1)] = []
        #         for j in range(n_unit_samplings):
        #             live_units['hidden{}'.format(i+1)].append(torch.nonzero(test_target_activations['hidden{}'.format(i+1)][:,j].abs().sum(0) != 0).squeeze())
        #             test_live_unit_fracs['hidden{}'.format(i+1)][j] = len(live_units['hidden{}'.format(i+1)][j])/n_units
        #     print(test_live_unit_fracs['hidden{}'.format(i+1)])
        live_units = {}

        target = test_target_activations['readout']
        mean = target.mean(0, True)
        total_var = (target - mean).pow(2).mean(0)
        live_units['readout'] = torch.nonzero((total_var > total_var_thr)).squeeze()
        test_live_unit_fracs['readout'] = len(live_units['readout'])/target_model.n_classes

        for i in range(4):
            if (i+1) == 4:
                target = test_target_activations['hidden4']
                mean = target.mean(0, True)
                total_var = (target - mean).pow(2).mean(0)
                live_units['hidden4'] = torch.nonzero((total_var > total_var_thr)).squeeze()
                test_live_unit_fracs['hidden4'] = len(live_units['hidden4'])/target_model.hidden4
            else:
                live_units['hidden{}'.format(i+1)] = []
                for j in range(n_unit_samplings):
                    target = test_target_activations['hidden{}'.format(i+1)][:,j]
                    mean = target.mean(0, True)
                    total_var = (target - mean).pow(2).mean(0)
                    live_units['hidden{}'.format(i+1)].append(torch.nonzero(total_var > total_var_thr).squeeze())
                    test_live_unit_fracs['hidden{}'.format(i+1)][j] = len(live_units['hidden{}'.format(i+1)][j])/n_units


        # mse, median_r2
        pred = test_pred_activations['readout'][:,live_units['readout']]
        target = test_target_activations['readout'][:,live_units['readout']]
        mean = target.mean(0, True)
        total_var = (target - mean).pow(2).mean(0)
        mse = (target - pred).pow(2).mean(0)
        test_median_r2s['readout'] = (1-(mse/total_var)).median()
        test_mses['readout'] = mse.mean()

        for i in range(4):
            if (i+1) == 4:
                pred = test_pred_activations['hidden4'][:,live_units['hidden4']]
                target = test_target_activations['hidden4'][:,live_units['hidden4']]
                mean = target.mean(0, True)
                total_var = (target - mean).pow(2).mean(0)
                mse = (target - pred).pow(2).mean(0)
                test_median_r2s['hidden4'] = (1-(mse/total_var)).median()
                test_mses['hidden4'] = mse.mean()
            else:
                for j in range(n_unit_samplings):
                    pred = test_pred_activations['hidden{}'.format(i+1)][:,j][:,live_units['hidden{}'.format(i+1)][j]]
                    target = test_target_activations['hidden{}'.format(i+1)][:,j][:,live_units['hidden{}'.format(i+1)][j]]
                    mean = target.mean(0, True)
                    total_var = (target - mean).pow(2).mean(0)
                    mse = (target - pred).pow(2).mean(0)
                    test_median_r2s['hidden{}'.format(i+1)][j] = (1-(mse/total_var)).median()
                    test_mses['hidden{}'.format(i+1)][j] = mse.mean()



        # pcc, median_pccs
        pred = test_pred_activations['readout'][:,live_units['readout']]
        target = test_target_activations['readout'][:,live_units['readout']]
        pred_mean = pred.mean(0, True)
        target_mean = target.mean(0, True)
        pred_centered = pred - pred_mean
        target_centered = target - target_mean
        cov = (pred_centered*target_centered).mean(0)
        std_pred = pred_centered.pow(2).mean(0).pow(0.5)
        std_target = target_centered.pow(2).mean(0).pow(0.5)
        pccs = cov/(std_pred*std_target)
        test_median_pccs['readout'] = pccs.median() 

        for i in range(4):
            if (i+1) == 4:
                pred = test_pred_activations['hidden4'][:,live_units['hidden4']]
                target = test_target_activations['hidden4'][:,live_units['hidden4']]
                pred_mean = pred.mean(0, True)
                target_mean = target.mean(0, True)
                pred_centered = pred - pred_mean
                target_centered = target - target_mean
                cov = (pred_centered*target_centered).mean(0)
                std_pred = pred_centered.pow(2).mean(0).pow(0.5)
                std_target = target_centered.pow(2).mean(0).pow(0.5)
                pccs = cov/(std_pred*std_target)
                test_median_pccs['hidden4'] = pccs.median() 
            else:
                for j in range(n_unit_samplings):
                    pred = test_pred_activations['hidden{}'.format(i+1)][:,j][:,live_units['hidden{}'.format(i+1)][j]]
                    target = test_target_activations['hidden{}'.format(i+1)][:,j][:,live_units['hidden{}'.format(i+1)][j]]
                    pred_mean = pred.mean(0, True)
                    target_mean = target.mean(0, True)
                    pred_centered = pred - pred_mean
                    target_centered = target - target_mean
                    cov = (pred_centered*target_centered).mean(0)
                    std_pred = pred_centered.pow(2).mean(0).pow(0.5)
                    std_target = target_centered.pow(2).mean(0).pow(0.5)
                    pccs = cov/(std_pred*std_target)
                    test_median_pccs['hidden{}'.format(i+1)][j] = pccs.median()

        return test_mses, test_median_r2s, test_median_pccs, test_live_unit_fracs



    def test_linear_regressor_aggregate_over_model_instances(self, pose_type, no_occ_data, source_model_name, target_model_name, readout, n_model_instances=5, n_units=256, n_unit_samplings=3, n_iter=5, total_var_thr=0):
        if no_occ_data:
            no_occ_data_name = 'NoOcc'
        else:
            no_occ_data_name = 'Occ'
        test_result_save_path = os.path.join(self.test_result_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name)
        if not os.path.exists(test_result_save_path):
            os.makedirs(test_result_save_path, exist_ok=True)

        mses = {}
        if readout:
            if 'FD' in source_model_name:
                mses['readout'] = np.zeros((n_model_instances, n_model_instances))
            elif 'RC' in source_model_name:
                mses['readout'] = np.zeros((n_model_instances, n_model_instances, n_iter))
        else:
            for i in range(4):
                if 'FD' in source_model_name:
                    if (i+1) == 4:
                        mses['hidden4'] = np.zeros((n_model_instances, n_model_instances))
                    else:
                        mses['hidden{}'.format(i+1)] = np.zeros((n_model_instances, n_model_instances, n_unit_samplings))
                elif 'RC' in source_model_name:
                    if (i+1) == 4:
                        mses['hidden4'] = np.zeros((n_model_instances, n_model_instances, n_iter))
                    else:
                        mses['hidden{}'.format(i+1)] = np.zeros((n_model_instances, n_model_instances, n_unit_samplings, n_iter))

        median_r2s = {}
        if readout:
            if 'FD' in source_model_name:
                median_r2s['readout'] = np.zeros((n_model_instances, n_model_instances))
            elif 'RC' in source_model_name:
                median_r2s['readout'] = np.zeros((n_model_instances, n_model_instances, n_iter))
        else:
            for i in range(4):
                if 'FD' in source_model_name:
                    if (i+1) == 4:
                        median_r2s['hidden4'] = np.zeros((n_model_instances, n_model_instances))
                    else:
                        median_r2s['hidden{}'.format(i+1)] = np.zeros((n_model_instances, n_model_instances, n_unit_samplings))
                elif 'RC' in source_model_name:
                    if (i+1) == 4:
                        median_r2s['hidden4'] = np.zeros((n_model_instances, n_model_instances, n_iter))
                    else:
                        median_r2s['hidden{}'.format(i+1)] = np.zeros((n_model_instances, n_model_instances, n_unit_samplings, n_iter))

        median_pccs = {}
        if readout:
            if 'FD' in source_model_name:
                median_pccs['readout'] = np.zeros((n_model_instances, n_model_instances))
            elif 'RC' in source_model_name:
                median_pccs['readout'] = np.zeros((n_model_instances, n_model_instances, n_iter))
        else:
            for i in range(4):
                if 'FD' in source_model_name:
                    if (i+1) == 4:
                        median_pccs['hidden4'] = np.zeros((n_model_instances, n_model_instances))
                    else:
                        median_pccs['hidden{}'.format(i+1)] = np.zeros((n_model_instances, n_model_instances, n_unit_samplings))
                elif 'RC' in source_model_name:
                    if (i+1) == 4:
                        median_pccs['hidden4'] = np.zeros((n_model_instances, n_model_instances, n_iter))
                    else:
                        median_pccs['hidden{}'.format(i+1)] = np.zeros((n_model_instances, n_model_instances, n_unit_samplings, n_iter))

        live_unit_fracs = {}
        if readout:
            if 'FD' in source_model_name:
                live_unit_fracs['readout'] = np.zeros((n_model_instances, n_model_instances))
            elif 'RC' in source_model_name:
                live_unit_fracs['readout'] = np.zeros((n_model_instances, n_model_instances, n_iter))
        else:
            for i in range(4):
                if 'FD' in source_model_name:
                    if (i+1) == 4:
                        live_unit_fracs['hidden4'] = np.zeros((n_model_instances, n_model_instances))
                    else:
                        live_unit_fracs['hidden{}'.format(i+1)] = np.zeros((n_model_instances, n_model_instances, n_unit_samplings))
                elif 'RC' in source_model_name:
                    if (i+1) == 4:
                        live_unit_fracs['hidden4'] = np.zeros((n_model_instances, n_model_instances, n_iter))
                    else:
                        live_unit_fracs['hidden{}'.format(i+1)] = np.zeros((n_model_instances, n_model_instances, n_unit_samplings, n_iter))


        for source_idx in range(n_model_instances):
            for target_idx in range(n_model_instances):
                print('')
                print('source_idx: {}'.format(source_idx+1))                        
                print('target_idx: {}'.format(target_idx+1))
                print('')
                test_mses, test_median_r2s, test_median_pccs, test_live_unit_fracs = self.test_linear_regressor(pose_type, no_occ_data, source_model_name, target_model_name, 
                                                                        source_idx, target_idx, n_units, n_unit_samplings, total_var_thr, readout)
                if readout:
                    mses['readout'][source_idx, target_idx] = test_mses['readout']
                    median_r2s['readout'][source_idx, target_idx] = test_median_r2s['readout']
                    median_pccs['readout'][source_idx, target_idx] = test_median_pccs['readout']
                    live_unit_fracs['readout'][source_idx, target_idx] = test_live_unit_fracs['readout']
                else:
                    for i in range(4):
                        mses['hidden{}'.format(i+1)][source_idx, target_idx] = test_mses['hidden{}'.format(i+1)]
                        median_r2s['hidden{}'.format(i+1)][source_idx, target_idx] = test_median_r2s['hidden{}'.format(i+1)]
                        median_pccs['hidden{}'.format(i+1)][source_idx, target_idx] = test_median_pccs['hidden{}'.format(i+1)]
                        live_unit_fracs['hidden{}'.format(i+1)][source_idx, target_idx] = test_live_unit_fracs['hidden{}'.format(i+1)]

        if readout:
            np.savez_compressed(os.path.join(test_result_save_path, 'readout_mses'),**mses)
            np.savez_compressed(os.path.join(test_result_save_path, 'readout_median_r2s_thr{:.1e}'.format(total_var_thr)),**median_r2s)
            np.savez_compressed(os.path.join(test_result_save_path, 'readout_median_pccs_thr{:.1e}'.format(total_var_thr)),**median_pccs)
            np.savez_compressed(os.path.join(test_result_save_path, 'readout_live_unit_fracs'),**live_unit_fracs)
        else:
            np.savez_compressed(os.path.join(test_result_save_path, 'mses'),**mses)
            np.savez_compressed(os.path.join(test_result_save_path, 'median_r2s_thr{:.1e}'.format(total_var_thr)),**median_r2s)
            np.savez_compressed(os.path.join(test_result_save_path, 'median_pccs_thr{:.1e}'.format(total_var_thr)),**median_pccs)
            np.savez_compressed(os.path.join(test_result_save_path, 'live_unit_fracs'),**live_unit_fracs)

    def test_linear_regressor(self, pose_type, no_occ_data, source_model_name, target_model_name, source_idx, target_idx, n_units, n_unit_samplings, total_var_thr, readout):
        if pose_type =='gt':
            input_size = 28
        elif pose_type == 'original':
            input_size = 50

        if source_model_name in ['FDClassifier', 'NoOccFDClassifier']:
            source_model = FDClassifier(input_size=input_size)
        elif source_model_name in ['RCClassifier', 'NoOccRCClassifier']:
            if readout:
                source_model = RCClassifier(input_size=input_size, record_readouts=True)
            else:                
                source_model = RCClassifier(input_size=input_size, record_activations=True)
                
        if target_model_name in ['FDClassifier', 'NoOccFDClassifier']:
            target_model = FDClassifier(input_size=input_size)
        elif target_model_name in ['RCClassifier', 'NoOccRCClassifier']:
            if readout:
                target_model = RCClassifier(input_size=input_size, record_readouts=True)
            else:                
                target_model = RCClassifier(input_size=input_size, record_activations=True)

        source_model_save_path = os.path.join(os.path.join(self.classifier_dir, pose_type, source_model_name), 
                                    'trial_{}'.format(source_idx+1), source_model_name+'_best_val_acc.pth.tar')
        target_model_save_path = os.path.join(os.path.join(self.classifier_dir, pose_type, target_model_name), 
                                    'trial_{}'.format(target_idx+1), target_model_name+'_best_val_acc.pth.tar')

        source_model.load_state_dict(torch.load(source_model_save_path, map_location=lambda storage, loc: storage))
        target_model.load_state_dict(torch.load(target_model_save_path, map_location=lambda storage, loc: storage))

        if source_model_name in ['FDClassifier', 'NoOccFDClassifier']:
            test_mses, test_median_r2s, test_median_pccs, test_live_unit_fracs = self.FD_test_linear_regressor_helper(pose_type, no_occ_data, source_model, target_model, source_model_name, target_model_name, 
                                                source_idx, target_idx, n_units, n_unit_samplings, total_var_thr, readout)
        elif source_model_name in ['RCClassifier', 'NoOccRCClassifier']:
            test_mses, test_median_r2s, test_median_pccs, test_live_unit_fracs = self.RC_test_linear_regressor_helper(pose_type, no_occ_data, source_model, target_model, source_model_name, target_model_name, 
                                                source_idx, target_idx, n_units, n_unit_samplings, total_var_thr, readout)

        return test_mses, test_median_r2s, test_median_pccs, test_live_unit_fracs



    def RC_test_linear_regressor_helper(self, pose_type, no_occ_data, source_model, target_model, source_model_name, target_model_name, source_idx, target_idx, n_units, n_unit_samplings, total_var_thr, readout):
        '''
        Return: 

        test_r2s
        test_mses
        test_pccs
        test_median_pccs
        '''

        source_model.eval()
        target_model.eval()

        if no_occ_data:
            no_occ_data_name = 'NoOcc'
        else:
            no_occ_data_name = 'Occ'

        model_save_path = os.path.join(self.model_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name, 'source_idx{}_target_idx{}'.format(source_idx+1, target_idx+1))

        if readout:
            regressors = {}
            regressors['readout'] = nn.Linear(source_model.n_classes, target_model.n_classes)
            regressors['readout'].load_state_dict(torch.load(os.path.join(model_save_path, 'readout_best_val_loss.pth.tar'),
                                                             map_location=lambda storage, loc: storage))

            if self.gpu_mode:
                regressors['readout'].cuda()
        else:
            # Determine units to sample.
            sample_unit_path = os.path.join(self.sample_unit_dir, self.data_dir, pose_type, target_model_name, 'target_idx{}'.format(target_idx+1))
            if not os.path.exists(sample_unit_path):
                os.makedirs(sample_unit_path, exist_ok=True)

            if os.path.isfile(os.path.join(sample_unit_path, 'sample_indices.pt')):
                sample_indices = torch.load(os.path.join(sample_unit_path, 'sample_indices.pt'),
                                    map_location=lambda storage, loc: storage)
            else:
                C = source_model.hidden1
                H = source_model.input_size//2
                W = H
                sample_indices_hidden1 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

                C = source_model.hidden2
                H = source_model.input_size//4
                W = H
                sample_indices_hidden2 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

                C = source_model.hidden3
                H = source_model.input_size//8
                W = H
                sample_indices_hidden3 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

                sample_indices = torch.stack([sample_indices_hidden1, sample_indices_hidden2, sample_indices_hidden3], 0) # (n_layer, n_unit_samplings, n_units)
                torch.save(sample_indices, os.path.join(sample_unit_path, 'sample_indices.pt'))

            if self.gpu_mode:
                sample_indices = sample_indices.cuda()

            regressors = {}
            regressors['hidden1'] = []
            regressors['hidden2'] = []
            regressors['hidden3'] = []

            for i in range(n_unit_samplings):
                regressors['hidden1'].append(nn.Linear(source_model.hidden1*(source_model.input_size//2)**2, n_units))
                regressors['hidden2'].append(nn.Linear(source_model.hidden2*(source_model.input_size//4)**2, n_units))
                regressors['hidden3'].append(nn.Linear(source_model.hidden3*(source_model.input_size//8)**2, n_units))

                regressors['hidden1'][i].load_state_dict(torch.load(os.path.join(model_save_path, 'hidden1_sample{}_best_val_loss.pth.tar'.format(i+1)),
                                                             map_location=lambda storage, loc: storage))
                regressors['hidden2'][i].load_state_dict(torch.load(os.path.join(model_save_path, 'hidden2_sample{}_best_val_loss.pth.tar'.format(i+1)),
                                                             map_location=lambda storage, loc: storage))
                regressors['hidden3'][i].load_state_dict(torch.load(os.path.join(model_save_path, 'hidden3_sample{}_best_val_loss.pth.tar'.format(i+1)),
                                                             map_location=lambda storage, loc: storage))

                if self.gpu_mode:
                    regressors['hidden1'][i].cuda()
                    regressors['hidden2'][i].cuda()
                    regressors['hidden3'][i].cuda()

            regressors['hidden4'] = nn.Linear(source_model.hidden4, target_model.hidden4)
            regressors['hidden4'].load_state_dict(torch.load(os.path.join(model_save_path, 'hidden4_best_val_loss.pth.tar'),
                                                             map_location=lambda storage, loc: storage))

            if self.gpu_mode:
                regressors['hidden4'].cuda()

        if self.gpu_mode:
            source_model.cuda()
            target_model.cuda()

        # Validation of the main model and z_classifier.
        test_mses = {}
        test_median_r2s = {}
        test_median_pccs = {}
        test_live_unit_fracs ={}
        if readout:
            test_mses['readout'] = np.zeros(source_model.n_iter)
            test_median_r2s['readout'] = np.zeros(source_model.n_iter)
            test_median_pccs['readout'] = np.zeros(source_model.n_iter)
            test_live_unit_fracs['readout'] = np.zeros(source_model.n_iter)            
        else:
            for i in range(4):
                if (i+1) == 4:
                    test_mses['hidden4'] = np.zeros(source_model.n_iter)
                    test_median_r2s['hidden4'] = np.zeros(source_model.n_iter)
                    test_median_pccs['hidden4'] = np.zeros(source_model.n_iter)
                    test_live_unit_fracs['hidden4'] = np.zeros(source_model.n_iter)
                else:
                    test_mses['hidden{}'.format(i+1)] = np.zeros((n_unit_samplings, source_model.n_iter))
                    test_median_r2s['hidden{}'.format(i+1)] = np.zeros((n_unit_samplings, source_model.n_iter))
                    test_median_pccs['hidden{}'.format(i+1)] = np.zeros((n_unit_samplings, source_model.n_iter))
                    test_live_unit_fracs['hidden{}'.format(i+1)] = np.zeros((n_unit_samplings, source_model.n_iter))

        test_pred_activations = {}
        if readout:
            test_pred_activations['readout'] = torch.zeros(len(self.test_loader.dataset), source_model.n_iter, target_model.n_classes)
            if self.gpu_mode:
                test_pred_activations['readout'] = test_pred_activations['readout'].cuda()
        else:
            for i in range(4):
                if (i+1) == 4:
                    test_pred_activations['hidden4'] = torch.zeros(len(self.test_loader.dataset), source_model.n_iter, target_model.hidden4)
                else:
                    test_pred_activations['hidden{}'.format(i+1)] = torch.zeros(len(self.test_loader.dataset), n_unit_samplings, source_model.n_iter, n_units)
                if self.gpu_mode:
                    test_pred_activations['hidden{}'.format(i+1)] = test_pred_activations['hidden{}'.format(i+1)].cuda()


        test_target_activations = {}
        if readout:
            test_target_activations['readout'] = torch.zeros(len(self.test_loader.dataset), source_model.n_iter, target_model.n_classes)
            if self.gpu_mode:
                test_target_activations['readout'] = test_target_activations['readout'].cuda()
        else:
            for i in range(4):
                if (i+1) == 4:
                    test_target_activations['hidden4'] = torch.zeros(len(self.test_loader.dataset), source_model.n_iter, target_model.hidden4)
                else:
                    test_target_activations['hidden{}'.format(i+1)] = torch.zeros(len(self.test_loader.dataset), n_unit_samplings, source_model.n_iter, n_units)
                if self.gpu_mode:
                    test_target_activations['hidden{}'.format(i+1)] = test_target_activations['hidden{}'.format(i+1)].cuda()


        iter_start_time = time.time()
        for iter, (data, ind_objs, ind_ps, _) in enumerate(self.test_loader):
            N, C, _, _ = data.size()
            data = Variable(data, volatile=True)
            ind_objs = Variable(ind_objs, volatile=True)
            ind_ps = Variable(ind_ps, volatile=True)
            if self.gpu_mode:
                data = data.cuda()
                ind_objs = ind_objs.cuda()
                ind_ps = ind_ps.cuda()

            if pose_type == 'gt':
                aff_ps = self.affine_form(ind_ps[:,1])
                grid = F.affine_grid(aff_ps, torch.Size([N, C, self.in_size, self.in_size]))

            if pose_type == 'gt':                
                if no_occ_data:
                    x = F.grid_sample(ind_objs[:,1], grid)
                else:
                    x = F.grid_sample(data, grid)
            elif pose_type == 'original':
                if no_occ_data:
                    x = ind_objs[:,1]
                else:
                    x = data

            _ = source_model(x)
            _ = target_model(x)

            if readout:
                regressor = regressors['readout']
                for t in range(source_model.n_iter):
                    pred = regressor(source_model.readouts[t])
                    test_pred_activations['readout'][iter*N:(iter+1)*N, t] = pred.data
                    test_target_activations['readout'][iter*N:(iter+1)*N, t] = target_model.readouts[t].data                
            else:
                for i in range(4):
                    regressor = regressors['hidden{}'.format(i+1)]
                    if (i+1) == 4:
                        for t in range(source_model.n_iter):
                            pred = regressor(source_model.activations_hidden4[t])
                            test_pred_activations['hidden4'][iter*N:(iter+1)*N, t] = pred.data
                            test_target_activations['hidden4'][iter*N:(iter+1)*N, t] = target_model.activations_hidden4[t].data
                    else:
                        for j in range(n_unit_samplings):
                            for t in range(source_model.n_iter):
                                sampled_target_activations = self.sample_units(getattr(target_model, 'activations_hidden{}'.format(i+1))[t], sample_indices[i,j])
                                pred = regressor[j](self.flatten(getattr(source_model, 'activations_hidden{}'.format(i+1))[t]))

                                test_pred_activations['hidden{}'.format(i+1)][iter*N:(iter+1)*N, j, t] = pred.data
                                test_target_activations['hidden{}'.format(i+1)][iter*N:(iter+1)*N, j, t] = sampled_target_activations.data


        # identify live units.
        # live_units = {}
        # eps = 1e-4
        # for i in range(4):
        #     if (i+1) == 4:
        #         live_units['hidden4'] = torch.nonzero((test_target_activations['hidden4'].abs().max(0)[0].min(0)[0] > eps)).squeeze()
        #         test_live_unit_fracs['hidden4'] = len(live_units['hidden4'])/target_model.hidden4
        #     else:
        #         live_units['hidden{}'.format(i+1)] = []
        #         for j in range(n_unit_samplings):
        #             live_units['hidden{}'.format(i+1)].append(torch.nonzero(test_target_activations['hidden{}'.format(i+1)][:,j].abs().max(0)[0].min(0)[0] > eps).squeeze())
        #             test_live_unit_fracs['hidden{}'.format(i+1)][j] = len(live_units['hidden{}'.format(i+1)][j])/n_units
        #     print(test_live_unit_fracs['hidden{}'.format(i+1)])

        live_units = {}
        if readout:
            target = test_target_activations['readout']
            mean = target.mean(0, True)
            total_var = (target - mean).pow(2).mean(0)
            live_units['readout'] = torch.nonzero((total_var.min(0)[0] > total_var_thr)).squeeze()
            test_live_unit_fracs['readout'] = len(live_units['readout'])/target_model.n_classes
        else:
            for i in range(4):
                if (i+1) == 4:
                    target = test_target_activations['hidden4']
                    mean = target.mean(0, True)
                    total_var = (target - mean).pow(2).mean(0)
                    live_units['hidden4'] = torch.nonzero((total_var.min(0)[0] > total_var_thr)).squeeze()
                    test_live_unit_fracs['hidden4'] = len(live_units['hidden4'])/target_model.hidden4
                else:
                    live_units['hidden{}'.format(i+1)] = []
                    for j in range(n_unit_samplings):
                        target = test_target_activations['hidden{}'.format(i+1)][:,j]
                        mean = target.mean(0, True)
                        total_var = (target - mean).pow(2).mean(0)
                        live_units['hidden{}'.format(i+1)].append(torch.nonzero(total_var.min(0)[0] > total_var_thr).squeeze())
                        test_live_unit_fracs['hidden{}'.format(i+1)][j] = len(live_units['hidden{}'.format(i+1)][j])/n_units

        # mse, r2
        if readout:
            pred = test_pred_activations['readout'][:,:,live_units['readout']]
            target = test_target_activations['readout'][:,:,live_units['readout']]
            mean = target.mean(0, True)
            total_var = (target - mean).pow(2).mean(0)
            mse = (target - pred).pow(2).mean(0)
            test_median_r2s['readout'] = (1-(mse/total_var)).median(1)[0].cpu().numpy()
            test_mses['readout'] = mse.mean(1).cpu().numpy()
        else:
            for i in range(4):
                if (i+1) == 4:
                    pred = test_pred_activations['hidden4'][:,:,live_units['hidden4']]
                    target = test_target_activations['hidden4'][:,:,live_units['hidden4']]
                    mean = target.mean(0, True)
                    total_var = (target - mean).pow(2).mean(0)
                    mse = (target - pred).pow(2).mean(0)
                    test_median_r2s['hidden4'] = (1-(mse/total_var)).median(1)[0].cpu().numpy()
                    test_mses['hidden4'] = mse.mean(1).cpu().numpy()
                else:
                    for j in range(n_unit_samplings):
                        pred = test_pred_activations['hidden{}'.format(i+1)][:,j][:,:,live_units['hidden{}'.format(i+1)][j]]
                        target = test_target_activations['hidden{}'.format(i+1)][:,j][:,:,live_units['hidden{}'.format(i+1)][j]]
                        mean = target.mean(0, True)
                        total_var = (target - mean).pow(2).mean(0)
                        mse = (target - pred).pow(2).mean(0)
                        test_median_r2s['hidden{}'.format(i+1)][j] = (1-(mse/total_var)).median(1)[0].cpu().numpy()
                        test_mses['hidden{}'.format(i+1)][j] = mse.mean(1).cpu().numpy()


        # pcc, median_pccs
        if readout:
            pred = test_pred_activations['readout'][:,:,live_units['readout']]
            target = test_target_activations['readout'][:,:,live_units['readout']]
            pred_mean = pred.mean(0, True)
            target_mean = target.mean(0, True)
            pred_centered = pred - pred_mean
            target_centered = target - target_mean
            cov = (pred_centered*target_centered).mean(0)
            std_pred = pred_centered.pow(2).mean(0).pow(0.5)
            std_target = target_centered.pow(2).mean(0).pow(0.5)
            pccs = cov/(std_pred*std_target)
            test_median_pccs['readout'] = pccs.median(1)[0].cpu().numpy()            
        else:
            for i in range(4):
                if (i+1) == 4:
                    pred = test_pred_activations['hidden4'][:,:,live_units['hidden4']]
                    target = test_target_activations['hidden4'][:,:,live_units['hidden4']]
                    pred_mean = pred.mean(0, True)
                    target_mean = target.mean(0, True)
                    pred_centered = pred - pred_mean
                    target_centered = target - target_mean
                    cov = (pred_centered*target_centered).mean(0)
                    std_pred = pred_centered.pow(2).mean(0).pow(0.5)
                    std_target = target_centered.pow(2).mean(0).pow(0.5)
                    pccs = cov/(std_pred*std_target)
                    test_median_pccs['hidden4'] = pccs.median(1)[0].cpu().numpy()
                else:
                    for j in range(n_unit_samplings):
                        pred = test_pred_activations['hidden{}'.format(i+1)][:,j][:,:,live_units['hidden{}'.format(i+1)][j]]
                        target = test_target_activations['hidden{}'.format(i+1)][:,j][:,:,live_units['hidden{}'.format(i+1)][j]]
                        pred_mean = pred.mean(0, True)
                        target_mean = target.mean(0, True)
                        pred_centered = pred - pred_mean
                        target_centered = target - target_mean
                        cov = (pred_centered*target_centered).mean(0)
                        std_pred = pred_centered.pow(2).mean(0).pow(0.5)
                        std_target = target_centered.pow(2).mean(0).pow(0.5)
                        pccs = cov/(std_pred*std_target)
                        test_median_pccs['hidden{}'.format(i+1)][j] = pccs.median(1)[0].cpu().numpy()

        return test_mses, test_median_r2s, test_median_pccs, test_live_unit_fracs


    def FD_test_linear_regressor_helper(self, pose_type, no_occ_data, source_model, target_model, source_model_name, target_model_name, source_idx, target_idx, n_units, n_unit_samplings, total_var_thr, readout):
        '''
        Return: 

        test_r2s
        test_mses
        test_pccs
        test_median_pccs
        '''
        activations = {}
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook

        if readout:
            source_model.model[14].register_forward_hook(get_activation('source_readout'))
            target_model.model[14].register_forward_hook(get_activation('target_readout'))
        else:
            source_model.model[2].register_forward_hook(get_activation('source_hidden1'))
            target_model.model[2].register_forward_hook(get_activation('target_hidden1'))

            source_model.model[5].register_forward_hook(get_activation('source_hidden2'))
            target_model.model[5].register_forward_hook(get_activation('target_hidden2'))

            source_model.model[8].register_forward_hook(get_activation('source_hidden3'))
            target_model.model[8].register_forward_hook(get_activation('target_hidden3'))

            source_model.model[12].register_forward_hook(get_activation('source_hidden4'))
            target_model.model[12].register_forward_hook(get_activation('target_hidden4'))

        source_model.eval()
        target_model.eval()

        if no_occ_data:
            no_occ_data_name = 'NoOcc'
        else:
            no_occ_data_name = 'Occ'

        model_save_path = os.path.join(self.model_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name, 'source_idx{}_target_idx{}'.format(source_idx+1, target_idx+1))

        if readout:
            regressors = {}
            regressors['readout'] = nn.Linear(source_model.n_classes, target_model.n_classes)
            regressors['readout'].load_state_dict(torch.load(os.path.join(model_save_path, 'readout_best_val_loss.pth.tar'),
                                                             map_location=lambda storage, loc: storage))

            if self.gpu_mode:
                regressors['readout'].cuda()
        else:
            # Determine units to sample.
            sample_unit_path = os.path.join(self.sample_unit_dir, self.data_dir, pose_type, target_model_name, 'target_idx{}'.format(target_idx+1))
            if not os.path.exists(sample_unit_path):
                os.makedirs(sample_unit_path, exist_ok=True)

            if os.path.isfile(os.path.join(sample_unit_path, 'sample_indices.pt')):
                sample_indices = torch.load(os.path.join(sample_unit_path, 'sample_indices.pt'),
                                    map_location=lambda storage, loc: storage)
            else:
                C = source_model.hidden1
                H = source_model.input_size//2
                W = H
                sample_indices_hidden1 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

                C = source_model.hidden2
                H = source_model.input_size//4
                W = H
                sample_indices_hidden2 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

                C = source_model.hidden3
                H = source_model.input_size//8
                W = H
                sample_indices_hidden3 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

                sample_indices = torch.stack([sample_indices_hidden1, sample_indices_hidden2, sample_indices_hidden3], 0) # (n_layer, n_unit_samplings, n_units)
                torch.save(sample_indices, os.path.join(sample_unit_path, 'sample_indices.pt'))

            if self.gpu_mode:
                sample_indices = sample_indices.cuda()
            regressors = {}
            regressors['hidden1'] = []
            regressors['hidden2'] = []
            regressors['hidden3'] = []

            for i in range(n_unit_samplings):
                regressors['hidden1'].append(nn.Linear(source_model.hidden1*(source_model.input_size//2)**2, n_units))
                regressors['hidden2'].append(nn.Linear(source_model.hidden2*(source_model.input_size//4)**2, n_units))
                regressors['hidden3'].append(nn.Linear(source_model.hidden3*(source_model.input_size//8)**2, n_units))


                regressors['hidden1'][i].load_state_dict(torch.load(os.path.join(model_save_path, 'hidden1_sample{}_best_val_loss.pth.tar'.format(i+1)),
                                                             map_location=lambda storage, loc: storage))
                regressors['hidden2'][i].load_state_dict(torch.load(os.path.join(model_save_path, 'hidden2_sample{}_best_val_loss.pth.tar'.format(i+1)),
                                                             map_location=lambda storage, loc: storage))
                regressors['hidden3'][i].load_state_dict(torch.load(os.path.join(model_save_path, 'hidden3_sample{}_best_val_loss.pth.tar'.format(i+1)),
                                                             map_location=lambda storage, loc: storage))


                if self.gpu_mode:
                    regressors['hidden1'][i].cuda()
                    regressors['hidden2'][i].cuda()
                    regressors['hidden3'][i].cuda()

            regressors['hidden4'] = nn.Linear(source_model.hidden4, target_model.hidden4)
            regressors['hidden4'].load_state_dict(torch.load(os.path.join(model_save_path, 'hidden4_best_val_loss.pth.tar'),
                                                             map_location=lambda storage, loc: storage))

            if self.gpu_mode:
                regressors['hidden4'].cuda()

        if self.gpu_mode:
            source_model.cuda()
            target_model.cuda()

        test_mses = {}
        test_median_r2s = {}
        test_median_pccs = {}
        test_live_unit_fracs ={}
        if readout:
            test_mses['readout'] = 0.0
            test_median_r2s['readout'] = 0.0
            test_median_pccs['readout'] = 0.0
            test_live_unit_fracs['readout'] = 0.0
        else:
            for i in range(4):
                if (i+1) == 4:
                    test_mses['hidden4'] = 0.0
                    test_median_r2s['hidden4'] = 0.0
                    test_median_pccs['hidden4'] = 0.0
                    test_live_unit_fracs['hidden4'] = 0.0
                else:
                    test_mses['hidden{}'.format(i+1)] = np.zeros((n_unit_samplings,))
                    test_median_r2s['hidden{}'.format(i+1)] = np.zeros((n_unit_samplings,))
                    test_median_pccs['hidden{}'.format(i+1)] = np.zeros((n_unit_samplings,))
                    test_live_unit_fracs['hidden{}'.format(i+1)] = np.zeros((n_unit_samplings,))


        test_pred_activations = {}
        if readout:
            test_pred_activations['readout'] = torch.zeros(len(self.test_loader.dataset), target_model.n_classes)
            if self.gpu_mode:
                test_pred_activations['readout'] = test_pred_activations['readout'].cuda()
        else:
            for i in range(4):
                if (i+1) == 4:
                    test_pred_activations['hidden4'] = torch.zeros(len(self.test_loader.dataset), target_model.hidden4)
                else:
                    test_pred_activations['hidden{}'.format(i+1)] = torch.zeros(len(self.test_loader.dataset), n_unit_samplings, n_units)
                if self.gpu_mode:
                    test_pred_activations['hidden{}'.format(i+1)] = test_pred_activations['hidden{}'.format(i+1)].cuda()


        test_target_activations = {}
        if readout:
            test_target_activations['readout'] = torch.zeros(len(self.test_loader.dataset), target_model.n_classes)            
            if self.gpu_mode:
                test_target_activations['readout'] = test_target_activations['readout'].cuda()
        else:
            for i in range(4):
                if (i+1) == 4:
                    test_target_activations['hidden4'] = torch.zeros(len(self.test_loader.dataset), target_model.hidden4)
                else:
                    test_target_activations['hidden{}'.format(i+1)] = torch.zeros(len(self.test_loader.dataset), n_unit_samplings, n_units)
                if self.gpu_mode:
                    test_target_activations['hidden{}'.format(i+1)] = test_target_activations['hidden{}'.format(i+1)].cuda()


        iter_start_time = time.time()
        for iter, (data, ind_objs, ind_ps, _) in enumerate(self.test_loader):
            N, C, _, _ = data.size()
            data = Variable(data, volatile=True)
            ind_objs = Variable(ind_objs, volatile=True)
            ind_ps = Variable(ind_ps, volatile=True)
            if self.gpu_mode:
                data = data.cuda()
                ind_objs = ind_objs.cuda()
                ind_ps = ind_ps.cuda()

            if pose_type == 'gt':
                aff_ps = self.affine_form(ind_ps[:,1])
                grid = F.affine_grid(aff_ps, torch.Size([N, C, self.in_size, self.in_size]))

            if pose_type == 'gt':                
                if no_occ_data:
                    x = F.grid_sample(ind_objs[:,1], grid)
                else:
                    x = F.grid_sample(data, grid)
            elif pose_type == 'original':
                if no_occ_data:
                    x = ind_objs[:,1]
                else:
                    x = data

            _ = source_model(x)
            _ = target_model(x)

            if readout:
                regressor = regressors['readout']
                pred = regressor(activations['source_readout'])
                test_pred_activations['readout'][iter*N:(iter+1)*N] = pred.data
                test_target_activations['readout'][iter*N:(iter+1)*N] = activations['target_readout'].data                
            else:
                for i in range(4):
                    regressor = regressors['hidden{}'.format(i+1)]
                    if (i+1) == 4:
                        pred = regressor(activations['source_hidden4'])
                        test_pred_activations['hidden4'][iter*N:(iter+1)*N] = pred.data
                        test_target_activations['hidden4'][iter*N:(iter+1)*N] = activations['target_hidden4'].data
                    else:
                        for j in range(n_unit_samplings):
                            sampled_target_activations = self.sample_units(activations['target_hidden{}'.format(i+1)], sample_indices[i,j])
                            pred = regressor[j](self.flatten(activations['source_hidden{}'.format(i+1)]))
                            test_pred_activations['hidden{}'.format(i+1)][iter*N:(iter+1)*N, j] = pred.data
                            test_target_activations['hidden{}'.format(i+1)][iter*N:(iter+1)*N, j] = sampled_target_activations.data


        # identify live units.
        # live_units = {}
        # for i in range(4):
        #     if (i+1) == 4:
        #         live_units['hidden4'] = torch.nonzero((test_target_activations['hidden4'].abs().sum(0) != 0)).squeeze()
        #         test_live_unit_fracs['hidden4'] = len(live_units['hidden4'])/target_model.hidden4
        #     else:
        #         live_units['hidden{}'.format(i+1)] = []
        #         for j in range(n_unit_samplings):
        #             live_units['hidden{}'.format(i+1)].append(torch.nonzero(test_target_activations['hidden{}'.format(i+1)][:,j].abs().sum(0) != 0).squeeze())
        #             test_live_unit_fracs['hidden{}'.format(i+1)][j] = len(live_units['hidden{}'.format(i+1)][j])/n_units
        #     print(test_live_unit_fracs['hidden{}'.format(i+1)])
        live_units = {}
        if readout:
            target = test_target_activations['readout']
            mean = target.mean(0, True)
            total_var = (target - mean).pow(2).mean(0)
            live_units['readout'] = torch.nonzero((total_var > total_var_thr)).squeeze()
            test_live_unit_fracs['readout'] = len(live_units['readout'])/target_model.n_classes
        else:
            for i in range(4):
                if (i+1) == 4:
                    target = test_target_activations['hidden4']
                    mean = target.mean(0, True)
                    total_var = (target - mean).pow(2).mean(0)
                    live_units['hidden4'] = torch.nonzero((total_var > total_var_thr)).squeeze()
                    test_live_unit_fracs['hidden4'] = len(live_units['hidden4'])/target_model.hidden4
                else:
                    live_units['hidden{}'.format(i+1)] = []
                    for j in range(n_unit_samplings):
                        target = test_target_activations['hidden{}'.format(i+1)][:,j]
                        mean = target.mean(0, True)
                        total_var = (target - mean).pow(2).mean(0)
                        live_units['hidden{}'.format(i+1)].append(torch.nonzero(total_var > total_var_thr).squeeze())
                        test_live_unit_fracs['hidden{}'.format(i+1)][j] = len(live_units['hidden{}'.format(i+1)][j])/n_units


        # mse, median_r2
        if readout:
            pred = test_pred_activations['readout'][:,live_units['readout']]
            target = test_target_activations['readout'][:,live_units['readout']]
            mean = target.mean(0, True)
            total_var = (target - mean).pow(2).mean(0)
            mse = (target - pred).pow(2).mean(0)
            test_median_r2s['readout'] = (1-(mse/total_var)).median()
            test_mses['readout'] = mse.mean()
        else:
            for i in range(4):
                if (i+1) == 4:
                    pred = test_pred_activations['hidden4'][:,live_units['hidden4']]
                    target = test_target_activations['hidden4'][:,live_units['hidden4']]
                    mean = target.mean(0, True)
                    total_var = (target - mean).pow(2).mean(0)
                    mse = (target - pred).pow(2).mean(0)
                    test_median_r2s['hidden4'] = (1-(mse/total_var)).median()
                    test_mses['hidden4'] = mse.mean()
                else:
                    for j in range(n_unit_samplings):
                        pred = test_pred_activations['hidden{}'.format(i+1)][:,j][:,live_units['hidden{}'.format(i+1)][j]]
                        target = test_target_activations['hidden{}'.format(i+1)][:,j][:,live_units['hidden{}'.format(i+1)][j]]
                        mean = target.mean(0, True)
                        total_var = (target - mean).pow(2).mean(0)
                        mse = (target - pred).pow(2).mean(0)
                        test_median_r2s['hidden{}'.format(i+1)][j] = (1-(mse/total_var)).median()
                        test_mses['hidden{}'.format(i+1)][j] = mse.mean()



        # pcc, median_pccs
        if readout:
            pred = test_pred_activations['readout'][:,live_units['readout']]
            target = test_target_activations['readout'][:,live_units['readout']]
            pred_mean = pred.mean(0, True)
            target_mean = target.mean(0, True)
            pred_centered = pred - pred_mean
            target_centered = target - target_mean
            cov = (pred_centered*target_centered).mean(0)
            std_pred = pred_centered.pow(2).mean(0).pow(0.5)
            std_target = target_centered.pow(2).mean(0).pow(0.5)
            pccs = cov/(std_pred*std_target)
            test_median_pccs['readout'] = pccs.median() 
        else:
            for i in range(4):
                if (i+1) == 4:
                    pred = test_pred_activations['hidden4'][:,live_units['hidden4']]
                    target = test_target_activations['hidden4'][:,live_units['hidden4']]
                    pred_mean = pred.mean(0, True)
                    target_mean = target.mean(0, True)
                    pred_centered = pred - pred_mean
                    target_centered = target - target_mean
                    cov = (pred_centered*target_centered).mean(0)
                    std_pred = pred_centered.pow(2).mean(0).pow(0.5)
                    std_target = target_centered.pow(2).mean(0).pow(0.5)
                    pccs = cov/(std_pred*std_target)
                    test_median_pccs['hidden4'] = pccs.median() 
                else:
                    for j in range(n_unit_samplings):
                        pred = test_pred_activations['hidden{}'.format(i+1)][:,j][:,live_units['hidden{}'.format(i+1)][j]]
                        target = test_target_activations['hidden{}'.format(i+1)][:,j][:,live_units['hidden{}'.format(i+1)][j]]
                        pred_mean = pred.mean(0, True)
                        target_mean = target.mean(0, True)
                        pred_centered = pred - pred_mean
                        target_centered = target - target_mean
                        cov = (pred_centered*target_centered).mean(0)
                        std_pred = pred_centered.pow(2).mean(0).pow(0.5)
                        std_target = target_centered.pow(2).mean(0).pow(0.5)
                        pccs = cov/(std_pred*std_target)
                        test_median_pccs['hidden{}'.format(i+1)][j] = pccs.median()

        return test_mses, test_median_r2s, test_median_pccs, test_live_unit_fracs


    def determine_sample_units(self, pose_type, target_model_name, target_idx, n_units=256, n_unit_samplings=3):
        sample_unit_path = os.path.join(self.sample_unit_dir, self.data_dir, pose_type, target_model_name, 'target_idx{}'.format(target_idx+1))
        if not os.path.exists(sample_unit_path):
            os.makedirs(sample_unit_path, exist_ok=True)


        if os.path.isfile(os.path.join(sample_unit_path, 'sample_indices.pt')):
            print('')
            print('sample_indices.pt already exists.')
            print('')
        else:

            if pose_type =='gt':
                input_size = 28
            elif pose_type == 'original':
                input_size = 50

            if target_model_name in ['FDClassifier', 'NoOccFDClassifier']:
                target_model = FDClassifier(input_size=input_size)
            elif target_model_name in ['RCClassifier', 'NoOccRCClassifier']:
                target_model = RCClassifier(input_size=input_size)
            elif target_model_name in ['TDRCClassifier', 'NoOccTDRCClassifier']:
                target_model = TDRCClassifier(input_size=input_size)

            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123)

            C = target_model.hidden1
            H = target_model.input_size//2
            W = H
            sample_indices_hidden1 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

            C = target_model.hidden2
            H = target_model.input_size//4
            W = H
            sample_indices_hidden2 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

            C = target_model.hidden3
            H = target_model.input_size//8
            W = H
            sample_indices_hidden3 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

            sample_indices = torch.stack([sample_indices_hidden1, sample_indices_hidden2, sample_indices_hidden3], 0) # (n_layer, n_unit_samplings, n_units)
            torch.save(sample_indices, os.path.join(sample_unit_path, 'sample_indices.pt'))


    # all layers instead of doing readout and the other layers separately.
    def train_linear_regressor_all_layers(self, pose_type, no_occ_data, source_model_name, target_model_name, source_idx, target_idx):
        if pose_type =='gt':
            input_size = 28
        elif pose_type == 'original':
            input_size = 50

        if source_model_name in ['FDClassifier', 'NoOccFDClassifier']:
            source_model = FDClassifier(input_size=input_size)
        elif source_model_name in ['RCClassifier', 'NoOccRCClassifier']:
            source_model = RCClassifier(input_size=input_size, record_activations=True, record_readouts=True)
        elif source_model_name in ['TDRCClassifier', 'NoOccTDRCClassifier']:
            source_model = TDRCClassifier(input_size=input_size, record_activations=True, record_readouts=True)

                
        if target_model_name in ['FDClassifier', 'NoOccFDClassifier']:
            target_model = FDClassifier(input_size=input_size)
        elif target_model_name in ['RCClassifier', 'NoOccRCClassifier']:
            target_model = RCClassifier(input_size=input_size, record_activations=True, record_readouts=True)
        elif target_model_name in ['TDRCClassifier', 'NoOccTDRCClassifier']:
            target_model = TDRCClassifier(input_size=input_size, record_activations=True, record_readouts=True)


        source_model_save_path = os.path.join(os.path.join(self.classifier_dir, pose_type, source_model_name), 
                                    'trial_{}'.format(source_idx+1), source_model_name+'_best_val_acc.pth.tar')
        target_model_save_path = os.path.join(os.path.join(self.classifier_dir, pose_type, target_model_name), 
                                    'trial_{}'.format(target_idx+1), target_model_name+'_best_val_acc.pth.tar')

        source_model.load_state_dict(torch.load(source_model_save_path, map_location=lambda storage, loc: storage))
        target_model.load_state_dict(torch.load(target_model_save_path, map_location=lambda storage, loc: storage))

        if source_model_name in ['FDClassifier', 'NoOccFDClassifier']:
            self.FD_train_linear_regressor_helper_all_layers(pose_type, no_occ_data, source_model, target_model, source_model_name, target_model_name, source_idx, target_idx)
        elif source_model_name in ['RCClassifier', 'NoOccRCClassifier', 'TDRCClassifier', 'NoOccTDRCClassifier']:
            self.RC_train_linear_regressor_helper_all_layers(pose_type, no_occ_data, source_model, target_model, source_model_name, target_model_name, source_idx, target_idx)


    def RC_train_linear_regressor_helper_all_layers(self, pose_type, no_occ_data, source_model, target_model, source_model_name, target_model_name, source_idx, target_idx, n_units=256, n_unit_samplings=3):
        if no_occ_data:
            no_occ_data_name = 'NoOcc'
        else:
            no_occ_data_name = 'Occ'
        result_save_path = os.path.join(self.result_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name, 'source_idx{}_target_idx{}'.format(source_idx+1, target_idx+1))
        model_save_path = os.path.join(self.model_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name, 'source_idx{}_target_idx{}'.format(source_idx+1, target_idx+1))
        if not os.path.exists(result_save_path):
            os.makedirs(result_save_path, exist_ok=True)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)

        hist = {}

        hist['readout'] = {}
        hist['readout']['train_loss'] = []
        hist['readout']['train_R2'] = []
        hist['readout']['val_loss'] = []
        hist['readout']['val_R2'] = []

        for i in range(4):
            hist['hidden{}'.format(i+1)] = {}
            if (i+1) == 4:
                hist['hidden{}'.format(i+1)]['train_loss'] = []
                hist['hidden{}'.format(i+1)]['train_R2'] = []
                hist['hidden{}'.format(i+1)]['val_loss'] = []
                hist['hidden{}'.format(i+1)]['val_R2'] = []
            else:
                for j in range(n_unit_samplings):
                    hist['hidden{}'.format(i+1)][j] = {}
                    hist['hidden{}'.format(i+1)][j]['train_loss'] = []
                    hist['hidden{}'.format(i+1)][j]['train_R2'] = []
                    hist['hidden{}'.format(i+1)][j]['val_loss'] = []
                    hist['hidden{}'.format(i+1)][j]['val_R2'] = []
        hist['per_epoch_time'] = []

        source_model.eval()
        target_model.eval()

        # Readout
        regressors = {}
        regressors['readout'] = nn.Linear(source_model.n_classes, target_model.n_classes)
        optimizers = {}
        optimizers['readout'] = optim.Adam(regressors['readout'].parameters(), lr=self.lr)
        if self.gpu_mode:
            regressors['readout'].cuda()

        # Non-readoout layers
        # Determine units to sample.
        sample_unit_path = os.path.join(self.sample_unit_dir, self.data_dir, pose_type, target_model_name, 'target_idx{}'.format(target_idx+1))
        if not os.path.exists(sample_unit_path):
            os.makedirs(sample_unit_path, exist_ok=True)

        if os.path.isfile(os.path.join(sample_unit_path, 'sample_indices.pt')):
            sample_indices = torch.load(os.path.join(sample_unit_path, 'sample_indices.pt'),
                                map_location=lambda storage, loc: storage)
        else:
            C = source_model.hidden1
            H = source_model.input_size//2
            W = H
            sample_indices_hidden1 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

            C = source_model.hidden2
            H = source_model.input_size//4
            W = H
            sample_indices_hidden2 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

            C = source_model.hidden3
            H = source_model.input_size//8
            W = H
            sample_indices_hidden3 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

            sample_indices = torch.stack([sample_indices_hidden1, sample_indices_hidden2, sample_indices_hidden3], 0) # (n_layer, n_unit_samplings, n_units)
            torch.save(sample_indices, os.path.join(sample_unit_path, 'sample_indices.pt'))

        if self.gpu_mode:
            sample_indices = sample_indices.cuda()

        regressors['hidden1'] = []
        regressors['hidden2'] = []
        regressors['hidden3'] = []

        optimizers['hidden1'] = []
        optimizers['hidden2'] = []
        optimizers['hidden3'] = []

        for i in range(n_unit_samplings):
            regressors['hidden1'].append(nn.Linear(source_model.hidden1*(source_model.input_size//2)**2, n_units))
            regressors['hidden2'].append(nn.Linear(source_model.hidden2*(source_model.input_size//4)**2, n_units))
            regressors['hidden3'].append(nn.Linear(source_model.hidden3*(source_model.input_size//8)**2, n_units))

            optimizers['hidden1'].append(optim.Adam(regressors['hidden1'][-1].parameters(), lr=self.lr))
            optimizers['hidden2'].append(optim.Adam(regressors['hidden2'][-1].parameters(), lr=self.lr))
            optimizers['hidden3'].append(optim.Adam(regressors['hidden3'][-1].parameters(), lr=self.lr))

            if self.gpu_mode:
                regressors['hidden1'][-1].cuda()
                regressors['hidden2'][-1].cuda()
                regressors['hidden3'][-1].cuda()

        regressors['hidden4'] = nn.Linear(source_model.hidden4, target_model.hidden4)
        optimizers['hidden4'] = optim.Adam(regressors['hidden4'].parameters(), lr=self.lr)

        if self.gpu_mode:
            regressors['hidden4'].cuda()


        if self.gpu_mode:
            source_model.cuda()
            target_model.cuda()

        reg_loss = nn.MSELoss()

        print('training start!')
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            # Train the main model.
            print('')
            print('Train the linear regressor.')
            print('')
            iter_start_time = time.time()
            for iter, (data, ind_objs, ind_ps, _) in enumerate(self.train_loader):
                N, C, _, _ = data.size()
                data = Variable(data)
                ind_objs = Variable(ind_objs)
                ind_ps = Variable(ind_ps)
                if self.gpu_mode:
                    data = data.cuda()
                    ind_objs = ind_objs.cuda()
                    ind_ps = ind_ps.cuda()

                if pose_type == 'gt':
                    aff_ps = self.affine_form(ind_ps[:,1])
                    grid = F.affine_grid(aff_ps, torch.Size([N, C, self.in_size, self.in_size]))

                if pose_type == 'gt':                
                    if no_occ_data:
                        x = F.grid_sample(ind_objs[:,1], grid)
                    else:
                        x = F.grid_sample(data, grid)
                elif pose_type == 'original':
                    if no_occ_data:
                        x = ind_objs[:,1]
                    else:
                        x = data

                _ = source_model(x)
                _ = target_model(x)

                train_R2s = []

                regressor = regressors['readout']
                optimizer = optimizers['readout']
                optimizer.zero_grad()

                train_losses_iter = []
                train_R2s_iter = []
                for t in range(source_model.n_iter):
                    pred = regressor(source_model.readouts[t])
                    train_loss = reg_loss(pred, target_model.readouts[t])
                    train_losses_iter.append(train_loss)
                    mean = target_model.readouts[t].data.mean(0, True)
                    total_var = (target_model.readouts[t].data-mean).pow(2).mean()
                    train_R2 = 1 - train_loss.data[0]/total_var
                    train_R2s_iter.append(train_R2)

                train_loss = torch.stack(train_losses_iter, 0).mean()
                train_loss.backward()
                optimizer.step()

                train_R2 = np.array(train_R2s_iter).mean()

                hist['readout']['train_loss'].append(train_loss.data[0])
                hist['readout']['train_R2'].append(train_R2)
                train_R2s.append(train_R2)

                for i in range(4):
                    regressor = regressors['hidden{}'.format(i+1)]
                    optimizer = optimizers['hidden{}'.format(i+1)]
                    if (i+1) == 4:
                        optimizer.zero_grad()

                        train_losses_iter = []
                        train_R2s_iter = []
                        for t in range(source_model.n_iter):
                            pred = regressor(source_model.activations_hidden4[t])
                            train_loss = reg_loss(pred, target_model.activations_hidden4[t])
                            train_losses_iter.append(train_loss)
                            mean = target_model.activations_hidden4[t].data.mean(0, True)
                            total_var = (target_model.activations_hidden4[t].data-mean).pow(2).mean()
                            train_R2 = 1 - train_loss.data[0]/total_var
                            train_R2s_iter.append(train_R2)


                        train_loss = torch.stack(train_losses_iter, 0).mean()
                        train_loss.backward()
                        optimizer.step()

                        train_R2 = np.array(train_R2s_iter).mean()

                        hist['hidden4']['train_loss'].append(train_loss.data[0])
                        hist['hidden4']['train_R2'].append(train_R2)
                        train_R2s.append(train_R2)
                    else:
                        for j in range(n_unit_samplings):
                            optimizer[j].zero_grad()

                            train_losses_iter = []
                            train_R2s_iter = []
                            for t in range(source_model.n_iter):
                                sampled_target_activations = self.sample_units(getattr(target_model, 'activations_hidden{}'.format(i+1))[t], sample_indices[i,j])
                                pred = regressor[j](self.flatten(getattr(source_model, 'activations_hidden{}'.format(i+1))[t]))
                                train_loss = reg_loss(pred, sampled_target_activations)
                                train_losses_iter.append(train_loss)
                                mean = sampled_target_activations.data.mean(0, True)
                                total_var = (sampled_target_activations.data-mean).pow(2).mean()
                                train_R2 = 1 - train_loss.data[0]/total_var
                                train_R2s_iter.append(train_R2)


                            train_loss = torch.stack(train_losses_iter, 0).mean()
                            train_loss.backward()
                            optimizer[j].step()

                            train_R2 = np.array(train_R2s_iter).mean()

                            hist['hidden{}'.format(i+1)][j]['train_loss'].append(train_loss.data[0])
                            hist['hidden{}'.format(i+1)][j]['train_R2'].append(train_R2)
                            train_R2s.append(train_R2)

                mean_train_R2 = np.array(train_R2s).mean()
                std_train_R2 = np.array(train_R2s).std()

                if (iter + 1) % 100 == 0:
                    iter_time = time.time() - iter_start_time
                    iter_start_time = time.time()
                    print("Epoch {}: [{}/{}] mean_train_R2: {:.3f} std_train_R2: {:.3f} ({:.3f} sec)".format((epoch + 1), (iter + 1), 
                            len(self.train_loader), mean_train_R2, std_train_R2, iter_time))


            # Validation of the main model and z_classifier.
            print('')
            print('Validation begins.')
            print('')

            val_sums = {}

            val_sums['readout'] = {}
            val_sums['readout']['val_loss'] = 0
            val_sums['readout']['val_R2'] = 0

            for i in range(4):
                val_sums['hidden{}'.format(i+1)] = {}
                if (i+1) == 4:
                    val_sums['hidden{}'.format(i+1)]['val_loss'] = 0
                    val_sums['hidden{}'.format(i+1)]['val_R2'] = 0
                else:
                    for j in range(n_unit_samplings):
                        val_sums['hidden{}'.format(i+1)][j] = {}
                        val_sums['hidden{}'.format(i+1)][j]['val_loss'] = 0
                        val_sums['hidden{}'.format(i+1)][j]['val_R2'] = 0


            iter_start_time = time.time()
            for iter, (data, ind_objs, ind_ps, _) in enumerate(self.val_loader):
                N, C, _, _ = data.size()
                data = Variable(data, volatile=True)
                ind_objs = Variable(ind_objs, volatile=True)
                ind_ps = Variable(ind_ps, volatile=True)
                if self.gpu_mode:
                    data = data.cuda()
                    ind_objs = ind_objs.cuda()
                    ind_ps = ind_ps.cuda()

                if pose_type == 'gt':
                    aff_ps = self.affine_form(ind_ps[:,1])
                    grid = F.affine_grid(aff_ps, torch.Size([N, C, self.in_size, self.in_size]))

                if pose_type == 'gt':                
                    if no_occ_data:
                        x = F.grid_sample(ind_objs[:,1], grid)
                    else:
                        x = F.grid_sample(data, grid)
                elif pose_type == 'original':
                    if no_occ_data:
                        x = ind_objs[:,1]
                    else:
                        x = data

                _ = source_model(x)
                _ = target_model(x)

                regressor = regressors['readout']
                optimizer = optimizers['readout']

                val_losses_iter = []
                val_R2s_iter = []
                for t in range(source_model.n_iter):
                    pred = regressor(source_model.readouts[t])
                    val_loss = reg_loss(pred, target_model.readouts[t])
                    val_losses_iter.append(val_loss)
                    mean = target_model.readouts[t].data.mean(0, True)
                    total_var = (target_model.readouts[t].data-mean).pow(2).mean()
                    val_R2 = 1 - val_loss.data[0]/total_var
                    val_R2s_iter.append(val_R2)

                val_loss = torch.stack(val_losses_iter, 0).mean()
                val_sums['readout']['val_loss'] += val_loss.data[0]

                val_R2 = np.array(val_R2s_iter).mean()
                val_sums['readout']['val_R2'] += val_R2

                for i in range(4):
                    regressor = regressors['hidden{}'.format(i+1)]
                    optimizer = optimizers['hidden{}'.format(i+1)]
                    if (i+1) == 4:

                        val_losses_iter = []
                        val_R2s_iter = []
                        for t in range(source_model.n_iter):
                            pred = regressor(source_model.activations_hidden4[t])
                            val_loss = reg_loss(pred, target_model.activations_hidden4[t])
                            val_losses_iter.append(val_loss)
                            mean = target_model.activations_hidden4[t].data.mean(0, True)
                            total_var = (target_model.activations_hidden4[t].data-mean).pow(2).mean()
                            val_R2 = 1 - val_loss.data[0]/total_var
                            val_R2s_iter.append(val_R2)

                        val_loss = torch.stack(val_losses_iter, 0).mean()
                        val_sums['hidden4']['val_loss'] += val_loss.data[0]

                        val_R2 = np.array(val_R2s_iter).mean()
                        val_sums['hidden4']['val_R2'] += val_R2
                    else:
                        for j in range(n_unit_samplings):
                            val_losses_iter = []
                            val_R2s_iter = []
                            for t in range(source_model.n_iter):
                                sampled_target_activations = self.sample_units(getattr(target_model, 'activations_hidden{}'.format(i+1))[t], sample_indices[i,j])
                                pred = regressor[j](self.flatten(getattr(source_model, 'activations_hidden{}'.format(i+1))[t]))
                                val_loss = reg_loss(pred, sampled_target_activations)
                                val_losses_iter.append(val_loss)
                                mean = sampled_target_activations.data.mean(0, True)
                                total_var = (sampled_target_activations.data-mean).pow(2).mean()
                                val_R2 = 1 - val_loss.data[0]/total_var
                                val_R2s_iter.append(val_R2)


                            val_loss = torch.stack(val_losses_iter, 0).mean()
                            val_sums['hidden{}'.format(i+1)][j]['val_loss'] += val_loss.data[0]

                            val_R2 = np.array(val_R2s_iter).mean()
                            val_sums['hidden{}'.format(i+1)][j]['val_R2'] += val_R2

            val_R2s = []

            val_sums['readout']['val_loss'] /= len(self.val_loader)
            val_sums['readout']['val_R2'] /= len(self.val_loader)
            hist['readout']['val_loss'].append(val_sums['readout']['val_loss'])
            hist['readout']['val_R2'].append(val_sums['readout']['val_R2'])
            val_R2s.append(val_sums['readout']['val_R2'])                

            for i in range(4):
                if (i+1) == 4:
                    val_sums['hidden{}'.format(i+1)]['val_loss'] /= len(self.val_loader)
                    val_sums['hidden{}'.format(i+1)]['val_R2'] /= len(self.val_loader)
                    hist['hidden{}'.format(i+1)]['val_loss'].append(val_sums['hidden{}'.format(i+1)]['val_loss'])
                    hist['hidden{}'.format(i+1)]['val_R2'].append(val_sums['hidden{}'.format(i+1)]['val_R2'])
                    val_R2s.append(val_sums['hidden{}'.format(i+1)]['val_R2'])
                else:
                    for j in range(n_unit_samplings):
                        val_sums['hidden{}'.format(i+1)][j]['val_loss'] /= len(self.val_loader)
                        val_sums['hidden{}'.format(i+1)][j]['val_R2'] /= len(self.val_loader)
                        hist['hidden{}'.format(i+1)][j]['val_loss'].append(val_sums['hidden{}'.format(i+1)][j]['val_loss'])
                        hist['hidden{}'.format(i+1)][j]['val_R2'].append(val_sums['hidden{}'.format(i+1)][j]['val_R2'])
                        val_R2s.append(val_sums['hidden{}'.format(i+1)][j]['val_R2'])

            mean_val_R2 = np.array(val_R2s).mean()
            std_val_R2 = np.array(val_R2s).std()

            print("Epoch {}: mean_val_R2: {:.3f} std_val_R2s: {:.3f}".format((epoch + 1), mean_val_R2, std_val_R2))


            hist['per_epoch_time'].append(time.time() - epoch_start_time)
            print('per_epoch_time: {:.6f}'.format(hist['per_epoch_time'][-1]))


            if hist['readout']['val_loss'][-1] < min(hist['readout']['val_loss'][:-1] + [float('inf')]):
                self.save_model(regressors['readout'].state_dict(), model_save_path, 'readout_best_val_loss.pth.tar')
                self.save_as_pkl(hist['readout'], model_save_path, 'readout_best_val_loss_history.pkl') 

            for i in range(4):
                if (i+1) == 4:
                    if hist['hidden4']['val_loss'][-1] < min(hist['hidden4']['val_loss'][:-1] + [float('inf')]):
                        self.save_model(regressors['hidden4'].state_dict(), model_save_path, 'hidden4_best_val_loss.pth.tar')
                        self.save_as_pkl(hist['hidden4'], model_save_path, 'hidden4_best_val_loss_history.pkl') 
                else:
                    for j in range(n_unit_samplings):
                        if hist['hidden{}'.format(i+1)][j]['val_loss'][-1] < min(hist['hidden{}'.format(i+1)][j]['val_loss'][:-1] + [float('inf')]):
                            self.save_model(regressors['hidden{}'.format(i+1)][j].state_dict(), model_save_path, 'hidden{}_sample{}_best_val_loss.pth.tar'.format(i+1, j+1))
                            self.save_as_pkl(hist['hidden{}'.format(i+1)][j], model_save_path, 'hidden{}_sample{}_best_val_loss_history.pkl'.format(i+1, j+1)) 




    def FD_train_linear_regressor_helper_all_layers(self, pose_type, no_occ_data, source_model, target_model, source_model_name, target_model_name, source_idx, target_idx, n_units=256, n_unit_samplings=3):
        if no_occ_data:
            no_occ_data_name = 'NoOcc'
        else:
            no_occ_data_name = 'Occ'
        result_save_path = os.path.join(self.result_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name, 'source_idx{}_target_idx{}'.format(source_idx+1, target_idx+1))
        model_save_path = os.path.join(self.model_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name, 'source_idx{}_target_idx{}'.format(source_idx+1, target_idx+1))
        if not os.path.exists(result_save_path):
            os.makedirs(result_save_path, exist_ok=True)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)

        hist = {}

        hist['readout'] = {}
        hist['readout']['train_loss'] = []
        hist['readout']['train_R2'] = []
        hist['readout']['val_loss'] = []
        hist['readout']['val_R2'] = []

        for i in range(4):
            hist['hidden{}'.format(i+1)] = {}
            if (i+1) == 4:
                hist['hidden{}'.format(i+1)]['train_loss'] = []
                hist['hidden{}'.format(i+1)]['train_R2'] = []
                hist['hidden{}'.format(i+1)]['val_loss'] = []
                hist['hidden{}'.format(i+1)]['val_R2'] = []
            else:
                for j in range(n_unit_samplings):
                    hist['hidden{}'.format(i+1)][j] = {}
                    hist['hidden{}'.format(i+1)][j]['train_loss'] = []
                    hist['hidden{}'.format(i+1)][j]['train_R2'] = []
                    hist['hidden{}'.format(i+1)][j]['val_loss'] = []
                    hist['hidden{}'.format(i+1)][j]['val_R2'] = []

        hist['per_epoch_time'] = []

        activations = {}
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook

        source_model.model[14].register_forward_hook(get_activation('source_readout'))
        target_model.model[14].register_forward_hook(get_activation('target_readout'))

        source_model.model[2].register_forward_hook(get_activation('source_hidden1'))
        target_model.model[2].register_forward_hook(get_activation('target_hidden1'))

        source_model.model[5].register_forward_hook(get_activation('source_hidden2'))
        target_model.model[5].register_forward_hook(get_activation('target_hidden2'))

        source_model.model[8].register_forward_hook(get_activation('source_hidden3'))
        target_model.model[8].register_forward_hook(get_activation('target_hidden3'))

        source_model.model[12].register_forward_hook(get_activation('source_hidden4'))
        target_model.model[12].register_forward_hook(get_activation('target_hidden4'))

        source_model.eval()
        target_model.eval()

        # Readout
        regressors = {}
        regressors['readout'] = nn.Linear(source_model.n_classes, target_model.n_classes)
        optimizers = {}
        optimizers['readout'] = optim.Adam(regressors['readout'].parameters(), lr=self.lr)
        if self.gpu_mode:
            regressors['readout'].cuda()

        # Non-readout layers
        # Determine units to sample.
        sample_unit_path = os.path.join(self.sample_unit_dir, self.data_dir, pose_type, target_model_name, 'target_idx{}'.format(target_idx+1))
        if not os.path.exists(sample_unit_path):
            os.makedirs(sample_unit_path, exist_ok=True)

        if os.path.isfile(os.path.join(sample_unit_path, 'sample_indices.pt')):
            sample_indices = torch.load(os.path.join(sample_unit_path, 'sample_indices.pt'),
                                map_location=lambda storage, loc: storage)
        else:
            C = source_model.hidden1
            H = source_model.input_size//2
            W = H
            sample_indices_hidden1 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

            C = source_model.hidden2
            H = source_model.input_size//4
            W = H
            sample_indices_hidden2 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

            C = source_model.hidden3
            H = source_model.input_size//8
            W = H
            sample_indices_hidden3 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

            sample_indices = torch.stack([sample_indices_hidden1, sample_indices_hidden2, sample_indices_hidden3], 0) # (n_layer, n_unit_samplings, n_units)
            torch.save(sample_indices, os.path.join(sample_unit_path, 'sample_indices.pt'))

        if self.gpu_mode:
            sample_indices = sample_indices.cuda()

        regressors['hidden1'] = []
        regressors['hidden2'] = []
        regressors['hidden3'] = []

        optimizers['hidden1'] = []
        optimizers['hidden2'] = []
        optimizers['hidden3'] = []

        for i in range(n_unit_samplings):
            regressors['hidden1'].append(nn.Linear(source_model.hidden1*(source_model.input_size//2)**2, n_units))
            regressors['hidden2'].append(nn.Linear(source_model.hidden2*(source_model.input_size//4)**2, n_units))
            regressors['hidden3'].append(nn.Linear(source_model.hidden3*(source_model.input_size//8)**2, n_units))

            optimizers['hidden1'].append(optim.Adam(regressors['hidden1'][-1].parameters(), lr=self.lr))
            optimizers['hidden2'].append(optim.Adam(regressors['hidden2'][-1].parameters(), lr=self.lr))
            optimizers['hidden3'].append(optim.Adam(regressors['hidden3'][-1].parameters(), lr=self.lr))

            if self.gpu_mode:
                regressors['hidden1'][-1].cuda()
                regressors['hidden2'][-1].cuda()
                regressors['hidden3'][-1].cuda()

        regressors['hidden4'] = nn.Linear(source_model.hidden4, target_model.hidden4)
        optimizers['hidden4'] = optim.Adam(regressors['hidden4'].parameters(), lr=self.lr)

        if self.gpu_mode:
            regressors['hidden4'].cuda()


        if self.gpu_mode:
            source_model.cuda()
            target_model.cuda()

        reg_loss = nn.MSELoss()

        print('training start!')
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            # Train the main model.
            print('')
            print('Train the linear regressor.')
            print('')
            iter_start_time = time.time()
            for iter, (data, ind_objs, ind_ps, _) in enumerate(self.train_loader):
                N, C, _, _ = data.size()
                data = Variable(data)
                ind_objs = Variable(ind_objs)
                ind_ps = Variable(ind_ps)
                if self.gpu_mode:
                    data = data.cuda()
                    ind_objs = ind_objs.cuda()
                    ind_ps = ind_ps.cuda()

                if pose_type == 'gt':
                    aff_ps = self.affine_form(ind_ps[:,1])
                    grid = F.affine_grid(aff_ps, torch.Size([N, C, self.in_size, self.in_size]))

                if pose_type == 'gt':                
                    if no_occ_data:
                        x = F.grid_sample(ind_objs[:,1], grid)
                    else:
                        x = F.grid_sample(data, grid)
                elif pose_type == 'original':
                    if no_occ_data:
                        x = ind_objs[:,1]
                    else:
                        x = data

                _ = source_model(x)
                _ = target_model(x)

                train_R2s = []

                regressor = regressors['readout']
                optimizer = optimizers['readout']
                optimizer.zero_grad()
                pred = regressor(activations['source_readout'])
                train_loss = reg_loss(pred, activations['target_readout'])

                train_loss.backward()
                optimizer.step()

                hist['readout']['train_loss'].append(train_loss.data[0])

                mean = activations['target_readout'].data.mean(0, True)
                total_var = (activations['target_readout'].data-mean).pow(2).mean()
                train_R2 = 1 - train_loss.data[0]/total_var
                hist['readout']['train_R2'].append(train_R2)
                train_R2s.append(train_R2)

                for i in range(4):
                    regressor = regressors['hidden{}'.format(i+1)]
                    optimizer = optimizers['hidden{}'.format(i+1)]
                    if (i+1) == 4:
                        optimizer.zero_grad()

                        pred = regressor(activations['source_hidden4'])
                        train_loss = reg_loss(pred, activations['target_hidden4'])

                        train_loss.backward()
                        optimizer.step()

                        hist['hidden4']['train_loss'].append(train_loss.data[0])

                        mean = activations['target_hidden4'].data.mean(0, True)
                        total_var = (activations['target_hidden4'].data-mean).pow(2).mean()
                        train_R2 = 1 - train_loss.data[0]/total_var
                        hist['hidden4']['train_R2'].append(train_R2)
                        train_R2s.append(train_R2)
                    else:
                        for j in range(n_unit_samplings):
                            optimizer[j].zero_grad()

                            sampled_target_activations = self.sample_units(activations['target_hidden{}'.format(i+1)], sample_indices[i,j])
                            pred = regressor[j](self.flatten(activations['source_hidden{}'.format(i+1)]))
                            train_loss = reg_loss(pred, sampled_target_activations)

                            train_loss.backward()
                            optimizer[j].step()

                            hist['hidden{}'.format(i+1)][j]['train_loss'].append(train_loss.data[0])
                            
                            mean = sampled_target_activations.data.mean(0, True)
                            total_var = (sampled_target_activations.data-mean).pow(2).mean()
                            train_R2 = 1 - train_loss.data[0]/total_var 
                            hist['hidden{}'.format(i+1)][j]['train_R2'].append(train_R2)
                            train_R2s.append(train_R2)

                mean_train_R2 = np.array(train_R2s).mean()
                std_train_R2 = np.array(train_R2s).std()

                if (iter + 1) % 100 == 0:
                    iter_time = time.time() - iter_start_time
                    iter_start_time = time.time()
                    print("Epoch {}: [{}/{}] mean_train_R2: {:.3f} std_train_R2: {:.3f} ({:.3f} sec)".format((epoch + 1), (iter + 1), 
                            len(self.train_loader), mean_train_R2, std_train_R2, iter_time))


            # Validation of the main model and z_classifier.
            print('')
            print('Validation begins.')
            print('')

            val_sums = {}

            val_sums['readout'] = {}
            val_sums['readout']['val_loss'] = 0
            val_sums['readout']['val_R2'] = 0

            for i in range(4):
                val_sums['hidden{}'.format(i+1)] = {}
                if (i+1) == 4:
                    val_sums['hidden{}'.format(i+1)]['val_loss'] = 0
                    val_sums['hidden{}'.format(i+1)]['val_R2'] = 0
                else:
                    for j in range(n_unit_samplings):
                        val_sums['hidden{}'.format(i+1)][j] = {}
                        val_sums['hidden{}'.format(i+1)][j]['val_loss'] = 0
                        val_sums['hidden{}'.format(i+1)][j]['val_R2'] = 0


            iter_start_time = time.time()
            for iter, (data, ind_objs, ind_ps, _) in enumerate(self.val_loader):
                N, C, _, _ = data.size()
                data = Variable(data, volatile=True)
                ind_objs = Variable(ind_objs, volatile=True)
                ind_ps = Variable(ind_ps, volatile=True)
                if self.gpu_mode:
                    data = data.cuda()
                    ind_objs = ind_objs.cuda()
                    ind_ps = ind_ps.cuda()

                if pose_type == 'gt':
                    aff_ps = self.affine_form(ind_ps[:,1])
                    grid = F.affine_grid(aff_ps, torch.Size([N, C, self.in_size, self.in_size]))

                if pose_type == 'gt':                
                    if no_occ_data:
                        x = F.grid_sample(ind_objs[:,1], grid)
                    else:
                        x = F.grid_sample(data, grid)
                elif pose_type == 'original':
                    if no_occ_data:
                        x = ind_objs[:,1]
                    else:
                        x = data

                _ = source_model(x)
                _ = target_model(x)

                regressor = regressors['readout']
                optimizer = optimizers['readout']
                pred = regressor(activations['source_readout'])
                val_loss = reg_loss(pred, activations['target_readout'])

                val_sums['readout']['val_loss'] += val_loss.data[0]

                mean = activations['target_readout'].data.mean(0, True)
                total_var = (activations['target_readout'].data-mean).pow(2).mean()
                val_R2 = 1 - val_loss.data[0]/total_var
                val_sums['readout']['val_R2'] += val_R2

                for i in range(4):
                    regressor = regressors['hidden{}'.format(i+1)]
                    optimizer = optimizers['hidden{}'.format(i+1)]
                    if (i+1) == 4:
                        pred = regressor(activations['source_hidden4'])
                        val_loss = reg_loss(pred, activations['target_hidden4'])

                        val_sums['hidden4']['val_loss'] += val_loss.data[0]

                        mean = activations['target_hidden4'].data.mean(0, True)
                        total_var = (activations['target_hidden4'].data-mean).pow(2).mean()
                        val_R2 = 1 - val_loss.data[0]/total_var
                        val_sums['hidden4']['val_R2'] += val_R2
                    else:
                        for j in range(n_unit_samplings):
                            sampled_target_activations = self.sample_units(activations['target_hidden{}'.format(i+1)], sample_indices[i,j])
                            pred = regressor[j](self.flatten(activations['source_hidden{}'.format(i+1)]))
                            val_loss = reg_loss(pred, sampled_target_activations)

                            val_sums['hidden{}'.format(i+1)][j]['val_loss'] += val_loss.data[0]
                            
                            mean = sampled_target_activations.data.mean(0, True)
                            total_var = (sampled_target_activations.data-mean).pow(2).mean()
                            val_R2 = 1 - val_loss.data[0]/total_var 
                            val_sums['hidden{}'.format(i+1)][j]['val_R2'] += val_R2

            val_R2s = []

            val_sums['readout']['val_loss'] /= len(self.val_loader)
            val_sums['readout']['val_R2'] /= len(self.val_loader)
            hist['readout']['val_loss'].append(val_sums['readout']['val_loss'])
            hist['readout']['val_R2'].append(val_sums['readout']['val_R2'])
            val_R2s.append(val_sums['readout']['val_R2'])                

            for i in range(4):
                if (i+1) == 4:
                    val_sums['hidden{}'.format(i+1)]['val_loss'] /= len(self.val_loader)
                    val_sums['hidden{}'.format(i+1)]['val_R2'] /= len(self.val_loader)
                    hist['hidden{}'.format(i+1)]['val_loss'].append(val_sums['hidden{}'.format(i+1)]['val_loss'])
                    hist['hidden{}'.format(i+1)]['val_R2'].append(val_sums['hidden{}'.format(i+1)]['val_R2'])
                    val_R2s.append(val_sums['hidden{}'.format(i+1)]['val_R2'])
                else:
                    for j in range(n_unit_samplings):
                        val_sums['hidden{}'.format(i+1)][j]['val_loss'] /= len(self.val_loader)
                        val_sums['hidden{}'.format(i+1)][j]['val_R2'] /= len(self.val_loader)
                        hist['hidden{}'.format(i+1)][j]['val_loss'].append(val_sums['hidden{}'.format(i+1)][j]['val_loss'])
                        hist['hidden{}'.format(i+1)][j]['val_R2'].append(val_sums['hidden{}'.format(i+1)][j]['val_R2'])
                        val_R2s.append(val_sums['hidden{}'.format(i+1)][j]['val_R2'])

            mean_val_R2 = np.array(val_R2s).mean()
            std_val_R2 = np.array(val_R2s).std()

            print("Epoch {}: mean_val_R2: {:.3f} std_val_R2s: {:.3f}".format((epoch + 1), mean_val_R2, std_val_R2))


            hist['per_epoch_time'].append(time.time() - epoch_start_time)
            print('per_epoch_time: {:.6f}'.format(hist['per_epoch_time'][-1]))

            self.visualize_results(hist, result_save_path, n_unit_samplings, readout)

            if hist['readout']['val_loss'][-1] < min(hist['readout']['val_loss'][:-1] + [float('inf')]):
                self.save_model(regressors['readout'].state_dict(), model_save_path, 'readout_best_val_loss.pth.tar')
                self.save_as_pkl(hist['readout'], model_save_path, 'readout_best_val_loss_history.pkl') 

            for i in range(4):
                if (i+1) == 4:
                    if hist['hidden4']['val_loss'][-1] < min(hist['hidden4']['val_loss'][:-1] + [float('inf')]):
                        self.save_model(regressors['hidden4'].state_dict(), model_save_path, 'hidden4_best_val_loss.pth.tar')
                        self.save_as_pkl(hist['hidden4'], model_save_path, 'hidden4_best_val_loss_history.pkl') 
                else:
                    for j in range(n_unit_samplings):
                        if hist['hidden{}'.format(i+1)][j]['val_loss'][-1] < min(hist['hidden{}'.format(i+1)][j]['val_loss'][:-1] + [float('inf')]):
                            self.save_model(regressors['hidden{}'.format(i+1)][j].state_dict(), model_save_path, 'hidden{}_sample{}_best_val_loss.pth.tar'.format(i+1, j+1))
                            self.save_as_pkl(hist['hidden{}'.format(i+1)][j], model_save_path, 'hidden{}_sample{}_best_val_loss_history.pkl'.format(i+1, j+1)) 



    def train_linear_regressor(self, pose_type, no_occ_data, source_model_name, target_model_name, source_idx, target_idx, readout):
        if pose_type =='gt':
            input_size = 28
        elif pose_type == 'original':
            input_size = 50

        if source_model_name in ['FDClassifier', 'NoOccFDClassifier']:
            source_model = FDClassifier(input_size=input_size)
        elif source_model_name in ['RCClassifier', 'NoOccRCClassifier']:
            if readout:
                source_model = RCClassifier(input_size=input_size, record_readouts=True)
            else:                
                source_model = RCClassifier(input_size=input_size, record_activations=True)
                
        if target_model_name in ['FDClassifier', 'NoOccFDClassifier']:
            target_model = FDClassifier(input_size=input_size)
        elif target_model_name in ['RCClassifier', 'NoOccRCClassifier']:
            if readout:
                target_model = RCClassifier(input_size=input_size, record_readouts=True)
            else:                
                target_model = RCClassifier(input_size=input_size, record_activations=True)

        source_model_save_path = os.path.join(os.path.join(self.classifier_dir, pose_type, source_model_name), 
                                    'trial_{}'.format(source_idx+1), source_model_name+'_best_val_acc.pth.tar')
        target_model_save_path = os.path.join(os.path.join(self.classifier_dir, pose_type, target_model_name), 
                                    'trial_{}'.format(target_idx+1), target_model_name+'_best_val_acc.pth.tar')

        source_model.load_state_dict(torch.load(source_model_save_path, map_location=lambda storage, loc: storage))
        target_model.load_state_dict(torch.load(target_model_save_path, map_location=lambda storage, loc: storage))

        if source_model_name in ['FDClassifier', 'NoOccFDClassifier']:
            self.FD_train_linear_regressor_helper(pose_type, no_occ_data, source_model, target_model, source_model_name, target_model_name, source_idx, target_idx, readout)
        elif source_model_name in ['RCClassifier', 'NoOccRCClassifier']:
            self.RC_train_linear_regressor_helper(pose_type, no_occ_data, source_model, target_model, source_model_name, target_model_name, source_idx, target_idx, readout)


    def RC_train_linear_regressor_helper(self, pose_type, no_occ_data, source_model, target_model, source_model_name, target_model_name, source_idx, target_idx, readout, n_units=256, n_unit_samplings=3):
        if no_occ_data:
            no_occ_data_name = 'NoOcc'
        else:
            no_occ_data_name = 'Occ'
        result_save_path = os.path.join(self.result_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name, 'source_idx{}_target_idx{}'.format(source_idx+1, target_idx+1))
        model_save_path = os.path.join(self.model_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name, 'source_idx{}_target_idx{}'.format(source_idx+1, target_idx+1))
        if not os.path.exists(result_save_path):
            os.makedirs(result_save_path, exist_ok=True)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)

        hist = {}
        if readout:
            hist['readout'] = {}
            hist['readout']['train_loss'] = []
            hist['readout']['train_R2'] = []
            hist['readout']['val_loss'] = []
            hist['readout']['val_R2'] = []
        else:
            for i in range(4):
                hist['hidden{}'.format(i+1)] = {}
                if (i+1) == 4:
                    hist['hidden{}'.format(i+1)]['train_loss'] = []
                    hist['hidden{}'.format(i+1)]['train_R2'] = []
                    hist['hidden{}'.format(i+1)]['val_loss'] = []
                    hist['hidden{}'.format(i+1)]['val_R2'] = []
                else:
                    for j in range(n_unit_samplings):
                        hist['hidden{}'.format(i+1)][j] = {}
                        hist['hidden{}'.format(i+1)][j]['train_loss'] = []
                        hist['hidden{}'.format(i+1)][j]['train_R2'] = []
                        hist['hidden{}'.format(i+1)][j]['val_loss'] = []
                        hist['hidden{}'.format(i+1)][j]['val_R2'] = []
        hist['per_epoch_time'] = []

        source_model.eval()
        target_model.eval()

        if readout:
            regressors = {}
            regressors['readout'] = nn.Linear(source_model.n_classes, target_model.n_classes)
            optimizers = {}
            optimizers['readout'] = optim.Adam(regressors['readout'].parameters(), lr=self.lr)
            if self.gpu_mode:
                regressors['readout'].cuda()
        else:
            # Determine units to sample.
            sample_unit_path = os.path.join(self.sample_unit_dir, self.data_dir, pose_type, target_model_name, 'target_idx{}'.format(target_idx+1))
            if not os.path.exists(sample_unit_path):
                os.makedirs(sample_unit_path, exist_ok=True)

            if os.path.isfile(os.path.join(sample_unit_path, 'sample_indices.pt')):
                sample_indices = torch.load(os.path.join(sample_unit_path, 'sample_indices.pt'),
                                    map_location=lambda storage, loc: storage)
            else:
                C = source_model.hidden1
                H = source_model.input_size//2
                W = H
                sample_indices_hidden1 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

                C = source_model.hidden2
                H = source_model.input_size//4
                W = H
                sample_indices_hidden2 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

                C = source_model.hidden3
                H = source_model.input_size//8
                W = H
                sample_indices_hidden3 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

                sample_indices = torch.stack([sample_indices_hidden1, sample_indices_hidden2, sample_indices_hidden3], 0) # (n_layer, n_unit_samplings, n_units)
                torch.save(sample_indices, os.path.join(sample_unit_path, 'sample_indices.pt'))

            if self.gpu_mode:
                sample_indices = sample_indices.cuda()

            regressors = {}
            regressors['hidden1'] = []
            regressors['hidden2'] = []
            regressors['hidden3'] = []

            optimizers = {}
            optimizers['hidden1'] = []
            optimizers['hidden2'] = []
            optimizers['hidden3'] = []

            for i in range(n_unit_samplings):
                regressors['hidden1'].append(nn.Linear(source_model.hidden1*(source_model.input_size//2)**2, n_units))
                regressors['hidden2'].append(nn.Linear(source_model.hidden2*(source_model.input_size//4)**2, n_units))
                regressors['hidden3'].append(nn.Linear(source_model.hidden3*(source_model.input_size//8)**2, n_units))

                optimizers['hidden1'].append(optim.Adam(regressors['hidden1'][-1].parameters(), lr=self.lr))
                optimizers['hidden2'].append(optim.Adam(regressors['hidden2'][-1].parameters(), lr=self.lr))
                optimizers['hidden3'].append(optim.Adam(regressors['hidden3'][-1].parameters(), lr=self.lr))

                if self.gpu_mode:
                    regressors['hidden1'][-1].cuda()
                    regressors['hidden2'][-1].cuda()
                    regressors['hidden3'][-1].cuda()

            regressors['hidden4'] = nn.Linear(source_model.hidden4, target_model.hidden4)
            optimizers['hidden4'] = optim.Adam(regressors['hidden4'].parameters(), lr=self.lr)

            if self.gpu_mode:
                regressors['hidden4'].cuda()


        if self.gpu_mode:
            source_model.cuda()
            target_model.cuda()

        reg_loss = nn.MSELoss()

        print('training start!')
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            # Train the main model.
            print('')
            print('Train the linear regressor.')
            print('')
            iter_start_time = time.time()
            for iter, (data, ind_objs, ind_ps, _) in enumerate(self.train_loader):
                N, C, _, _ = data.size()
                data = Variable(data)
                ind_objs = Variable(ind_objs)
                ind_ps = Variable(ind_ps)
                if self.gpu_mode:
                    data = data.cuda()
                    ind_objs = ind_objs.cuda()
                    ind_ps = ind_ps.cuda()

                if pose_type == 'gt':
                    aff_ps = self.affine_form(ind_ps[:,1])
                    grid = F.affine_grid(aff_ps, torch.Size([N, C, self.in_size, self.in_size]))

                if pose_type == 'gt':                
                    if no_occ_data:
                        x = F.grid_sample(ind_objs[:,1], grid)
                    else:
                        x = F.grid_sample(data, grid)
                elif pose_type == 'original':
                    if no_occ_data:
                        x = ind_objs[:,1]
                    else:
                        x = data

                _ = source_model(x)
                _ = target_model(x)

                train_R2s = []

                if readout:
                    regressor = regressors['readout']
                    optimizer = optimizers['readout']
                    optimizer.zero_grad()

                    train_losses_iter = []
                    train_R2s_iter = []
                    for t in range(source_model.n_iter):
                        pred = regressor(source_model.readouts[t])
                        train_loss = reg_loss(pred, target_model.readouts[t])
                        train_losses_iter.append(train_loss)
                        mean = target_model.readouts[t].data.mean(0, True)
                        total_var = (target_model.readouts[t].data-mean).pow(2).mean()
                        train_R2 = 1 - train_loss.data[0]/total_var
                        train_R2s_iter.append(train_R2)

                    train_loss = torch.stack(train_losses_iter, 0).mean()
                    train_loss.backward()
                    optimizer.step()

                    train_R2 = np.array(train_R2s_iter).mean()

                    hist['readout']['train_loss'].append(train_loss.data[0])
                    hist['readout']['train_R2'].append(train_R2)
                    train_R2s.append(train_R2)
                else:
                    for i in range(4):
                        regressor = regressors['hidden{}'.format(i+1)]
                        optimizer = optimizers['hidden{}'.format(i+1)]
                        if (i+1) == 4:
                            optimizer.zero_grad()

                            train_losses_iter = []
                            train_R2s_iter = []
                            for t in range(source_model.n_iter):
                                pred = regressor(source_model.activations_hidden4[t])
                                train_loss = reg_loss(pred, target_model.activations_hidden4[t])
                                train_losses_iter.append(train_loss)
                                mean = target_model.activations_hidden4[t].data.mean(0, True)
                                total_var = (target_model.activations_hidden4[t].data-mean).pow(2).mean()
                                train_R2 = 1 - train_loss.data[0]/total_var
                                train_R2s_iter.append(train_R2)


                            train_loss = torch.stack(train_losses_iter, 0).mean()
                            train_loss.backward()
                            optimizer.step()

                            train_R2 = np.array(train_R2s_iter).mean()

                            hist['hidden4']['train_loss'].append(train_loss.data[0])
                            hist['hidden4']['train_R2'].append(train_R2)
                            train_R2s.append(train_R2)
                        else:
                            for j in range(n_unit_samplings):
                                optimizer[j].zero_grad()

                                train_losses_iter = []
                                train_R2s_iter = []
                                for t in range(source_model.n_iter):
                                    sampled_target_activations = self.sample_units(getattr(target_model, 'activations_hidden{}'.format(i+1))[t], sample_indices[i,j])
                                    pred = regressor[j](self.flatten(getattr(source_model, 'activations_hidden{}'.format(i+1))[t]))
                                    train_loss = reg_loss(pred, sampled_target_activations)
                                    train_losses_iter.append(train_loss)
                                    mean = sampled_target_activations.data.mean(0, True)
                                    total_var = (sampled_target_activations.data-mean).pow(2).mean()
                                    train_R2 = 1 - train_loss.data[0]/total_var
                                    train_R2s_iter.append(train_R2)


                                train_loss = torch.stack(train_losses_iter, 0).mean()
                                train_loss.backward()
                                optimizer[j].step()

                                train_R2 = np.array(train_R2s_iter).mean()

                                hist['hidden{}'.format(i+1)][j]['train_loss'].append(train_loss.data[0])
                                hist['hidden{}'.format(i+1)][j]['train_R2'].append(train_R2)
                                train_R2s.append(train_R2)

                mean_train_R2 = np.array(train_R2s).mean()
                std_train_R2 = np.array(train_R2s).std()

                if (iter + 1) % 100 == 0:
                    iter_time = time.time() - iter_start_time
                    iter_start_time = time.time()
                    print("Epoch {}: [{}/{}] mean_train_R2: {:.3f} std_train_R2: {:.3f} ({:.3f} sec)".format((epoch + 1), (iter + 1), 
                            len(self.train_loader), mean_train_R2, std_train_R2, iter_time))


            # Validation of the main model and z_classifier.
            print('')
            print('Validation begins.')
            print('')

            val_sums = {}
            if readout:
                val_sums['readout'] = {}
                val_sums['readout']['val_loss'] = 0
                val_sums['readout']['val_R2'] = 0
            else:
                for i in range(4):
                    val_sums['hidden{}'.format(i+1)] = {}
                    if (i+1) == 4:
                        val_sums['hidden{}'.format(i+1)]['val_loss'] = 0
                        val_sums['hidden{}'.format(i+1)]['val_R2'] = 0
                    else:
                        for j in range(n_unit_samplings):
                            val_sums['hidden{}'.format(i+1)][j] = {}
                            val_sums['hidden{}'.format(i+1)][j]['val_loss'] = 0
                            val_sums['hidden{}'.format(i+1)][j]['val_R2'] = 0


            iter_start_time = time.time()
            for iter, (data, ind_objs, ind_ps, _) in enumerate(self.val_loader):
                N, C, _, _ = data.size()
                data = Variable(data, volatile=True)
                ind_objs = Variable(ind_objs, volatile=True)
                ind_ps = Variable(ind_ps, volatile=True)
                if self.gpu_mode:
                    data = data.cuda()
                    ind_objs = ind_objs.cuda()
                    ind_ps = ind_ps.cuda()

                if pose_type == 'gt':
                    aff_ps = self.affine_form(ind_ps[:,1])
                    grid = F.affine_grid(aff_ps, torch.Size([N, C, self.in_size, self.in_size]))

                if pose_type == 'gt':                
                    if no_occ_data:
                        x = F.grid_sample(ind_objs[:,1], grid)
                    else:
                        x = F.grid_sample(data, grid)
                elif pose_type == 'original':
                    if no_occ_data:
                        x = ind_objs[:,1]
                    else:
                        x = data

                _ = source_model(x)
                _ = target_model(x)

                if readout:
                    regressor = regressors['readout']
                    optimizer = optimizers['readout']

                    val_losses_iter = []
                    val_R2s_iter = []
                    for t in range(source_model.n_iter):
                        pred = regressor(source_model.readouts[t])
                        val_loss = reg_loss(pred, target_model.readouts[t])
                        val_losses_iter.append(val_loss)
                        mean = target_model.readouts[t].data.mean(0, True)
                        total_var = (target_model.readouts[t].data-mean).pow(2).mean()
                        val_R2 = 1 - val_loss.data[0]/total_var
                        val_R2s_iter.append(val_R2)

                    val_loss = torch.stack(val_losses_iter, 0).mean()
                    val_sums['readout']['val_loss'] += val_loss.data[0]

                    val_R2 = np.array(val_R2s_iter).mean()
                    val_sums['readout']['val_R2'] += val_R2
                else:
                    for i in range(4):
                        regressor = regressors['hidden{}'.format(i+1)]
                        optimizer = optimizers['hidden{}'.format(i+1)]
                        if (i+1) == 4:

                            val_losses_iter = []
                            val_R2s_iter = []
                            for t in range(source_model.n_iter):
                                pred = regressor(source_model.activations_hidden4[t])
                                val_loss = reg_loss(pred, target_model.activations_hidden4[t])
                                val_losses_iter.append(val_loss)
                                mean = target_model.activations_hidden4[t].data.mean(0, True)
                                total_var = (target_model.activations_hidden4[t].data-mean).pow(2).mean()
                                val_R2 = 1 - val_loss.data[0]/total_var
                                val_R2s_iter.append(val_R2)

                            val_loss = torch.stack(val_losses_iter, 0).mean()
                            val_sums['hidden4']['val_loss'] += val_loss.data[0]

                            val_R2 = np.array(val_R2s_iter).mean()
                            val_sums['hidden4']['val_R2'] += val_R2
                        else:
                            for j in range(n_unit_samplings):
                                val_losses_iter = []
                                val_R2s_iter = []
                                for t in range(source_model.n_iter):
                                    sampled_target_activations = self.sample_units(getattr(target_model, 'activations_hidden{}'.format(i+1))[t], sample_indices[i,j])
                                    pred = regressor[j](self.flatten(getattr(source_model, 'activations_hidden{}'.format(i+1))[t]))
                                    val_loss = reg_loss(pred, sampled_target_activations)
                                    val_losses_iter.append(val_loss)
                                    mean = sampled_target_activations.data.mean(0, True)
                                    total_var = (sampled_target_activations.data-mean).pow(2).mean()
                                    val_R2 = 1 - val_loss.data[0]/total_var
                                    val_R2s_iter.append(val_R2)


                                val_loss = torch.stack(val_losses_iter, 0).mean()
                                val_sums['hidden{}'.format(i+1)][j]['val_loss'] += val_loss.data[0]

                                val_R2 = np.array(val_R2s_iter).mean()
                                val_sums['hidden{}'.format(i+1)][j]['val_R2'] += val_R2

            val_R2s = []
            if readout:
                val_sums['readout']['val_loss'] /= len(self.val_loader)
                val_sums['readout']['val_R2'] /= len(self.val_loader)
                hist['readout']['val_loss'].append(val_sums['readout']['val_loss'])
                hist['readout']['val_R2'].append(val_sums['readout']['val_R2'])
                val_R2s.append(val_sums['readout']['val_R2'])                
            else:
                for i in range(4):
                    if (i+1) == 4:
                        val_sums['hidden{}'.format(i+1)]['val_loss'] /= len(self.val_loader)
                        val_sums['hidden{}'.format(i+1)]['val_R2'] /= len(self.val_loader)
                        hist['hidden{}'.format(i+1)]['val_loss'].append(val_sums['hidden{}'.format(i+1)]['val_loss'])
                        hist['hidden{}'.format(i+1)]['val_R2'].append(val_sums['hidden{}'.format(i+1)]['val_R2'])
                        val_R2s.append(val_sums['hidden{}'.format(i+1)]['val_R2'])
                    else:
                        for j in range(n_unit_samplings):
                            val_sums['hidden{}'.format(i+1)][j]['val_loss'] /= len(self.val_loader)
                            val_sums['hidden{}'.format(i+1)][j]['val_R2'] /= len(self.val_loader)
                            hist['hidden{}'.format(i+1)][j]['val_loss'].append(val_sums['hidden{}'.format(i+1)][j]['val_loss'])
                            hist['hidden{}'.format(i+1)][j]['val_R2'].append(val_sums['hidden{}'.format(i+1)][j]['val_R2'])
                            val_R2s.append(val_sums['hidden{}'.format(i+1)][j]['val_R2'])

            mean_val_R2 = np.array(val_R2s).mean()
            std_val_R2 = np.array(val_R2s).std()

            print("Epoch {}: mean_val_R2: {:.3f} std_val_R2s: {:.3f}".format((epoch + 1), mean_val_R2, std_val_R2))


            hist['per_epoch_time'].append(time.time() - epoch_start_time)
            print('per_epoch_time: {:.6f}'.format(hist['per_epoch_time'][-1]))

            self.visualize_results(hist, result_save_path, n_unit_samplings, readout)


            if readout:
                if hist['readout']['val_loss'][-1] < min(hist['readout']['val_loss'][:-1] + [float('inf')]):
                    self.save_model(regressors['readout'].state_dict(), model_save_path, 'readout_best_val_loss.pth.tar')
                    self.save_as_pkl(hist['readout'], model_save_path, 'readout_best_val_loss_history.pkl') 
            else:
                for i in range(4):
                    if (i+1) == 4:
                        if hist['hidden4']['val_loss'][-1] < min(hist['hidden4']['val_loss'][:-1] + [float('inf')]):
                            self.save_model(regressors['hidden4'].state_dict(), model_save_path, 'hidden4_best_val_loss.pth.tar')
                            self.save_as_pkl(hist['hidden4'], model_save_path, 'hidden4_best_val_loss_history.pkl') 
                    else:
                        for j in range(n_unit_samplings):
                            if hist['hidden{}'.format(i+1)][j]['val_loss'][-1] < min(hist['hidden{}'.format(i+1)][j]['val_loss'][:-1] + [float('inf')]):
                                self.save_model(regressors['hidden{}'.format(i+1)][j].state_dict(), model_save_path, 'hidden{}_sample{}_best_val_loss.pth.tar'.format(i+1, j+1))
                                self.save_as_pkl(hist['hidden{}'.format(i+1)][j], model_save_path, 'hidden{}_sample{}_best_val_loss_history.pkl'.format(i+1, j+1)) 


    def FD_train_linear_regressor_helper(self, pose_type, no_occ_data, source_model, target_model, source_model_name, target_model_name, source_idx, target_idx, readout, n_units=256, n_unit_samplings=3):
        if no_occ_data:
            no_occ_data_name = 'NoOcc'
        else:
            no_occ_data_name = 'Occ'
        result_save_path = os.path.join(self.result_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name, 'source_idx{}_target_idx{}'.format(source_idx+1, target_idx+1))
        model_save_path = os.path.join(self.model_dir, self.data_dir, pose_type, no_occ_data_name, source_model_name, target_model_name, 'source_idx{}_target_idx{}'.format(source_idx+1, target_idx+1))
        if not os.path.exists(result_save_path):
            os.makedirs(result_save_path, exist_ok=True)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)

        hist = {}
        if readout:
            hist['readout'] = {}
            hist['readout']['train_loss'] = []
            hist['readout']['train_R2'] = []
            hist['readout']['val_loss'] = []
            hist['readout']['val_R2'] = []
        else:
            for i in range(4):
                hist['hidden{}'.format(i+1)] = {}
                if (i+1) == 4:
                    hist['hidden{}'.format(i+1)]['train_loss'] = []
                    hist['hidden{}'.format(i+1)]['train_R2'] = []
                    hist['hidden{}'.format(i+1)]['val_loss'] = []
                    hist['hidden{}'.format(i+1)]['val_R2'] = []
                else:
                    for j in range(n_unit_samplings):
                        hist['hidden{}'.format(i+1)][j] = {}
                        hist['hidden{}'.format(i+1)][j]['train_loss'] = []
                        hist['hidden{}'.format(i+1)][j]['train_R2'] = []
                        hist['hidden{}'.format(i+1)][j]['val_loss'] = []
                        hist['hidden{}'.format(i+1)][j]['val_R2'] = []
        hist['per_epoch_time'] = []

        activations = {}
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook

        if readout:
            source_model.model[14].register_forward_hook(get_activation('source_readout'))
            target_model.model[14].register_forward_hook(get_activation('target_readout'))
        else:
            source_model.model[2].register_forward_hook(get_activation('source_hidden1'))
            target_model.model[2].register_forward_hook(get_activation('target_hidden1'))

            source_model.model[5].register_forward_hook(get_activation('source_hidden2'))
            target_model.model[5].register_forward_hook(get_activation('target_hidden2'))

            source_model.model[8].register_forward_hook(get_activation('source_hidden3'))
            target_model.model[8].register_forward_hook(get_activation('target_hidden3'))

            source_model.model[12].register_forward_hook(get_activation('source_hidden4'))
            target_model.model[12].register_forward_hook(get_activation('target_hidden4'))

        source_model.eval()
        target_model.eval()

        if readout:
            regressors = {}
            regressors['readout'] = nn.Linear(source_model.n_classes, target_model.n_classes)
            optimizers = {}
            optimizers['readout'] = optim.Adam(regressors['readout'].parameters(), lr=self.lr)
            if self.gpu_mode:
                regressors['readout'].cuda()
        else:
            # Determine units to sample.
            sample_unit_path = os.path.join(self.sample_unit_dir, self.data_dir, pose_type, target_model_name, 'target_idx{}'.format(target_idx+1))
            if not os.path.exists(sample_unit_path):
                os.makedirs(sample_unit_path, exist_ok=True)

            if os.path.isfile(os.path.join(sample_unit_path, 'sample_indices.pt')):
                sample_indices = torch.load(os.path.join(sample_unit_path, 'sample_indices.pt'),
                                    map_location=lambda storage, loc: storage)
            else:
                C = source_model.hidden1
                H = source_model.input_size//2
                W = H
                sample_indices_hidden1 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

                C = source_model.hidden2
                H = source_model.input_size//4
                W = H
                sample_indices_hidden2 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

                C = source_model.hidden3
                H = source_model.input_size//8
                W = H
                sample_indices_hidden3 = self.generate_sample_indices(C, H, W, n_units, n_unit_samplings)

                sample_indices = torch.stack([sample_indices_hidden1, sample_indices_hidden2, sample_indices_hidden3], 0) # (n_layer, n_unit_samplings, n_units)
                torch.save(sample_indices, os.path.join(sample_unit_path, 'sample_indices.pt'))

            if self.gpu_mode:
                sample_indices = sample_indices.cuda()

            regressors = {}
            regressors['hidden1'] = []
            regressors['hidden2'] = []
            regressors['hidden3'] = []

            optimizers = {}
            optimizers['hidden1'] = []
            optimizers['hidden2'] = []
            optimizers['hidden3'] = []

            for i in range(n_unit_samplings):
                regressors['hidden1'].append(nn.Linear(source_model.hidden1*(source_model.input_size//2)**2, n_units))
                regressors['hidden2'].append(nn.Linear(source_model.hidden2*(source_model.input_size//4)**2, n_units))
                regressors['hidden3'].append(nn.Linear(source_model.hidden3*(source_model.input_size//8)**2, n_units))

                optimizers['hidden1'].append(optim.Adam(regressors['hidden1'][-1].parameters(), lr=self.lr))
                optimizers['hidden2'].append(optim.Adam(regressors['hidden2'][-1].parameters(), lr=self.lr))
                optimizers['hidden3'].append(optim.Adam(regressors['hidden3'][-1].parameters(), lr=self.lr))

                if self.gpu_mode:
                    regressors['hidden1'][-1].cuda()
                    regressors['hidden2'][-1].cuda()
                    regressors['hidden3'][-1].cuda()

            regressors['hidden4'] = nn.Linear(source_model.hidden4, target_model.hidden4)
            optimizers['hidden4'] = optim.Adam(regressors['hidden4'].parameters(), lr=self.lr)

            if self.gpu_mode:
                regressors['hidden4'].cuda()


        if self.gpu_mode:
            source_model.cuda()
            target_model.cuda()

        reg_loss = nn.MSELoss()

        print('training start!')
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            # Train the main model.
            print('')
            print('Train the linear regressor.')
            print('')
            iter_start_time = time.time()
            for iter, (data, ind_objs, ind_ps, _) in enumerate(self.train_loader):
                N, C, _, _ = data.size()
                data = Variable(data)
                ind_objs = Variable(ind_objs)
                ind_ps = Variable(ind_ps)
                if self.gpu_mode:
                    data = data.cuda()
                    ind_objs = ind_objs.cuda()
                    ind_ps = ind_ps.cuda()

                if pose_type == 'gt':
                    aff_ps = self.affine_form(ind_ps[:,1])
                    grid = F.affine_grid(aff_ps, torch.Size([N, C, self.in_size, self.in_size]))

                if pose_type == 'gt':                
                    if no_occ_data:
                        x = F.grid_sample(ind_objs[:,1], grid)
                    else:
                        x = F.grid_sample(data, grid)
                elif pose_type == 'original':
                    if no_occ_data:
                        x = ind_objs[:,1]
                    else:
                        x = data

                _ = source_model(x)
                _ = target_model(x)

                train_R2s = []

                if readout:
                    regressor = regressors['readout']
                    optimizer = optimizers['readout']
                    optimizer.zero_grad()
                    pred = regressor(activations['source_readout'])
                    train_loss = reg_loss(pred, activations['target_readout'])

                    train_loss.backward()
                    optimizer.step()

                    hist['readout']['train_loss'].append(train_loss.data[0])

                    mean = activations['target_readout'].data.mean(0, True)
                    total_var = (activations['target_readout'].data-mean).pow(2).mean()
                    train_R2 = 1 - train_loss.data[0]/total_var
                    hist['readout']['train_R2'].append(train_R2)
                    train_R2s.append(train_R2)
                else:
                    for i in range(4):
                        regressor = regressors['hidden{}'.format(i+1)]
                        optimizer = optimizers['hidden{}'.format(i+1)]
                        if (i+1) == 4:
                            optimizer.zero_grad()

                            pred = regressor(activations['source_hidden4'])
                            train_loss = reg_loss(pred, activations['target_hidden4'])

                            train_loss.backward()
                            optimizer.step()

                            hist['hidden4']['train_loss'].append(train_loss.data[0])

                            mean = activations['target_hidden4'].data.mean(0, True)
                            total_var = (activations['target_hidden4'].data-mean).pow(2).mean()
                            train_R2 = 1 - train_loss.data[0]/total_var
                            hist['hidden4']['train_R2'].append(train_R2)
                            train_R2s.append(train_R2)
                        else:
                            for j in range(n_unit_samplings):
                                optimizer[j].zero_grad()

                                sampled_target_activations = self.sample_units(activations['target_hidden{}'.format(i+1)], sample_indices[i,j])
                                pred = regressor[j](self.flatten(activations['source_hidden{}'.format(i+1)]))
                                train_loss = reg_loss(pred, sampled_target_activations)

                                train_loss.backward()
                                optimizer[j].step()

                                hist['hidden{}'.format(i+1)][j]['train_loss'].append(train_loss.data[0])
                                
                                mean = sampled_target_activations.data.mean(0, True)
                                total_var = (sampled_target_activations.data-mean).pow(2).mean()
                                train_R2 = 1 - train_loss.data[0]/total_var 
                                hist['hidden{}'.format(i+1)][j]['train_R2'].append(train_R2)
                                train_R2s.append(train_R2)

                mean_train_R2 = np.array(train_R2s).mean()
                std_train_R2 = np.array(train_R2s).std()

                if (iter + 1) % 100 == 0:
                    iter_time = time.time() - iter_start_time
                    iter_start_time = time.time()
                    print("Epoch {}: [{}/{}] mean_train_R2: {:.3f} std_train_R2: {:.3f} ({:.3f} sec)".format((epoch + 1), (iter + 1), 
                            len(self.train_loader), mean_train_R2, std_train_R2, iter_time))


            # Validation of the main model and z_classifier.
            print('')
            print('Validation begins.')
            print('')

            val_sums = {}
            if readout:
                val_sums['readout'] = {}
                val_sums['readout']['val_loss'] = 0
                val_sums['readout']['val_R2'] = 0
            else:
                for i in range(4):
                    val_sums['hidden{}'.format(i+1)] = {}
                    if (i+1) == 4:
                        val_sums['hidden{}'.format(i+1)]['val_loss'] = 0
                        val_sums['hidden{}'.format(i+1)]['val_R2'] = 0
                    else:
                        for j in range(n_unit_samplings):
                            val_sums['hidden{}'.format(i+1)][j] = {}
                            val_sums['hidden{}'.format(i+1)][j]['val_loss'] = 0
                            val_sums['hidden{}'.format(i+1)][j]['val_R2'] = 0


            iter_start_time = time.time()
            for iter, (data, ind_objs, ind_ps, _) in enumerate(self.val_loader):
                N, C, _, _ = data.size()
                data = Variable(data, volatile=True)
                ind_objs = Variable(ind_objs, volatile=True)
                ind_ps = Variable(ind_ps, volatile=True)
                if self.gpu_mode:
                    data = data.cuda()
                    ind_objs = ind_objs.cuda()
                    ind_ps = ind_ps.cuda()

                if pose_type == 'gt':
                    aff_ps = self.affine_form(ind_ps[:,1])
                    grid = F.affine_grid(aff_ps, torch.Size([N, C, self.in_size, self.in_size]))

                if pose_type == 'gt':                
                    if no_occ_data:
                        x = F.grid_sample(ind_objs[:,1], grid)
                    else:
                        x = F.grid_sample(data, grid)
                elif pose_type == 'original':
                    if no_occ_data:
                        x = ind_objs[:,1]
                    else:
                        x = data

                _ = source_model(x)
                _ = target_model(x)

                if readout:
                    regressor = regressors['readout']
                    optimizer = optimizers['readout']
                    pred = regressor(activations['source_readout'])
                    val_loss = reg_loss(pred, activations['target_readout'])

                    val_sums['readout']['val_loss'] += val_loss.data[0]

                    mean = activations['target_readout'].data.mean(0, True)
                    total_var = (activations['target_readout'].data-mean).pow(2).mean()
                    val_R2 = 1 - val_loss.data[0]/total_var
                    val_sums['readout']['val_R2'] += val_R2
                else:
                    for i in range(4):
                        regressor = regressors['hidden{}'.format(i+1)]
                        optimizer = optimizers['hidden{}'.format(i+1)]
                        if (i+1) == 4:
                            pred = regressor(activations['source_hidden4'])
                            val_loss = reg_loss(pred, activations['target_hidden4'])

                            val_sums['hidden4']['val_loss'] += val_loss.data[0]

                            mean = activations['target_hidden4'].data.mean(0, True)
                            total_var = (activations['target_hidden4'].data-mean).pow(2).mean()
                            val_R2 = 1 - val_loss.data[0]/total_var
                            val_sums['hidden4']['val_R2'] += val_R2
                        else:
                            for j in range(n_unit_samplings):
                                sampled_target_activations = self.sample_units(activations['target_hidden{}'.format(i+1)], sample_indices[i,j])
                                pred = regressor[j](self.flatten(activations['source_hidden{}'.format(i+1)]))
                                val_loss = reg_loss(pred, sampled_target_activations)

                                val_sums['hidden{}'.format(i+1)][j]['val_loss'] += val_loss.data[0]
                                
                                mean = sampled_target_activations.data.mean(0, True)
                                total_var = (sampled_target_activations.data-mean).pow(2).mean()
                                val_R2 = 1 - val_loss.data[0]/total_var 
                                val_sums['hidden{}'.format(i+1)][j]['val_R2'] += val_R2

            val_R2s = []
            if readout:
                val_sums['readout']['val_loss'] /= len(self.val_loader)
                val_sums['readout']['val_R2'] /= len(self.val_loader)
                hist['readout']['val_loss'].append(val_sums['readout']['val_loss'])
                hist['readout']['val_R2'].append(val_sums['readout']['val_R2'])
                val_R2s.append(val_sums['readout']['val_R2'])                
            else:
                for i in range(4):
                    if (i+1) == 4:
                        val_sums['hidden{}'.format(i+1)]['val_loss'] /= len(self.val_loader)
                        val_sums['hidden{}'.format(i+1)]['val_R2'] /= len(self.val_loader)
                        hist['hidden{}'.format(i+1)]['val_loss'].append(val_sums['hidden{}'.format(i+1)]['val_loss'])
                        hist['hidden{}'.format(i+1)]['val_R2'].append(val_sums['hidden{}'.format(i+1)]['val_R2'])
                        val_R2s.append(val_sums['hidden{}'.format(i+1)]['val_R2'])
                    else:
                        for j in range(n_unit_samplings):
                            val_sums['hidden{}'.format(i+1)][j]['val_loss'] /= len(self.val_loader)
                            val_sums['hidden{}'.format(i+1)][j]['val_R2'] /= len(self.val_loader)
                            hist['hidden{}'.format(i+1)][j]['val_loss'].append(val_sums['hidden{}'.format(i+1)][j]['val_loss'])
                            hist['hidden{}'.format(i+1)][j]['val_R2'].append(val_sums['hidden{}'.format(i+1)][j]['val_R2'])
                            val_R2s.append(val_sums['hidden{}'.format(i+1)][j]['val_R2'])

            mean_val_R2 = np.array(val_R2s).mean()
            std_val_R2 = np.array(val_R2s).std()

            print("Epoch {}: mean_val_R2: {:.3f} std_val_R2s: {:.3f}".format((epoch + 1), mean_val_R2, std_val_R2))


            hist['per_epoch_time'].append(time.time() - epoch_start_time)
            print('per_epoch_time: {:.6f}'.format(hist['per_epoch_time'][-1]))

            self.visualize_results(hist, result_save_path, n_unit_samplings, readout)

            if readout:
                if hist['readout']['val_loss'][-1] < min(hist['readout']['val_loss'][:-1] + [float('inf')]):
                    self.save_model(regressors['readout'].state_dict(), model_save_path, 'readout_best_val_loss.pth.tar')
                    self.save_as_pkl(hist['readout'], model_save_path, 'readout_best_val_loss_history.pkl') 
            else:
                for i in range(4):
                    if (i+1) == 4:
                        if hist['hidden4']['val_loss'][-1] < min(hist['hidden4']['val_loss'][:-1] + [float('inf')]):
                            self.save_model(regressors['hidden4'].state_dict(), model_save_path, 'hidden4_best_val_loss.pth.tar')
                            self.save_as_pkl(hist['hidden4'], model_save_path, 'hidden4_best_val_loss_history.pkl') 
                    else:
                        for j in range(n_unit_samplings):
                            if hist['hidden{}'.format(i+1)][j]['val_loss'][-1] < min(hist['hidden{}'.format(i+1)][j]['val_loss'][:-1] + [float('inf')]):
                                self.save_model(regressors['hidden{}'.format(i+1)][j].state_dict(), model_save_path, 'hidden{}_sample{}_best_val_loss.pth.tar'.format(i+1, j+1))
                                self.save_as_pkl(hist['hidden{}'.format(i+1)][j], model_save_path, 'hidden{}_sample{}_best_val_loss_history.pkl'.format(i+1, j+1)) 


    def affine_form(self, p):
        '''
        Args:
        p: (N, 4). s = p[0], a = p[1], x = p[2], y = p[3].
        s, a, x, y: respectively scale, angle, x, y translations. They all have shapes (N,).
        a, x, y are unconstrained.
        
        Return:
        aff_p: (N, 2, 3)-shaped affine parameters corresponding to them.
        '''
        N = p.size(0)
        aff_p = torch.zeros(N, 2, 3)
        if isinstance(p, Variable):
            aff_p = Variable(aff_p)
        if p.is_cuda:
            aff_p = aff_p.cuda()
        s = p[:,0]
        aff_p[:,0,0] = s*torch.cos(p[:,1])
        aff_p[:,0,1] = -s*torch.sin(p[:,1])
        aff_p[:,1,0] = s*torch.sin(p[:,1])
        aff_p[:,1,1] = s*torch.cos(p[:,1])
        aff_p[:,0,2] = p[:,2]
        aff_p[:,1,2] = p[:,3]

        return aff_p

    def visualize_results(self, hist, result_save_path, n_unit_samplings, readout):
        if readout:
            self.visualize_results_helper(hist['readout']['train_loss'], result_save_path, 'train_losses_readout.png',
             x_label='Iter', y_label='L2_Loss')
            self.visualize_results_helper(hist['readout']['val_loss'], result_save_path, 'val_losses_readout.png',
             x_label='Epoch', y_label='L2_Loss')
            self.visualize_results_helper(hist['readout']['train_R2'], result_save_path, 'train_R2_readout.png', 
                x_label='Iter', y_label='R2')
            self.visualize_results_helper(hist['readout']['val_R2'], result_save_path, 'val_R2_readout.png', x_label='Epoch', y_label='R2')            
        else:
            for i in range(4):
                if (i+1) == 4:
                    self.visualize_results_helper(hist['hidden4']['train_loss'], result_save_path, 'train_losses_hidden4.png',
                     x_label='Iter', y_label='L2_Loss')
                    self.visualize_results_helper(hist['hidden4']['val_loss'], result_save_path, 'val_losses_hidden4.png',
                     x_label='Epoch', y_label='L2_Loss')
                    self.visualize_results_helper(hist['hidden4']['train_R2'], result_save_path, 'train_R2_hidden4.png', 
                        x_label='Iter', y_label='R2')
                    self.visualize_results_helper(hist['hidden4']['val_R2'], result_save_path, 'val_R2_hidden4.png', x_label='Epoch', y_label='R2')
                else:
                    self.visualize_results_helper2(hist['hidden{}'.format(i+1)], result_save_path, i, n_unit_samplings)

    def visualize_results_helper2(self, result_list, result_save_path, layer_idx, n_unit_samplings):
        # For train results
        fig, ax = plt.subplots()
        x = range(1, len(result_list[0]['train_loss'])+1)
        for j in range(n_unit_samplings):
            ax.plot(x, result_list[j]['train_loss'], label='sample #{}'.format(j+1))
        ax.set_xlabel('Iter')
        ax.set_ylabel('L2_Loss')
        ax.legend(loc=1)
        ax.grid(True)
        save_name = 'train_losses_hidden{}.png'.format(layer_idx+1)
        fig.savefig(os.path.join(result_save_path, save_name))
        plt.close(fig)

        fig, ax = plt.subplots()
        x = range(1, len(result_list[0]['train_R2'])+1)
        for j in range(n_unit_samplings):
            ax.plot(x, result_list[j]['train_R2'], label='sample #{}'.format(j+1))
        ax.set_xlabel('Iter')
        ax.set_ylabel('R2')
        ax.legend(loc=1)
        ax.grid(True)
        save_name = 'train_R2_hidden{}.png'.format(layer_idx+1)
        fig.savefig(os.path.join(result_save_path, save_name))
        plt.close(fig)

        # For val results
        fig, ax = plt.subplots()
        x = range(1, len(result_list[0]['val_loss'])+1)
        for j in range(n_unit_samplings):
            ax.plot(x, result_list[j]['val_loss'], label='sample #{}'.format(j+1))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('L2_Loss')
        ax.legend(loc=1)
        ax.grid(True)
        save_name = 'val_losses_hidden{}.png'.format(layer_idx+1)
        fig.savefig(os.path.join(result_save_path, save_name))
        plt.close(fig)

        fig, ax = plt.subplots()
        x = range(1, len(result_list[0]['val_R2'])+1)
        for j in range(n_unit_samplings):
            ax.plot(x, result_list[j]['val_R2'], label='sample #{}'.format(j+1))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('R2')
        ax.legend(loc=1)
        ax.grid(True)
        save_name = 'val_R2_hidden{}.png'.format(layer_idx+1)
        fig.savefig(os.path.join(result_save_path, save_name))
        plt.close(fig)


    def visualize_results_helper(self, result_list, result_save_path, save_name, x_label='Iter', y_label='L2_Loss'):
        fig = plt.figure()
        x = range(1,len(result_list)+1)
        plt.plot(x, result_list)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        #plt.legend(loc=1)
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(result_save_path, save_name))
        plt.close(fig)

    def save_model(self, model_state_dict, model_save_path, save_name):
        torch.save(model_state_dict, os.path.join(model_save_path, save_name))

    def save_as_pkl(self, obj, model_save_path, save_name):  
        with open(os.path.join(model_save_path, save_name), 'wb') as f:
            pickle.dump(obj, f)

