import math, utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim.lr_scheduler as lr_scheduler

from new_pose_net import *

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class PoseNetExp(object):
    def __init__(self, args):
        self.debug = args.debug
        self.train_baseline_flag = args.train_baseline
        self.train_flag = args.train
        self.test_flag = args.test
        self.train_refiner_flag = args.train_refiner
        self.plot_flag = args.plot

        self.data_name = args.data_name

        if self.data_name == 'light_occ':
            self.data_type = 'fashion_mnist'
            self.data_rotate = True
            self.valid_classes = '0289'
            self.scale_min = 2.2
            self.scale_max = 2.2
            self.angle_min = -math.pi
            self.angle_max = math.pi
            self.trans_min = -0.55
            self.trans_max = 0.55
            self.n_objs = 2
            self.in_size = 28
            self.out_size = 50
            self.vr_min = 0.40
            self.vr_max = 0.90
            self.vr_bin_size = 0.10
            self.eps = 1e-2
            self.eps2 = 1e-1
        elif self.data_name == 'heavy_occ':
            self.data_type = 'fashion_mnist'
            self.data_rotate = True
            self.valid_classes = '0289'
            self.scale_min = 2.2
            self.scale_max = 2.2
            self.angle_min = -math.pi
            self.angle_max = math.pi
            self.trans_min = -0.55
            self.trans_max = 0.55
            self.n_objs = 2
            self.in_size = 28
            self.out_size = 50
            self.vr_min = 0.25
            self.vr_max = 0.50
            self.vr_bin_size = 0.05
            self.eps = 1e-2
            self.eps2 = 1e-1

        self.baseline_type = args.baseline_type
        self.baseline_lr = args.baseline_lr
        self.baseline_batch_size = args.baseline_batch_size
        self.baseline_test_batch_size = args.baseline_test_batch_size
        self.baseline_epoch = args.baseline_epoch

        self.model_type = args.model_type
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.epoch = args.epoch

        self.refiner_type = args.refiner_type
        self.refiner_lr = args.refiner_lr
        self.refiner_batch_size = args.refiner_batch_size
        self.refiner_test_batch_size = args.refiner_test_batch_size
        self.refiner_epoch = args.refiner_epoch

        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        
        self.n_vis = min(args.n_vis, args.test_batch_size)

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

        if not self.test_flag:
            self.init_dataset()


        # Note that we need a baseline for the training of self.model.
        if self.baseline_type == 'Baseline':
            self.baseline = Baseline()

        self.baseline_optimizer = optim.Adam(self.baseline.parameters(), lr=self.baseline_lr)
        self.cls_loss = nn.CrossEntropyLoss()
        self.recon_loss = utils.BKLDivLoss(size_average=True)
        if self.gpu_mode:
                self.baseline.cuda()


        if self.train_flag or self.test_flag:
            if self.model_type == 'PoseNetBasic':
                self.model = PoseNetBasic()
            elif self.model_type == 'PoseNetBasicDetached':
                self.model = PoseNetBasicDetached()

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            #self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=9, verbose=True, threshold=0.1, threshold_mode='rel')              
            self.reg_loss = nn.MSELoss()
            if self.gpu_mode:
                self.model.cuda()


        if self.train_refiner_flag:
            # We need a pretrained PoseNet.
            if self.model_type == 'PoseNetBasic':
                self.model = PoseNetBasic()
            elif self.model_type == 'PoseNetBasicDetached':
                self.model = PoseNetBasicDetached()

            model_save_dir = os.path.join(self.save_dir, self.data_dir, self.model_type)
            self.model.load_state_dict(torch.load(os.path.join(model_save_dir, self.model_type+'_best_val_loss.pth.tar')))
            print('Loaded a pretrained PoseNet for a PoseNet Refiner.')
            self.model.eval()

            if self.refiner_type == 'PoseNetRefiner':
                self.refiner = PoseNetRefiner()

            self.refiner_optimizer = optim.Adam(self.refiner.parameters(), lr=self.refiner_lr)
            self.reg_loss = nn.MSELoss()
            if self.gpu_mode:
                self.model.cuda()
                self.refiner.cuda()


        if not self.test_flag:
            # write settings and architecture into a txt file and save it in a result/log/save folder.
            for dir in [self.save_dir,self.result_dir]:
                if self.train_baseline_flag:
                    record_dir = os.path.join(dir, self.data_dir, self.baseline_type)
                    record_name = self.baseline_type + '_specs.txt' 
                elif self.train_flag:
                    record_dir = os.path.join(dir, self.data_dir, self.model_type)
                    record_name = self.model_type + '_specs.txt' 
                elif self.train_refiner_flag:
                    record_dir = os.path.join(dir, self.data_dir, self.refiner_type)
                    record_name = self.refiner_type + '_specs.txt' 

                if not os.path.exists(record_dir):
                    os.makedirs(record_dir)
                # Output a text file summarizing the parameters that generated the dataset.
                # We can record not only fixed, initial hps, but also constantly changing ones, such as decaying lr, if we decay lr.
                with open(os.path.join(record_dir, record_name), 'w') as summary:
                    summary.write('CUDA enabled: {}\n'.format(self.gpu_mode))
                    if self.gpu_mode:
                        summary.write('GPU Device: {}\n'.format(torch.cuda.current_device()))
                    summary.write('data_type: {}\n'.format(self.data_type))
                    summary.write('data_rotate: {}\n'.format(self.data_rotate))
                    summary.write('valid_classes: {}\n'.format(self.valid_classes))
                    summary.write('scale_min: {:.2f}\n'.format(self.scale_min))
                    summary.write('scale_max: {:.2f}\n'.format(self.scale_max))
                    summary.write('angle_min: {:.2f}\n'.format(self.angle_min*180/math.pi))
                    summary.write('angle_max: {:.2f}\n'.format(self.angle_max*180/math.pi))
                    summary.write('trans_min: {:.2f}\n'.format(self.trans_min))
                    summary.write('trans_max: {:.2f}\n'.format(self.trans_max))
                    summary.write('n_objs: {}\n'.format(self.n_objs))
                    summary.write('in_size: {}\n'.format(self.in_size))
                    summary.write('out_size: {}\n'.format(self.out_size))
                    summary.write('vr_min: {:.2f}\n'.format(self.vr_min))
                    summary.write('vr_max: {:.2f}\n'.format(self.vr_max))
                    summary.write('vr_bin_size: {:.2f}\n'.format(self.vr_bin_size))
                    summary.write('eps: {:.0e}\n'.format(self.eps))
                    summary.write('eps2: {:.0e}\n'.format(self.eps2))

                    summary.write('baseline_model: {}\n'.format(self.baseline_type))
                    summary.write('baseline_lr: {:.1e}\n'.format(self.baseline_lr))
                    summary.write('baseline_batch_size: {}\n'.format(self.baseline_batch_size))
                    summary.write('baseline_test_batch_size: {}\n'.format(self.baseline_test_batch_size))
                    summary.write('baseline_epoch: {}\n'.format(self.baseline_epoch))

                    if self.train_flag:
                        summary.write('Model: {}\n'.format(self.model_type))
                        summary.write('lr: {:.1e}\n'.format(self.lr))
                        summary.write('batch_size: {}\n'.format(self.batch_size))
                        summary.write('test_batch_size: {}\n'.format(self.test_batch_size))
                        summary.write('epoch: {}\n'.format(self.epoch))
                    elif self.train_refiner_flag:
                        summary.write('Pretrained PoseNet Model: {}\n'.format(self.model_type))
                        summary.write('lr: {:.1e}\n'.format(self.lr))
                        summary.write('batch_size: {}\n'.format(self.batch_size))
                        summary.write('test_batch_size: {}\n'.format(self.test_batch_size))
                        summary.write('epoch: {}\n'.format(self.epoch))

                        summary.write('PoseNet Refiner Model: {}\n'.format(self.refiner_type))
                        summary.write('lr: {:.1e}\n'.format(self.refiner_lr))
                        summary.write('batch_size: {}\n'.format(self.refiner_batch_size))
                        summary.write('test_batch_size: {}\n'.format(self.refiner_test_batch_size))
                        summary.write('epoch: {}\n'.format(self.refiner_epoch))

                    summary.write('n_vis: {}'.format(self.n_vis))
                    summary.write('baseline architecture: \n')
                    summary.write(repr(self.baseline))
                    
                    if self.train_flag:
                        summary.write('model architecture: \n')
                        summary.write(repr(self.model))
                    elif self.train_refiner_flag:
                        summary.write('Pretrained PoseNet architecture: \n')
                        summary.write(repr(self.model))
                        summary.write('PoseNet Refiner architecture: \n')
                        summary.write(repr(self.refiner))



    # def init_dataset(self, data_dir=None):
    #     '''
    #     I believe it is a good practice to not separately define a Dataset instance in case it initalizes
    #     a huge data. For example, in case I use a superimposed rgb or gray FashionMNIST, the initialized
    #     dataset may take up almost as large as 12GB, train and test set combined.
    #     '''
    #     if data_dir is None:
    #         data_dir = self.data_dir

    #     kwargs = {'num_workers': 8, 'pin_memory': True} if self.gpu_mode else {}
    #     self.train_loader = DataLoader(utils.MultiObjectDataset(data_dir, data_type='train', requires_ind_objs=True), shuffle=True, batch_size=self.batch_size, 
    #         drop_last=True, **kwargs)
    #     self.val_loader = DataLoader(utils.MultiObjectDataset(data_dir, data_type='val', requires_ind_objs=True), shuffle=True, batch_size=self.test_batch_size, 
    #         drop_last=False, **kwargs)



    def init_dataset(self, data_dir=None, batch_size=None, test_batch_size=None, test_set=False, vr_min=None, vr_max=None):
        '''
        I believe it is a good practice to not separately define a Dataset instance in case it initalizes
        a huge data. For example, in case I use a superimposed rgb or gray FashionMNIST, the initialized
        dataset may take up almost as large as 12GB, train and test set combined.
        '''
        if data_dir is None:
            data_dir = self.data_dir
        if batch_size is None:
            batch_size = self.batch_size
        if test_batch_size is None:
            test_batch_size = self.test_batch_size

        kwargs = {'num_workers': 8, 'pin_memory': True} if self.gpu_mode else {}
        if not test_set:
            self.train_loader = DataLoader(utils.Gen3MultiObjectDataset(data_dir, data_type='train', requires_ind_objs=True, vr_min=vr_min, vr_max=vr_max), shuffle=True, batch_size=batch_size, 
                drop_last=True, **kwargs)
            self.val_loader = DataLoader(utils.Gen3MultiObjectDataset(data_dir, data_type='val', requires_ind_objs=True, vr_min=vr_min, vr_max=vr_max), shuffle=True, batch_size=test_batch_size, 
                drop_last=False, **kwargs)
        else:
            self.test_loader = DataLoader(utils.Gen3MultiObjectDataset(data_dir, data_type='test', requires_ind_objs=True, vr_min=vr_min, vr_max=vr_max), shuffle=False, batch_size=test_batch_size, 
                drop_last=False, **kwargs)
        print('initialized source dataset.')



    def train_baseline(self, trial_idx=None):
        hist = {}
        hist['total_loss'] = []
        hist['total_ind_obj_loss'] = []
        hist['total_acc'] = []
        for i in range(self.n_objs):
            hist['ind_loss{}'.format(i+1)] = []
            hist['ind_acc{}'.format(i+1)] = []

        hist['val_total_loss'] = []
        hist['val_total_ind_obj_loss'] = []
        hist['val_total_acc'] = []
        for i in range(self.n_objs):
            hist['val_ind_loss{}'.format(i+1)] = []
            hist['val_ind_acc{}'.format(i+1)] = []

        hist['per_epoch_time'] = []
        hist['total_time'] = []

        print('training start!')
        start_time = time.time()
        for epoch in range(self.baseline_epoch):
            self.cur_epoch = epoch + 1
            epoch_start_time = time.time()

            # Train the main model.
            print('')
            print('Train the baseline.')
            print('')

            self.baseline.train()
            iter_start_time = time.time()
            for iter, (data, ind_objs, _, labels) in enumerate(self.train_loader):
                # note that even though ind_objs are not used for backprop, we still make it a Variable, because it will enter into self.loss_fct.recon_loss_fct, which is nn.Module.
                data = Variable(data)
                ind_objs = Variable(ind_objs)
                labels = Variable(labels)
                if self.gpu_mode:
                    data = data.cuda()
                    ind_objs = ind_objs.cuda()
                    labels = labels.cuda()
                '''
                First, we train the model on ind_objs.
                '''
                self.baseline_optimizer.zero_grad()
                N, n_objs, C, H, W = ind_objs.size()
                scores, unocc_ind_obj_recons = self.baseline(ind_objs.view(-1, C, H, W), single_obj=True)

                ind_obj_recon_loss = self.recon_loss(unocc_ind_obj_recons, ind_objs.view(-1, C, H, W))
                ind_obj_cls_loss = self.cls_loss(scores, labels.view(-1))
                total_ind_obj_loss = ind_obj_recon_loss + ind_obj_cls_loss 

                total_ind_obj_loss.backward()
                self.baseline_optimizer.step()

                '''
                Next, we train the model on data, which are occluded object images.
                '''
                self.baseline_optimizer.zero_grad()
                scores, ind_obj_recons = self.baseline(data, single_obj=False)

                ind_recon_loss1 = self.recon_loss(ind_obj_recons[:,0], ind_objs[:,0])
                ind_cls_loss1 = self.cls_loss(scores[:,0], labels[:,0])
                ind_loss1 = ind_recon_loss1 + ind_cls_loss1
                ind_recon_loss2 = self.recon_loss(ind_obj_recons[:,1], ind_objs[:,1])
                ind_cls_loss2 = self.cls_loss(scores[:,1], labels[:,1])
                ind_loss2 = ind_recon_loss2 + ind_cls_loss2

                total_loss = ind_loss1 + ind_loss2
                total_loss.backward()
                self.baseline_optimizer.step()

                ind_acc1 = (scores[:,0].max(1)[1] == labels[:,0]).sum().data[0]/len(labels)
                ind_acc2 = (scores[:,1].max(1)[1] == labels[:,1]).sum().data[0]/len(labels)
                total_acc = ((scores[:,0].max(1)[1] == labels[:,0])*(scores[:,1].max(1)[1] == labels[:,1])).sum().data[0]/len(labels)


                hist['total_loss'].append(total_loss.data[0])
                hist['total_ind_obj_loss'].append(total_ind_obj_loss.data[0])
                hist['ind_loss1'].append(ind_loss1.data[0])
                hist['ind_loss2'].append(ind_loss2.data[0])
                hist['total_acc'].append(total_acc)
                hist['ind_acc1'].append(ind_acc1)
                hist['ind_acc2'].append(ind_acc2)


                if ((iter + 1) % 100) == 0:
                    iter_time = time.time() - iter_start_time
                    iter_start_time = time.time()
                    print("Epoch {}: [{}/{}] total_ind_obj_loss: {:.3f} total_loss: {:.3f} ind_loss1: {:.3f} ind_loss2: {:.3f} total_acc: {:.3f} ind_acc1: {:.3f} ind_acc2: {:.3f} ({:.3f} sec)".format(
                        (epoch + 1), (iter + 1), len(self.train_loader), total_ind_obj_loss.data[0], total_loss.data[0], ind_loss1.data[0], ind_loss2.data[0], total_acc, ind_acc1, ind_acc2, iter_time))

            print('')
            print('Validation begins for the baseline')
            print('')
            total_ind_obj_loss = 0.0
            total_loss = 0.0
            ind_loss1 = 0.0
            ind_loss2 = 0.0

            total_acc = 0.0
            ind_acc1 = 0.0
            ind_acc2 = 0.0
            
            if self.plot_flag:
                '''
                We want to visualize the following things: (data, recons), (ind_objs, unocc_ind_obj_recons, ind_obj_recons. G_ind_objs_recon2).
                '''
                self.vis_data = None
                self.vis_ind_objs = None
                self.vis_unocc_ind_obj_recons = None
                self.vis_ind_obj_recons = None

            self.baseline.eval()
            for iter, (data, ind_objs, _, labels) in enumerate(self.val_loader):
                data = Variable(data, volatile=True)
                ind_objs = Variable(ind_objs, volatile=True)
                labels = Variable(labels, volatile=True)
                if self.gpu_mode:
                    data = data.cuda()
                    ind_objs = ind_objs.cuda()
                    labels = labels.cuda()

                '''
                First we train the model on ind_objs.
                ''' 
                N, n_objs, C, H, W = ind_objs.size()
                scores, unocc_ind_obj_recons = self.baseline(ind_objs.view(-1, C, H, W), single_obj=True)

                ind_obj_recon_loss = self.recon_loss(unocc_ind_obj_recons, ind_objs.view(-1, C, H, W)).data[0]
                ind_obj_cls_loss = self.cls_loss(scores, labels.view(-1)).data[0]
                total_ind_obj_loss += ind_obj_recon_loss + ind_obj_cls_loss 

                '''
                Next, we train the model on data, which are occluded object images.
                '''
                scores, ind_obj_recons = self.baseline(data, single_obj=False)

                ind_recon_loss1 = self.recon_loss(ind_obj_recons[:,0], ind_objs[:,0]).data[0]
                ind_cls_loss1 = self.cls_loss(scores[:,0], labels[:,0]).data[0]
                ind_loss1 += ind_recon_loss1 + ind_cls_loss1
                ind_recon_loss2 = self.recon_loss(ind_obj_recons[:,1], ind_objs[:,1]).data[0]
                ind_cls_loss2 = self.cls_loss(scores[:,1], labels[:,1]).data[0]
                ind_loss2 += ind_recon_loss2 + ind_cls_loss2

                total_loss = ind_loss1 + ind_loss2

                ind_acc1 += (scores[:,0].max(1)[1] == labels[:,0]).sum().data[0]/len(labels)
                ind_acc2 += (scores[:,1].max(1)[1] == labels[:,1]).sum().data[0]/len(labels)
                total_acc += ((scores[:,0].max(1)[1] == labels[:,0])*(scores[:,1].max(1)[1] == labels[:,1])).sum().data[0]/len(labels)

                if self.plot_flag:
                    if (iter + 1) == len(self.val_loader):
                        '''
                        We want to visualize the following things: (data, recons), (ind_objs, unocc_ind_obj_recons, ind_obj_recons. G_ind_objs_recon2).
                        '''
                        self.vis_data = data[:self.n_vis].data.cpu().numpy()
                        self.vis_ind_objs = ind_objs[:self.n_vis].data.cpu().numpy()
                        self.vis_unocc_ind_obj_recons = unocc_ind_obj_recons.view(-1, n_objs, C, H, W)[:self.n_vis].data.cpu().numpy()
                        self.vis_ind_obj_recons = ind_obj_recons[:self.n_vis].data.cpu().numpy()


            total_ind_obj_loss /= len(self.val_loader)
            total_loss /= len(self.val_loader)
            ind_loss1 /= len(self.val_loader)
            ind_loss2 /= len(self.val_loader)

            total_acc /= len(self.val_loader)
            ind_acc1 /= len(self.val_loader)
            ind_acc2 /= len(self.val_loader)

            #self.scheduler.step(total_acc, (epoch+1))

            hist['val_total_loss'].append(total_loss)
            hist['val_total_ind_obj_loss'].append(total_ind_obj_loss)
            hist['val_ind_loss1'].append(ind_loss1)
            hist['val_ind_loss2'].append(ind_loss2)

            hist['val_total_acc'].append(total_acc)
            hist['val_ind_acc1'].append(ind_acc1)
            hist['val_ind_acc2'].append(ind_acc2)

            hist['per_epoch_time'].append(time.time() - epoch_start_time)
            print('per_epoch_time: {:.6f}'.format(hist['per_epoch_time'][-1]))

            print("Epoch {}: total_ind_obj_loss: {:.3f} total_loss: {:.3f} ind_loss1: {:.3f} ind_loss2: {:.3f} total_acc: {:.3f} ind_acc1: {:.3f} ind_acc2: {:.3f}".format(
                (epoch + 1), total_ind_obj_loss, total_loss, ind_loss1, ind_loss2, total_acc, ind_acc1, ind_acc2))
        

            if self.plot_flag:
                self.visualize_train_baseline_results(hist, (epoch+1))

            # Save the model. Since it's not clear which measure of performance defines the 'best' model, we save the best model for each measure of performance.
            # Checkpoint
            self.save_model(model_state_dict=self.baseline.state_dict(), save_dir=os.path.join(self.save_dir, self.data_dir, self.baseline_type), 
                save_name=self.baseline_type+'_checkpoint.pth.tar', trial_idx=trial_idx)
            self.save_as_pkl(obj=hist, save_dir=os.path.join(self.save_dir, self.data_dir, self.baseline_type), 
                save_name=self.baseline_type+'_checkpoint'+'_history.pkl', trial_idx=trial_idx)

            # Best total loss
            if hist['val_total_loss'][-1] < min(hist['val_total_loss'][:-1] + [float('inf')]): # [float('inf')] takes care of the case when [:-1] returns an empty list, as for epoch 1.
                self.save_model(model_state_dict=self.baseline.state_dict(), save_dir=os.path.join(self.save_dir, self.data_dir, self.baseline_type), 
                    save_name=self.baseline_type +'_best_val_loss'+'.pth.tar', trial_idx=trial_idx)
                self.save_as_pkl(obj=hist, save_dir=os.path.join(self.save_dir, self.data_dir, self.baseline_type), 
                    save_name=self.baseline_type +'_best_val_loss' +'_history.pkl', trial_idx=trial_idx)
                self.record_hps_and_results(save_dir=os.path.join(self.save_dir, self.data_dir, self.baseline_type), 
                    save_name=self.baseline_type +'_best_total_loss'+'_hps.txt', trial_idx=trial_idx)

        hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: {:.2f}, total {} epochs time: {:.2f}".format(np.mean(hist['per_epoch_time']),
              self.baseline_epoch, hist['total_time'][0]))
        print("Training finish!... save training results")


    def train(self, trial_idx=None):
        '''
        Load a pretrained baseline.
        '''
        baseline_save_dir = os.path.join(self.save_dir, self.data_dir, self.baseline_type)
        if trial_idx is not None:
            baseline_save_dir = os.path.join(baseline_save_dir, 'trial_{}'.format(trial_idx+1))
        self.baseline.load_state_dict(torch.load(os.path.join(baseline_save_dir, self.baseline_type + '_best_val_loss.pth.tar')))
        print('Loaded a pretrained baseline.')
        self.baseline.eval()




        self.hist = {}
        self.hist['total_loss'] = []
        self.hist['total_ind_obj_loss'] = []
        self.hist['val_total_loss'] = []
        self.hist['val_total_ind_obj_loss'] = []

        self.hist['per_epoch_time'] = []
        self.hist['total_time'] = []

        print('training start!')
        start_time = time.time()

        for epoch in range(self.epoch):
            self.cur_epoch = epoch + 1
            epoch_start_time = time.time()

            # Train the main model.
            print('')
            print('Train the main model.')
            print('')

            self.model.train()
            iter_start_time = time.time()
            for iter, (data, ind_objs, ps, labels) in enumerate(self.train_loader):
                data = Variable(data)
                ind_objs = Variable(ind_objs)
                ps = Variable(ps)
                labels = Variable(labels)
                if self.gpu_mode:
                    data = data.cuda()
                    ind_objs = ind_objs.cuda()
                    ps = ps.cuda()
                    labels = labels.cuda()

                '''
                First, train PoseNet on ind_objs.
                '''
                self.optimizer.zero_grad()
                pred_ps = self.model(ind_objs.view(-1, 1, self.out_size, self.out_size))
                total_ind_obj_loss = self.reg_loss(pred_ps.view(-1, self.n_objs, 4), ps)
                total_ind_obj_loss.backward()
                self.optimizer.step()
                '''
                Next, train it on x_b from the baseline.
                '''
                self.optimizer.zero_grad()
                _, ind_obj_recons = self.baseline(data, single_obj=False)
                pred_ps = self.model(ind_obj_recons[:,1].detach())
                total_loss = self.reg_loss(pred_ps, ps[:,1])
                total_loss.backward()
                self.optimizer.step()

                self.hist['total_loss'].append(total_loss.data[0])
                self.hist['total_ind_obj_loss'].append(total_ind_obj_loss.data[0])

                if (iter + 1) % 100 == 0:
                    iter_time = time.time() - iter_start_time
                    iter_start_time = time.time()
                    print("Epoch {}: [{}/{}] total_ind_obj_loss: {:.3f} total_loss: {:.3f} ({:.3f} sec)".format((epoch + 1), (iter + 1), 
                        len(self.train_loader), total_ind_obj_loss.data[0], total_loss.data[0], iter_time))


            # Validation of the main model and z_classifier.
            print('')
            print('Validation begins.')
            print('')
            total_ind_obj_loss = 0.0
            total_loss = 0.0


            if self.plot_flag:
                self.n_types_vis = 7

                self.vis_data = None
                self.vis_pred_x_b = None
                self.vis_unocc_canon_data =None
                self.vis_unocc_canon_x_b = None
                self.vis_canon_data = None 
                self.vis_canon_x_b = None 
                self.vis_gt_canon_x_b = None


            self.model.eval()
            iter_start_time = time.time()
            for iter, (data, ind_objs, ps, labels) in enumerate(self.val_loader):
                data = Variable(data, volatile=True)
                ind_objs = Variable(ind_objs, volatile=True)
                ps = Variable(ps, volatile=True)
                labels = Variable(labels, volatile=True)
                if self.gpu_mode:
                    data = data.cuda()
                    ind_objs = ind_objs.cuda()
                    ps = ps.cuda()
                    labels = labels.cuda()

                '''
                First, train PoseNet on ind_objs.
                '''
                C = data.size(1)
                unocc_pred_ps = self.model(ind_objs.view(-1, C, self.out_size, self.out_size)).view(-1, self.n_objs, 4)
                total_ind_obj_loss += self.reg_loss(unocc_pred_ps, ps).data[0]
                '''
                Next, train it on x_b from the baseline.
                '''
                _, ind_obj_recons = self.baseline(data, single_obj=False)
                pred_ps = self.model(ind_obj_recons[:,1].detach())
                total_loss += self.reg_loss(pred_ps, ps[:,1]).data[0]

                if self.plot_flag:
                    if (iter + 1) == len(self.val_loader):
                        C = data.size(1)
                        '''
                        First, visualize outputs from ind_objs.
                        '''
                        unocc_pred_aff_ps = self.affine_form(unocc_pred_ps[:self.n_vis,1])
                        unocc_grid = F.affine_grid(unocc_pred_aff_ps, torch.Size([self.n_vis, C, self.in_size, self.in_size]))
                        unocc_canon_x_b = F.grid_sample(ind_objs[:self.n_vis,1], unocc_grid)
                        unocc_canon_data = F.grid_sample(data[:self.n_vis], unocc_grid)

                        '''
                        Next, visualize outputs generated from x_b from the baseline.
                        '''
                        pred_x_b = ind_obj_recons[:self.n_vis,1]
                        # We only visualize the pose prediction for x_b from the baseline.
                        pred_aff_ps = self.affine_form(pred_ps[:self.n_vis]) # (N, 2, 3)
                        grid = F.affine_grid(pred_aff_ps, torch.Size([self.n_vis, C, self.in_size, self.in_size]))
                        canon_x_b = F.grid_sample(ind_objs[:self.n_vis,1], grid)
                        canon_data = F.grid_sample(data[:self.n_vis], grid)

                        # canon_pred_x_b.
                        canon_pred_x_b = F.grid_sample(pred_x_b, grid)

                        # gt_canon_x_b.
                        gt_aff_ps = self.affine_form(ps[:self.n_vis,1])
                        gt_grid = F.affine_grid(gt_aff_ps, torch.Size([self.n_vis, C, self.in_size, self.in_size]))
                        gt_canon_x_b = F.grid_sample(ind_objs[:self.n_vis,1], gt_grid)

                        # gt_canon_pred_x_b.
                        gt_canon_pred_x_b = F.grid_sample(pred_x_b, gt_grid)

                        self.vis_data = data[:self.n_vis].data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                        self.vis_pred_x_b = pred_x_b.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                        self.vis_unocc_canon_data = unocc_canon_data.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze() 
                        self.vis_unocc_canon_x_b = unocc_canon_x_b.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()  
                        self.vis_canon_data = canon_data.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                        self.vis_canon_x_b = canon_x_b.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                        self.vis_gt_canon_x_b = gt_canon_x_b.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()


            total_ind_obj_loss /= len(self.val_loader)
            total_loss /= len(self.val_loader)

            print("Epoch {}: val_total_ind_obj_loss: {:.3f} val_total_loss: {:.3f}".format((epoch + 1), total_ind_obj_loss, total_loss))

            self.hist['val_total_loss'].append(total_loss)
            self.hist['val_total_ind_obj_loss'].append(total_ind_obj_loss)

            #self.lr_scheduler.step(total_loss, (epoch+1))
            if self.plot_flag:
                self.visualize_train_results((epoch+1))

            self.hist['per_epoch_time'].append(time.time() - epoch_start_time)
            print('per_epoch_time: {:.6f}'.format(self.hist['per_epoch_time'][-1]))

            self.save_model(save_name=self.model_type+'_checkpoint.pth.tar', trial_idx=trial_idx)
            self.save_as_pkl(obj=self.hist, save_name=self.model_type+'_checkpoint'+'_history.pkl', trial_idx=trial_idx)

            # Best total loss
            if self.hist['val_total_loss'][-1] < min(self.hist['val_total_loss'][:-1] + [float('inf')]): # [float('inf')] takes care of the case when [:-1] returns an empty list, as for epoch 1.
                self.save_model(save_name=self.model_type+'_best_val_loss.pth.tar', trial_idx=trial_idx)
                self.save_as_pkl(obj=self.hist, save_name=self.model_type+'_best_val_loss'+'_history.pkl', trial_idx=trial_idx)


        self.hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: {:.2f}, total {} epochs time: {:.2f}".format(np.mean(self.hist['per_epoch_time']),
              self.epoch, self.hist['total_time'][0]))
        print("Training finish!... save training results")


    def test(self):
        '''
        Load a pretrained baseline.
        '''
        self.baseline_save_dir = os.path.join(self.save_dir, self.data_dir, self.baseline_type)
        self.baseline.load_state_dict(torch.load(os.path.join(self.baseline_save_dir, self.baseline_type + '_best_val_loss.pth.tar')))
        print('Loaded a pretrained baseline.')
        self.baseline.eval()

        model_save_dir = os.path.join(self.save_dir, self.data_dir, self.model_type)
        self.model.load_state_dict(torch.load(os.path.join(model_save_dir, self.model_type+'_best_val_loss.pth.tar')))
        self.model.eval()

        self.init_dataset(test_set=True)

        # Validation of the main model and z_classifier.
        print('')
        print('Test begins.')
        print('')
        total_ind_obj_loss = 0.0
        total_loss = 0.0


        self.n_types_vis = 7

        self.vis_data = None
        self.vis_pred_x_b = None
        self.vis_unocc_canon_data =None
        self.vis_unocc_canon_x_b = None
        self.vis_canon_data = None 
        self.vis_canon_x_b = None 
        self.vis_gt_canon_x_b = None


        iter_start_time = time.time()
        for iter, (data, ind_objs, ps, labels) in enumerate(self.test_loader):
            data = Variable(data, volatile=True)
            ind_objs = Variable(ind_objs, volatile=True)
            ps = Variable(ps, volatile=True)
            labels = Variable(labels, volatile=True)
            if self.gpu_mode:
                data = data.cuda()
                ind_objs = ind_objs.cuda()
                ps = ps.cuda()
                labels = labels.cuda()
            '''
            First, train PoseNet on ind_objs.
            '''
            C = data.size(1)
            unocc_pred_ps = self.model(ind_objs.view(-1, C, self.out_size, self.out_size)).view(-1, self.n_objs, 4)
            total_ind_obj_loss += self.reg_loss(unocc_pred_ps, ps).data[0]
            '''
            Next, train it on x_b from the baseline.
            '''
            _, ind_obj_recons = self.baseline(data, single_obj=False)
            pred_ps = self.model(ind_obj_recons[:,1].detach())
            total_loss += self.reg_loss(pred_ps, ps[:,1]).data[0]

            if (iter + 1) == len(self.test_loader):
                C = data.size(1)
                '''
                First, visualize outputs from ind_objs.
                '''
                unocc_pred_aff_ps = self.affine_form(unocc_pred_ps[:self.n_vis,1])
                unocc_grid = F.affine_grid(unocc_pred_aff_ps, torch.Size([self.n_vis, C, self.in_size, self.in_size]))
                unocc_canon_x_b = F.grid_sample(ind_objs[:self.n_vis,1], unocc_grid)
                unocc_canon_data = F.grid_sample(data[:self.n_vis], unocc_grid)

                '''
                Next, visualize outputs generated from x_b from the baseline.
                '''
                pred_x_b = ind_obj_recons[:self.n_vis,1]
                # We only visualize the pose prediction for x_b from the baseline.
                pred_aff_ps = self.affine_form(pred_ps[:self.n_vis]) # (N, 2, 3)
                grid = F.affine_grid(pred_aff_ps, torch.Size([self.n_vis, C, self.in_size, self.in_size]))
                canon_x_b = F.grid_sample(ind_objs[:self.n_vis,1], grid)
                canon_data = F.grid_sample(data[:self.n_vis], grid)

                # canon_pred_x_b.
                canon_pred_x_b = F.grid_sample(pred_x_b, grid)

                # gt_canon_x_b.
                gt_aff_ps = self.affine_form(ps[:self.n_vis,1])
                gt_grid = F.affine_grid(gt_aff_ps, torch.Size([self.n_vis, C, self.in_size, self.in_size]))
                gt_canon_x_b = F.grid_sample(ind_objs[:self.n_vis,1], gt_grid)

                # gt_canon_pred_x_b.
                gt_canon_pred_x_b = F.grid_sample(pred_x_b, gt_grid)

                self.vis_data = data[:self.n_vis].data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                self.vis_pred_x_b = pred_x_b.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                self.vis_unocc_canon_data = unocc_canon_data.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze() 
                self.vis_unocc_canon_x_b = unocc_canon_x_b.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()  
                self.vis_canon_data = canon_data.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                self.vis_canon_x_b = canon_x_b.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                self.vis_gt_canon_x_b = gt_canon_x_b.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()

        total_ind_obj_loss /= len(self.test_loader)
        total_loss /= len(self.test_loader)

        print("total_ind_obj_loss: {:.3f} total_loss: {:.3f}".format(total_ind_obj_loss, total_loss))

        #self.lr_scheduler.step(total_loss, (epoch+1))

        self.visualize_test_results()


    def train_refiner(self):
        self.hist = {}
        self.hist['total_loss'] = []
        self.hist['loss_diff'] = []        
        self.hist['val_total_loss'] = []
        self.hist['val_loss_diff'] = []

        self.hist['per_epoch_time'] = []
        self.hist['total_time'] = []

        print('training start!')
        start_time = time.time()

        for epoch in range(self.epoch):
            self.cur_epoch = epoch + 1
            epoch_start_time = time.time()

            # Train the main model.
            print('')
            print('Train the refiner.')
            print('')

            self.refiner.train()
            iter_start_time = time.time()
            for iter, (data, ind_objs, ps, labels) in enumerate(self.train_loader):
                data = Variable(data)
                ind_objs = Variable(ind_objs)
                ps = Variable(ps)
                labels = Variable(labels)
                if self.gpu_mode:
                    data = data.cuda()
                    ind_objs = ind_objs.cuda()
                    ps = ps.cuda()
                    labels = labels.cuda()
                '''
                First, forward pass through the baseline and PoseNet.
                '''
                _, ind_obj_recons = self.baseline(data, single_obj=False)
                pred_ps = self.model(ind_obj_recons[:,1].detach())

                '''
                Next, train the refiner.
                '''
                self.refiner_optimizer.zero_grad()
                refined_ps = self.refiner(ind_obj_recons[:,1].detach(), ind_obj_recons[:,0].detach(), data.detach(), pred_ps.detach())
                total_loss = self.reg_loss(refined_ps, ps[:,1])
                total_loss.backward()
                self.refiner_optimizer.step()

                loss_diff = total_loss.data[0] - self.reg_loss(pred_ps, ps[:,1]).data[0]


                self.hist['total_loss'].append(total_loss.data[0])
                self.hist['loss_diff'].append(loss_diff)

                if (iter + 1) % 100 == 0:
                    iter_time = time.time() - iter_start_time
                    iter_start_time = time.time()
                    print("Epoch {}: [{}/{}] total_loss: {:.3f} ({:.3f} sec)".format((epoch + 1), (iter + 1), 
                        len(self.train_loader), total_loss.data[0], iter_time))


            # Validation of the main model and z_classifier.
            print('')
            print('Validation begins for the refiner.')
            print('')
            total_loss = 0.0
            loss_diff = 0.0

            self.n_types_vis = 10

            self.vis_x = None
            self.vis_pred_x_f = None
            self.vis_pred_x_b = None

            self.vis_canon_x = None 
            self.vis_canon_pred_x_f = None
            self.vis_canon_pred_x_b = None

            self.vis_refined_canon_x = None 
            self.vis_refined_canon_pred_x_f = None
            self.vis_refined_canon_pred_x_b = None

            self.vis_gt_canon_x_b = None


            self.model.eval()
            iter_start_time = time.time()
            for iter, (data, ind_objs, ps, labels) in enumerate(self.val_loader):
                data = Variable(data, volatile=True)
                ind_objs = Variable(ind_objs, volatile=True)
                ps = Variable(ps, volatile=True)
                labels = Variable(labels, volatile=True)
                if self.gpu_mode:
                    data = data.cuda()
                    ind_objs = ind_objs.cuda()
                    ps = ps.cuda()
                    labels = labels.cuda()

                '''
                First, forward pass through the baseline and PoseNet.
                '''
                _, ind_obj_recons = self.baseline(data, single_obj=False)
                pred_ps = self.model(ind_obj_recons[:,1].detach())

                '''
                Next, train the refiner.
                '''
                refined_ps = self.refiner(ind_obj_recons[:,1].detach(), ind_obj_recons[:,0].detach(), data.detach(), pred_ps.detach())
                total_loss += self.reg_loss(refined_ps, ps[:,1]).data[0]
                loss_diff += self.reg_loss(refined_ps, ps[:,1]).data[0] - self.reg_loss(pred_ps, ps[:,1]).data[0]


                if (iter + 1) == len(self.val_loader):
                    C = data.size(1)
                    # x, pred_x_f, pred_x_b
                    x = data[:self.n_vis]
                    pred_x_f = ind_obj_recons[:self.n_vis, 0]
                    pred_x_b = ind_obj_recons[:self.n_vis, 1]

                    # canon_x, canon_pred_x_f, canon_pred_x_b
                    pred_aff_ps = self.affine_form(pred_ps[:self.n_vis])
                    grid = F.affine_grid(pred_aff_ps, torch.Size([self.n_vis, C, self.in_size, self.in_size]))
                    canon_x = F.grid_sample(x, grid)
                    canon_pred_x_f = F.grid_sample(pred_x_f, grid)
                    canon_pred_x_b = F.grid_sample(pred_x_b, grid)

                    # refined_canon_x, refined_canon_pred_x_f, refined_canon_pred_x_b
                    refined_aff_ps = self.affine_form(refined_ps[:self.n_vis])
                    grid = F.affine_grid(refined_aff_ps, torch.Size([self.n_vis, C, self.in_size, self.in_size]))
                    refined_canon_x = F.grid_sample(x, grid)
                    refined_canon_pred_x_f = F.grid_sample(pred_x_f, grid)
                    refined_canon_pred_x_b = F.grid_sample(pred_x_b, grid)

                    # gt_canon_x_b
                    gt_aff_ps = self.affine_form(ps[:self.n_vis,1])
                    grid = F.affine_grid(gt_aff_ps, torch.Size([self.n_vis, C, self.in_size, self.in_size]))
                    gt_canon_x_b = F.grid_sample(ind_objs[:self.n_vis,1], grid)

                    self.vis_x = x.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                    self.vis_pred_x_f = pred_x_f.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                    self.vis_pred_x_b = pred_x_b.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()

                    self.vis_canon_x = canon_x.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                    self.vis_canon_pred_x_f = canon_pred_x_f.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                    self.vis_canon_pred_x_b = canon_pred_x_b.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()

                    self.vis_refined_canon_x = refined_canon_x.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                    self.vis_refined_canon_pred_x_f = refined_canon_pred_x_f.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                    self.vis_refined_canon_pred_x_b = refined_canon_pred_x_b.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()

                    self.vis_gt_canon_x_b = gt_canon_x_b.data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()


            total_loss /= len(self.val_loader)
            loss_diff /= len(self.val_loader)

            print("Epoch {}: val_total_loss: {:.3f}".format((epoch + 1), total_loss))

            self.hist['val_total_loss'].append(total_loss)
            self.hist['val_loss_diff'].append(loss_diff)

            #self.lr_scheduler.step(total_loss, (epoch+1))

            self.visualize_train_refiner_results((epoch+1))

            self.hist['per_epoch_time'].append(time.time() - epoch_start_time)
            print('per_epoch_time: {:.6f}'.format(self.hist['per_epoch_time'][-1]))

            self.save_model(save_dir=os.path.join(self.save_dir, self.data_dir, self.refiner_type), 
                save_name=self.refiner_type+'_checkpoint.pth.tar')
            self.save_as_pkl(obj=self.hist, save_dir=os.path.join(self.save_dir, self.data_dir, self.refiner_type),
                save_name=self.refiner_type+'_checkpoint'+'_history.pkl')

            # Best total loss
            if self.hist['val_total_loss'][-1] < min(self.hist['val_total_loss'][:-1] + [float('inf')]): # [float('inf')] takes care of the case when [:-1] returns an empty list, as for epoch 1.
                self.save_model(save_dir=os.path.join(self.save_dir, self.data_dir, self.refiner_type), 
                    save_name=self.refiner_type+'_best_val_loss.pth.tar')
                self.save_as_pkl(obj=self.hist, save_dir=os.path.join(self.save_dir, self.data_dir, self.refiner_type),  
                    save_name=self.refiner_type+'_best_val_loss_history.pkl')


        self.hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: {:.2f}, total {} epochs time: {:.2f}".format(np.mean(self.hist['per_epoch_time']),
              self.epoch, self.hist['total_time'][0]))
        print("Training finish!... save training results")


    # def train_pfc(self):
    #     self.hist = {}
    #     self.hist['total_loss'] = []
    #     self.hist['val_total_loss'] = []

    #     self.hist['per_epoch_time'] = []
    #     self.hist['total_time'] = []

    #     print('training start!')
    #     start_time = time.time()

    #     for epoch in range(self.pfc_epoch):
    #         self.cur_epoch = epoch + 1
    #         epoch_start_time = time.time()

    #         # Train the main model.
    #         print('')
    #         print('Train the main model.')
    #         print('')

    #         self.pfc.train()
    #         # self.model.G.eval() # We do not set G to eval mode to make the gradient flow better.
    #         iter_start_time = time.time()
    #         for iter, (data, ind_objs, ps, _) in enumerate(self.train_loader):
    #             N, C, _, _ = data.size()
    #             data = Variable(data)
    #             ind_objs = Variable(ind_objs)
    #             ps = Variable(ind_ps)
    #             if self.gpu_mode:
    #                 data = data.cuda()
    #                 ind_objs = ind_objs.cuda()
    #                 ps = ind_ps.cuda()

    #             # Since the parameters of G in self.model are not in the param group of the optimizer,
    #             # we should do self.model.zero_grad() instead of the usual self.optimizer.zero_grad()
    #             self.pfc.zero_grad()
    #             if self.pfc_type in 'PFC1Clipping':




    #                 grid = F.affine_grid(aff_bgd_ps, torch.Size([N, C, self.in_size, self.in_size]))
    #                 canon_x = F.grid_sample(data, grid)
    #                 canon_x_f = F.grid_sample(ind_objs[:,0], grid)
    #                 canon_x_b = F.grid_sample(ind_objs[:,1], grid)
    #                 # Returned variable is the last recon.
    #                 _ = self.model(canon_x, canon_x_f)
    #                 # We minimize mse_loss between the entire recon sequence and the target.
    #                 # Note that we do not optimize self.recons[:,0] because it corresponds to z0.
    #                 total_loss = self.mse_loss(self.model.recons[:,1:], canon_x_b.unsqueeze(1).expand(-1, self.n_iter, -1, -1, -1))

    #                 _ = self.model(data, ind_objs[:,0])
    #                 total_loss = self.mse_loss(self.model.recons[:,1:], canon_x_b.unsqueeze(1).expand(-1, self.n_iter, -1, -1, -1))

    #             total_loss.backward()
    #             self.optimizer.step()

    #             self.hist['total_loss'].append(total_loss.data[0])

    #             if (iter + 1) % 100 == 0:
    #                 iter_time = time.time() - iter_start_time
    #                 iter_start_time = time.time()
    #                 print("Epoch {}: [{}/{}] total_loss: {:.3f} ({:.3f} sec)".format((epoch + 1), (iter + 1), len(self.train_loader), total_loss.data[0], iter_time))


    #         # Validation of the main model and z_classifier.
    #         print('')
    #         print('Validation begins.')
    #         print('')
    #         total_loss = 0.0

    #         self.vis_x = None
    #         self.vis_x_f = None
    #         self.vis_recons = None
    #         self.vis_target = None

    #         self.model.eval()
    #         iter_start_time = time.time()
    #         for iter, (data, ind_objs, ind_ps, _) in enumerate(self.val_loader):
    #             N, C, _, _ = data.size()
    #             data = Variable(data, volatile=True)
    #             ind_objs = Variable(ind_objs, volatile=True)
    #             ind_ps = Variable(ind_ps, volatile=True)
    #             if self.gpu_mode:
    #                 data = data.cuda()
    #                 ind_objs = ind_objs.cuda()
    #                 ind_ps = ind_ps.cuda()


    #             if self.model_type in ['PFC1', 'PFC1Clipping']:
    #                 # Map data and the foreground image, ind_objs[:,0], into the canonical pose.
    #                 aff_bgd_ps = self.affine_form(ind_ps[:,1])
    #                 grid = F.affine_grid(aff_bgd_ps, torch.Size([N, C, self.in_size, self.in_size]))
    #                 canon_x = F.grid_sample(data, grid)
    #                 canon_x_f = F.grid_sample(ind_objs[:,0], grid)
    #                 canon_x_b = F.grid_sample(ind_objs[:,1], grid)
    #                 # Returned variable is the last recon.
    #                 _ = self.model(canon_x, canon_x_f)
    #                 # We minimize mse_loss between the entire recon sequence and the target.
    #                 # Note that we do not optimize self.recons[:,0] because it corresponds to z0.
    #                 total_loss += self.mse_loss(self.model.recons[:,1:], canon_x_b.unsqueeze(1).expand(-1, self.n_iter, -1, -1, -1)).data[0]
    #             elif self.model_type in ['PFC2', 'PFC3', 'PFC4']:
    #                 aff_bgd_ps = self.affine_form(ind_ps[:,1])
    #                 grid = F.affine_grid(aff_bgd_ps, torch.Size([N, C, self.in_size, self.in_size]))
    #                 canon_x_b = F.grid_sample(ind_objs[:,1], grid)
                    
    #                 _ = self.model(data, ind_objs[:,0])
    #                 total_loss += self.mse_loss(self.model.recons[:,1:], canon_x_b.unsqueeze(1).expand(-1, self.n_iter, -1, -1, -1)).data[0]


    #             if (iter + 1) == len(self.val_loader):
    #                 if self.model_type in ['PFC1', 'PFC1Clipping']:
    #                     self.vis_x = canon_x[:self.n_vis].data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
    #                     self.vis_x_f = canon_x_f[:self.n_vis].data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
    #                 elif self.model_type == 'PFC2':
    #                     self.vis_x = data[:self.n_vis].data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
    #                     self.vis_x_f = ind_objs[:self.n_vis,0].data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
    #                 elif self.model_type in ['PFC3', 'PFC4']:
    #                     self.vis_x = self.model.canon_x[:self.n_vis].data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
    #                     self.vis_x_f = self.model.canon_x_f[:self.n_vis].data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
    #                 self.vis_recons = self.model.recons[:self.n_vis].data.cpu().numpy().transpose(0, 1, 3, 4, 2).squeeze()
    #                 self.vis_target = canon_x_b[:self.n_vis].data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()

    #         total_loss /= len(self.val_loader)

    #         print("Epoch {}: val_total_loss {:.3f}".format((epoch + 1), total_loss))
    #         self.hist['val_total_loss'].append(total_loss)

    #         self.hist['per_epoch_time'].append(time.time() - epoch_start_time)
    #         print('per_epoch_time: {:.6f}'.format(self.hist['per_epoch_time'][-1]))

    #         self.visualize_train_results((epoch+1))

    #         self.save_model(save_name=self.model_type+'_checkpoint.pth.tar')
    #         self.save_as_pkl(obj=self.hist, save_name=self.model_type+'_checkpoint'+'_history.pkl')

    #         if self.hist['val_total_loss'][-1] < min(self.hist['val_total_loss'][:-1] + [float('inf')]):
    #             self.save_model(save_name=self.model_type+'_best_val_total_loss.pth.tar')
    #             self.save_as_pkl(obj=self.hist, save_name=self.model_type+'_best_val_total_loss'+'_history.pkl')


    #     self.hist['total_time'].append(time.time() - start_time)
    #     print("Avg one epoch time: {:.2f}, total {} epochs time: {:.2f}".format(np.mean(self.hist['per_epoch_time']),
    #           self.epoch, self.hist['total_time'][0]))
    #     print("Training finish!... save training results")



    def affine_form(self, p, scale_positive=True, eps=1e-6):
        '''
        Args:
        p: (N, 4). s = p[0], a = p[1], x = p[2], y = p[3].
        s, a, x, y: respectively scale, angle, x, y translations. They all have shapes (N,).
        a, x, y are unconstrained.
        
        Return:
        aff_p: (N, 2, 3)-shaped affine parameters corresponding to them.
        '''
        N = p.size(0)
        aff_p = Variable(torch.zeros(N, 2, 3))
        if p.is_cuda:
            aff_p = aff_p.cuda()
        if scale_positive:
            s = p[:,0]
        else:
            s = eps + self.softplus(p[:,0])
        aff_p[:,0,0] = s*torch.cos(p[:,1])
        aff_p[:,0,1] = -s*torch.sin(p[:,1])
        aff_p[:,1,0] = s*torch.sin(p[:,1])
        aff_p[:,1,1] = s*torch.cos(p[:,1])
        aff_p[:,0,2] = p[:,2]
        aff_p[:,1,2] = p[:,3]

        return aff_p

    def visualize_train_baseline_results(self, hist, epoch):
        plot_dir = os.path.join(self.result_dir, self.data_dir, self.baseline_type)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        utils.show_ind_obj_recons(self.vis_ind_objs, self.vis_unocc_ind_obj_recons, self.vis_ind_obj_recons, None, plot_dir, 'ind_obj_recons_epoch{}.png'.format(epoch))

        utils.multi_object_classifier_loss_plot(hist, plot_dir, self.baseline_type)
        utils.val_multi_object_classifier_loss_plot(hist, plot_dir, self.baseline_type)
        utils.multi_object_classifier_acc_plot(hist, plot_dir, self.baseline_type)
        utils.val_multi_object_classifier_acc_plot(hist, plot_dir, self.baseline_type)
        utils.per_epoch_time_plot(hist, plot_dir, self.baseline_type)



    def visualize_train_results(self, epoch):
        save_dir = os.path.join(self.result_dir, self.data_dir, self.model_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Images
        fig = plt.figure(figsize=(40, 30))
        if self.debug:
            self.n_vis_per_column = 1
        else:
            self.n_vis_per_column = 5
        outer = gridspec.GridSpec(self.n_vis//self.n_vis_per_column, self.n_vis_per_column, wspace=0.2, hspace=0.2)
        for i in range(self.n_vis):
            self.visualize_train_results_helper(fig, outer[i], i)        

        fig.savefig(os.path.join(save_dir, 'vis_epoch{}.png'.format(epoch)))
        plt.close(fig)


        # plot train and val loss.
        loss_plot_path = os.path.join(self.result_dir, self.data_dir, self.model_type)
        if not os.path.exists(loss_plot_path):
            os.makedirs(loss_plot_path)
        utils.train_loss_plot(self.hist, loss_plot_path, self.model_type)        
        utils.val_loss_plot(self.hist, loss_plot_path, self.model_type)        

    def visualize_train_results_helper(self, fig, outer, idx):
        inner = gridspec.GridSpecFromSubplotSpec(1, self.n_types_vis, subplot_spec=outer, wspace=0.2, hspace=0.2)
        for j in range(self.n_types_vis):
            ax = fig.add_subplot(inner[j])
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            if j == 0:
                ax.imshow(self.vis_data[idx], vmin=0, vmax=1)
            elif j == 1:
                ax.imshow(self.vis_pred_x_b[idx], vmin=0, vmax=1)
            elif j == 2:
                ax.imshow(self.vis_unocc_canon_data[idx], vmin=0, vmax=1)
            elif j == 3:
                ax.imshow(self.vis_unocc_canon_x_b[idx], vmin=0, vmax=1)
            elif j == 4:
                ax.imshow(self.vis_canon_data[idx], vmin=0, vmax=1)
            elif j == 5:
                ax.imshow(self.vis_canon_x_b[idx], vmin=0, vmax=1)
            elif j == 6:
                ax.imshow(self.vis_gt_canon_x_b[idx], vmin=0, vmax=1)


    def visualize_test_results(self):
        save_dir = os.path.join(self.result_dir, self.data_dir, self.model_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Images
        fig = plt.figure(figsize=(40, 30))
        if self.debug:
            self.n_vis_per_column = 1
        else:
            self.n_vis_per_column = 5
        outer = gridspec.GridSpec(self.n_vis//self.n_vis_per_column, self.n_vis_per_column, wspace=0.2, hspace=0.2)
        for i in range(self.n_vis):
            self.visualize_train_results_helper(fig, outer[i], i)

        fig.savefig(os.path.join(save_dir, 'test_set_vis.png'))
        plt.close(fig)


    def visualize_train_refiner_results(self, epoch):
        save_dir = os.path.join(self.result_dir, self.data_dir, self.refiner_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Images
        fig = plt.figure(figsize=(40, 30))
        if self.debug:
            self.n_vis_per_column = 1
        else:
            self.n_vis_per_column = 2
        outer = gridspec.GridSpec(self.n_vis//self.n_vis_per_column, self.n_vis_per_column, wspace=0.2, hspace=0.2)
        for i in range(self.n_vis):
            self.visualize_train_refiner_results_helper(fig, outer[i], i)        

        fig.savefig(os.path.join(save_dir, 'vis_epoch{}.png'.format(epoch)))
        plt.close(fig)


        # plot train and val loss.
        loss_plot_path = os.path.join(self.result_dir, self.data_dir, self.refiner_type)
        if not os.path.exists(loss_plot_path):
            os.makedirs(loss_plot_path)
        utils.train_loss_plot(self.hist, loss_plot_path, self.refiner_type)        
        utils.val_loss_plot(self.hist, loss_plot_path, self.refiner_type)        


    def visualize_train_refiner_results_helper(self, fig, outer, idx):
        inner = gridspec.GridSpecFromSubplotSpec(1, self.n_types_vis, subplot_spec=outer, wspace=0.2, hspace=0.2)
        for j in range(self.n_types_vis):
            ax = fig.add_subplot(inner[j])
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            if j == 0:
                ax.imshow(self.vis_x[idx], vmin=0, vmax=1)
            elif j == 1:
                ax.imshow(self.vis_pred_x_f[idx], vmin=0, vmax=1)
            elif j == 2:
                ax.imshow(self.vis_pred_x_b[idx], vmin=0, vmax=1)
            elif j == 3:
                ax.imshow(self.vis_canon_x[idx], vmin=0, vmax=1)
            elif j == 4:
                ax.imshow(self.vis_canon_pred_x_f[idx], vmin=0, vmax=1)
            elif j == 5:
                ax.imshow(self.vis_canon_pred_x_b[idx], vmin=0, vmax=1)
            elif j == 6:
                ax.imshow(self.vis_refined_canon_x[idx], vmin=0, vmax=1)
            elif j == 7:
                ax.imshow(self.vis_refined_canon_pred_x_f[idx], vmin=0, vmax=1)
            elif j == 8:
                ax.imshow(self.vis_refined_canon_pred_x_b[idx], vmin=0, vmax=1)
            elif j == 9:
                ax.imshow(self.vis_gt_canon_x_b[idx], vmin=0, vmax=1)



    def save_model(self, model_state_dict=None, save_dir=None, save_name='.pth.tar', trial_idx=None):
        if model_state_dict is None:
            model_state_dict = self.model.state_dict()
        if save_dir is None:
            save_dir = os.path.join(self.save_dir, self.data_dir, self.model_type)
        if trial_idx is not None:
            save_dir = os.path.join(save_dir, 'trial_{}'.format(trial_idx+1))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model_state_dict, os.path.join(save_dir, save_name))

    def save_as_pkl(self, obj, save_dir=None, save_name='.pkl', trial_idx=None):
        if save_dir is None:
                save_dir = os.path.join(self.save_dir, self.data_dir, self.model_type)
        if trial_idx is not None:
            save_dir = os.path.join(save_dir, 'trial_{}'.format(trial_idx+1))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, save_name), 'wb') as f:
            pickle.dump(obj, f)

    def record_hps_and_results(self, save_dir=None, save_name='.txt', trial_idx=None):
        if save_dir is None:
            save_dir = os.path.join(self.save_dir, self.data_dir, self.model_type)
        if trial_idx is not None:
            save_dir = os.path.join(save_dir, 'trial_{}'.format(trial_idx+1))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Output a text file summarizing the parameters that generated the dataset.
        # We can record not only fixed, initial hps, but also constantly changing ones, such as decaying lr, if we decay lr.
        with open(os.path.join(save_dir, save_name), 'w') as summary:
            summary.write('epochs trained: {}'.format(self.cur_epoch))