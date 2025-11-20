import utils, torch, time, os, pickle
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from search_space_canon_gans import generator
from pfc_models import *
from pfc_classifiers import *
from new_pose_net import *


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class PFCModelsExp(object):
    def __init__(self, args):
        self.train_flag = args.train
        self.debug = args.debug
        self.plot_flag = args.plot
        self.n_trials = args.n_trials

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

        self.pose_type = args.pose_type

        if self.pose_type == 'pred':
            self.baseline_type = args.baseline_type
            self.baseline_save_dir = args.baseline_save_dir

            self.posenet_type = args.posenet_type
            self.posenet_save_dir = args.posenet_save_dir

        self.model_type = args.model_type
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.epoch = args.epoch

        self.gan_model_type = args.gan_model_type
        self.gan_save_dir = args.gan_save_dir

        self.reduced_data_dir = args.reduced_data_dir
        
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


        self.init_dataset()


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

    def train(self, trial_idx=None):

        if self.pose_type == 'pred':
            # For training PFC, we need a pretrained Baseline and PoseNet.
            if self.baseline_type == 'Baseline':
                self.baseline = Baseline()        
            self.baseline_save_dir = os.path.join(self.baseline_save_dir, self.data_dir, self.baseline_type)
            if trial_idx is not None:
                self.baseline_save_dir = os.path.join(self.baseline_save_dir, 'trial_{}'.format(trial_idx+1))
            self.baseline.load_state_dict(torch.load(os.path.join(self.baseline_save_dir, self.baseline_type + '_best_val_loss.pth.tar')))

            if self.posenet_type == 'PoseNetBasicDetached':
                self.posenet = PoseNetBasicDetached()
            self.posenet_save_dir = os.path.join(self.posenet_save_dir, self.data_dir, self.posenet_type)
            if trial_idx is not None:
                self.posenet_save_dir = os.path.join(self.posenet_save_dir, 'trial_{}'.format(trial_idx+1))
            self.posenet.load_state_dict(torch.load(os.path.join(self.posenet_save_dir, self.posenet_type + '_best_val_loss.pth.tar')))

            if self.gpu_mode:
                self.baseline.cuda()
                self.posenet.cuda()

        # load a pretrained gan.
        if self.gan_model_type == 'GAN':
            self.G = generator()

        self.gan_save_dir = os.path.join(self.gan_save_dir, self.reduced_data_dir, self.gan_model_type)
        if trial_idx is not None:
            self.gan_save_dir = os.path.join(self.gan_save_dir, 'trial_{}'.format(trial_idx+1))
        self.G.load_state_dict(torch.load(os.path.join(self.gan_save_dir, self.gan_model_type + '_G.pth.tar')))

        if self.pose_type in ['gt', 'pred']:
            input_size = 28
        elif self.pose_type == 'original':
            input_size = 50


        if self.model_type == 'PFC':
            self.model = PFC1Clipping(G=self.G, input_size=input_size)
            self.n_iter = self.model.n_iter
        elif self.model_type == 'PFCNoFG':
            self.model = PFC1ClippingNoFG(G=self.G, input_size=input_size)
            self.n_iter = self.model.n_iter   

        self.optimizer = optim.Adam(list(set(self.model.parameters())-set(self.model.G.parameters())), lr=self.lr)       
        self.recon_loss = nn.MSELoss()

        if self.gpu_mode:
            self.model.cuda()


        self.hist = {}
        self.hist['total_loss'] = []
        self.hist['val_total_loss'] = []

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
            # self.model.G.eval() # We do not set G to eval mode to make the gradient flow better.
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

                self.model.zero_grad()

                # We need gt pose for the target, canon_x_b for all pose types.
                aff_gt_ps = self.affine_form(ind_ps[:,1])
                gt_grid = F.affine_grid(aff_gt_ps, torch.Size([N, C, self.in_size, self.in_size]))
                canon_x_b = F.grid_sample(ind_objs[:,1], gt_grid)

                # Estimate the pose.
                if self.pose_type == 'gt':
                    if self.model_type == 'PFC':
                        canon_x = F.grid_sample(data, gt_grid)
                        canon_x_f = F.grid_sample(ind_objs[:,0], gt_grid)
                        # Returned variable is the last recon.
                        _ = self.model(canon_x, canon_x_f)
                        total_loss = self.recon_loss(self.model.recons[:,1:], canon_x_b.unsqueeze(1).expand(-1, self.n_iter, -1, -1, -1))
                    elif self.model_type == 'PFCNoFG':
                        canon_x = F.grid_sample(data, gt_grid)
                        _ = self.model(canon_x)
                        total_loss = self.recon_loss(self.model.recons[:,1:], canon_x_b.unsqueeze(1).expand(-1, self.n_iter, -1, -1, -1))
                elif self.pose_type == 'pred':
                    _, ind_obj_recons = self.baseline(data, single_obj=False)
                    pred_ps = self.posenet(ind_obj_recons[:,1].detach())
                    aff_ps = self.affine_form(pred_ps.detach())
                    grid = F.affine_grid(aff_ps, torch.Size([N, C, self.in_size, self.in_size]))
                    if self.model_type == 'PFC':
                        canon_x = F.grid_sample(data, grid)
                        canon_x_f = F.grid_sample(ind_obj_recons[:,0], grid)
                        _ = self.model(canon_x, canon_x_f)
                        total_loss = self.recon_loss(self.model.recons[:,1:], canon_x_b.unsqueeze(1).expand(-1, self.n_iter, -1, -1, -1))
                    elif self.model_type == 'PFCNoFG':
                        canon_x = F.grid_sample(data, grid)
                        _ = self.model(canon_x)
                        total_loss = self.recon_loss(self.model.recons[:,1:], canon_x_b.unsqueeze(1).expand(-1, self.n_iter, -1, -1, -1))
                elif self.pose_type == 'original':
                    if self.model_type == 'PFC':
                        _ = self.model(data, ind_objs[:,0])
                        total_loss = self.recon_loss(self.model.recons[:,1:], canon_x_b.unsqueeze(1).expand(-1, self.n_iter, -1, -1, -1))
                    elif self.model_type == 'PFCNoFG':
                        _ = self.model(data)
                        total_loss = self.recon_loss(self.model.recons[:,1:], canon_x_b.unsqueeze(1).expand(-1, self.n_iter, -1, -1, -1))


                total_loss.backward()
                self.optimizer.step()

                self.hist['total_loss'].append(total_loss.data[0])

                if (iter + 1) % 100 == 0:
                    iter_time = time.time() - iter_start_time
                    iter_start_time = time.time()
                    print("Epoch {}: [{}/{}] total_loss: {:.3f} ({:.3f} sec)".format((epoch + 1), (iter + 1), len(self.train_loader), total_loss.data[0], iter_time))


            # Validation of the main model and z_classifier.
            print('')
            print('Validation begins.')
            print('')
            total_loss = 0.0

            if self.plot_flag:
                self.vis_x = None
                self.vis_x_f = None
                self.vis_recons = None
                self.vis_target = None

            self.model.eval()
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


                # We need gt pose for the target, canon_x_b for all pose types.
                aff_gt_ps = self.affine_form(ind_ps[:,1])
                gt_grid = F.affine_grid(aff_gt_ps, torch.Size([N, C, self.in_size, self.in_size]))
                canon_x_b = F.grid_sample(ind_objs[:,1], gt_grid)

                # Estimate the pose.
                if self.pose_type == 'gt':
                    if self.model_type == 'PFC':
                        canon_x = F.grid_sample(data, gt_grid)
                        canon_x_f = F.grid_sample(ind_objs[:,0], gt_grid)
                        # Returned variable is the last recon.
                        _ = self.model(canon_x, canon_x_f)
                        total_loss += self.recon_loss(self.model.recons[:,1:], canon_x_b.unsqueeze(1).expand(-1, self.n_iter, -1, -1, -1)).data[0]
                    elif self.model_type == 'PFCNoFG':
                        canon_x = F.grid_sample(data, gt_grid)
                        _ = self.model(canon_x)
                        total_loss += self.recon_loss(self.model.recons[:,1:], canon_x_b.unsqueeze(1).expand(-1, self.n_iter, -1, -1, -1)).data[0]
                elif self.pose_type == 'pred':
                    _, ind_obj_recons = self.baseline(data, single_obj=False)
                    pred_ps = self.posenet(ind_obj_recons[:,1].detach())
                    aff_ps = self.affine_form(pred_ps.detach())
                    grid = F.affine_grid(aff_ps, torch.Size([N, C, self.in_size, self.in_size]))
                    if self.model_type == 'PFC':
                        canon_x = F.grid_sample(data, grid)
                        canon_x_f = F.grid_sample(ind_obj_recons[:,0], grid)
                        _ = self.model(canon_x, canon_x_f)
                        total_loss += self.recon_loss(self.model.recons[:,1:], canon_x_b.unsqueeze(1).expand(-1, self.n_iter, -1, -1, -1)).data[0]
                    elif self.model_type == 'PFCNoFG':
                        canon_x = F.grid_sample(data, grid)
                        _ = self.model(canon_x)
                        total_loss += self.recon_loss(self.model.recons[:,1:], canon_x_b.unsqueeze(1).expand(-1, self.n_iter, -1, -1, -1)).data[0]
                elif self.pose_type == 'original':
                    if self.model_type == 'PFC':
                        canon_x = data
                        canon_x_f = ind_objs[:,0]
                        _ = self.model(canon_x, canon_x_f)
                        total_loss += self.recon_loss(self.model.recons[:,1:], canon_x_b.unsqueeze(1).expand(-1, self.n_iter, -1, -1, -1)).data[0]
                    elif self.model_type == 'PFCNoFG':
                        canon_x = data
                        _ = self.model(canon_x)
                        total_loss += self.recon_loss(self.model.recons[:,1:], canon_x_b.unsqueeze(1).expand(-1, self.n_iter, -1, -1, -1)).data[0]

                if self.plot_flag:
                    if (iter + 1) == len(self.val_loader):
                        if self.model_type == 'PFC':
                            self.vis_x = canon_x[:self.n_vis].data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                            self.vis_x_f = canon_x_f[:self.n_vis].data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                        elif self.model_type == 'PFCNoFG':
                            self.vis_x = canon_x[:self.n_vis].data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                        self.vis_recons = self.model.recons[:self.n_vis].data.cpu().numpy().transpose(0, 1, 3, 4, 2).squeeze()
                        self.vis_target = canon_x_b[:self.n_vis].data.cpu().numpy().transpose(0, 2, 3, 1).squeeze()



            total_loss /= len(self.val_loader)

            print("Epoch {}: val_total_loss {:.3f}".format((epoch + 1), total_loss))
            self.hist['val_total_loss'].append(total_loss)

            self.hist['per_epoch_time'].append(time.time() - epoch_start_time)
            print('per_epoch_time: {:.6f}'.format(self.hist['per_epoch_time'][-1]))

            if self.plot_flag:
                self.visualize_train_results((epoch+1), trial_idx=trial_idx)

            self.save_model(save_name=self.model_type+'_checkpoint.pth.tar', trial_idx=trial_idx)
            self.save_as_pkl(obj=self.hist, save_name=self.model_type+'_checkpoint'+'_history.pkl', trial_idx=trial_idx)

            if self.hist['val_total_loss'][-1] < min(self.hist['val_total_loss'][:-1] + [float('inf')]):
                self.save_model(save_name=self.model_type+'_best_val_total_loss.pth.tar', trial_idx=trial_idx)
                self.save_as_pkl(obj=self.hist, save_name=self.model_type+'_best_val_total_loss'+'_history.pkl', trial_idx=trial_idx)


        self.hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: {:.2f}, total {} epochs time: {:.2f}".format(np.mean(self.hist['per_epoch_time']),
              self.epoch, self.hist['total_time'][0]))
        print("Training finish!... save training results")
    



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


    def visualize_train_results(self, epoch, trial_idx=None):
        save_path = os.path.join(self.result_dir, self.pose_type, self.data_name, self.model_type)
        if trial_idx is not None:
            save_path = os.path.join(save_path, 'trial_{}'.format(trial_idx+1))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fig = plt.figure(figsize=(30,50))
        outer = gridspec.GridSpec(self.n_vis, self.n_iter+4, wspace=0.2, hspace=0.2) # 4 for canon_x, canon_x_f, G(z_0), and target.
        for i in range(self.n_vis):
            ax = fig.add_subplot(outer[i, 0])
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.imshow(self.vis_x[i], vmin=0, vmax=1)

            if self.model_type == 'PFC':
                ax = fig.add_subplot(outer[i, 1])
                ax.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                ax.imshow(self.vis_x_f[i], vmin=0, vmax=1)

            for j in range(self.n_iter+1):
                ax = fig.add_subplot(outer[i, j+2])
                ax.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                ax.imshow(self.vis_recons[i, j], vmin=0, vmax=1)

            ax = fig.add_subplot(outer[i, self.n_iter+3])
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.imshow(self.vis_target[i], vmin=0, vmax=1)   

        fig.savefig(os.path.join(save_path, 'vis_epoch{}.png'.format(epoch)))
        plt.close(fig)

        # plot train and val loss.
        loss_plot_path = os.path.join(self.result_dir, self.pose_type, self.data_name, self.model_type)
        if trial_idx is not None:
            loss_plot_path = os.path.join(loss_plot_path, str(trial_idx+1))
        if not os.path.exists(loss_plot_path):
            os.makedirs(loss_plot_path)
        utils.train_loss_plot(self.hist, loss_plot_path, self.model_type)        
        utils.val_loss_plot(self.hist, loss_plot_path, self.model_type)


    def save_model(self, model_state_dict=None, save_dir=None, save_name='.pth.tar', trial_idx=None):
        if model_state_dict is None:
            model_state_dict = self.model.state_dict()
        if save_dir is None:
            save_dir = os.path.join(self.save_dir, self.pose_type, self.data_name, self.model_type)
        if trial_idx is not None:
            save_dir = os.path.join(save_dir, 'trial_{}'.format(trial_idx+1))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model_state_dict, os.path.join(save_dir, save_name))

    def save_as_pkl(self, obj, save_dir=None, save_name='.pkl', trial_idx=None):
        if save_dir is None:
            save_dir = os.path.join(self.save_dir, self.pose_type, self.data_name, self.model_type)
        if trial_idx is not None:
            save_dir = os.path.join(save_dir, 'trial_{}'.format(trial_idx+1))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)                
        with open(os.path.join(save_dir, save_name), 'wb') as f:
            pickle.dump(obj, f)