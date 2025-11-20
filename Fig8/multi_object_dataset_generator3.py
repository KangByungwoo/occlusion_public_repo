'''
Unlike multi_object_dataset_generator3.py, we transform a 28 by 28 FashionMNIST image into a 50 by 50 image directly using Spatial Transformer,
using a randomly sampled pose p. We also keep the label for p.
'''
import utils, torch, time, os, pickle, argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import itertools
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'

def parse_args():
    parser = argparse.ArgumentParser()
    # As of 03/27/18, we distinguish different settings, including both training and architectural hps, of the same underlying model as different numbers.
    parser.add_argument('--debug', action='store_true', default=False)
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
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_train', type=int, default=50000) # For the actual dataset: 50000
    parser.add_argument('--n_val', type=int, default=10000) # For the actual dataset: 10000
    parser.add_argument('--n_test', type=int, default=10000) # For the actual dataset: 10000
    parser.add_argument('--n_vis', type=int, default=100)
    parser.add_argument('--reduced_data_dir', type=str, default='./reduced_fashion_mnist')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--gpu_mode', action='store_true', default=True)
    parser.add_argument('--gpu_idx', type=int, default=1)

    return check_args(parser.parse_args())

def check_args(args):
    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --gpu_mode:
    if args.gpu_mode:
        try:
            assert torch.cuda.is_available()
        except:
            print('cuda is not available.')

    return args


class MultiObjectDatasetGenerator3(object):
    def __init__(self, args):
        self.debug = args.debug
        self.data_type = args.data_type
        self.data_rotate = args.data_rotate
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
        self.batch_size = args.batch_size
        if self.debug:
            self.n_train = 50
            self.n_val = 10
            self.n_test = 10
        else:
            self.n_train = args.n_train
            self.n_val = args.n_val
            self.n_test = args.n_test
        self.n_vis = args.n_vis
        self.save_dir = args.save_dir
        self.reduced_data_dir = args.reduced_data_dir
        self.gpu_mode = args.gpu_mode
        self.gpu_idx = args.gpu_idx
        # set gpu device
        if self.gpu_mode:
            torch.cuda.set_device(self.gpu_idx)
        self.padding = ((self.out_size - self.in_size)//2,)*4

        self.n_vr_bins = round((self.vr_max-self.vr_min)/self.vr_bin_size)
        self.vr_bins = list(np.arange(self.vr_min, self.vr_max, self.vr_bin_size)) # for e.g. [0.2, 0.25, 0.3, 0.35] if self.vr_min = 0.2, sefl.vr_max = 0.4, self.vr_bin_size = 0.05.
        self.valid_classes_flag = args.valid_classes 
        self.valid_classes = [int(k) for k in self.valid_classes_flag] # This gives [0, 2, 8, 9] for '0289'.

        if self.data_rotate:
            self.data_rotate_flag = '_vr_{:.2f}_{:.2f}_scale_{:.1f}_{:.1f}_angle_{:.0f}_{:.0f}_trans_{:.2f}_{:.2f}'.format(self.vr_min, self.vr_max, self.scale_min, self.scale_max, 
                self.angle_min*180/math.pi, self.angle_max*180/math.pi, self.trans_min, self.trans_max)
        else:
            self.data_rotate_flag = ''
        
        if self.debug:
            self.save_dir = os.path.join('debugging_gen3_multi_' + self.data_type + self.data_rotate_flag + '_' + self.valid_classes_flag,
                                        '{}_{}_{}_{:.0e}_{:.0e}'.format(self.n_objs, self.in_size, self.out_size, self.eps, self.eps2))
        else:
            self.save_dir = os.path.join('gen3_multi_' + self.data_type + self.data_rotate_flag + '_' + self.valid_classes_flag,
                                        '{}_{}_{}_{:.0e}_{:.0e}'.format(self.n_objs, self.in_size, self.out_size, self.eps, self.eps2))


        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.reduced_data_dir = os.path.join(self.reduced_data_dir, self.valid_classes_flag)
        if not os.path.exists(self.reduced_data_dir):
            os.makedirs(self.reduced_data_dir)        

    def init_dataset(self):
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.gpu_mode else {}
        self.train_loader = DataLoader(utils.BasicDataset(self.reduced_data_dir, data_type='train'), shuffle=True, batch_size=self.n_objs*self.batch_size, 
            drop_last=True, **kwargs)
        self.test_loader = DataLoader(utils.BasicDataset(self.reduced_data_dir, data_type='test'), shuffle=True, batch_size=self.n_objs*self.batch_size,
            drop_last=True, **kwargs)
        print('initialized source dataset.')


    def reduce_dataset(self):
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.gpu_mode else {}
        if self.data_type == 'fashion_mnist':
            train_loader = DataLoader(datasets.FashionMNIST('./fashion_mnist', train=True, transform=transforms.ToTensor(), download=True), shuffle=True, batch_size=self.batch_size, 
                drop_last=True, **kwargs)
            test_loader = DataLoader(datasets.FashionMNIST('./fashion_mnist', train=False, transform=transforms.ToTensor()), shuffle=True, batch_size=self.batch_size,
                drop_last=True, **kwargs)
        self.reduce_dataset_helper(train_loader, os.path.join(self.reduced_data_dir, 'train'))
        self.reduce_dataset_helper(test_loader, os.path.join(self.reduced_data_dir, 'test'))

    def reduce_dataset_helper(self, loader, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        out_data = None
        out_labels = None
        for iter, (data, labels) in enumerate(loader):
            if self.gpu_mode:
                data = data.cuda()
                labels = labels.cuda()

            for valid_cls_iter, valid_cls in enumerate(self.valid_classes):
                valid = (labels == valid_cls)
                labels[torch.nonzero(valid).squeeze()] = valid_cls_iter  
                if valid_cls_iter == 0:
                    total_valid = valid 
                else: 
                    total_valid += valid
            total_valid_idx = torch.nonzero(total_valid).squeeze()
            data = data[total_valid_idx]
            labels = labels[total_valid_idx]

            if len(total_valid_idx) == 0:
                continue

            if out_data is None:
                out_data = data
                out_labels = labels
            else:
                out_data = torch.cat([out_data, data], 0)
                out_labels = torch.cat([out_labels, labels], 0)

        torch.save(out_data, os.path.join(save_dir, 'data.pt'))
        torch.save(out_labels, os.path.join(save_dir, 'labels.pt'))
        with open(os.path.join(save_dir, 'data_spec.txt'), 'w') as summary:
            summary.write('n_data: {}'.format(len(out_data)))


    def generate_dataset(self):
        # initialize the source dataset.
        self.init_dataset()

        print('train set generation begins.')
        print('n_train//n_vr_bins: {}'.format(self.n_train//self.n_vr_bins))
        for vr_bin_start in self.vr_bins:
            self.generate_dataset_helper(self.train_loader, os.path.join(self.save_dir, 'train'), self.n_train//self.n_vr_bins, vr_bin_start, vr_bin_start + self.vr_bin_size)
        # Aggregate them.
        self.aggregate_dataset(self.n_train, os.path.join(self.save_dir, 'train'))

        print('val set generation begins.')
        print('n_val//n_vr_bins: {}'.format(self.n_val//self.n_vr_bins))
        for vr_bin_start in self.vr_bins:
            self.generate_dataset_helper(self.train_loader, os.path.join(self.save_dir, 'val'), self.n_val//self.n_vr_bins, vr_bin_start, vr_bin_start + self.vr_bin_size)
        # Aggregate them.
        self.aggregate_dataset(self.n_val, os.path.join(self.save_dir, 'val'))

        print('test set generation begins.')
        print('n_test//n_vr_bins: {}'.format(self.n_test//self.n_vr_bins))
        for vr_bin_start in self.vr_bins:
            self.generate_dataset_helper(self.test_loader, os.path.join(self.save_dir, 'test'), self.n_test//self.n_vr_bins, vr_bin_start, vr_bin_start + self.vr_bin_size)
        # Aggregate them.
        self.aggregate_dataset(self.n_test, os.path.join(self.save_dir, 'test'))

    def generate_dataset_helper(self, loader, save_dir, n_data, vr_min, vr_max):
        '''
        Return:
            out_data: (N, C, H, W)
            out_ind_objs: (N, n_objs, C, H, W) # This doesn't contain any bgd.
            out_ind_ps: (N, n_objs, 4) # We want it to be 4-dim instead of 6-dim, because PoseNet will output a 4-dim vector, and we want to use this directly as a label for it. 
            We define it such that
                grid = F.affine_grid(p, torch.Size([N, C, 28, 28]))
                canon_pose_data = F.grid_sample(data, grid), where canon_pose_data is the original FashionMNIST.
            out_labels: (N, n_objs)
        '''
        if n_data == 0:
            return

        out_data = None
        out_ind_objs = None
        out_ind_ps = None
        out_labels = None
        out_vr = None

        n_valid_classes = len(self.valid_classes)
        n_generated = torch.zeros(n_valid_classes, n_valid_classes).long()
        if self.gpu_mode:
            n_generated = n_generated.cuda()

        outer_iter = 0

        n_data_per_class = n_data//(n_valid_classes**2) # 156 for n_data = 10000 and 4 valid classes and 4 vr_bins.
        n_data_per_class_remainder = n_data%(n_valid_classes**2) # 4 for the same condition as above.
        n_target = torch.ones_like(n_generated)*n_data_per_class
        permed_idx = np.random.permutation(n_valid_classes**2)
        for i in range(n_data_per_class_remainder):
            n_target.view(-1)[permed_idx[i]] += 1

        while True:
            outer_start_time = time.time()
            for iter, (data, labels) in enumerate(loader):
                inner_start_time = time.time()
                N, C, _, _ = data.size()
                if self.gpu_mode:
                    data = data.cuda()
                    labels = labels.cuda()

                # Randomly sample p_inv.
                scales = (self.scale_max-self.scale_min)*torch.rand(N) + self.scale_min
                angles = (self.angle_max-self.angle_min)*torch.rand(N) + self.angle_min
                trans_x = (self.trans_max-self.trans_min)*torch.rand(N) + self.trans_min
                trans_y = (self.trans_max-self.trans_min)*torch.rand(N) + self.trans_min
                if self.gpu_mode:
                    scales = scales.cuda()
                    angles = angles.cuda()
                    trans_x = trans_x.cuda()
                    trans_y = trans_y.cuda()


                p_invs = torch.stack([scales, angles, trans_x, trans_y], 1)
                aff_p_invs = self.affine_form(p_invs)

                grid = F.affine_grid(aff_p_invs, torch.Size([N, C, self.out_size, self.out_size]))
                data = F.grid_sample(data, grid).data

                ind_objs = []
                masks = []
                ind_ps = []
                ind_obj_labels = []
                for chunked_data, chunked_labels, chuncked_ps in zip(torch.chunk(data, self.n_objs, 0), torch.chunk(labels, self.n_objs, 0), torch.chunk(p_invs, self.n_objs, 0)):
                    ind_objs.append(chunked_data)
                    ind_ps.append(chuncked_ps)
                    mask = (chunked_data.mean(1).unsqueeze(1) > self.eps)
                    # We perform the following operation to get a refined mask.
                    # First, get a rough object mask by simple thresholding as before.
                    # Second, find the mean_intensity of pixels on that mask, which has shape (N). Note that we also average over channels.
                    # Third, select the subset of pixels in the mask which have a value that is above the self.eps2*mean_intensity.
                    N, C, _, _ = chunked_data.size()
                    mask_intensity_sum = (chunked_data*mask.float()).view(N, C, -1).sum(2) # (N, C)
                    mask_intensity_mean = mask_intensity_sum/mask.float().view(N, 1, -1).sum(2) # (N, C)
                    mask_intensity_mean = mask_intensity_mean.mean(1) # (N)
                    mask = (chunked_data.mean(1).unsqueeze(1) > self.eps2*mask_intensity_mean.view(-1, 1, 1, 1))
                    masks.append(mask)
                    ind_obj_labels.append(chunked_labels)


                # Next create data.
                composed_img = torch.zeros_like(ind_objs[0]) # (N, 1, H, W). Note that .mean(1) averages over RGB channel.
                for ind_obj, mask in reversed(list(zip(ind_objs, masks))):
                    composed_img = mask.float()*ind_obj + (1-mask.float())*composed_img

                # Calculate visible ratio. Assume n_objs = 2
                for mask_idx, mask in enumerate(masks):
                    if mask_idx == 0:
                        intsec = mask
                    else:
                        intsec *= mask

                N = intsec.size(0)
                vr = 1.0 - intsec.float().view(N, -1).sum(1)/masks[1].float().view(N, -1).sum(1) # (N)
                vr_satisfied = (vr >= vr_min)*(vr < vr_max) # a Byte Tensor, a binary mask of(N)
                n_vr_sat = vr_satisfied.sum()
                if n_vr_sat == 0:
                    continue
                vr_sat_idx = torch.nonzero(vr_satisfied).squeeze()

                composed_img = composed_img[vr_sat_idx].clone()
                ind_objs = torch.stack(ind_objs, 1)[vr_sat_idx].clone() # (n_vr_sat, n_objs, C, H, W)
                ind_ps = torch.stack(ind_ps, 1)[vr_sat_idx].clone() # (n_vr_sat, n_objs, 4)
                labels = torch.stack(ind_obj_labels, 1)[vr_sat_idx].clone() # (n_vr_sat, n_objs)
                vr = vr[vr_sat_idx].clone()

                # We separate these out according to their labels.
                target_label_idxs = []


                for i in range(n_valid_classes):
                    for j in range(n_valid_classes):
                        n_remaining = n_target[i,j] - n_generated[i,j]  
                        if n_remaining == 0:
                            continue
                        target_label = torch.LongTensor([i,j])
                        if self.gpu_mode:
                            target_label = target_label.cuda()
                        target_label_idx = torch.nonzero(((labels - target_label.unsqueeze(0)).abs().sum(1) == 0)).squeeze()
                        if target_label_idx.dim() == 0:
                            continue
                        target_label_idxs.append(target_label_idx[:n_remaining].clone())
                        n_generated[i,j] += len(target_label_idx[:n_remaining])

                if len(target_label_idxs) == 0:
                    continue

                target_label_idxs = torch.cat(target_label_idxs, 0)

                composed_img = composed_img[target_label_idxs].clone()
                ind_objs = ind_objs[target_label_idxs].clone()
                ind_ps = ind_ps[target_label_idxs].clone()
                labels = labels[target_label_idxs].clone()
                vr = vr[target_label_idxs].clone()


                if out_data is None:
                    out_data = composed_img
                    out_ind_objs = ind_objs
                    out_ind_ps = ind_ps
                    out_labels = labels
                    out_vr = vr
                else:
                    out_data = torch.cat([out_data, composed_img], 0)
                    out_ind_objs = torch.cat([out_ind_objs, ind_objs], 0)
                    out_ind_ps = torch.cat([out_ind_ps, ind_ps], 0)
                    out_labels = torch.cat([out_labels, labels], 0)
                    out_vr = torch.cat([out_vr, vr], 0)


                if (iter + 1) % 100 == 0:
                    inner_end_time = time.time() - inner_start_time
                    print('(vr_min: {:.2f} vr_max: {:.2f}) Outer Iter [{}]: iter [{}/{}] finished. {} generated so far ( {:.3f} sec)'.format(vr_min, vr_max, (outer_iter+1), (iter+1), len(loader), n_generated.sum(), inner_end_time))

                if (n_generated == n_target).all():
                    save_dir = os.path.join(save_dir, '{:.2f}_{:.2f}'.format(vr_min, vr_max))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    # I noticed that affine_inv takes a lot of time. So I decided to do it only after we are ready to save the data.
                    # out_ind_ps has size (N, n_objs, 4)
                    out_ind_ps = self.simple_form_inv(out_ind_ps.view(-1, 4)).view(-1, self.n_objs, 4)

                    torch.save(out_data[:n_data].clone(), os.path.join(save_dir ,'data.pt'))
                    torch.save(out_ind_objs[:n_data].clone(), os.path.join(save_dir, 'ind_objs.pt'))
                    torch.save(out_ind_ps[:n_data].clone(), os.path.join(save_dir, 'ind_ps.pt'))
                    torch.save(out_labels[:n_data].clone(), os.path.join(save_dir, 'labels.pt'))
                    torch.save(out_vr[:n_data].clone(), os.path.join(save_dir, 'vr.pt'))

                    with open(os.path.join(save_dir, 'data_spec.txt'), 'w') as summary:
                        summary.write('data_type: {}\n'.format(self.data_type))
                        summary.write('data_rotate: {}\n'.format(self.data_rotate))
                        summary.write('scale_min: {:.2f}\n'.format(self.scale_min))
                        summary.write('scale_max: {:.2f}\n'.format(self.scale_max))
                        summary.write('angle_min: {:.2f}\n'.format(self.angle_min*180/math.pi))
                        summary.write('angle_max: {:.2f}\n'.format(self.angle_max*180/math.pi))
                        summary.write('trans_min: {:.2f}\n'.format(self.trans_min))
                        summary.write('trans_max: {:.2f}\n'.format(self.trans_max))
                        summary.write('n_objs: {}\n'.format(self.n_objs))
                        summary.write('in_size: {}\n'.format(self.in_size))
                        summary.write('out_size: {}\n'.format(self.out_size))
                        summary.write('eps: {}\n'.format(self.eps))
                        summary.write('eps2: {}\n'.format(self.eps2))
                        summary.write('vr_min: {:.2f}\n'.format(vr_min))
                        summary.write('vr_max: {:.2f}\n'.format(vr_max))
                        summary.write('n_data: {}\n'.format(n_data))

                    return
            
            outer_end_time = time.time() - outer_start_time
            print('Outer Iter [{}] ended ({:.3f} sec)'.format(outer_iter+1, outer_end_time))    
            outer_iter += 1                    


    def aggregate_dataset(self, n_data, save_dir):
        if n_data == 0:
            return

        total_data = None
        total_ind_objs = None
        total_ind_ps = None
        total_labels = None
        total_vr = None

        for vr_bin_start in self.vr_bins:
            subdir = os.path.join(save_dir, '{:.2f}_{:.2f}'.format(vr_bin_start, vr_bin_start + self.vr_bin_size))
            
            data = torch.load(os.path.join(subdir, 'data.pt'))
            ind_objs = torch.load(os.path.join(subdir, 'ind_objs.pt'))
            ind_ps = torch.load(os.path.join(subdir, 'ind_ps.pt'))
            labels = torch.load(os.path.join(subdir, 'labels.pt'))
            vr = torch.load(os.path.join(subdir, 'vr.pt'))
            
            if total_data is None:
                total_data = data
                total_ind_objs = ind_objs
                total_ind_ps = ind_ps
                total_labels = labels
                total_vr = vr
            else:
                total_data = torch.cat([total_data, data], 0)
                total_ind_objs= torch.cat([total_ind_objs, ind_objs], 0)
                total_ind_ps = torch.cat([total_ind_ps, ind_ps], 0)
                total_labels = torch.cat([total_labels, labels], 0)
                total_vr = torch.cat([total_vr, vr], 0)

        torch.save(total_data, os.path.join(save_dir, 'data.pt'))
        torch.save(total_ind_objs, os.path.join(save_dir, 'ind_objs.pt'))
        torch.save(total_ind_ps, os.path.join(save_dir, 'ind_ps.pt'))
        torch.save(total_labels, os.path.join(save_dir, 'labels.pt'))
        torch.save(total_vr, os.path.join(save_dir, 'vr.pt'))


    def visualize_dataset(self):
        self.visualize_dataset_helper(os.path.join(self.save_dir, 'train'), self.n_train, self.n_vis)
        self.visualize_dataset_helper(os.path.join(self.save_dir, 'val'), self.n_val, self.n_vis)
        self.visualize_dataset_helper(os.path.join(self.save_dir, 'test'), self.n_test, self.n_vis)

    def visualize_dataset_helper(self, save_dir, n_data, n_vis):
        if n_data == 0:
            return

        indices = torch.from_numpy(np.random.permutation(n_data)[:n_vis]).long()
        data = torch.load(os.path.join(save_dir, 'data.pt'), map_location=lambda storage, loc: storage)[indices].cpu().numpy()

        ind_objs = torch.load(os.path.join(save_dir, 'ind_objs.pt'), map_location=lambda storage, loc: storage)[indices]
        
        ind_ps = torch.load(os.path.join(save_dir, 'ind_ps.pt'), map_location=lambda storage, loc: storage)[indices]

        labels_total = torch.load(os.path.join(save_dir, 'labels.pt'), map_location=lambda storage, loc: storage)

        vr_total = torch.load(os.path.join(save_dir, 'vr.pt'), map_location=lambda storage, loc: storage)
        vr = vr_total[indices].cpu().numpy()
        vr_total = vr_total.cpu().numpy()

        # mask generation
        masks = (ind_objs.mean(2).unsqueeze(2) > self.eps) # (N, n_objs, 1, H, W)
        N, n_objs, C, _, _ = ind_objs.size()
        mask_intensity_sum = (ind_objs*masks.float()).view(N, n_objs, C, -1).sum(3) # (N, n_objs. C)
        mask_intensity_mean = mask_intensity_sum/masks.float().view(N, n_objs, 1, -1).sum(3) # (N, n_objs, C)
        mask_intensity_mean = mask_intensity_mean.mean(2) # (N, n_objs)
        masks = (ind_objs.mean(2).unsqueeze(2) > self.eps2*mask_intensity_mean.view(N, n_objs, 1, 1, 1)) # (N, n_objs, 1, H, W)

        # canonical pose ind_objs
        aff_ind_ps = self.affine_form(ind_ps.view(-1, 4)) # (N*n_objs, 2, 3)
        canon_grid = F.affine_grid(aff_ind_ps, torch.Size([N*n_objs, C, self.in_size, self.in_size]))
        canon_ind_objs = F.grid_sample(ind_objs.view(-1, C, self.out_size, self.out_size), canon_grid).view(N, n_objs, C, self.in_size, self.in_size).data

        ind_objs = ind_objs.cpu().numpy()
        masks = masks.cpu().numpy()
        canon_ind_objs = canon_ind_objs.cpu().numpy()

        # flat_labels
        flat_labels_total = torch.zeros_like(labels_total[:,0]) # (N)
        for i in range(len(self.valid_classes)):
            for j in range(len(self.valid_classes)):
                target_label = torch.LongTensor([i,j])
                target_label_idx = torch.nonzero(((labels_total - target_label.unsqueeze(0)).abs().sum(1) == 0)).squeeze()
                if target_label_idx.dim() == 0:
                    continue
                flat_labels_total[target_label_idx] = i*len(self.valid_classes) + j  


        labels = labels_total[indices].cpu().numpy()
        labels_total = labels_total.cpu().numpy() 
        flat_labels_total = flat_labels_total.cpu().numpy()

        # Images
        fig = plt.figure(figsize=(40,30))
        outer = gridspec.GridSpec(1, 5*5, wspace=0.2, hspace=0.2)
        for i in range(5):
            utils.show_images_inner3(data[n_vis//5*i:n_vis//5*(i+1)], fig, outer[5*i])        
            utils.show_images_inner3(ind_objs[n_vis//5*i:n_vis//5*(i+1)], fig, outer[5*i+1:5*i+3])
            utils.show_images_inner3(masks[n_vis//5*i:n_vis//5*(i+1)], fig, outer[5*i+3:5*i+5])

        fig.savefig(os.path.join(save_dir, 'dataset_vis.png'))

        # Labels
        fig2 = plt.figure()
        outer2 = gridspec.GridSpec(1, 20, wspace=0.2, hspace=0.2)
        label_dict = {}
        label_dict[0] = 'T-shirt/top'
        label_dict[1] = 'Trouser'
        label_dict[2] = 'Pullover'
        label_dict[3] = 'Dress'
        label_dict[4] = 'Coat'
        label_dict[5] = 'Sandal'
        label_dict[6] = 'Shirt'
        label_dict[7] = 'Sneaker'
        label_dict[8] = 'Bag'
        label_dict[9] = 'Ankle boot'
        reduced_label_dict = {}
        for i, valid_cls in enumerate(self.valid_classes):
            reduced_label_dict[i] = label_dict[valid_cls]

        for i in range(5):
            utils.show_labels(labels[n_vis//5*i:n_vis//5*(i+1)], reduced_label_dict, fig2, outer2[4*i+1:4*i+3])

        fig2.savefig(os.path.join(save_dir, 'dataset_vis_labels.png'))

        # vr
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.hist(vr_total, self.vr_bins + [self.vr_max])
        fig3.savefig(os.path.join(save_dir, 'dataset_vr_total.png'))
        with open(os.path.join(save_dir, 'vr_counts.txt'), 'w') as summary:
            for vr_bin_start in self.vr_bins:
                count = np.count_nonzero((vr_total >= vr_bin_start)*(vr_total < vr_bin_start+self.vr_bin_size))
                summary.write('{:.2f}_{:.2f}: {}\n'.format(vr_bin_start, vr_bin_start+self.vr_bin_size, count))


        fig4 = plt.figure()
        outer4 = gridspec.GridSpec(1, 5, wspace=0.2, hspace=0.2)

        for i in range(5):
            utils.show_vrs(vr[n_vis//5*i:n_vis//5*(i+1)], fig4, outer4[i])

        fig4.savefig(os.path.join(save_dir, 'dataset_vis_vr_image_by_image.png'))


        # Labels count
        fig5 = plt.figure()

        ax51 = fig5.add_subplot(131)
        d = np.diff(np.unique(labels_total[:,0])).min()
        left_of_first_bin = labels_total[:,0].min() - float(d)/2
        right_of_last_bin = labels_total[:,0].max() + float(d)/2
        ax51.hist(labels_total[:,0], np.arange(left_of_first_bin, right_of_last_bin + d, d))

        ax52 = fig5.add_subplot(132)
        d = np.diff(np.unique(labels_total[:,1])).min()
        left_of_first_bin = labels_total[:,1].min() - float(d)/2
        right_of_last_bin = labels_total[:,1].max() + float(d)/2
        ax52.hist(labels_total[:,1], np.arange(left_of_first_bin, right_of_last_bin + d, d))

        ax53 = fig5.add_subplot(133)
        d = np.diff(np.unique(flat_labels_total)).min()
        left_of_first_bin = flat_labels_total.min() - float(d)/2
        right_of_last_bin = flat_labels_total.max() + float(d)/2
        ax53.hist(flat_labels_total, np.arange(left_of_first_bin, right_of_last_bin + d, d))
        for i in range(16):
            print('number of images with {}-th flat label: {}'.format((i+1), np.count_nonzero((flat_labels_total - i) == 0)))

        fig5.savefig(os.path.join(save_dir, 'dataset_labels_total.png'))

        # canon_ind_objs
        fig6 = plt.figure(figsize=(40,30))
        outer = gridspec.GridSpec(1, 5*2, wspace=0.2, hspace=0.2)
        for i in range(5):
            utils.show_images_inner3(ind_objs[n_vis//5*i:n_vis//5*(i+1)], fig6, outer[2*i])        
            utils.show_images_inner3(canon_ind_objs[n_vis//5*i:n_vis//5*(i+1)], fig6, outer[2*i+1])

        fig6.savefig(os.path.join(save_dir, 'canon_ind_objs_vis.png'))


        plt.close(fig)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)
        plt.close(fig5)
        plt.close(fig6)


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
        aff_p = torch.zeros(N, 2, 3)
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

    def simple_form(self, aff_p):
        '''
        Args:
        aff_p: (N, 2, 3)-shaped affine parameters corresponding to them.

        Return:
        p: (N, 4). s = p[0], a = p[1], x = p[2], y = p[3].
        s, a, x, y: respectively scale, angle, x, y translations. They all have shapes (N,).
        a, x, y are unconstrained.
        '''
        N = aff_p.size(0)
        p = torch.zeros(N, 4)
        if aff_p.is_cuda:
            p = p.cuda()


        p[:,0] = torch.sqrt(aff_p[:,0,0]**2+aff_p[:,0,1]**2)
        abs_angle = torch.acos(aff_p[:,0,0]/p[:,0]) # torch.acos has range from 0 to pi, while we assume angle is between -pi and pi.
        angle_sign = torch.sign(aff_p[:,1,0])
        p[:,1] = abs_angle*angle_sign
        p[:,2] = aff_p[:,0,2]
        p[:,3] = aff_p[:,1,2]

        return p

    def affine_inv(self, aff_p):
        '''
        Args:
        p: (N, 2, 3)
        Let p define y = Ax + b. Then, p_inv corresponds to x = A^(-1)y + A^(-1)(-b)
        Return:
        p_inv: (N, 2, 3)

        '''
        mat = aff_p[:,:,:2]
        bias = aff_p[:,:,2]

        inv_mat = torch.stack([torch.inverse(mat[i]) for i in range(len(mat))],0)
        inv_bias = inv_mat.matmul(-bias.contiguous().view(-1, 2, 1))
        return torch.cat([inv_mat, inv_bias], 2)

    def simple_form_inv(self, p):
        '''
        Args:
        p: (N, 4). s = p[0], a = p[1], x = p[2], y = p[3].
        Return:
        p_inv: (N, 4)
        '''
        p_inv = torch.zeros_like(p)
        p_inv[:,0] = p[:,0].pow(-1)
        p_inv[:,1] = -p[:,1]
        p_inv[:,2] = -p[:,0].pow(-1)*(p[:,2]*torch.cos(p[:,1]) + p[:,3]*torch.sin(p[:,1]))
        p_inv[:,3] = -p[:,0].pow(-1)*(-p[:,2]*torch.sin(p[:,1]) + p[:,3]*torch.cos(p[:,1]))

        return p_inv


def main():
    args = parse_args()
    if args is None:
        exit()

    generator = MultiObjectDatasetGenerator3(args)
    # generator.reduce_dataset()
    generator.generate_dataset()
    generator.visualize_dataset()

if __name__ == '__main__':
    main()