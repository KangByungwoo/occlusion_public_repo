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
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_train', type=int, default=50000)
    parser.add_argument('--n_val', type=int, default=10000)
    parser.add_argument('--n_test', type=int, default=10000)
    parser.add_argument('--n_vis', type=int, default=100)
    parser.add_argument('--reduced_data_dir', type=str, default='./reduced_fashion_mnist')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--gpu_mode', action='store_true', default=True)
    parser.add_argument('--gpu_idx', type=int, default=0)

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


class MultiObjectDatasetGenerator2(object):
    def __init__(self, args):
        self.debug = args.debug
        self.data_type = args.data_type
        self.data_rotate = args.data_rotate
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

        self.n_vr_bins = int((self.vr_max-self.vr_min)/self.vr_bin_size)
        self.vr_bins = list(np.arange(self.vr_min, self.vr_max, self.vr_bin_size)) # for e.g. [0.2, 0.25, 0.3, 0.35] if self.vr_min = 0.2, sefl.vr_max = 0.4, self.vr_bin_size = 0.05.
        self.valid_classes_flag = args.valid_classes 
        self.valid_classes = [int(k) for k in self.valid_classes_flag] # This gives [0, 2, 8, 9] for '0289'.

        if self.data_rotate:
            self.data_rotate_flag = '_rotated_{:.0f}_{:.0f}_trans_{:.2f}_{:.2f}'.format(self.angle_min*180/math.pi, self.angle_max*180/math.pi, self.trans_min, self.trans_max)
        else:
            self.data_rotate_flag = ''
        
        if self.debug:
            self.save_dir = os.path.join('extensive_info_debugging_multi_' + self.data_type + self.data_rotate_flag + '_' + self.valid_classes_flag,
                                            '{}_{}_{}_{:.2f}_{:.2f}_{:.0e}_{:.0e}'.format(self.n_objs, self.in_size, self.out_size, self.vr_min, self.vr_max, self.eps, self.eps2))
        else:
            self.save_dir = os.path.join('extensive_info_multi_' + self.data_type + self.data_rotate_flag + '_' + self.valid_classes_flag,
                                            '{}_{}_{}_{:.2f}_{:.2f}_{:.0e}_{:.0e}'.format(self.n_objs, self.in_size, self.out_size, self.vr_min, self.vr_max, self.eps, self.eps2))

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
        for vr_bin_start in self.vr_bins:
            self.generate_dataset_helper(self.train_loader, os.path.join(self.save_dir, 'train'), self.n_train//self.n_vr_bins, vr_bin_start, vr_bin_start + self.vr_bin_size)
        # Aggregate them.
        self.aggregate_dataset(self.n_train, os.path.join(self.save_dir, 'train'))

        print('val set generation begins.')
        for vr_bin_start in self.vr_bins:
            self.generate_dataset_helper(self.train_loader, os.path.join(self.save_dir, 'val'), self.n_val//self.n_vr_bins, vr_bin_start, vr_bin_start + self.vr_bin_size)
        # Aggregate them.
        self.aggregate_dataset(self.n_val, os.path.join(self.save_dir, 'val'))

        print('test set generation begins.')
        for vr_bin_start in self.vr_bins:
            self.generate_dataset_helper(self.test_loader, os.path.join(self.save_dir, 'test'), self.n_test//self.n_vr_bins, vr_bin_start, vr_bin_start + self.vr_bin_size)
        # Aggregate them.
        self.aggregate_dataset(self.n_test, os.path.join(self.save_dir, 'test'))


    def generate_dataset_helper(self, loader, save_dir, n_data, vr_min, vr_max):
        '''
        Return:
            out_data: (N, C, H, W)
            out_ind_objs: (N, n_objs, C, H, W)
            out_labels: (N, n_objs)
            out_angles_trans: (N, n_objs, 3)  # [angle_in_radians, tx, ty] for each object
        '''
        if n_data == 0:
            return

        out_data = None
        out_ind_objs = None
        out_labels = None
        out_vr = None
        out_angles_trans = None  # NEW: to store [angle, tx, ty]

        n_valid_classes = len(self.valid_classes)
        n_generated = torch.zeros(n_valid_classes, n_valid_classes, dtype=torch.long,
                                device=("cuda" if self.gpu_mode else "cpu"))

        outer_iter = 0
        n_data_per_class = n_data // (n_valid_classes**2)
        n_data_per_class_remainder = n_data % (n_valid_classes**2)
        n_target = torch.ones_like(n_generated) * n_data_per_class
        permed_idx = np.random.permutation(n_valid_classes**2)
        for i in range(n_data_per_class_remainder):
            n_target.view(-1)[permed_idx[i]] += 1

        while True:
            for iter, (data, labels) in enumerate(loader):

                # Move to GPU if needed:
                if self.gpu_mode:
                    data = data.cuda()
                    labels = labels.cuda()

                # PADDING (no more .data usage):
                data = F.pad(data, self.padding)

                # === Build random transforms (angle & translation) ===
                if self.data_rotate:
                    # For the entire batch of size B = n_objs*batch_size
                    B = data.size(0)
                    angles = (self.angle_max - self.angle_min) * torch.rand(B) + self.angle_min

                    # translation for half of them, since n_objs=2
                    B_half = B // self.n_objs
                    trans_angles = 2 * math.pi * torch.rand(B_half)
                    trans_mag = (self.trans_max - self.trans_min) * torch.rand(B_half) + self.trans_min
                    trans = trans_mag.unsqueeze(1) * torch.stack([torch.cos(trans_angles), torch.sin(trans_angles)], 1)
                    trans_first = trans          # shape (B_half, 2)
                    trans_second = -1.0 * trans  # shape (B_half, 2)
                    trans = torch.cat([trans_first, trans_second], 0)  # shape (B, 2)

                    if self.gpu_mode:
                        angles = angles.cuda()
                        trans = trans.cuda()

                    cosines = torch.cos(angles)
                    sines = torch.sin(angles)

                    # Build the Nx2x3 affine matrices
                    affine = torch.zeros(B, 2, 3, device=data.device)
                    affine[:, 0, 0] = cosines
                    affine[:, 0, 1] = -sines
                    affine[:, 1, 0] = sines
                    affine[:, 1, 1] = cosines
                    # translations
                    affine[:, 0, 2] = trans[:, 0]
                    affine[:, 1, 2] = trans[:, 1]

                    # Create grid and sample:
                    grid = F.affine_grid(affine, data.size(), align_corners=False)
                    data = F.grid_sample(data, grid, align_corners=False)

                    # Build a tensor of shape (B, 3) = [angle, tx, ty]
                    transforms_batch = torch.stack([angles, trans[:, 0], trans[:, 1]], dim=1)
                else:
                    # No rotation/translation
                    transforms_batch = torch.zeros(data.size(0), 3, device=data.device)

                # === Now chunk the data, labels, and transforms for each object ===
                ind_objs_list = []
                masks = []
                ind_obj_labels = []
                transforms_list = []
                chunked_data = torch.chunk(data, self.n_objs, dim=0)
                chunked_labels = torch.chunk(labels, self.n_objs, dim=0)
                chunked_transforms = torch.chunk(transforms_batch, self.n_objs, dim=0)

                for cd, cl, ct in zip(chunked_data, chunked_labels, chunked_transforms):
                    # cd: (batch_size, C, H, W)
                    # ct: (batch_size, 3)  => [angle, tx, ty] per sample

                    # Build foreground mask:
                    mask = (cd.mean(1, keepdim=True) > self.eps)
                    # Further refine mask by average intensity:
                    N, C, H, W = cd.size()
                    mask_intensity_sum = (cd * mask.float()).view(N, C, -1).sum(2)  # (N, C)
                    mask_intensity_mean = mask_intensity_sum / mask.float().view(N, 1, -1).sum(2)  # (N, C)
                    mask_intensity_mean = mask_intensity_mean.mean(1)  # (N)
                    mask = (cd.mean(1, keepdim=True) > self.eps2 * mask_intensity_mean.view(-1, 1, 1, 1))

                    ind_objs_list.append(cd)
                    masks.append(mask)
                    ind_obj_labels.append(cl)
                    transforms_list.append(ct)

                # Compose the final multi-object image:
                composed_img = torch.zeros_like(ind_objs_list[0])
                for ind_obj, mask in reversed(list(zip(ind_objs_list, masks))):
                    composed_img = mask.float() * ind_obj + (1 - mask.float()) * composed_img

                # Compute visible ratio VR for n_objs=2:
                # intersection of both object masks:
                intsec = masks[0]
                for mk in masks[1:]:
                    intsec = intsec * mk
                N = intsec.size(0)
                vr = 1.0 - intsec.float().view(N, -1).sum(1) / \
                            masks[1].float().view(N, -1).sum(1)

                # Keep only items in vr_min <= VR < vr_max
                vr_satisfied = (vr >= vr_min) & (vr < vr_max)
                if not vr_satisfied.any():
                    continue
                vr_sat_idx = vr_satisfied.nonzero(as_tuple=True)[0]

                # Filter composed_img, ind_objs, transforms, labels, etc.
                composed_img = composed_img[vr_sat_idx]
                # shape -> (num_vr_sat, n_objs, C, H, W)
                ind_objs = torch.stack(ind_objs_list, dim=1)[vr_sat_idx]
                labels = torch.stack(ind_obj_labels, dim=1)[vr_sat_idx]
                vr = vr[vr_sat_idx]

                # Similarly, shape -> (num_vr_sat, n_objs, 3)
                transforms_per_sample = torch.stack(transforms_list, dim=1)[vr_sat_idx]

                # === Now do label filtering and final selection ===
                target_label_idxs = []
                for i_cls in range(n_valid_classes):
                    for j_cls in range(n_valid_classes):
                        n_remaining = n_target[i_cls, j_cls] - n_generated[i_cls, j_cls]
                        if n_remaining == 0:
                            continue
                        target_label = torch.tensor([i_cls, j_cls], device=labels.device)
                        # find samples in which labels == [i_cls, j_cls]
                        matched_idx = ((labels - target_label.unsqueeze(0)).abs().sum(1) == 0).nonzero(as_tuple=True)[0]
                        if matched_idx.numel() == 0:
                            continue
                        # take up to n_remaining from matched
                        chosen = matched_idx[:n_remaining]
                        n_generated[i_cls, j_cls] += chosen.size(0)
                        target_label_idxs.append(chosen)

                if len(target_label_idxs) == 0:
                    continue
                target_label_idxs = torch.cat(target_label_idxs, dim=0)

                # Filter final set
                composed_img = composed_img[target_label_idxs]
                ind_objs = ind_objs[target_label_idxs]
                labels = labels[target_label_idxs]
                vr = vr[target_label_idxs]
                transforms_per_sample = transforms_per_sample[target_label_idxs]

                # === Accumulate results ===
                if out_data is None:
                    out_data = composed_img
                    out_ind_objs = ind_objs
                    out_labels = labels
                    out_vr = vr
                    out_angles_trans = transforms_per_sample
                else:
                    out_data = torch.cat([out_data, composed_img], 0)
                    out_ind_objs = torch.cat([out_ind_objs, ind_objs], 0)
                    out_labels = torch.cat([out_labels, labels], 0)
                    out_vr = torch.cat([out_vr, vr], 0)
                    out_angles_trans = torch.cat([out_angles_trans, transforms_per_sample], 0)

                # Logging / debug print
                if (iter + 1) % 100 == 0:
                    print('(vr_min: {:.2f} vr_max: {:.2f}) Outer Iter [{}]: iter [{}/{}] '
                        'finished. {} generated so far'.format(
                            vr_min, vr_max, (outer_iter + 1), (iter + 1),
                            len(loader), n_generated.sum()))

                # === Check if we have enough data for this VR bin ===
                if (n_generated == n_target).all():
                    save_subdir = os.path.join(save_dir, '{:.2f}_{:.2f}'.format(vr_min, vr_max))
                    os.makedirs(save_subdir, exist_ok=True)

                    # Only save up to n_data items
                    torch.save(out_data[:n_data],       os.path.join(save_subdir, 'data.pt'))
                    torch.save(out_ind_objs[:n_data],   os.path.join(save_subdir, 'ind_objs.pt'))
                    torch.save(out_labels[:n_data],     os.path.join(save_subdir, 'labels.pt'))
                    torch.save(out_vr[:n_data],         os.path.join(save_subdir, 'vr.pt'))
                    # NEW: Save the transformations
                    torch.save(out_angles_trans[:n_data],
                            os.path.join(save_subdir, 'angles_trans.pt'))

                    # Write metadata
                    with open(os.path.join(save_subdir, 'data_spec.txt'), 'w') as summary:
                        summary.write('data_type: {}\n'.format(self.data_type))
                        summary.write('data_rotate: {}\n'.format(self.data_rotate))
                        summary.write('angle_min: {:.2f}\n'.format(self.angle_min*180/math.pi))
                        summary.write('angle_max: {:.2f}\n'.format(self.angle_max*180/math.pi))
                        summary.write('trans_min: {:.2f}\n'.format(self.trans_min))
                        summary.write('trans_max: {:.2f}\n'.format(self.trans_max))
                        summary.write('n_objs: {}\n'.format(self.n_objs))
                        summary.write('in_size: {}\n'.format(self.in_size))
                        summary.write('out_size: {}\n'.format(self.out_size))
                        summary.write('eps: {}\n'.format(self.eps))
                        summary.write('vr_min: {:.2f}\n'.format(vr_min))
                        summary.write('vr_max: {:.2f}\n'.format(vr_max))
                        summary.write('n_data: {}\n'.format(n_data))

                    return

            outer_iter += 1



    def aggregate_dataset(self, n_data, save_dir):
        if n_data == 0:
            return

        total_data = None
        total_ind_objs = None
        total_labels = None
        total_vr = None

        for vr_bin_start in self.vr_bins:
            subdir = os.path.join(save_dir, '{:.2f}_{:.2f}'.format(vr_bin_start, vr_bin_start + self.vr_bin_size))
            
            data = torch.load(os.path.join(subdir, 'data.pt'))
            ind_objs = torch.load(os.path.join(subdir, 'ind_objs.pt'))
            labels = torch.load(os.path.join(subdir, 'labels.pt'))
            vr = torch.load(os.path.join(subdir, 'vr.pt'))
            
            if total_data is None:
                total_data = data
                total_ind_objs = ind_objs
                total_labels = labels
                total_vr = vr
            else:
                total_data = torch.cat([total_data, data], 0)
                total_ind_objs= torch.cat([total_ind_objs, ind_objs], 0)
                total_labels = torch.cat([total_labels, labels], 0)
                total_vr = torch.cat([total_vr, vr], 0)

        torch.save(total_data, os.path.join(save_dir, 'data.pt'))
        torch.save(total_ind_objs, os.path.join(save_dir, 'ind_objs.pt'))
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
        
        labels_total = torch.load(os.path.join(save_dir, 'labels.pt'), map_location=lambda storage, loc: storage)

        vr_total = torch.load(os.path.join(save_dir, 'vr.pt'), map_location=lambda storage, loc: storage)
        vr = vr_total[indices].cpu().numpy()
        vr_total = vr_total.cpu().numpy()

        masks = (ind_objs.mean(2).unsqueeze(2) > self.eps) # (N, n_objs, 1, H, W)
        N, n_objs, C, _, _ = ind_objs.size()
        mask_intensity_sum = (ind_objs*masks.float()).view(N, n_objs, C, -1).sum(3) # (N, n_objs. C)
        mask_intensity_mean = mask_intensity_sum/masks.float().view(N, n_objs, 1, -1).sum(3) # (N, n_objs, C)
        mask_intensity_mean = mask_intensity_mean.mean(2) # (N, n_objs)
        masks = (ind_objs.mean(2).unsqueeze(2) > self.eps2*mask_intensity_mean.view(N, n_objs, 1, 1, 1)) # (N, n_objs, 1, H, W)

        ind_objs = ind_objs.cpu().numpy()
        masks = masks.cpu().numpy()

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

        fig.savefig(os.path.join(save_dir, 'dataset_vis.eps'))

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

        fig2.savefig(os.path.join(save_dir, 'dataset_vis_labels.eps'))

        # vr
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.hist(vr_total, self.vr_bins + [self.vr_max])
        fig3.savefig(os.path.join(save_dir, 'dataset_vr_total.eps'))
        with open(os.path.join(save_dir, 'vr_counts.txt'), 'w') as summary:
            for vr_bin_start in self.vr_bins:
                count = np.count_nonzero((vr_total >= vr_bin_start)*(vr_total < vr_bin_start+self.vr_bin_size))
                summary.write('{:.2f}_{:.2f}: {}\n'.format(vr_bin_start, vr_bin_start+self.vr_bin_size, count))


        fig4 = plt.figure()
        outer4 = gridspec.GridSpec(1, 5, wspace=0.2, hspace=0.2)

        for i in range(5):
            utils.show_vrs(vr[n_vis//5*i:n_vis//5*(i+1)], fig4, outer4[i])

        fig4.savefig(os.path.join(save_dir, 'dataset_vis_vr_image_by_image.eps'))


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

        fig5.savefig(os.path.join(save_dir, 'dataset_labels_total.eps'))


        plt.close(fig)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)
        plt.close(fig5)


def main():
    args = parse_args()
    if args is None:
        exit()

    start_time = time.time()
    generator = MultiObjectDatasetGenerator2(args)
    # generator.reduce_dataset()
    generator.generate_dataset()
    generator.visualize_dataset()
    print('Total time: {}'.format(time.time()-start_time))

if __name__ == '__main__':
    main()