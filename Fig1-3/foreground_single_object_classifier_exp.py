import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T 
import torch.nn as nn
import itertools
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from foreground_single_object_classifier import *

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class ForegroundSingleObjectClassifierExp(object):
    def __init__(self, args):
        # parameters
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

        self.model_type = args.model_type
        self.epoch = args.epoch
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size
        self.dropout_p = args.dropout_p
        self.test_batch_size = args.test_batch_size
        self.vis_wrong_cls1 = args.vis_wrong_cls1
        self.n_vis = args.n_vis
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.hp_dir = args.hp_dir
        self.best_hp_test_dir = args.best_hp_test_dir

        self.gpu_mode = args.gpu_mode
        self.gpu_idx = args.gpu_idx
        # set gpu device
        if self.gpu_mode:
            torch.cuda.set_device(args.gpu_idx)
        self.valid_classes_flag = args.valid_classes 
        self.valid_classes = [int(k) for k in self.valid_classes_flag] # This gives [0, 2, 8, 9] for '0289'.

        self.original_label_dict = {}
        self.original_label_dict[0] = 'T-shirt/top'
        self.original_label_dict[1] = 'Trouser'
        self.original_label_dict[2] = 'Pullover'
        self.original_label_dict[3] = 'Dress'
        self.original_label_dict[4] = 'Coat'
        self.original_label_dict[5] = 'Sandal'
        self.original_label_dict[6] = 'Shirt'
        self.original_label_dict[7] = 'Sneaker'
        self.original_label_dict[8] = 'Bag'
        self.original_label_dict[9] = 'Ankle boot'
        self.label_dict = {}
        for i, valid_cls in enumerate(self.valid_classes):
            self.label_dict[i] = self.original_label_dict[valid_cls]


        #save names
        self.best_total_loss_save_name = self.model_type+'_best_total_loss'


        # networks init
        if self.model_type == 'FD':
            self.model = FDForegroundSingleObjectClassifier(n_classes=len(self.valid_classes), dropout_p=self.dropout_p)
        elif self.model_type == 'FDFat':
            self.model = FDForegroundSingleObjectClassifier(hidden1=128, hidden2=128, hidden3=128, hidden4=2048, n_classes=len(self.valid_classes), dropout_p=self.dropout_p)
        elif self.model_type == 'RC':
            self.model = RCForegroundSingleObjectClassifier(n_classes=len(self.valid_classes), dropout_p=self.dropout_p)
        elif self.model_type == 'RC5':
            self.model = RCForegroundSingleObjectClassifier(n_iter=5, n_classes=len(self.valid_classes), dropout_p=self.dropout_p)
        elif self.model_type == 'RCControl':
            self.model = RCControlForegroundSingleObjectClassifier(n_classes=len(self.valid_classes), dropout_p=self.dropout_p)
        elif self.model_type == 'FDFatterTaller2':
            self.model = FDTallerForegroundSingleObjectClassifier(hidden=256, hidden_fc=1024, n_classes=len(self.valid_classes), dropout_p=self.dropout_p)                 
        elif self.model_type == 'WeightSharedFDFatterTaller2':
            self.model = WeightSharedFDTallerForegroundSingleObjectClassifier(hidden=256, hidden_fc=1024, n_classes=len(self.valid_classes), dropout_p=self.dropout_p)                 
        elif self.model_type == 'TDRCTypeC':
            self.model = TDRCForegroundSingleObjectClassifierC(n_classes=len(self.valid_classes), dropout_p=self.dropout_p)
        elif self.model_type == 'TDRCTypeC5':
            self.model = TDRCForegroundSingleObjectClassifierC(n_iter=5, n_classes=len(self.valid_classes), dropout_p=self.dropout_p)
        elif self.model_type == 'TDRCTypeD':
            self.model = TDRCForegroundSingleObjectClassifierD(n_classes=len(self.valid_classes), dropout_p=self.dropout_p)
        elif self.model_type == 'TDRCTypeD5':
            self.model = TDRCForegroundSingleObjectClassifierD(n_iter=5, n_classes=len(self.valid_classes), dropout_p=self.dropout_p)



        weights = []
        biases = []
        for name, params in self.model.named_parameters():
            if 'bias' in name or 'h0' in name or 'c0' in name:
                biases.append(params)
            else:
                weights.append(params)

        self.optimizer = optim.Adam([{'params': weights}, {'params': biases, 'weight_decay': 0.0}], lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=self.lr_decay, patience=4, verbose=True, threshold=0.005, threshold_mode='abs', cooldown=0, 
                                                            min_lr=0, eps=1e-8)
        # size_average option is not available, and this loss always return the sum over obsevations across minibatch.
        # This is because KLD_element is always summed, not averaged over observations.
        self.loss_fct = nn.CrossEntropyLoss(size_average=False)

        if self.gpu_mode:
            self.model.cuda()
            self.loss_fct.cuda() # Since self.loss_fct doesn't have a parameter, we don't have to do this, but this seems like a good practice, since it's a nn.Module.



        if self.data_rotate:
            self.data_rotate_flag = '_rotated_{:.0f}_{:.0f}_trans_{:.2f}_{:.2f}'.format(self.angle_min*180/math.pi, self.angle_max*180/math.pi, self.trans_min, self.trans_max)
        else:
            self.data_rotate_flag = ''
        
        if not self.debug:
            self.data_dir = os.path.join('multi_' + self.data_type + self.data_rotate_flag + '_' + self.valid_classes_flag,
                                        '{}_{}_{}_{:.2f}_{:.2f}_{:.0e}_{:.0e}'.format(self.n_objs, self.in_size, self.out_size, self.vr_min, self.vr_max, self.eps, self.eps2))
        else:
            self.data_dir = os.path.join('debugging_multi_' + self.data_type + self.data_rotate_flag + '_' + self.valid_classes_flag,
                                        '{}_{}_{}_{:.2f}_{:.2f}_{:.0e}_{:.0e}'.format(self.n_objs, self.in_size, self.out_size, self.vr_min, self.vr_max, self.eps, self.eps2))


        self.init_dataset()


        self.hp_dir = os.path.join(self.hp_dir, self.data_dir, self.model_type)
        self.hp_model_save_name = 'lr_{:.1e}_wd_{:.1e}_bs_{}_dp_{:.2f}'.format(self.lr, self.weight_decay, self.batch_size, self.dropout_p) + '.pth.tar'
        self.hp_hist_save_name = 'lr_{:.1e}_wd_{:.1e}_bs_{}_dp_{:.2f}'.format(self.lr, self.weight_decay, self.batch_size, self.dropout_p) + '_history.pkl'

        self.best_hp_test_dir = os.path.join(self.best_hp_test_dir, self.data_dir, self.model_type)

        # print settings
        print('')
        print('CUDA enabled: {}'.format(self.gpu_mode))
        if self.gpu_mode:
            print('Current GPU Device: {}'.format(torch.cuda.current_device()))
        print('data_type: {}'.format(self.data_type))
        print('data_rotate: {}'.format(self.data_rotate))
        print('valid_classes: {}'.format(self.valid_classes_flag))
        print('angle_min: {:.2f}'.format(self.angle_min*180/math.pi))
        print('angle_max: {:.2f}'.format(self.angle_max*180/math.pi))
        print('trans_min: {:.2f}'.format(self.trans_min))
        print('trans_max: {:.2f}'.format(self.trans_max))
        print('n_objs: {}'.format(self.n_objs))
        print('in_size: {}'.format(self.in_size))
        print('out_size: {}'.format(self.out_size))
        print('eps: {:.0e}'.format(self.eps))
        print('eps2: {:.0e}'.format(self.eps2))
        print('vr_min: {:.2f}'.format(self.vr_min))
        print('vr_max: {:.2f}'.format(self.vr_max))

        print('Model: {}'.format(self.model_type))
        print('lr: {:.1e}'.format(self.lr))
        print('lr_decay: {:.1e}'.format(self.lr_decay))
        print('batch_size: {}'.format(self.batch_size))
        print('test_batch_size: {}'.format(self.test_batch_size))
        print('epoch: {}'.format(self.epoch))
        if self.vis_wrong_cls1:
            print('n_vis: {}'.format(self.n_vis))

        # write settings and architecture into a txt file and save it in a result/log/save folder.
        for dir in [self.save_dir,self.result_dir, self.log_dir]:
            record_dir = os.path.join(dir, self.data_dir, self.model_type)
            if not os.path.exists(record_dir):
                os.makedirs(record_dir)
            record_name = self.model_type + '_specs.txt' 
            # Output a text file summarizing the parameters that generated the dataset.
            # We can record not only fixed, initial hps, but also constantly changing ones, such as decaying lr, if we decay lr.
            with open(os.path.join(record_dir, record_name), 'w') as summary:
                summary.write('CUDA enabled: {}\n'.format(self.gpu_mode))
                if self.gpu_mode:
                    summary.write('GPU Device: {}\n'.format(torch.cuda.current_device()))
                summary.write('data_type: {}\n'.format(self.data_type))
                summary.write('data_rotate: {}\n'.format(self.data_rotate))
                summary.write('valid_classes: {}\n'.format(self.valid_classes_flag))
                summary.write('angle_min: {:.2f}\n'.format(self.angle_min*180/math.pi))
                summary.write('angle_max: {:.2f}\n'.format(self.angle_max*180/math.pi))
                summary.write('trans_min: {:.2f}\n'.format(self.trans_min))
                summary.write('trans_max: {:.2f}\n'.format(self.trans_max))
                summary.write('n_objs: {}\n'.format(self.n_objs))
                summary.write('in_size: {}\n'.format(self.in_size))
                summary.write('out_size: {}\n'.format(self.out_size))
                summary.write('eps: {:.0e}\n'.format(self.eps))
                summary.write('eps2: {:.0e}\n'.format(self.eps2))
                summary.write('vr_min: {:.2f}\n'.format(self.vr_min))
                summary.write('vr_max: {:.2f}\n'.format(self.vr_max))

                summary.write('Model: {}\n'.format(self.model_type))
                summary.write('lr: {:.1e}\n'.format(self.lr))
                summary.write('lr_decay: {:.1e}\n'.format(self.lr_decay))
                summary.write('batch_size: {}\n'.format(self.batch_size))
                summary.write('test_batch_size: {}\n'.format(self.test_batch_size))
                summary.write('epoch: {}\n'.format(self.epoch))
                if self.vis_wrong_cls1:
                    summary.write('n_vis: {}'.format(self.n_vis))
                summary.write('model architecture: \n')
                summary.write(repr(self.model))


    def init_dataset(self, data_dir=None):
        '''
        I believe it is a good practice to not separately define a Dataset instance in case it initalizes
        a huge data. For example, in case I use a superimposed rgb or gray FashionMNIST, the initialized
        dataset may take up almost as large as 12GB, train and test set combined.
        '''
        if data_dir is None:
            data_dir = self.data_dir

        kwargs = {'num_workers': 8, 'pin_memory': True} if self.gpu_mode else {}
        self.train_loader = DataLoader(utils.MultiObjectDataset(data_dir, data_type='train'), shuffle=True, batch_size=self.batch_size, 
            drop_last=True, **kwargs)
        self.val_loader = DataLoader(utils.MultiObjectDataset(data_dir, data_type='val'), shuffle=True, batch_size=self.test_batch_size, 
            drop_last=False, **kwargs)


    def train(self):
        '''
        As of 03/29/18, This has become the new train method.
        Compared to train_old, following things have been changed:
        1. we measure ind_obj_recon_loss by loss_fct.recon_loss_fct (so that if we change the type of recon function used in the main loss_fct, ind_obj_recon_loss is accordingly measured by the changed recon_loss function).
        2. we determine the permutation that minimizes the ind_obj_recon_loss.
        3. we record the minimum ind_obj_recon_loss and use the corresponding permutation to provide true_labels for z classification and order prediction.
        This seems to be a more reasonable way to decide which predicted ind obj corresponds to which actual ind obj than somewhat ad-hoc normed_mask_overlap.
        4. note that now, there is no need to define and worry about z_valid, because we find an optimal permutation for every example, and define true label using it.
        Similarly, we don't need to define no_duplicate for order prediction. However, if there is no overlap between objects, occlusion ordering is truly ambiguous and therefore we ignore those examples.
        5. we no longer calculate gray_diff when evaluating order prediction. All the new datasets are generated such that all the objects in an image have distinct grayscales/colors for the applicable datasets. This is because
        an object can potentially completely lie inside the support of another object and if the two objects have exactly the same grayscale in this case, it looks exactly as if there is a single object. Since this can potentially
        interfere with the training of the network, we do not generate such images in the new datasets. 
        '''
        self.hist = {}
        self.hist['total_loss'] = []
        self.hist['total_acc'] = []

        self.hist['val_total_loss'] = []
        self.hist['val_total_acc'] = []

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
            self.loss_fct.train()
            iter_start_time = time.time()
            for iter, (data, labels) in enumerate(self.train_loader):
                # note that even though ind_objs are not used for backprop, we still make it a Variable, because it will enter into self.loss_fct.recon_loss_fct, which is nn.Module.
                data = Variable(data)
                labels = Variable(labels)
                if self.gpu_mode:
                    data = data.cuda()
                    labels = labels.cuda()

                self.optimizer.zero_grad()
                scores = self.model(data)
                total_loss = self.loss_fct(scores, labels[:,0])/len(data)
                total_loss.backward()
                self.optimizer.step()

                total_acc = (scores.max(1)[1] == labels[:,0]).sum().data[0]/len(labels)

                self.hist['total_loss'].append(total_loss.data[0])
                self.hist['total_acc'].append(total_acc)

                if ((iter + 1) % 100) == 0:
                    iter_time = time.time() - iter_start_time
                    iter_start_time = time.time()
                    print("Epoch {}: [{}/{}] total_loss: {:.3f} total_acc: {:.3f} ({:.3f} sec)".format(
                        (epoch + 1), (iter + 1), len(self.train_loader), total_loss.data[0], total_acc, iter_time))


            # Validation of the main model and z_classifier.
            print('')
            print('Validation begins.')
            print('')
            total_loss = 0.0
            total_acc = 0.0

            self.vis_data = None
            self.vis_pred = None
            self.vis_labels = None

            self.model.eval()
            self.loss_fct.eval()
            iter_start_time = time.time()
            for iter, (data, labels) in enumerate(self.val_loader):
                data = Variable(data, volatile=True)
                labels = Variable(labels, volatile=True)
                if self.gpu_mode:
                    data = data.cuda()
                    labels = labels.cuda()

                scores= self.model(data)
                total_loss += self.loss_fct(scores, labels[:,0]).data[0]/len(data)

                total_acc += (scores.max(1)[1] == labels[:,0]).sum().data[0]/len(labels)


                if self.vis_wrong_cls1:
                    # Visualize some exmaples where the model failed to correctly classify the background object.
                    wrong_cls2_idx = torch.nonzero((scores.max(1)[1] != labels[:,0])).squeeze()
                    if wrong_cls2_idx.dim() != 0:
                        if self.vis_data is None:
                            n_select = self.n_vis
                        elif len(self.vis_data) < self.n_vis:
                            n_select = self.n_vis - len(self.vis_data)
                        selected_data = data[wrong_cls2_idx[:n_select]].data
                        selected_pred = scores.max(1)[1][wrong_cls2_idx[:n_select]].data
                        selected_labels = labels[wrong_cls2_idx[:n_select]].data

                        if self.vis_data is None:
                            self.vis_data = selected_data
                            self.vis_pred = selected_pred
                            self.vis_labels = selected_labels
                        elif len(self.vis_data) < self.n_vis:
                            self.vis_data = torch.cat([self.vis_data, selected_data], 0)
                            self.vis_pred = torch.cat([self.vis_pred, selected_pred], 0)
                            self.vis_labels = torch.cat([self.vis_labels, selected_labels], 0)


                if ((iter + 1) % 100) == 0:
                    iter_time = time.time() - iter_start_time
                    iter_start_time = time.time()
                    print("Epoch {}: [{}/{}] total_loss: {:.3f} total_acc: {:.3f} ({:.3f} sec)".format(
                        (epoch + 1), (iter + 1), len(self.val_loader), total_loss/(iter+1), total_acc/(iter+1), iter_time))
 

            total_loss /= len(self.val_loader)
            total_acc /= len(self.val_loader)

            self.scheduler.step(total_acc)

            self.hist['val_total_loss'].append(total_loss)
            self.hist['val_total_acc'].append(total_acc)

            self.hist['per_epoch_time'].append(time.time() - epoch_start_time)
            print('per_epoch_time: {:.6f}'.format(self.hist['per_epoch_time'][-1]))

            self.visualize_train_results((epoch+1))

            # Save the model. Since it's not clear which measure of performance defines the 'best' model, we save the best model for each measure of performance.
            # Checkpoint
            self.save_model(save_name=self.model_type+'_checkpoint.pth.tar')
            self.save_as_pkl(obj=self.hist, save_name=self.model_type+'_checkpoint'+'_history.pkl')

            # Best total loss
            if self.hist['val_total_loss'][-1] < min(self.hist['val_total_loss'][:-1] + [float('inf')]): # [float('inf')] takes care of the case when [:-1] returns an empty list, as for epoch 1.
                self.save_model(save_name=self.best_total_loss_save_name+'.pth.tar')
                self.save_as_pkl(obj=self.hist, save_name=self.best_total_loss_save_name+'_history.pkl')
                self.record_hps_and_results(save_name=self.best_total_loss_save_name+'_hps.txt')

        self.hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: {:.2f}, total {} epochs time: {:.2f}".format(np.mean(self.hist['per_epoch_time']),
              self.epoch, self.hist['total_time'][0]))
        print("Training finish!... save training results")

    def hp_search(self, save_dir=None):
        '''
        Copied and made minor modifications to train(self).
        '''

        if save_dir is None:
            save_dir = self.hp_dir

        self.hist = {}
        self.hist['total_loss'] = []
        self.hist['total_acc'] = []

        self.hist['val_total_loss'] = []
        self.hist['val_total_acc'] = []

        self.hist['per_epoch_time'] = []
        self.hist['total_time'] = []

        num_bad_epochs = 0

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
            self.loss_fct.train()
            iter_start_time = time.time()
            for iter, (data, labels) in enumerate(self.train_loader):
                # note that even though ind_objs are not used for backprop, we still make it a Variable, because it will enter into self.loss_fct.recon_loss_fct, which is nn.Module.
                data = Variable(data)
                labels = Variable(labels)
                if self.gpu_mode:
                    data = data.cuda()
                    labels = labels.cuda()

                self.optimizer.zero_grad()
                scores = self.model(data)
                total_loss = self.loss_fct(scores, labels[:,0])/len(data)
                total_loss.backward()
                self.optimizer.step()

                total_acc = (scores.max(1)[1] == labels[:,0]).sum().data[0]/len(labels)

                self.hist['total_loss'].append(total_loss.data[0])
                self.hist['total_acc'].append(total_acc)

                if ((iter + 1) % 100) == 0:
                    iter_time = time.time() - iter_start_time
                    iter_start_time = time.time()
                    print("Epoch {}: [{}/{}] total_loss: {:.3f} total_acc: {:.3f} ({:.3f} sec)".format(
                        (epoch + 1), (iter + 1), len(self.train_loader), total_loss.data[0], total_acc, iter_time))


            # Validation of the main model and z_classifier.
            print('')
            print('Validation begins.')
            print('')
            total_loss = 0.0
            total_acc = 0.0

            self.vis_data = None
            self.vis_pred = None
            self.vis_labels = None

            self.model.eval()
            self.loss_fct.eval()
            iter_start_time = time.time()
            for iter, (data, labels) in enumerate(self.val_loader):
                data = Variable(data, volatile=True)
                labels = Variable(labels, volatile=True)
                if self.gpu_mode:
                    data = data.cuda()
                    labels = labels.cuda()

                scores = self.model(data)
                total_loss += self.loss_fct(scores, labels[:,0]).data[0]/len(data)

                total_acc += (scores.max(1)[1] == labels[:,0]).sum().data[0]/len(labels)

                if ((iter + 1) % 100) == 0:
                    iter_time = time.time() - iter_start_time
                    iter_start_time = time.time()
                    print('model_type: {} / lr: {:.1e} / weight_decay: {:.1e} / batch_size: {}'.format(self.model_type, self.lr, self.weight_decay, self.batch_size))
                    print("Epoch {}: [{}/{}] total_loss: {:.3f} total_acc: {:.3f} ({:.3f} sec)".format(
                        (epoch + 1), (iter + 1), len(self.val_loader), total_loss/(iter+1), total_acc/(iter+1), iter_time))
 

            total_loss /= len(self.val_loader)
            total_acc /= len(self.val_loader)

            self.hist['val_total_loss'].append(total_loss)
            self.hist['val_total_acc'].append(total_acc)

            self.hist['per_epoch_time'].append(time.time() - epoch_start_time)
            print('per_epoch_time: {:.6f}'.format(self.hist['per_epoch_time'][-1]))

            # Save the model. Since it's not clear which measure of performance defines the 'best' model, we save the best model for each measure of performance.
            # Checkpoint

            # Best total loss
            if self.hist['val_total_acc'][-1] > max(self.hist['val_total_acc'][:-1] + [float('-inf')]): # [float('inf')] takes care of the case when [:-1] returns an empty list, as for epoch 1.
                self.save_model(save_dir=save_dir, save_name=self.hp_model_save_name)
                self.save_as_pkl(obj=self.hist, save_dir=save_dir, save_name=self.hp_hist_save_name)

            if self.scheduler.is_better(total_acc, self.scheduler.best):
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1

            # Stop hp search if the following criteria are met.
            if len(self.hist['val_total_acc']) == 10 and max(self.hist['val_total_acc']) < 0.6:
                print('')
                print('val_total_acc did not exceed 0.6 in 10 epochs.')
                print('model_type: {}'.format(self.model_type))
                print('lr: {:.1e}'.format(self.lr))
                print('weight_decay: {:.1e}'.format(self.weight_decay))
                print('batch_size: {}'.format(self.batch_size))
                print('dropout_p: {:.2f}'.format(self.dropout_p))
                print('best_acc: {:.3f}'.format(self.hist['val_total_acc'][-1]))
                print('')
                os.rename(os.path.join(save_dir, self.hp_model_save_name), os.path.join(save_dir, 'acc_{:.3f}_'.format(max(self.hist['val_total_acc'])) + self.hp_model_save_name))
                os.rename(os.path.join(save_dir, self.hp_hist_save_name), os.path.join(save_dir, 'acc_{:.3f}_'.format(max(self.hist['val_total_acc'])) + self.hp_hist_save_name))
                return max(self.hist['val_total_acc'])
            elif num_bad_epochs == 10:
                print('')
                print('num_bad_epochs reached 10.')
                print('model_type: {}'.format(self.model_type))
                print('lr: {:.1e}'.format(self.lr))
                print('weight_decay: {:.1e}'.format(self.weight_decay))
                print('batch_size: {}'.format(self.batch_size))
                print('dropout_p: {:.2f}'.format(self.dropout_p))
                print('best_acc: {:.3f}'.format(self.hist['val_total_acc'][-1]))
                print('')
                os.rename(os.path.join(save_dir, self.hp_model_save_name), os.path.join(save_dir,'acc_{:.3f}_'.format(max(self.hist['val_total_acc'])) + self.hp_model_save_name))
                os.rename(os.path.join(save_dir, self.hp_hist_save_name), os.path.join(save_dir,'acc_{:.3f}_'.format(max(self.hist['val_total_acc'])) + self.hp_hist_save_name))
                return max(self.hist['val_total_acc'])

            self.scheduler.step(total_acc, (epoch + 1))

        print('model_type: {}'.format(self.model_type))
        print('lr: {:.1e}'.format(self.lr))
        print('weight_decay: {:.1e}'.format(self.weight_decay))
        print('batch_size: {}'.format(self.batch_size))
        print('dropout_p: {:.2f}'.format(self.dropout_p))
        print('best_acc: {:.3f}'.format(self.hist['val_total_acc'][-1]))
        print('')

        self.hist['total_time'].append(time.time() - start_time)

        print("Avg one epoch time: {:.2f}, total {} epochs time: {:.2f}".format(np.mean(self.hist['per_epoch_time']),
              self.epoch, self.hist['total_time'][0]))
        print('')

        os.rename(os.path.join(save_dir, self.hp_model_save_name), os.path.join(save_dir,'acc_{:.3f}_'.format(max(self.hist['val_total_acc'])) + self.hp_model_save_name))
        os.rename(os.path.join(save_dir, self.hp_hist_save_name), os.path.join(save_dir,'acc_{:.3f}_'.format(max(self.hist['val_total_acc'])) + self.hp_hist_save_name))
        return max(self.hist['val_total_acc'])


    def best_hp_test(self, trial_idx, data_dir=None):
        
        if data_dir is None:
            data_dir = self.data_dir

        best_val_total_acc = self.hp_search(save_dir=self.best_hp_test_dir)

        prev_model_save_name = os.path.join(self.best_hp_test_dir, 'acc_{:.3f}_'.format(best_val_total_acc) + self.hp_model_save_name)
        new_model_save_name = os.path.join(self.best_hp_test_dir,'trial_{}_acc_{:.3f}_'.format(trial_idx, best_val_total_acc) + self.hp_model_save_name)

        prev_hist_save_name = os.path.join(self.best_hp_test_dir, 'acc_{:.3f}_'.format(best_val_total_acc) + self.hp_hist_save_name)
        new_hist_save_name = os.path.join(self.best_hp_test_dir,'trial_{}_acc_{:.3f}_'.format(trial_idx, best_val_total_acc) + self.hp_hist_save_name)

        os.rename(prev_model_save_name, new_model_save_name)
        os.rename(prev_hist_save_name, new_hist_save_name)

        # We should not use the latest model, but the one with best val acc.
        self.model.load_state_dict(torch.load(new_model_save_name))

        kwargs = {'num_workers': 8, 'pin_memory': True} if self.gpu_mode else {}
        self.test_loader = DataLoader(utils.MultiObjectDataset(data_dir, data_type='test'), shuffle=True, batch_size=self.test_batch_size, 
            drop_last=False, **kwargs)

        # Validation of the main model and z_classifier.
        print('')
        print('Test begins.')
        print('')
        total_loss = 0.0
        total_acc = 0.0

        self.model.eval()
        self.loss_fct.eval()
        iter_start_time = time.time()
        for iter, (data, labels) in enumerate(self.test_loader):
            data = Variable(data, volatile=True)
            labels = Variable(labels, volatile=True)
            if self.gpu_mode:
                data = data.cuda()
                labels = labels.cuda()

                scores = self.model(data)
                total_loss += self.loss_fct(scores, labels[:,0]).data[0]/len(data)

                total_acc += (scores.max(1)[1] == labels[:,0]).sum().data[0]/len(labels)

                if ((iter + 1) % 100) == 0:
                    iter_time = time.time() - iter_start_time
                    iter_start_time = time.time()
                    print('model_type: {} / lr: {:.1e} / weight_decay: {:.1e} / batch_size: {}'.format(self.model_type, self.lr, self.weight_decay, self.batch_size))
                    print("Test Result: [{}/{}] total_loss: {:.3f} total_acc: {:.3f} ({:.3f} sec)".format(
                        (iter + 1), len(self.test_loader), total_loss/(iter+1), total_acc/(iter+1), iter_time))

        total_acc /= len(self.test_loader)
        total_loss /= len(self.test_loader)
    
        with open(os.path.join(self.best_hp_test_dir, 'trial_{}_test_performance.txt'.format(trial_idx)), 'w') as f:
            f.write('total_acc: {:.4f}\n'.format(total_acc))
            f.write('total_loss: {:.4f}\n'.format(total_loss))

        return total_acc

    def visualize_train_results(self, epoch):
        plot_dir = os.path.join(self.log_dir, self.data_dir, self.model_type)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            
        if self.vis_wrong_cls1:
            self.show_wrong_cls_examples(self.vis_data.cpu().numpy(), self.vis_pred.cpu().numpy(), self.vis_labels.cpu().numpy(), self.label_dict, plot_dir, self.model_type, epoch)
        utils.multi_object_classifier_loss_plot(self.hist, plot_dir, self.model_type)
        utils.val_multi_object_classifier_loss_plot(self.hist, plot_dir, self.model_type)
        utils.multi_object_classifier_acc_plot(self.hist, plot_dir, self.model_type)
        utils.val_multi_object_classifier_acc_plot(self.hist, plot_dir, self.model_type)
        utils.per_epoch_time_plot(self.hist, plot_dir, self.model_type)


    def save_model(self, save_dir=None, save_name='.pth.tar'):
        if save_dir is None:
            save_dir = os.path.join(self.save_dir, self.data_dir, self.model_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model.state_dict(), os.path.join(save_dir, save_name))

    def save_as_pkl(self, obj, save_dir=None, save_name='.pkl'):
        if save_dir is None:
                save_dir = os.path.join(self.save_dir, self.data_dir, self.model_type)
        with open(os.path.join(save_dir, save_name), 'wb') as f:
            pickle.dump(obj, f)

    def record_hps_and_results(self, save_dir=None, save_name='.txt'):
        if save_dir is None:
            save_dir = os.path.join(self.save_dir, self.data_dir, self.model_type)
        # Output a text file summarizing the parameters that generated the dataset.
        # We can record not only fixed, initial hps, but also constantly changing ones, such as decaying lr, if we decay lr.
        with open(os.path.join(save_dir, save_name), 'w') as summary:
            summary.write('epochs trained: {}'.format(self.cur_epoch))

    def load(self, save_dir=None, save_name='.pth.tar'):
        if save_dir is None:
            save_dir = os.path.join(self.save_dir, self.data_dir, self.model_type)
        # This can be used after initializing an instance of ROAM_Exp.
        self.model.load_state_dict(torch.load(os.path.join(save_dir, save_name)))

    def show_wrong_cls_examples(self, images, pred, labels, label_dict, save_dir, model_name, epoch):
        fig = plt.figure(figsize=(30,10))
        outer = gridspec.GridSpec(1, 3, wspace=0.2, hspace=0.2)
        utils.show_images_inner(images, fig, outer[0], '')
        utils.show_labels_inner(pred, label_dict, fig, outer[1], '')
        utils.show_labels_inner(labels, label_dict, fig, outer[2], '')

        fig.savefig(os.path.join(save_dir, model_name + '_wrong_cls2_epoch{}.png'.format(epoch)))
        plt.close(fig)