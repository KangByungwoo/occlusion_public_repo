import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math


class PoseNetRefiner(nn.Module):
    '''
    Input: x_f, x_b, x and p_b, where x is the two-object image in the original pose, x_f and x_b the output from the Baseline,
    and p_b is the predicted canon pose of the x_b from PoseNet.
    Output: a refined_p_b, which is optimized with L2 loss with the ground truth p_b.
    '''
    def __init__(self, input_size=28, inter_size=28, input_channel=1, hidden1=128, hidden2=128, hidden3=128, hidden4=256, hidden5=128, n_p=4):
        super(PoseNetRefiner, self).__init__()
        self.input_size = input_size
        self.inter_size = inter_size
        self.input_channel = input_channel
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.hidden5 = hidden5
        self.n_p = n_p
        self.sxy_embedding = nn.Sequential(
            nn.Conv2d(3*input_channel, hidden1, 4, 2, 1),
            nn.BatchNorm2d(hidden1),
            nn.ReLU(),
            nn.Conv2d(hidden1, hidden2, 4, 2, 1),
            nn.BatchNorm2d(hidden2),
            nn.ReLU(),
            nn.Conv2d(hidden2, hidden3, 4, 2, 1),
            nn.BatchNorm2d(hidden3),
            nn.ReLU(),
            Flatten(),
            nn.Linear(hidden3*(input_size//8)**2, hidden4),
            nn.BatchNorm1d(hidden4),
            nn.ReLU(),
            nn.Linear(hidden4, hidden5),
            nn.BatchNorm1d(hidden5),
            nn.ReLU(),
            nn.Linear(hidden5, n_p-1),
            )

        self.angle_embedding = nn.Sequential(
            nn.Conv2d(3*input_channel, hidden1, 4, 2, 1),
            nn.BatchNorm2d(hidden1),
            nn.ReLU(),
            nn.Conv2d(hidden1, hidden2, 4, 2, 1),
            nn.BatchNorm2d(hidden2),
            nn.ReLU(),
            nn.Conv2d(hidden2, hidden3, 4, 2, 1),
            nn.BatchNorm2d(hidden3),
            nn.ReLU(),
            Flatten(),
            nn.Linear(hidden3*(inter_size//8)**2, hidden4),
            nn.BatchNorm1d(hidden4),
            nn.ReLU(),
            nn.Linear(hidden4, hidden5),
            nn.BatchNorm1d(hidden5),
            nn.ReLU(),
            nn.Linear(hidden5, 1),
            )

        self.init_parameters()


    def init_parameters(self):
        for name, param in self.sxy_embedding.named_parameters():
            if 'weight' in name:
                if param.size() == torch.Size([self.n_p - 1, self.hidden5]) and name[0] == '16':
                    init.constant(param, 0)
                    continue
                if param.dim() > 1:
                    init.kaiming_normal(param)
                elif param.dim() == 1:
                    init.constant(param, 1)
            elif 'bias' in name:
                if param.size(0) == self.n_p - 1 and name[0] == '16':
                    param.data = torch.Tensor([1, 0, 0])
                    continue
                init.constant(param, 0)

        for name, param in self.angle_embedding.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    init.kaiming_normal(param)
                elif param.dim() == 1:
                    init.constant(param, 1)
            elif 'bias' in name:
                init.constant(param, 0)

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

    def forward(self, x_b, x_f, x, p):
        N, C = x.size(0), x.size(1)
        # Make sure that the inputs are detached.
        x_b = x_b.detach()
        x_f = x_f.detach()
        x = x.detach()
        p = p.detach()
        '''
        Before the main forward pass begins, we transform x_b, x_f, and x into the pose p.
        '''
        aff_p = self.affine_form(p)
        grid = F.affine_grid(aff_p, torch.Size([N, C, self.input_size,self.input_size]))
        x_b = F.grid_sample(x_b, grid)
        x_f = F.grid_sample(x_f, grid)
        x = F.grid_sample(x, grid)

        '''
        Now, the main forward pass.
        '''
        dp_sxy = self.sxy_embedding(torch.cat([x_b, x_f, x], 1))

        zero_angle = Variable(torch.zeros(N, 1))
        if x.is_cuda:
            zero_angle = zero_angle.cuda()
        dp = torch.cat([dp_sxy[:,0:1].detach(), zero_angle, dp_sxy[:,1:].detach()], 1)

        aff_dp = self.affine_form(dp)

        grid = F.affine_grid(aff_dp, torch.Size([N, C, self.inter_size, self.inter_size]))
        x_b = F.grid_sample(x_b, grid)
        x_f = F.grid_sample(x_f, grid)
        x = F.grid_sample(x, grid)

        dp_a = self.angle_embedding(torch.cat([x_b, x_f, x], 1)).squeeze()

        refined_p = torch.stack([p[:,0]*dp_sxy[:,0], p[:,1]+dp_a, p[:,2]+dp_sxy[:,1], p[:,3]+dp_sxy[:,2]], 1)

        return refined_p


class PoseNetBasicDetached(nn.Module):
    '''
    Same as FDPoseNetIndObjBN2AngleLater3 in pose_net.py, and only differs from PoseNetBasic in detaching the input to the angle prediction module.
    
    Input: x (N, n_objs, H, W). (e.g. 50 by 50) Individual object images are in separate channels. 
    Output: p (N, n_objs, n_p), where n_p = 4.
    '''
    def __init__(self, input_size=50, inter_size=28, input_channel=1, hidden1=128, hidden2=128, hidden3=128, hidden4=256, hidden5=128, n_p=4):
        super(PoseNetBasicDetached, self).__init__()
        self.input_size = input_size
        self.inter_size = inter_size
        self.input_channel = input_channel
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.hidden5 = hidden5
        self.n_p = n_p
        self.sxy_embedding = nn.Sequential(
            nn.Conv2d(input_channel, hidden1, 4, 2, 1),
            nn.BatchNorm2d(hidden1),
            nn.ReLU(),
            nn.Conv2d(hidden1, hidden2, 4, 2, 1),
            nn.BatchNorm2d(hidden2),
            nn.ReLU(),
            nn.Conv2d(hidden2, hidden3, 4, 2, 1),
            nn.BatchNorm2d(hidden3),
            nn.ReLU(),
            Flatten(),
            nn.Linear(hidden3*(input_size//8)**2, hidden4),
            nn.BatchNorm1d(hidden4),
            nn.ReLU(),
            nn.Linear(hidden4, hidden5),
            nn.BatchNorm1d(hidden5),
            nn.ReLU(),
            nn.Linear(hidden5, n_p-1),
            )

        self.angle_embedding = nn.Sequential(
            nn.Conv2d(input_channel, hidden1, 4, 2, 1),
            nn.BatchNorm2d(hidden1),
            nn.ReLU(),
            nn.Conv2d(hidden1, hidden2, 4, 2, 1),
            nn.BatchNorm2d(hidden2),
            nn.ReLU(),
            nn.Conv2d(hidden2, hidden3, 4, 2, 1),
            nn.BatchNorm2d(hidden3),
            nn.ReLU(),
            Flatten(),            
            nn.Linear(hidden3*(inter_size//8)**2, hidden4),
            nn.BatchNorm1d(hidden4),
            nn.ReLU(),
            nn.Linear(hidden4, hidden5),
            nn.BatchNorm1d(hidden5),
            nn.ReLU(),
            nn.Linear(hidden5, 1),
            )

        self.init_parameters()


    def init_parameters(self):
        for name, param in self.sxy_embedding.named_parameters():
            if 'weight' in name:
                if param.size() == torch.Size([self.n_p - 1, self.hidden5]):
                    init.constant(param, 0)
                    continue
                if param.dim() > 1:
                    init.kaiming_normal(param)
                elif param.dim() == 1:
                    init.constant(param, 1)
            elif 'bias' in name:
                if param.size(0) == self.n_p - 1:
                    param.data = torch.Tensor([1, 0, 0])
                    continue
                init.constant(param, 0)

        for name, param in self.angle_embedding.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    init.kaiming_normal(param)
                elif param.dim() == 1:
                    init.constant(param, 1)
            elif 'bias' in name:
                init.constant(param, 0)


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

    def forward(self, x):
        N, C = x.size(0), x.size(1)
        p_sxy = self.sxy_embedding(x)
        zero_angle = Variable(torch.zeros(N, 1))
        if x.is_cuda:
            zero_angle = zero_angle.cuda()
        p = torch.cat([p_sxy[:,0:1].detach(), zero_angle, p_sxy[:,1:].detach()], 1)

        aff_p = self.affine_form(p)

        grid = F.affine_grid(aff_p, torch.Size([N, C, self.inter_size, self.inter_size]))
        inter_x = F.grid_sample(x, grid)

        angle = self.angle_embedding(inter_x)

        return torch.cat([p_sxy[:,0:1], angle, p_sxy[:,1:]], 1)





class PoseNetBasic(nn.Module):
    '''
    Same as FDPoseNetIndObjBN2AngleLater2 in pose_net.py
    
    Input: x (N, n_objs, H, W). (e.g. 50 by 50) Individual object images are in separate channels. 
    Output: p (N, n_objs, n_p), where n_p = 4.
    '''
    def __init__(self, input_size=50, inter_size=28, input_channel=1, hidden1=128, hidden2=128, hidden3=128, hidden4=256, hidden5=128, n_p=4):
        super(PoseNetBasic, self).__init__()
        self.input_size = input_size
        self.inter_size = inter_size
        self.input_channel = input_channel
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.hidden5 = hidden5
        self.n_p = n_p
        self.sxy_embedding = nn.Sequential(
            nn.Conv2d(input_channel, hidden1, 4, 2, 1),
            nn.BatchNorm2d(hidden1),
            nn.ReLU(),
            nn.Conv2d(hidden1, hidden2, 4, 2, 1),
            nn.BatchNorm2d(hidden2),
            nn.ReLU(),
            nn.Conv2d(hidden2, hidden3, 4, 2, 1),
            nn.BatchNorm2d(hidden3),
            nn.ReLU(),
            Flatten(),
            nn.Linear(hidden3*(input_size//8)**2, hidden4),
            nn.BatchNorm1d(hidden4),
            nn.ReLU(),
            nn.Linear(hidden4, hidden5),
            nn.BatchNorm1d(hidden5),
            nn.ReLU(),
            nn.Linear(hidden5, n_p-1),
            )

        self.angle_embedding = nn.Sequential(
            nn.Conv2d(input_channel, hidden1, 4, 2, 1),
            nn.BatchNorm2d(hidden1),
            nn.ReLU(),
            nn.Conv2d(hidden1, hidden2, 4, 2, 1),
            nn.BatchNorm2d(hidden2),
            nn.ReLU(),
            nn.Conv2d(hidden2, hidden3, 4, 2, 1),
            nn.BatchNorm2d(hidden3),
            nn.ReLU(),
            Flatten(),            
            nn.Linear(hidden3*(inter_size//8)**2, hidden4),
            nn.BatchNorm1d(hidden4),
            nn.ReLU(),
            nn.Linear(hidden4, hidden5),
            nn.BatchNorm1d(hidden5),
            nn.ReLU(),
            nn.Linear(hidden5, 1),
            )

        self.init_parameters()


    def init_parameters(self):
        for name, param in self.sxy_embedding.named_parameters():
            if 'weight' in name:
                if param.size() == torch.Size([self.n_p - 1, self.hidden5]):
                    init.constant(param, 0)
                    continue
                if param.dim() > 1:
                    init.kaiming_normal(param)
                elif param.dim() == 1:
                    init.constant(param, 1)
            elif 'bias' in name:
                if param.size(0) == self.n_p - 1:
                    param.data = torch.Tensor([1, 0, 0])
                    continue
                init.constant(param, 0)

        for name, param in self.angle_embedding.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    init.kaiming_normal(param)
                elif param.dim() == 1:
                    init.constant(param, 1)
            elif 'bias' in name:
                init.constant(param, 0)


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

    def forward(self, x):
        N, C = x.size(0), x.size(1)
        p_sxy = self.sxy_embedding(x)
        zero_angle = Variable(torch.zeros(N, 1))
        if x.is_cuda:
            zero_angle = zero_angle.cuda()
        p = torch.cat([p_sxy[:,0:1], zero_angle, p_sxy[:,1:]], 1)

        aff_p = self.affine_form(p)

        grid = F.affine_grid(aff_p, torch.Size([N, C, self.inter_size, self.inter_size]))
        inter_x = F.grid_sample(x, grid)

        angle = self.angle_embedding(inter_x)

        return torch.cat([p_sxy[:,0:1], angle, p_sxy[:,1:]], 1)





'''
We will use Det_FDBaseline4_latent_256_lighter of occlusion_pfc_prototype_exp.py as our baseline. This showed a reasonably good performance without being too huge.
In particular, in terms of reconstruction of the background object, I think it was not much worse than Det_FDBaseline4_latent_512, although the classification accuracy
was somewhat lower. But, for the PoseNet, all that matters will be the recon of the background object, so I will stick with Det_FDBaseline4_latent_256_lighter.
The default parameters of the modules below coincide with those of Det_FDBaseline4_latent_256_lighter.

One important distinction from Det_FDBaseline4_latent_256_lighter is that this only outputs the recon, not mask and app separately.
Acordingly, in new_pose_net_exp.py, we do not optimize the full recon loss, which requires separate masks and apps to compose the full recon.


'''

class Baseline(nn.Module):
    def __init__(self, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, latent=256, cls_hidden1=128, cls_hidden2=64, n_classes=4, dropout_p=0.5):
        super(Baseline, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.latent = latent
        self.cls_hidden1 = cls_hidden1
        self.cls_hidden2 = cls_hidden2
        self.n_classes = n_classes
        self.dropout_p = dropout_p

        self.encoder = BaselineEncoder(hidden1=2*hidden1, hidden2=2*hidden2, hidden3=2*hidden3, latent=2*latent)
        self.decoder = BaselineDecoder(hidden1=hidden1, hidden2=hidden2, hidden3=hidden3, latent=latent)

        # Linear classifier on top of the latent space.
        self.cls_fc1 = nn.Linear(latent, cls_hidden1)
        self.cls_fc1_bn = nn.BatchNorm2d(cls_hidden1)
        self.cls_fc2 = nn.Linear(cls_hidden1, cls_hidden2)
        self.cls_fc2_bn = nn.BatchNorm2d(cls_hidden2)
        self.cls_fc3 = nn.Linear(cls_hidden2, n_classes)

        if dropout_p is not None:
            assert isinstance(dropout_p, float)
            self.dropout = nn.Dropout(p=dropout_p)

        self.relu = nn.ReLU()
        # negative_slope set to 0.2 following https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/ops.py and the DCGAN paper.
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def classifier(self, z):
        z = self.relu(self.cls_fc1_bn(self.cls_fc1(z)))
        if self.dropout_p is not None:
            z = self.dropout(z)
        z = self.relu(self.cls_fc2_bn(self.cls_fc2(z)))
        if self.dropout_p is not None:
            z = self.dropout(z)            
        return self.cls_fc3(z)

    def forward(self, x, single_obj=False):
        scores = []
        recons = []
        if single_obj:
            self.zs = [self.encoder(x)[:,:self.latent]]
        else:
            self.zs = list(torch.split(self.encoder(x), self.latent, 1))


        for i in range(2):
            recon = self.decoder(self.zs[i])
            if self.dropout_p is not None:
                score = self.classifier(self.dropout(self.zs[i]))
            else:
                score = self.classifier(self.zs[i])
            if single_obj:
                self.zs = self.zs[0]
                return score, recon
            scores.append(score)
            recons.append(recon)

        scores = torch.stack(scores, 1)
        recons = torch.stack(recons, 1)

        self.zs = torch.stack(self.zs, 1)

        return scores, recons



class BaselineEncoder(nn.Module):
    def __init__(self, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, latent=50):
        '''
        Same as FDBaselineEncoder except that there is no stocahstic layer in this.
        '''
        super(BaselineEncoder, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.latent = latent

        self.enc_conv1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden1, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(in_channels=hidden1, out_channels=hidden2, kernel_size=4, stride=2, padding=1)
        self.enc_conv2_bn = nn.BatchNorm2d(hidden2)
        self.enc_conv3 = nn.Conv2d(in_channels=hidden2, out_channels=hidden3, kernel_size=4, stride=2, padding=1)
        self.enc_conv3_bn = nn.BatchNorm2d(hidden3)
        self.enc_fc = nn.Linear(int(input_size/8)**2*hidden3, latent)

        # negative_slope set to 0.2 following https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/ops.py and the DCGAN paper.
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # When hc0, cc0 are the initial states of ROOM, we need to take into account that they have
        # shapes (hidden_size,), not (batch_size, hidden_size). So, we just copy them along the batch dimension.
        h = self.lrelu(self.enc_conv1(x))
        h = self.lrelu(self.enc_conv2_bn(self.enc_conv2(h)))
        h = self.lrelu(self.enc_conv3_bn(self.enc_conv3(h))).view(-1, int(self.input_size/8)**2*self.hidden3)
        return self.enc_fc(h)




class BaselineDecoder(nn.Module):
    def __init__(self, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, latent=50, n_classes=4, eps=1e-6):
        super(BaselineDecoder, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.latent = latent
        self.n_classes = n_classes
        self.eps = eps


        self.dec_fc = nn.Linear(latent, int(input_size/8)**2*hidden3)
        self.dec_fc_bn = nn.BatchNorm1d(int(input_size/8)**2*hidden3)
        self.dec_conv1 = nn.ConvTranspose2d(in_channels=hidden3, out_channels=hidden2, kernel_size=4, stride=2, padding=1)
        self.dec_conv1_bn = nn.BatchNorm2d(hidden2)
        self.dec_conv2 = nn.ConvTranspose2d(in_channels=hidden2, out_channels=hidden1, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.dec_conv2_bn = nn.BatchNorm2d(hidden1)
        self.dec_conv3 = nn.ConvTranspose2d(in_channels=hidden1, out_channels=input_channels, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.relu(self.dec_fc_bn(self.dec_fc(z))).view(-1, self.hidden3, int(self.input_size/8), int(self.input_size/8))
        h = self.relu(self.dec_conv1_bn(self.dec_conv1(h)))
        h = self.relu(self.dec_conv2_bn(self.dec_conv2(h)))
        return self.sigmoid(self.dec_conv3(h)).clamp(self.eps, 1-self.eps)




class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)

