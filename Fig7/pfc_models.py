import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math

from search_space_canon_gans import generator

class PFC4(nn.Module):
    '''
    PFC4 is the same as PFC3 except that the output of the attention module is 6D.
    Input: raw canon occluded image x, the foreground object image x_f. Both (N, C, H, W).
    Output: a sequence of dzs. (N, n_iter, z_dim). the sum of dzs[:,i]'s up to time t will become the estimate of z at time t, z_t. 

    Note that this is not an RNN, as there is no hidden state passed between time steps.
    '''
    def __init__(self, G=None, n_iter=5, z_dim=62, input_size=50, patch_size=28, input_channel=1, n_att_iter=1, att_hidden1=128, att_hidden2=128, att_hidden3=128, 
        att_hidden4=256, att_hidden5=256, n_p=6, hidden1=256, hidden2=256, hidden3=128, hidden4=512, hidden5=512):
        super(PFC4, self).__init__()
        if G is None:
            self.G = generator()
        else:
            self.G = G
        self.n_iter = n_iter
        self.z_dim = z_dim
        self.input_size = input_size
        self.patch_size = patch_size
        self.input_channel = input_channel
        self.n_att_iter = n_att_iter
        self.att_hidden1 = att_hidden1
        self.att_hidden2 = att_hidden2
        self.att_hidden3 = att_hidden3
        self.att_hidden4 = att_hidden4
        self.att_hidden5 = att_hidden5
        self.n_p = n_p
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.hidden5 = hidden5

        self.attention1 = nn.Sequential(
            nn.Conv2d(input_channel*2, att_hidden1, 4, 2, 1),
            nn.BatchNorm2d(att_hidden1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(att_hidden1, att_hidden2, 4, 2, 1),
            nn.BatchNorm2d(att_hidden2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(att_hidden2, att_hidden3, 4, 2, 1),
            nn.BatchNorm2d(att_hidden3),
            nn.LeakyReLU(0.2),
            Flatten(),
            nn.Linear(att_hidden3*(input_size//8)**2, att_hidden4),
            nn.BatchNorm1d(att_hidden4),
            nn.LeakyReLU(0.2),
            )

        self.attention2 = nn.Sequential(
            nn.Linear(att_hidden4+n_p, att_hidden5),
            nn.BatchNorm1d(att_hidden5),
            nn.LeakyReLU(0.2),
            nn.Linear(att_hidden5, n_p)
            )

        # To embedding, a channel-wise concatenation of x, x_f, G(z_t).
        self.embedding = nn.Sequential(
            nn.Conv2d(input_channel*3, hidden1, 4, 2, 1),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden1, hidden2, 4, 2, 1),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden2, hidden3, 4, 2, 1),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2),
            Flatten(),
            nn.Linear(hidden3*(patch_size//8)**2, hidden4),
            nn.BatchNorm1d(hidden4),
            nn.LeakyReLU(0.2)
            )

        self.regressor = nn.Sequential(
            nn.Linear(hidden4+z_dim, hidden5),
            nn.BatchNorm1d(hidden5),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden5, z_dim)
            )

        self.z0 = Parameter(torch.Tensor(z_dim))
        self.p0 = Variable(torch.Tensor([[1, 0, 0], [0, 1, 0]]).view(n_p))
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        self.init_parameters()

    def init_parameters(self):
        for name, param in self.attention1.named_parameters():
            if 'weight' in name and param.dim() > 1:
                init.kaiming_normal(param)
            elif 'weight' in name and param.dim() == 1:
                # This corresponds to batchnorm layers.
                init.constant(param, 1.0)
            elif 'bias' in name:
                init.constant(param, 0.0)

        for name, param in self.attention2.named_parameters():
            # We make sure that initially the predicted pose is the identity.
            # Note that the output of self.attention2 is dp, so we want it to be initially zero.
            if name[0] == '3':
                if 'weight' in name:
                    init.constant(param, 0.0)
                elif 'bias' in name:
                    init.constant(param, 0.0)
                continue

            if 'weight' in name and param.dim() > 1:
                init.kaiming_normal(param)
            elif 'weight' in name and param.dim() == 1:
                # This corresponds to batchnorm layers.
                init.constant(param, 1.0)
            elif 'bias' in name:
                init.constant(param, 0.0)


        for name, param in self.embedding.named_parameters():
            if 'weight' in name and param.dim() > 1:
                init.kaiming_normal(param)
            elif 'weight' in name and param.dim() == 1:
                # This corresponds to batchnorm layers.
                init.constant(param, 1.0)
            elif 'bias' in name:
                init.constant(param, 0.0)

        for name, param in self.regressor.named_parameters():
            if 'weight' in name and param.dim() > 1:
                init.kaiming_normal(param)
            elif 'weight' in name and param.dim() == 1:
                # This corresponds to batchnorm layers.
                init.constant(param, 1.0)
            elif 'bias' in name:
                init.constant(param, 0.0)

        init.constant(self.z0, 0.0) # Note that we apply sigmoid to this in the forward method, so z0 is actually initialized to 0.5.


    def forward(self, x, x_f):
        N, C, _, _ = x.size()

        # Attenion module
        self.ps = []
        p = self.p0.expand(N, self.n_p)
        if x.is_cuda:
            p = p.cuda()
        self.ps.append(p)

        for i in range(self.n_att_iter): 
            att_features = self.attention1(torch.cat([x, x_f], 1))
            dp = self.attention2(torch.cat([att_features, p],1))
            p = p + dp
            if (i+1) != self.n_att_iter:
                grid = F.affine_grid(p.view(-1, 2, 3), x.size())
            else:
                grid = F.affine_grid(p.view(-1, 2, 3), torch.Size([N, C, self.patch_size, self.patch_size]))
            x = F.grid_sample(x, grid)
            x_f = F.grid_sample(x_f, grid)
            self.ps.append(p)

        self.ps = torch.stack(self.ps, 1)
        self.canon_x = x
        self.canon_x_f = x_f

        # PFC
        self.zs = []
        self.recons = []
        z = self.sigmoid(self.z0.expand(N, self.z_dim))
        recon = self.G(z)
        self.zs.append(z)
        self.recons.append(recon)

        for i in range(self.n_iter):
            features = self.embedding(torch.cat([x, x_f, recon],1))
            dz = self.regressor(torch.cat([features, z], 1))
            z = z + dz
            recon = self.G(z)
            self.zs.append(z)
            self.recons.append(recon)

        self.zs = torch.stack(self.zs, 1)
        self.recons = torch.stack(self.recons, 1)

        return recon

class PFC3(nn.Module):
    '''
    PFC3 is essentially attenion module + PFC2.
    Input: raw canon occluded image x, the foreground object image x_f. Both (N, C, H, W).
    Output: a sequence of dzs. (N, n_iter, z_dim). the sum of dzs[:,i]'s up to time t will become the estimate of z at time t, z_t. 

    Note that this is not an RNN, as there is no hidden state passed between time steps.
    '''
    def __init__(self, G=None, n_iter=5, z_dim=62, input_size=50, patch_size=28, input_channel=1, n_att_iter=1, att_hidden1=128, att_hidden2=128, att_hidden3=128, 
        att_hidden4=256, att_hidden5=256, n_p=4, hidden1=256, hidden2=256, hidden3=128, hidden4=512, hidden5=512):
        super(PFC3, self).__init__()
        if G is None:
            self.G = generator()
        else:
            self.G = G
        self.n_iter = n_iter
        self.z_dim = z_dim
        self.input_size = input_size
        self.patch_size = patch_size
        self.input_channel = input_channel
        self.n_att_iter = n_att_iter
        self.att_hidden1 = att_hidden1
        self.att_hidden2 = att_hidden2
        self.att_hidden3 = att_hidden3
        self.att_hidden4 = att_hidden4
        self.att_hidden5 = att_hidden5
        self.n_p = n_p
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.hidden5 = hidden5

        self.attention1 = nn.Sequential(
            nn.Conv2d(input_channel*2, att_hidden1, 4, 2, 1),
            nn.BatchNorm2d(att_hidden1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(att_hidden1, att_hidden2, 4, 2, 1),
            nn.BatchNorm2d(att_hidden2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(att_hidden2, att_hidden3, 4, 2, 1),
            nn.BatchNorm2d(att_hidden3),
            nn.LeakyReLU(0.2),
            Flatten(),
            nn.Linear(att_hidden3*(input_size//8)**2, att_hidden4),
            nn.BatchNorm1d(att_hidden4),
            nn.LeakyReLU(0.2),
            )

        self.attention2 = nn.Sequential(
            nn.Linear(att_hidden4+n_p, att_hidden5),
            nn.BatchNorm1d(att_hidden5),
            nn.LeakyReLU(0.2),
            nn.Linear(att_hidden5, n_p)
            )

        # To embedding, a channel-wise concatenation of x, x_f, G(z_t).
        self.embedding = nn.Sequential(
            nn.Conv2d(input_channel*3, hidden1, 4, 2, 1),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden1, hidden2, 4, 2, 1),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden2, hidden3, 4, 2, 1),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2),
            Flatten(),
            nn.Linear(hidden3*(patch_size//8)**2, hidden4),
            nn.BatchNorm1d(hidden4),
            nn.LeakyReLU(0.2)
            )

        self.regressor = nn.Sequential(
            nn.Linear(hidden4+z_dim, hidden5),
            nn.BatchNorm1d(hidden5),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden5, z_dim)
            )

        self.z0 = Parameter(torch.Tensor(z_dim))
        self.p0 = Variable(torch.Tensor([1.0, 0.0, 0.0, 0.0]))
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        self.init_parameters()

    def init_parameters(self):
        for name, param in self.attention1.named_parameters():
            if 'weight' in name and param.dim() > 1:
                init.kaiming_normal(param)
            elif 'weight' in name and param.dim() == 1:
                # This corresponds to batchnorm layers.
                init.constant(param, 1.0)
            elif 'bias' in name:
                init.constant(param, 0.0)

        for name, param in self.attention2.named_parameters():
            # We make sure that initially the predicted pose is the identity.
            # Note that the output of self.attention2 is dp, so we want it to be initially zero.
            if name[0] == '3':
                if 'weight' in name:
                    init.constant(param, 0.0)
                elif 'bias' in name:
                    init.constant(param, 0.0)
                continue

            if 'weight' in name and param.dim() > 1:
                init.kaiming_normal(param)
            elif 'weight' in name and param.dim() == 1:
                # This corresponds to batchnorm layers.
                init.constant(param, 1.0)
            elif 'bias' in name:
                init.constant(param, 0.0)


        for name, param in self.embedding.named_parameters():
            if 'weight' in name and param.dim() > 1:
                init.kaiming_normal(param)
            elif 'weight' in name and param.dim() == 1:
                # This corresponds to batchnorm layers.
                init.constant(param, 1.0)
            elif 'bias' in name:
                init.constant(param, 0.0)

        for name, param in self.regressor.named_parameters():
            if 'weight' in name and param.dim() > 1:
                init.kaiming_normal(param)
            elif 'weight' in name and param.dim() == 1:
                # This corresponds to batchnorm layers.
                init.constant(param, 1.0)
            elif 'bias' in name:
                init.constant(param, 0.0)

        init.constant(self.z0, 0.0) # Note that we apply sigmoid to this in the forward method, so z0 is actually initialized to 0.5.


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


    def forward(self, x, x_f):
        N, C, _, _ = x.size()

        # Attenion module
        self.ps = []
        p = self.p0.expand(N, self.n_p)
        if x.is_cuda:
            p = p.cuda()
        self.ps.append(p)

        for i in range(self.n_att_iter): 
            att_features = self.attention1(torch.cat([x, x_f], 1))
            dp = self.attention2(torch.cat([att_features, p],1))
            p = p + dp
            aff_p = self.affine_form(p)
            if (i+1) != self.n_att_iter:
                grid = F.affine_grid(aff_p, x.size())
            else:
                grid = F.affine_grid(aff_p, torch.Size([N, C, self.patch_size, self.patch_size]))
            x = F.grid_sample(x, grid)
            x_f = F.grid_sample(x_f, grid)
            self.ps.append(p)

        self.ps = torch.stack(self.ps, 1)
        self.canon_x = x
        self.canon_x_f = x_f

        # PFC
        self.zs = []
        self.recons = []
        z = self.sigmoid(self.z0.expand(N, self.z_dim))
        recon = self.G(z)
        self.zs.append(z)
        self.recons.append(recon)

        for i in range(self.n_iter):
            features = self.embedding(torch.cat([x, x_f, recon],1))
            dz = self.regressor(torch.cat([features, z], 1))
            z = z + dz
            recon = self.G(z)
            self.zs.append(z)
            self.recons.append(recon)

        self.zs = torch.stack(self.zs, 1)
        self.recons = torch.stack(self.recons, 1)

        return recon


class PFC2(nn.Module):
    '''
    Same as PFC1 except that it takes as input original pose images. In particular, it does not predict the pose.
    But, the target image to use for L2 loss is still a canonical pose image.
    At test time, though, only original pose images are used. 

    Input: raw canon occluded image x, the foreground object image x_f. Both (N, C, H, W).
    Output: a sequence of dzs. (N, n_iter, z_dim). the sum of dzs[:,i]'s up to time t will become the estimate of z at time t, z_t. 

    Note that this is not an RNN, as there is no hidden state passed between time steps.
    '''
    def __init__(self, G=None, n_iter=5, z_dim=62, input_size=50, input_channel=1, hidden1=256, hidden2=256, hidden3=128, hidden4=512, hidden5=512):
        super(PFC2, self).__init__()
        if G is None:
            self.G = generator()
        else:
            self.G = G
        self.n_iter = n_iter
        self.z_dim = z_dim
        self.input_size = input_size
        self.input_channel = input_channel
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.hidden5 = hidden5

        # To embedding, a channel-wise concatenation of x, x_f, G(z_t).
        self.embedding = nn.Sequential(
            nn.Conv2d(input_channel*3, hidden1, 4, 2, 1),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden1, hidden2, 4, 2, 1),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden2, hidden3, 4, 2, 1),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2),
            Flatten(),
            nn.Linear(hidden3*(input_size//8)**2, hidden4),
            nn.BatchNorm1d(hidden4),
            nn.LeakyReLU(0.2)
            )

        self.regressor = nn.Sequential(
            nn.Linear(hidden4+z_dim, hidden5),
            nn.BatchNorm1d(hidden5),
            nn.ReLU(),
            nn.Linear(hidden5, z_dim)
            )

        self.z0 = Parameter(torch.Tensor(z_dim))
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        self.init_parameters()

    def init_parameters(self):
        for name, param in self.embedding.named_parameters():
            if 'weight' in name and param.dim() > 1:
                init.kaiming_normal(param)
            elif 'weight' in name and param.dim() == 1:
                # This corresponds to batchnorm layers.
                init.constant(param, 1.0)
            elif 'bias' in name:
                init.constant(param, 0.0)

        for name, param in self.regressor.named_parameters():
            if 'weight' in name and param.dim() > 1:
                init.kaiming_normal(param)
            elif 'weight' in name and param.dim() == 1:
                # This corresponds to batchnorm layers.
                init.constant(param, 1.0)
            elif 'bias' in name:
                init.constant(param, 0.0)

        init.constant(self.z0, 0.0) # Note that we apply sigmoid to this in the forward method, so z0 is actually initialized to 0.5.


    def forward(self, x, x_f):
        self.zs = []
        self.recons = []
        N = x.size(0)
        z = self.sigmoid(self.z0.expand(N, self.z_dim))
        recon = self.G(z)
        self.zs.append(z)
        self.recons.append(recon)

        for i in range(self.n_iter):
            recon = F.upsample(recon, (self.input_size, self.input_size), mode='bilinear')
            features = self.embedding(torch.cat([x, x_f, recon],1))
            dz = self.regressor(torch.cat([features, z], 1))
            z = z + dz
            recon = self.G(z)
            self.zs.append(z)
            self.recons.append(recon)

        self.zs = torch.stack(self.zs, 1)
        self.recons = torch.stack(self.recons, 1)

        return recon



class PFC1ClippingNoFG(nn.Module):
    '''
    Same asPFC1Clipping except that it doesn't take as input the foreground object image x_f
    '''
    def __init__(self, G=None, n_iter=5, z_dim=62, input_size=28, input_channel=1, hidden1=256, hidden2=256, hidden3=128, hidden4=512, hidden5=512):
        super(PFC1ClippingNoFG, self).__init__()
        if G is None:
            self.G = generator()
        else:
            self.G = G
        self.n_iter = n_iter
        self.z_dim = z_dim
        self.input_size = input_size
        self.input_channel = input_channel
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.hidden5 = hidden5

        # To embedding, a channel-wise concatenation of x, x_f, G(z_t).
        self.embedding = nn.Sequential(
            nn.Conv2d(input_channel*2, hidden1, 4, 2, 1),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden1, hidden2, 4, 2, 1),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden2, hidden3, 3, 1, 1),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2),
            Flatten(),
            nn.Linear(hidden3*(input_size//4)**2, hidden4),
            nn.BatchNorm1d(hidden4),
            nn.LeakyReLU(0.2)
            )

        self.regressor = nn.Sequential(
            nn.Linear(hidden4+z_dim, hidden5),
            nn.BatchNorm1d(hidden5),
            nn.ReLU(),
            nn.Linear(hidden5, z_dim),
            nn.Tanh()
            )

        self.z0 = Parameter(torch.Tensor(z_dim))
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        self.init_parameters()

    def init_parameters(self):
        for name, param in self.embedding.named_parameters():
            if 'weight' in name and param.dim() > 1:
                init.kaiming_normal(param)
            elif 'weight' in name and param.dim() == 1:
                # This corresponds to batchnorm layers.
                init.constant(param, 1.0)
            elif 'bias' in name:
                init.constant(param, 0.0)

        for name, param in self.regressor.named_parameters():
            if 'weight' in name and param.dim() > 1:
                init.kaiming_normal(param)
            elif 'weight' in name and param.dim() == 1:
                # This corresponds to batchnorm layers.
                init.constant(param, 1.0)
            elif 'bias' in name:
                init.constant(param, 0.0)

        init.constant(self.z0, 0.0) # Note that we apply sigmoid to this in the forward method, so z0 is actually initialized to 0.5.


    def forward(self, x):
        self.zs = []
        self.recons = []
        N = x.size(0)
        z = self.sigmoid(self.z0.expand(N, self.z_dim))
        recon = self.G(z)
        self.zs.append(z)
        self.recons.append(recon)

        for i in range(self.n_iter):
            if self.input_size != 28:
                recon = F.upsample(recon, size=self.input_size, mode='bilinear')
            features = self.embedding(torch.cat([x, recon],1))
            dz = self.regressor(torch.cat([features, z], 1))
            z = torch.clamp(z + dz, 0, 1)
            recon = self.G(z)
            self.zs.append(z)
            self.recons.append(recon)

        self.zs = torch.stack(self.zs, 1)
        self.recons = torch.stack(self.recons, 1)

        return recon


class PFC1Clipping(nn.Module):
    '''
    Same as PFC1 except that we apply clippnig to dz so that the resulting z is always within the original valid range of GAN, which is the uniform distribution between 0 and 1.
    But, note that if we instead apply Tanh to dz, it is possible that z can go outside the original valid range.

    Input: raw canon occluded image x, the foreground object image x_f. Both (N, C, H, W).
    Output: a sequence of dzs. (N, n_iter, z_dim). the sum of dzs[:,i]'s up to time t will become the estimate of z at time t, z_t. 

    Note that this is not an RNN, as there is no hidden state passed between time steps.
    '''
    def __init__(self, G=None, n_iter=5, z_dim=62, input_size=28, input_channel=1, hidden1=256, hidden2=256, hidden3=128, hidden4=512, hidden5=512):
        super(PFC1Clipping, self).__init__()
        if G is None:
            self.G = generator()
        else:
            self.G = G
        self.n_iter = n_iter
        self.z_dim = z_dim
        self.input_size = input_size
        self.input_channel = input_channel
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.hidden5 = hidden5

        # To embedding, a channel-wise concatenation of x, x_f, G(z_t).
        self.embedding = nn.Sequential(
            nn.Conv2d(input_channel*3, hidden1, 4, 2, 1),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden1, hidden2, 4, 2, 1),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden2, hidden3, 3, 1, 1),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2),
            Flatten(),
            nn.Linear(hidden3*(input_size//4)**2, hidden4),
            nn.BatchNorm1d(hidden4),
            nn.LeakyReLU(0.2)
            )

        self.regressor = nn.Sequential(
            nn.Linear(hidden4+z_dim, hidden5),
            nn.BatchNorm1d(hidden5),
            nn.ReLU(),
            nn.Linear(hidden5, z_dim),
            nn.Tanh()
            )

        self.z0 = Parameter(torch.Tensor(z_dim))
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        self.init_parameters()

    def init_parameters(self):
        for name, param in self.embedding.named_parameters():
            if 'weight' in name and param.dim() > 1:
                init.kaiming_normal(param)
            elif 'weight' in name and param.dim() == 1:
                # This corresponds to batchnorm layers.
                init.constant(param, 1.0)
            elif 'bias' in name:
                init.constant(param, 0.0)

        for name, param in self.regressor.named_parameters():
            if 'weight' in name and param.dim() > 1:
                init.kaiming_normal(param)
            elif 'weight' in name and param.dim() == 1:
                # This corresponds to batchnorm layers.
                init.constant(param, 1.0)
            elif 'bias' in name:
                init.constant(param, 0.0)

        init.constant(self.z0, 0.0) # Note that we apply sigmoid to this in the forward method, so z0 is actually initialized to 0.5.


    def forward(self, x, x_f):
        self.zs = []
        self.recons = []
        N = x.size(0)
        z = self.sigmoid(self.z0.expand(N, self.z_dim))
        recon = self.G(z)
        self.zs.append(z)
        self.recons.append(recon)

        for i in range(self.n_iter):
            if self.input_size != 28:
                recon = F.upsample(recon, size=self.input_size, mode='bilinear')
            features = self.embedding(torch.cat([x, x_f, recon],1))
            dz = self.regressor(torch.cat([features, z], 1))
            z = torch.clamp(z + dz, 0, 1)
            recon = self.G(z)
            self.zs.append(z)
            self.recons.append(recon)

        self.zs = torch.stack(self.zs, 1)
        self.recons = torch.stack(self.recons, 1)

        return recon


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        if x.dim() == 4:
            N, C, H, W = x.size()
            x = x.view(N, -1)

        return x



class PFC1(nn.Module):
    '''
    Input: raw canon occluded image x, the foreground object image x_f. Both (N, C, H, W).
    Output: a sequence of dzs. (N, n_iter, z_dim). the sum of dzs[:,i]'s up to time t will become the estimate of z at time t, z_t. 

    Note that this is not an RNN, as there is no hidden state passed between time steps.
    '''
    def __init__(self, G=None, n_iter=5, z_dim=62, input_size=28, input_channel=1, hidden1=256, hidden2=256, hidden3=128, hidden4=512, hidden5=512):
        super(PFC1, self).__init__()
        if G is None:
            self.G = generator()
        else:
            self.G = G
        self.n_iter = n_iter
        self.z_dim = z_dim
        self.input_size = input_size
        self.input_channel = input_channel
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.hidden5 = hidden5

        # To embedding, a channel-wise concatenation of x, x_f, G(z_t).
        self.embedding = nn.Sequential(
            nn.Conv2d(input_channel*3, hidden1, 4, 2, 1),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden1, hidden2, 4, 2, 1),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden2, hidden3, 3, 1, 1),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2),
            Flatten(),
            nn.Linear(hidden3*(input_size//4)**2, hidden4),
            nn.BatchNorm1d(hidden4),
            nn.LeakyReLU(0.2)
            )

        self.regressor = nn.Sequential(
            nn.Linear(hidden4+z_dim, hidden5),
            nn.BatchNorm1d(hidden5),
            nn.ReLU(),
            nn.Linear(hidden5, z_dim)
            )

        self.z0 = Parameter(torch.Tensor(z_dim))
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        self.init_parameters()

    def init_parameters(self):
        for name, param in self.embedding.named_parameters():
            if 'weight' in name and param.dim() > 1:
                init.kaiming_normal(param)
            elif 'weight' in name and param.dim() == 1:
                # This corresponds to batchnorm layers.
                init.constant(param, 1.0)
            elif 'bias' in name:
                init.constant(param, 0.0)

        for name, param in self.regressor.named_parameters():
            if 'weight' in name and param.dim() > 1:
                init.kaiming_normal(param)
            elif 'weight' in name and param.dim() == 1:
                # This corresponds to batchnorm layers.
                init.constant(param, 1.0)
            elif 'bias' in name:
                init.constant(param, 0.0)

        init.constant(self.z0, 0.0) # Note that we apply sigmoid to this in the forward method, so z0 is actually initialized to 0.5.


    def forward(self, x, x_f):
        self.zs = []
        self.recons = []
        N = x.size(0)
        z = self.sigmoid(self.z0.expand(N, self.z_dim))
        recon = self.G(z)
        self.zs.append(z)
        self.recons.append(recon)

        for i in range(self.n_iter):
            features = self.embedding(torch.cat([x, x_f, recon],1))
            dz = self.regressor(torch.cat([features, z], 1))
            z = z + dz
            recon = self.G(z)
            self.zs.append(z)
            self.recons.append(recon)

        self.zs = torch.stack(self.zs, 1)
        self.recons = torch.stack(self.recons, 1)

        return recon


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        if x.dim() == 4:
            N, C, H, W = x.size()
            x = x.view(N, -1)

        return x
