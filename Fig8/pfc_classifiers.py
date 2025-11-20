import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math



class TDRCClassifier(nn.Module):
    '''
    The same architecture as TDRCTypeD
    '''
    def __init__(self, n_iter=5, input_size=28, input_channels=1, hidden1=64, hidden2=128, hidden3=128, hidden4=512, n_classes=4, record_activations=False, record_readouts=False):
        super(TDRCClassifier, self).__init__()
        self.n_iter = n_iter
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.n_classes = n_classes
        self.record_activations = record_activations
        self.record_readouts = record_readouts

        # We have additional input dimensions for the top-down feedbacks.
        self.convlstm1 = ConvLSTMCell(input_channels=input_channels*2, hidden_channels=hidden1, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.convlstm2 = ConvLSTMCell(input_channels=hidden1, hidden_channels=hidden2, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.convlstm3 = ConvLSTMCell(input_channels=hidden2, hidden_channels=hidden3, kernel_size=5)
        self.pool3 = nn.MaxPool2d(2,2)
        self.lstm = nn.LSTMCell(int(input_size/8)**2*hidden3, hidden4)
        self.fc = nn.Linear(hidden4, n_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        '''
        TD layers are labeled by the origin of the top-down feedback.
        '''
        self.td_convlstm31 = nn.Conv2d(in_channels=hidden3, out_channels=input_channels, kernel_size=1, stride=1, padding=0)

        self.convlstm1_h0 = Parameter(torch.Tensor(hidden1, input_size, input_size))
        self.convlstm1_c0 = Parameter(torch.Tensor(hidden1, input_size, input_size))

        self.convlstm2_h0 = Parameter(torch.Tensor(hidden2, int(input_size/2), int(input_size/2)))
        self.convlstm2_c0 = Parameter(torch.Tensor(hidden2, int(input_size/2), int(input_size/2)))

        self.convlstm3_h0 = Parameter(torch.Tensor(hidden3, int(input_size/4), int(input_size/4)))
        self.convlstm3_c0 = Parameter(torch.Tensor(hidden3, int(input_size/4), int(input_size/4)))

        self.lstm_h0 = Parameter(torch.Tensor(hidden4))
        self.lstm_c0 = Parameter(torch.Tensor(hidden4))

        self.init_parameters()

    def init_parameters(self):
        self.convlstm1_h0.data.fill_(0.0)
        self.convlstm1_c0.data.fill_(0.0)

        self.convlstm2_h0.data.fill_(0.0)
        self.convlstm2_c0.data.fill_(0.0)

        self.convlstm3_h0.data.fill_(0.0)
        self.convlstm3_c0.data.fill_(0.0)

        self.lstm_h0.data.fill_(0.0)
        self.lstm_c0.data.fill_(0.0)

        init.normal(self.convlstm1.conv.weight[:,:self.input_channels], 0.0, 1.0/math.sqrt(self.input_channels*(5**2))) # 5**2 is (kernel_size)**2.
        init.normal(self.convlstm1.conv.weight[:,self.input_channels:], 0.0, 1.0/math.sqrt(self.hidden1*(5**2))) # 5**2 is (kernel_size)**2.
        init.constant(self.convlstm1.conv.bias, 0.0)
        # We initialize the forget gate bias to 1.0.
        init.constant(self.convlstm1.conv.bias[self.hidden1:2*self.hidden1], 1.0)

        init.normal(self.convlstm2.conv.weight[:,:self.hidden1], 0.0, 1.0/math.sqrt(self.hidden1*(5**2))) # 5**2 is (kernel_size)**2.
        init.normal(self.convlstm2.conv.weight[:,self.hidden1:], 0.0, 1.0/math.sqrt(self.hidden2*(5**2))) # 5**2 is (kernel_size)**2.
        init.constant(self.convlstm2.conv.bias, 0.0)
        # We initialize the forget gate bias to 1.0.
        init.constant(self.convlstm2.conv.bias[self.hidden2:2*self.hidden2], 1.0)

        init.normal(self.convlstm3.conv.weight[:,:self.hidden2], 0.0, 1.0/math.sqrt(self.hidden2*(5**2))) # 5**2 is (kernel_size)**2.
        init.normal(self.convlstm3.conv.weight[:,self.hidden2:], 0.0, 1.0/math.sqrt(self.hidden3*(5**2))) # 5**2 is (kernel_size)**2.
        init.constant(self.convlstm3.conv.bias, 0.0)
        # We initialize the forget gate bias to 1.0.
        init.constant(self.convlstm3.conv.bias[self.hidden3:2*self.hidden3], 1.0)

        init.normal(self.lstm.weight_ih, 0.0, 1.0/math.sqrt(int(self.input_size/8)**2*self.hidden3))
        init.normal(self.lstm.weight_hh, 0.0, 1.0/math.sqrt(self.hidden4))
        init.constant(self.lstm.bias_ih, 0.0)
        # We initialize the forget gate bias to 1.0. Note that we need to modify only one of bias_ih or bias_hh.
        init.constant(self.lstm.bias_ih[self.hidden4:2*self.hidden4], 1.0)
        init.constant(self.lstm.bias_hh, 0.0)

        init.normal(self.fc.weight, 0.0, 1.0/math.sqrt(self.hidden4))
        init.constant(self.fc.bias, 0.0)

        init.normal(self.td_convlstm31.weight, 1.0/math.sqrt(self.hidden3))
        init.constant(self.td_convlstm31.bias, 0.0)

    def cell(self, predict, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0):
        # TD feedbacks.
        upsample1 = F.upsample(convlstm3_hidden0[0], scale_factor=2, mode='nearest')
        if self.input_size == 50:
            upsample1 = F.pad(upsample1, (0,1,0,1), mode='constant', value=0)
        td_convlstm31 = self.td_convlstm31(F.upsample(upsample1, scale_factor=2, mode='nearest'))

        convlstm1_h1, convlstm1_c1 = self.convlstm1(torch.cat([x, td_convlstm31], 1), convlstm1_hidden0) # input x: C=input_channels, H,W = input_size = 50.
        h = self.relu(self.pool1(convlstm1_h1))
        if self.record_activations:
            self.activation_hidden1 = h.detach()
        convlstm2_h1, convlstm2_c1 = self.convlstm2(h, convlstm2_hidden0) # input h: C=hidden1, H,W = int(input_size/2) = 25.
        h = self.relu(self.pool2(convlstm2_h1))
        if self.record_activations:
            self.activation_hidden2 = h.detach()
        convlstm3_h1, convlstm3_c1 = self.convlstm3(h, convlstm3_hidden0) # input h: C=hidden2, H,W = int(input_size/4) = 12.
        h = self.relu(self.pool3(convlstm3_h1))
        if self.record_activations:
            self.activation_hidden3 = h.detach()
        h = h.view(-1, int(self.input_size/8)**2*self.hidden3)
        h = self.dropout(h)
        lstm_h1, lstm_c1 = self.lstm(h, lstm_hidden0)
        if self.record_activations:
            self.activation_hidden4 = lstm_h1.detach()
        if predict:
            h = self.dropout(lstm_h1)
            scores = self.fc(h)
            if self.record_readouts:
                self.readout = scores.detach()
            return scores, (convlstm1_h1, convlstm1_c1), (convlstm2_h1, convlstm2_c1), (convlstm3_h1, convlstm3_c1), (lstm_h1, lstm_c1)
        else:
            if self.record_readouts:
                h = self.dropout(lstm_h1)
                scores = self.fc(h)
                self.readout = scores.detach()
            return (convlstm1_h1, convlstm1_c1), (convlstm2_h1, convlstm2_c1), (convlstm3_h1, convlstm3_c1), (lstm_h1, lstm_c1)

    def forward(self, x):
        if self.record_activations:
            self.activations_hidden1 = []
            self.activations_hidden2 = []
            self.activations_hidden3 = []
            self.activations_hidden4 = []

        if self.record_readouts:
            self.readouts = []

        N = x.size(0)
        convlstm1_hidden0 = (self.convlstm1_h0.expand(N, -1, -1, -1), self.convlstm1_c0.expand(N, -1, -1, -1))
        convlstm2_hidden0 = (self.convlstm2_h0.expand(N, -1, -1, -1), self.convlstm2_c0.expand(N, -1, -1, -1))
        convlstm3_hidden0 = (self.convlstm3_h0.expand(N, -1, -1, -1), self.convlstm3_c0.expand(N, -1, -1, -1))
        lstm_hidden0 = (self.lstm_h0.expand(N, -1), self.lstm_c0.expand(N, -1))

        self.all_scores = []
        for i in range(self.n_iter):
            scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)
            self.all_scores.append(scores)
            if self.record_activations:
                self.activations_hidden1.append(self.activation_hidden1)
                self.activations_hidden2.append(self.activation_hidden2)
                self.activations_hidden3.append(self.activation_hidden3)
                self.activations_hidden4.append(self.activation_hidden4)

            if self.record_readouts:
                self.readouts.append(self.readout)



        return scores


class RCClassifier(nn.Module):
    '''
    We only make the very first convolutional layer a ConvLSTM layer. I think even this could outperform FD and PFC + FD models.
    If this doesn't outperform them, then I can consider making all layers recurrent.
    n_iter is set to 5 to make a fair comparision with PFC models.

    Input: Occluded image. (N, C, H, W)
    '''
    def __init__(self, n_iter=5, input_size=28, input_channels=1, hidden1=64, hidden2=128, hidden3=128, hidden4=512, n_classes=4, record_activations=False, record_readouts=False):
        super(RCClassifier, self).__init__()
        self.n_iter = n_iter
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.n_classes = n_classes
        self.record_activations = record_activations
        self.record_readouts = record_readouts

        self.convlstm1 = ConvLSTMCell(input_channels=input_channels, hidden_channels=hidden1, kernel_size=5)
        self.convlstm1_h0 = Parameter(torch.Tensor(hidden1, input_size, input_size))
        self.convlstm1_c0 = Parameter(torch.Tensor(hidden1, input_size, input_size))


        self.convlstm2 = ConvLSTMCell(input_channels=hidden1, hidden_channels=hidden2, kernel_size=5)
        self.convlstm2_h0 = Parameter(torch.Tensor(hidden2, input_size//2, input_size//2))
        self.convlstm2_c0 = Parameter(torch.Tensor(hidden2, input_size//2, input_size//2))


        self.convlstm3 = ConvLSTMCell(input_channels=hidden2, hidden_channels=hidden3, kernel_size=5)
        self.convlstm3_h0 = Parameter(torch.Tensor(hidden3, input_size//4, input_size//4))
        self.convlstm3_c0 = Parameter(torch.Tensor(hidden3, input_size//4, input_size//4))

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
         
        self.lstm = nn.LSTMCell(hidden3*(input_size//8)**2, hidden4)
        self.lstm_h0 = Parameter(torch.Tensor(hidden4))
        self.lstm_c0 = Parameter(torch.Tensor(hidden4))

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden4, n_classes)

        self.init_parameters()


    def init_parameters(self):
        self.convlstm1_h0.data.fill_(0.0)
        self.convlstm1_c0.data.fill_(0.0)
        
        self.convlstm2_h0.data.fill_(0.0)
        self.convlstm2_c0.data.fill_(0.0)
        
        self.convlstm3_h0.data.fill_(0.0)
        self.convlstm3_c0.data.fill_(0.0)

        self.lstm_h0.data.fill_(0.0)
        self.lstm_c0.data.fill_(0.0)

        init.normal(self.convlstm1.conv.weight[:,:self.input_channels], 0.0, 1.0/math.sqrt(self.input_channels*(5**2))) # 5**2 is (kernel_size)**2.
        init.normal(self.convlstm1.conv.weight[:,self.input_channels:], 0.0, 1.0/math.sqrt(self.hidden1*(5**2))) # 5**2 is (kernel_size)**2.
        init.constant(self.convlstm1.conv.bias, 0.0)
        # We initialize the forget gate bias to 1.0.
        init.constant(self.convlstm1.conv.bias[self.hidden1:2*self.hidden1], 1.0)

        init.normal(self.convlstm2.conv.weight[:,:self.hidden1], 0.0, 1.0/math.sqrt(self.hidden1*(5**2))) # 5**2 is (kernel_size)**2.
        init.normal(self.convlstm2.conv.weight[:,self.hidden1:], 0.0, 1.0/math.sqrt(self.hidden2*(5**2))) # 5**2 is (kernel_size)**2.
        init.constant(self.convlstm2.conv.bias, 0.0)
        # We initialize the forget gate bias to 1.0.
        init.constant(self.convlstm2.conv.bias[self.hidden2:2*self.hidden2], 1.0)

        init.normal(self.convlstm3.conv.weight[:,:self.hidden2], 0.0, 1.0/math.sqrt(self.hidden2*(5**2))) # 5**2 is (kernel_size)**2.
        init.normal(self.convlstm3.conv.weight[:,self.hidden2:], 0.0, 1.0/math.sqrt(self.hidden3*(5**2))) # 5**2 is (kernel_size)**2.
        init.constant(self.convlstm3.conv.bias, 0.0)
        # We initialize the forget gate bias to 1.0.
        init.constant(self.convlstm3.conv.bias[self.hidden3:2*self.hidden3], 1.0)

        init.normal(self.lstm.weight_ih, 0.0, 1.0/math.sqrt((self.input_size//8)**2*self.hidden3))
        init.normal(self.lstm.weight_hh, 0.0, 1.0/math.sqrt(self.hidden4))
        init.constant(self.lstm.bias_ih, 0.0)
        # We initialize the forget gate bias to 1.0. Note that we need to modify only one of bias_ih or bias_hh.
        init.constant(self.lstm.bias_ih[self.hidden4:2*self.hidden4], 1.0)
        init.constant(self.lstm.bias_hh, 0.0)

        init.normal(self.fc.weight)
        init.constant(self.fc.bias, 0)


    def cell(self, predict, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0):
        # When hc0, cc0 are the initial states of ROOM, we need to take into account that they have
        # shapes (hidden_size,), not (batch_size, hidden_size). So, we just copy them along the batch dimension.
        convlstm1_h1, convlstm1_c1 = self.convlstm1(x, convlstm1_hidden0)
        h = self.relu(self.pool(convlstm1_h1))
        if self.record_activations:
            self.activation_hidden1 = h.detach()
        convlstm2_h1, convlstm2_c1 = self.convlstm2(h, convlstm2_hidden0)
        h = self.relu(self.pool(convlstm2_h1))
        if self.record_activations:
            self.activation_hidden2 = h.detach()
        convlstm3_h1, convlstm3_c1 = self.convlstm3(h, convlstm3_hidden0)
        h = self.relu(self.pool(convlstm3_h1))
        if self.record_activations:
            self.activation_hidden3 = h.detach()
        h = h.view(-1, int(self.input_size/8)**2*self.hidden3)
        h = self.dropout(h)
        lstm_h1, lstm_c1 = self.lstm(h, lstm_hidden0)
        if self.record_activations:
            self.activation_hidden4 = lstm_h1.detach()
        if predict:
            h = self.dropout(lstm_h1)
            scores = self.fc(h)
            if self.record_readouts:
                self.readout = scores.detach()
            return scores, (convlstm1_h1, convlstm1_c1), (convlstm2_h1, convlstm2_c1), (convlstm3_h1, convlstm3_c1), (lstm_h1, lstm_c1)
        else:
            if self.record_readouts:
                h = self.dropout(lstm_h1)
                scores = self.fc(h)
                self.readout = scores.detach()                
            return (convlstm1_h1, convlstm1_c1), (convlstm2_h1, convlstm2_c1), (convlstm3_h1, convlstm3_c1), (lstm_h1, lstm_c1)


    def forward(self, x):
        if self.record_activations:
            self.activations_hidden1 = []
            self.activations_hidden2 = []
            self.activations_hidden3 = []
            self.activations_hidden4 = []

        if self.record_readouts:
            self.readouts = []

        N = x.size(0)
        convlstm1_hidden0 = (self.convlstm1_h0.expand(N, -1, -1, -1), self.convlstm1_c0.expand(N, -1, -1, -1))
        convlstm2_hidden0 = (self.convlstm2_h0.expand(N, -1, -1, -1), self.convlstm2_c0.expand(N, -1, -1, -1))
        convlstm3_hidden0 = (self.convlstm3_h0.expand(N, -1, -1, -1), self.convlstm3_c0.expand(N, -1, -1, -1))
        lstm_hidden0 = (self.lstm_h0.expand(N, -1), self.lstm_c0.expand(N, -1))

        self.all_scores = []
        for i in range(self.n_iter):
            scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0) 
            self.all_scores.append(scores)
            if self.record_activations:
                self.activations_hidden1.append(self.activation_hidden1)
                self.activations_hidden2.append(self.activation_hidden2)
                self.activations_hidden3.append(self.activation_hidden3)
                self.activations_hidden4.append(self.activation_hidden4)

            if self.record_readouts:
                self.readouts.append(self.readout)

        return scores



class RCClassifierOld2(nn.Module):
    '''
    We only make the very first convolutional layer a ConvLSTM layer. I think even this could outperform FD and PFC + FD models.
    If this doesn't outperform them, then I can consider making all layers recurrent.
    n_iter is set to 5 to make a fair comparision with PFC models.

    Input: Occluded image. (N, C, H, W)
    '''
    def __init__(self, n_iter=5, input_size=28, input_channels=1, hidden1=64, hidden2=128, hidden3=128, hidden4=512, n_classes=4):
        super(RCClassifierOld2, self).__init__()
        self.n_iter = n_iter
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.n_classes = n_classes

        self.fd_layers1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden1, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(hidden1, hidden2, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(hidden2, hidden3, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            Flatten(),
            nn.Dropout(p=0.5),
            )
         
        self.lstm = nn.LSTMCell(hidden3*(input_size//8)**2, hidden4)
        self.lstm_h0 = Parameter(torch.Tensor(hidden4))
        self.lstm_c0 = Parameter(torch.Tensor(hidden4))

        self.fd_layers2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden4, n_classes)
            )

        self.init_parameters()

    def init_parameters(self):
        for name, param in self.fd_layers1.named_parameters():
            if 'weight' in name:
                init.kaiming_normal(param)
            elif 'bias' in name:
                init.constant(param, 0)                


        self.lstm_h0.data.fill_(0.0)
        self.lstm_c0.data.fill_(0.0)

        init.normal(self.lstm.weight_ih, 0.0, 1.0/math.sqrt((self.input_size//8)**2*self.hidden3))
        init.normal(self.lstm.weight_hh, 0.0, 1.0/math.sqrt(self.hidden4))
        init.constant(self.lstm.bias_ih, 0.0)
        # We initialize the forget gate bias to 1.0. Note that we need to modify only one of bias_ih or bias_hh.
        init.constant(self.lstm.bias_ih[self.hidden4:2*self.hidden4], 1.0)
        init.constant(self.lstm.bias_hh, 0.0)


        for name, param in self.fd_layers2.named_parameters():
            if 'weight' in name:
                init.kaiming_normal(param)
            elif 'bias' in name:
                init.constant(param, 0)                



    def forward(self, x):
        N = x.size(0)
        h0, c0 = self.lstm_h0.expand(N, -1), self.lstm_c0.expand(N, -1)
        self.all_scores = []
        for i in range(self.n_iter):
            h = self.fd_layers1(x)
            h0, c0 = self.lstm(h, (h0, c0))
            scores = self.fd_layers2(h0)
            self.all_scores.append(scores)

        return scores


class RCClassifierOld(nn.Module):
    '''
    We only make the very first convolutional layer a ConvLSTM layer. I think even this could outperform FD and PFC + FD models.
    If this doesn't outperform them, then I can consider making all layers recurrent.
    n_iter is set to 5 to make a fair comparision with PFC models.

    Input: Occluded image. (N, C, H, W)
    '''
    def __init__(self, n_iter=5, input_size=28, input_channels=1, hidden1=64, hidden2=128, hidden3=128, hidden4=512, n_classes=4):
        super(RCClassifierOld, self).__init__()
        self.n_iter = n_iter
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.n_classes = n_classes

        self.convlstm1 = ConvLSTMCell(input_channels=input_channels, hidden_channels=hidden1, kernel_size=5)
        self.convlstm1_h0 = Parameter(torch.Tensor(hidden1, input_size, input_size))
        self.convlstm1_c0 = Parameter(torch.Tensor(hidden1, input_size, input_size))
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fd_layers = nn.Sequential(
            nn.Conv2d(hidden1, hidden2, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(hidden2, hidden3, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden3*(input_size//8)**2, hidden4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden4, n_classes)
            )

        self.init_parameters()

    def init_parameters(self):
        self.convlstm1_h0.data.fill_(0.0)
        self.convlstm1_c0.data.fill_(0.0)

        init.normal(self.convlstm1.conv.weight[:,:self.input_channels], 0.0, 1.0/math.sqrt(self.input_channels*(5**2))) # 5**2 is (kernel_size)**2.
        init.normal(self.convlstm1.conv.weight[:,self.input_channels:], 0.0, 1.0/math.sqrt(self.hidden1*(5**2))) # 5**2 is (kernel_size)**2.
        init.constant(self.convlstm1.conv.bias, 0.0)
        # We initialize the forget gate bias to 1.0.
        init.constant(self.convlstm1.conv.bias[self.hidden1:2*self.hidden1], 1.0)

        for name, param in self.fd_layers.named_parameters():
            if 'weight' in name:
                init.kaiming_normal(param)
            elif 'bias' in name:
                init.constant(param, 0)                

    def forward(self, x):
        N = x.size(0)
        h0, c0 = self.convlstm1_h0.expand(N, -1, -1, -1), self.convlstm1_c0.expand(N, -1, -1, -1)
        for i in range(self.n_iter):
            h0, c0 = self.convlstm1(x, (h0, c0))
            h = self.relu(self.pool(h0))
            scores = self.fd_layers(h)

        return scores


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size-1) // 2
        # bk: self.input_channels + self.hidden_channels in the first argument because we assume that the spatial size of the input and hidden
        # is identical.
        self.conv = nn.Conv2d(in_channels=self.input_channels+self.hidden_channels, out_channels=4*self.hidden_channels, 
            kernel_size=self.kernel_size, stride=1, padding=self.padding)

    def forward(self, input, hidden):
        h, c = hidden
        combined = torch.cat((input, h), dim=1)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.chunk(A, 4, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c


class FDClassifier(nn.Module):
    '''
    The only difference from ClassifierA is that we removed batch norm layers from it, to compare with RCClassifier.
    Note that I accordingly removed the part in init_parameters that initialize batch norm layers.

    Input: Occluded image. (N, C, H, W)
    '''
    def __init__(self, input_size=28, input_channels=1, hidden1=64, hidden2=128, hidden3=128, hidden4=512, n_classes=4):
        super(FDClassifier, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.n_classes = n_classes

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, hidden1, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(hidden1, hidden2, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(hidden2, hidden3, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden3*(input_size//8)**2, hidden4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden4, n_classes)
            )

        self.init_parameters()

    def init_parameters(self):
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                init.kaiming_normal(param)
            elif 'bias' in name:
                init.constant(param, 0)                

    def forward(self, x):
        return self.model(x)



class ClassifierA(nn.Module):
    '''
    Input: Occluded image. (N, C, H, W)
    '''
    def __init__(self, input_size=28, input_channels=1, hidden1=64, hidden2=128, hidden3=128, hidden4=512, n_classes=4):
        super(ClassifierA, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.n_classes = n_classes

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, hidden1, 5, 1, 2),
            nn.BatchNorm2d(hidden1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(hidden1, hidden2, 5, 1, 2),
            nn.BatchNorm2d(hidden2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(hidden2, hidden3, 5, 1, 2),
            nn.BatchNorm2d(hidden3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden3*(input_size//8)**2, hidden4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden4, n_classes)
            )

        self.init_parameters()

    def init_parameters(self):
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if param.dim() == 1: # BN
                    init.constant(param, 1)
                else: # Conv or Linear
                    init.kaiming_normal(param)
            elif 'bias' in name:
                init.constant(param, 0)                

    def forward(self, x):
        return self.model(x)




class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        if x.dim() == 4:
            N, C, H, W = x.size()
            x = x.view(N, -1)

        return x
