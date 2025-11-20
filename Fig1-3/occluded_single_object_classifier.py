import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
import itertools
from copy import deepcopy

class FDOccludedSingleObjectClassifier(nn.Module):
    def __init__(self, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden4=1024, n_classes=10, dropout_p=0.5):
        super(FDOccludedSingleObjectClassifier, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.n_classes = n_classes
        self.dropout_p = dropout_p
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden1, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=hidden1, out_channels=hidden2, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(in_channels=hidden2, out_channels=hidden3, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(int(input_size/8)**2*hidden3, hidden4)
        self.fc2 = nn.Linear(hidden4, n_classes)
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

        self.init_parameters()

    def init_parameters(self):
        init.kaiming_normal(self.conv1.weight)
        init.constant(self.conv1.bias, 0.0)

        init.kaiming_normal(self.conv2.weight)
        init.constant(self.conv2.bias, 0.0)

        init.kaiming_normal(self.conv3.weight)
        init.constant(self.conv3.bias, 0.0)

        init.kaiming_normal(self.fc1.weight)
        init.constant(self.fc1.bias, 0.0)

        init.normal(self.fc2.weight, 0.0, 1.0/math.sqrt(self.hidden4))
        init.constant(self.fc2.bias, 0.0)
    
    def forward(self, x):
        h = self.relu(self.pool1(self.conv1(x)))
        h = self.relu(self.pool2(self.conv2(h)))
        h = self.relu(self.pool3(self.conv3(h))).view(-1, int(self.input_size/8)**2*self.hidden3)
        h = self.dropout(h)
        h = self.relu(self.fc1(h))
        h = self.dropout(h)
        scores = self.fc2(h)

        return scores


class FDTallerOccludedSingleObjectClassifier(nn.Module):
    def __init__(self, input_size=50, n_layers=12, input_channels=1, hidden=64, hidden_fc=1024, n_classes=10, dropout_p=0.5):
        super(FDTallerOccludedSingleObjectClassifier, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden = hidden
        self.hidden_fc = hidden_fc

        self.n_classes = n_classes
        self.dropout_p = dropout_p
        
        self.n_layers = n_layers

        for i in range(n_layers):
            if i == 0:
                setattr(self, 'conv{}'.format(i+1), nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden, kernel_size=5, stride=1, padding=2))
            else:
                setattr(self, 'conv{}'.format(i+1), nn.Conv2d(in_channels=self.hidden, out_channels=self.hidden, kernel_size=5, stride=1, padding=2))

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(int(input_size/8)**2*hidden, hidden_fc)
        self.fc2 = nn.Linear(hidden_fc, n_classes)
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

        self.init_parameters()

    def init_parameters(self):
        for i in range(self.n_layers):
            init.kaiming_normal(getattr(self, 'conv{}'.format(i+1)).weight)
            init.constant(getattr(self, 'conv{}'.format(i+1)).bias, 0.0)

        init.kaiming_normal(self.fc1.weight)
        init.constant(self.fc1.bias, 0.0)

        init.normal(self.fc2.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.fc2.bias, 0.0)
    
    def forward(self, x):
        for i in range(self.n_layers):
            x = self.relu(getattr(self, 'conv{}'.format(i+1))(x))
            if (i+1) % (self.n_layers//3) == 0:
                x = self.pool(x)

        x = self.dropout(x.view(-1, int(self.input_size/8)**2*self.hidden))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class WeightSharedFDTallerOccludedSingleObjectClassifier(nn.Module):
    def __init__(self, input_size=50, n_layers=12, input_channels=1, hidden=64, hidden_fc=1024, n_classes=10, dropout_p=0.5):
        super(WeightSharedFDTallerOccludedSingleObjectClassifier, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden = hidden
        self.hidden_fc = hidden_fc

        self.n_classes = n_classes
        self.dropout_p = dropout_p
        
        self.n_layers = n_layers

        for i in range(n_layers):
            if (i+1) == 1:
                setattr(self, 'conv{}'.format(i+1), nn.Conv2d(in_channels=input_channels, out_channels=hidden, kernel_size=5, stride=1, padding=2))
            elif (i+1) % (n_layers//3) == 0:
                setattr(self, 'conv{}'.format(i+1), nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=5, stride=1, padding=2))                

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(int(input_size/8)**2*hidden, hidden_fc)
        self.fc2 = nn.Linear(hidden_fc, n_classes)
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

        self.init_parameters()

    def init_parameters(self):
        for i in range(self.n_layers):
            if (i+1) == 1 or (i+1) % (self.n_layers//3) == 0:
                init.kaiming_normal(getattr(self, 'conv{}'.format(i+1)).weight)
                init.constant(getattr(self, 'conv{}'.format(i+1)).bias, 0.0)

        init.kaiming_normal(self.fc1.weight)
        init.constant(self.fc1.bias, 0.0)

        init.normal(self.fc2.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.fc2.bias, 0.0)
    
    def forward(self, x):
        for i in range(self.n_layers):
            if (i+1) == 1:
                x = self.relu(getattr(self, 'conv{}'.format(i+1))(x))
            elif (i+1) % (self.n_layers//3) == 0:
                x = self.relu(getattr(self, 'conv{}'.format(i+1))(x))
                x = self.pool(x)
            else:
                k = ((i+1)//(self.n_layers//3) + 1)*(self.n_layers//3)
                x = self.relu(getattr(self, 'conv{}'.format(k))(x))

        x = self.dropout(x.view(-1, int(self.input_size/8)**2*self.hidden))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)



class TDRCOccludedSingleObjectClassifierC(nn.Module):
    def __init__(self, reverse=False, n_iter=2, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden_lstm=512, n_classes=10, dropout_p=0.5):
        super(TDRCOccludedSingleObjectClassifierC, self).__init__()
        self.reverse = reverse
        self.n_iter = n_iter
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden_lstm = hidden_lstm
        self.n_classes = n_classes
        self.dropout_p = dropout_p

        # We have additional input dimensions for the top-down feedbacks.
        self.convlstm1 = ConvLSTMCell(input_channels=input_channels*2, hidden_channels=hidden1, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.convlstm2 = ConvLSTMCell(input_channels=hidden1, hidden_channels=hidden2, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.convlstm3 = ConvLSTMCell(input_channels=hidden2, hidden_channels=hidden3, kernel_size=5)
        self.pool3 = nn.MaxPool2d(2,2)
        self.lstm = nn.LSTMCell(int(input_size/8)**2*hidden3, hidden_lstm)
        self.fc = nn.Linear(hidden_lstm, n_classes)
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

        '''
        TD layers are labeled by the origin of the top-down feedback.
        '''
        self.td_convlstm2 = nn.Conv2d(in_channels=hidden2, out_channels=input_channels, kernel_size=1, stride=1, padding=0)

        self.convlstm1_h0 = Parameter(torch.Tensor(hidden1, input_size, input_size))
        self.convlstm1_c0 = Parameter(torch.Tensor(hidden1, input_size, input_size))

        self.convlstm2_h0 = Parameter(torch.Tensor(hidden2, int(input_size/2), int(input_size/2)))
        self.convlstm2_c0 = Parameter(torch.Tensor(hidden2, int(input_size/2), int(input_size/2)))

        self.convlstm3_h0 = Parameter(torch.Tensor(hidden3, int(input_size/4), int(input_size/4)))
        self.convlstm3_c0 = Parameter(torch.Tensor(hidden3, int(input_size/4), int(input_size/4)))

        self.lstm_h0 = Parameter(torch.Tensor(hidden_lstm))
        self.lstm_c0 = Parameter(torch.Tensor(hidden_lstm))

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
        init.normal(self.lstm.weight_hh, 0.0, 1.0/math.sqrt(self.hidden_lstm))
        init.constant(self.lstm.bias_ih, 0.0)
        # We initialize the forget gate bias to 1.0. Note that we need to modify only one of bias_ih or bias_hh.
        init.constant(self.lstm.bias_ih[self.hidden_lstm:2*self.hidden_lstm], 1.0)
        init.constant(self.lstm.bias_hh, 0.0)

        init.normal(self.fc.weight, 0.0, 1.0/math.sqrt(self.hidden_lstm))
        init.constant(self.fc.bias, 0.0)

        init.normal(self.td_convlstm2.weight, 1.0/math.sqrt(self.hidden2))
        init.constant(self.td_convlstm2.bias, 0.0)


    def cell(self, predict, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0):
        # TD feedbacks.
        td_convlstm2 = self.td_convlstm2(F.upsample(convlstm2_hidden0[0], scale_factor=2, mode='nearest')) # The linear size goes from 25 to 50.

        convlstm1_h1, convlstm1_c1 = self.convlstm1(torch.cat([x, td_convlstm2], 1), convlstm1_hidden0) # input x: C=input_channels, H,W = input_size = 50.
        h = self.relu(self.pool1(convlstm1_h1))
        convlstm2_h1, convlstm2_c1 = self.convlstm2(h, convlstm2_hidden0) # input h: C=hidden1, H,W = int(input_size/2) = 25.
        h = self.relu(self.pool2(convlstm2_h1))
        convlstm3_h1, convlstm3_c1 = self.convlstm3(h, convlstm3_hidden0) # input h: C=hidden2, H,W = int(input_size/4) = 12.
        h = self.relu(self.pool3(convlstm3_h1)).view(-1, int(self.input_size/8)**2*self.hidden3)
        h = self.dropout(h)

        lstm_h1, lstm_c1 = self.lstm(h, lstm_hidden0)
        if predict:
            h = self.dropout(lstm_h1)
            scores = self.fc(h)
            return scores, (convlstm1_h1, convlstm1_c1), (convlstm2_h1, convlstm2_c1), (convlstm3_h1, convlstm3_c1), (lstm_h1, lstm_c1)
        else:
            return (convlstm1_h1, convlstm1_c1), (convlstm2_h1, convlstm2_c1), (convlstm3_h1, convlstm3_c1), (lstm_h1, lstm_c1)

    def forward(self, x):
        N = x.size(0)
        convlstm1_hidden0 = (self.convlstm1_h0.expand(N, -1, -1, -1), self.convlstm1_c0.expand(N, -1, -1, -1))
        convlstm2_hidden0 = (self.convlstm2_h0.expand(N, -1, -1, -1), self.convlstm2_c0.expand(N, -1, -1, -1))
        convlstm3_hidden0 = (self.convlstm3_h0.expand(N, -1, -1, -1), self.convlstm3_c0.expand(N, -1, -1, -1))
        lstm_hidden0 = (self.lstm_h0.expand(N, -1), self.lstm_c0.expand(N, -1))

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)
            else:
                convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(False, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)


        return scores



class TDRCOccludedSingleObjectClassifierD(nn.Module):
    def __init__(self, reverse=False, n_iter=2, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden_lstm=512, n_classes=10, dropout_p=0.5):
        super(TDRCOccludedSingleObjectClassifierD, self).__init__()
        self.reverse = reverse
        self.n_iter = n_iter
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden_lstm = hidden_lstm
        self.n_classes = n_classes
        self.dropout_p = dropout_p

        # We have additional input dimensions for the top-down feedbacks.
        self.convlstm1 = ConvLSTMCell(input_channels=input_channels*2, hidden_channels=hidden1, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.convlstm2 = ConvLSTMCell(input_channels=hidden1, hidden_channels=hidden2, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.convlstm3 = ConvLSTMCell(input_channels=hidden2, hidden_channels=hidden3, kernel_size=5)
        self.pool3 = nn.MaxPool2d(2,2)
        self.lstm = nn.LSTMCell(int(input_size/8)**2*hidden3, hidden_lstm)
        self.fc = nn.Linear(hidden_lstm, n_classes)
        self.dropout = nn.Dropout(p=dropout_p)
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

        self.lstm_h0 = Parameter(torch.Tensor(hidden_lstm))
        self.lstm_c0 = Parameter(torch.Tensor(hidden_lstm))

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
        init.normal(self.lstm.weight_hh, 0.0, 1.0/math.sqrt(self.hidden_lstm))
        init.constant(self.lstm.bias_ih, 0.0)
        # We initialize the forget gate bias to 1.0. Note that we need to modify only one of bias_ih or bias_hh.
        init.constant(self.lstm.bias_ih[self.hidden_lstm:2*self.hidden_lstm], 1.0)
        init.constant(self.lstm.bias_hh, 0.0)

        init.normal(self.fc.weight, 0.0, 1.0/math.sqrt(self.hidden_lstm))
        init.constant(self.fc.bias, 0.0)

        init.normal(self.td_convlstm31.weight, 1.0/math.sqrt(self.hidden3))
        init.constant(self.td_convlstm31.bias, 0.0)

    def cell(self, predict, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0):
        # TD feedbacks.
        upsample1 = F.upsample(convlstm3_hidden0[0], scale_factor=2, mode='nearest')
        upsample1 = F.pad(upsample1, (0,1,0,1), mode='constant', value=0)
        td_convlstm31 = self.td_convlstm31(F.upsample(upsample1, scale_factor=2, mode='nearest'))

        convlstm1_h1, convlstm1_c1 = self.convlstm1(torch.cat([x, td_convlstm31], 1), convlstm1_hidden0) # input x: C=input_channels, H,W = input_size = 50.
        h = self.relu(self.pool1(convlstm1_h1))
        convlstm2_h1, convlstm2_c1 = self.convlstm2(h, convlstm2_hidden0) # input h: C=hidden1, H,W = int(input_size/2) = 25.
        h = self.relu(self.pool2(convlstm2_h1))
        convlstm3_h1, convlstm3_c1 = self.convlstm3(h, convlstm3_hidden0) # input h: C=hidden2, H,W = int(input_size/4) = 12.
        h = self.relu(self.pool3(convlstm3_h1)).view(-1, int(self.input_size/8)**2*self.hidden3)
        h = self.dropout(h)

        lstm_h1, lstm_c1 = self.lstm(h, lstm_hidden0)
        if predict:
            h = self.dropout(lstm_h1)
            scores = self.fc(h)
            return scores, (convlstm1_h1, convlstm1_c1), (convlstm2_h1, convlstm2_c1), (convlstm3_h1, convlstm3_c1), (lstm_h1, lstm_c1)
        else:
            return (convlstm1_h1, convlstm1_c1), (convlstm2_h1, convlstm2_c1), (convlstm3_h1, convlstm3_c1), (lstm_h1, lstm_c1)

    def forward(self, x):
        N = x.size(0)
        convlstm1_hidden0 = (self.convlstm1_h0.expand(N, -1, -1, -1), self.convlstm1_c0.expand(N, -1, -1, -1))
        convlstm2_hidden0 = (self.convlstm2_h0.expand(N, -1, -1, -1), self.convlstm2_c0.expand(N, -1, -1, -1))
        convlstm3_hidden0 = (self.convlstm3_h0.expand(N, -1, -1, -1), self.convlstm3_c0.expand(N, -1, -1, -1))
        lstm_hidden0 = (self.lstm_h0.expand(N, -1), self.lstm_c0.expand(N, -1))

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)
            else:
                convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(False, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)


        return scores



class RCOccludedSingleObjectClassifier(nn.Module):
    def __init__(self, n_iter=2, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden_lstm=512, n_classes=10, dropout_p=0.5):
        super(RCOccludedSingleObjectClassifier, self).__init__()
        self.n_iter = n_iter
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden_lstm = hidden_lstm
        self.dropout_p = dropout_p
        self.n_classes = n_classes

        self.convlstm1 = ConvLSTMCell(input_channels=input_channels, hidden_channels=hidden1, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.convlstm2 = ConvLSTMCell(input_channels=hidden1, hidden_channels=hidden2, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.convlstm3 = ConvLSTMCell(input_channels=hidden2, hidden_channels=hidden3, kernel_size=5)
        self.pool3 = nn.MaxPool2d(2,2)
        self.lstm = nn.LSTMCell(int(input_size/8)**2*hidden3, hidden_lstm)
        self.fc = nn.Linear(hidden_lstm, n_classes)
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

        self.convlstm1_h0 = Parameter(torch.Tensor(hidden1, input_size, input_size))
        self.convlstm1_c0 = Parameter(torch.Tensor(hidden1, input_size, input_size))

        self.convlstm2_h0 = Parameter(torch.Tensor(hidden2, int(input_size/2), int(input_size/2)))
        self.convlstm2_c0 = Parameter(torch.Tensor(hidden2, int(input_size/2), int(input_size/2)))

        self.convlstm3_h0 = Parameter(torch.Tensor(hidden3, int(input_size/4), int(input_size/4)))
        self.convlstm3_c0 = Parameter(torch.Tensor(hidden3, int(input_size/4), int(input_size/4)))

        self.lstm_h0 = Parameter(torch.Tensor(hidden_lstm))
        self.lstm_c0 = Parameter(torch.Tensor(hidden_lstm))

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
        init.normal(self.lstm.weight_hh, 0.0, 1.0/math.sqrt(self.hidden_lstm))
        init.constant(self.lstm.bias_ih, 0.0)
        # We initialize the forget gate bias to 1.0. Note that we need to modify only one of bias_ih or bias_hh.
        init.constant(self.lstm.bias_ih[self.hidden_lstm:2*self.hidden_lstm], 1.0)
        init.constant(self.lstm.bias_hh, 0.0)

        init.normal(self.fc.weight, 0.0, 1.0/math.sqrt(self.hidden_lstm))
        init.constant(self.fc.bias, 0.0)

    def cell(self, predict, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0):
        # When hc0, cc0 are the initial states of ROOM, we need to take into account that they have
        # shapes (hidden_size,), not (batch_size, hidden_size). So, we just copy them along the batch dimension.
        convlstm1_h1, convlstm1_c1 = self.convlstm1(x, convlstm1_hidden0)
        h = self.relu(self.pool1(convlstm1_h1))
        convlstm2_h1, convlstm2_c1 = self.convlstm2(h, convlstm2_hidden0)
        h = self.relu(self.pool2(convlstm2_h1))
        convlstm3_h1, convlstm3_c1 = self.convlstm3(h, convlstm3_hidden0)
        h = self.relu(self.pool3(convlstm3_h1)).view(-1, int(self.input_size/8)**2*self.hidden3)
        h = self.dropout(h)
        lstm_h1, lstm_c1 = self.lstm(h, lstm_hidden0)
        if predict:
            h = self.dropout(lstm_h1)
            scores = self.fc(h)
            return scores, (convlstm1_h1, convlstm1_c1), (convlstm2_h1, convlstm2_c1), (convlstm3_h1, convlstm3_c1), (lstm_h1, lstm_c1)
        else:
            return (convlstm1_h1, convlstm1_c1), (convlstm2_h1, convlstm2_c1), (convlstm3_h1, convlstm3_c1), (lstm_h1, lstm_c1)

    def forward(self, x):
        N = x.size(0)
        convlstm1_hidden0 = (self.convlstm1_h0.expand(N, -1, -1, -1), self.convlstm1_c0.expand(N, -1, -1, -1))
        convlstm2_hidden0 = (self.convlstm2_h0.expand(N, -1, -1, -1), self.convlstm2_c0.expand(N, -1, -1, -1))
        convlstm3_hidden0 = (self.convlstm3_h0.expand(N, -1, -1, -1), self.convlstm3_c0.expand(N, -1, -1, -1))
        lstm_hidden0 = (self.lstm_h0.expand(N, -1), self.lstm_c0.expand(N, -1))

        for i in range(self.n_iter):
            if i != self.n_iter-1:
                convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(False, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0) 
            elif i == self.n_iter-1:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0) 

        return scores



class RCControlOccludedSingleObjectClassifier(nn.Module):
    def __init__(self, n_iter=2, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden_lstm=512, n_classes=10, dropout_p=0.5):
        super(RCControlOccludedSingleObjectClassifier, self).__init__()
        self.n_iter = n_iter
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden_lstm = hidden_lstm
        self.n_classes = n_classes
        self.dropout_p = dropout_p

        self.convlstm1 = ConvLSTMCell(input_channels=input_channels, hidden_channels=hidden1, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.convlstm2 = ConvLSTMCell(input_channels=hidden1, hidden_channels=hidden2, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.convlstm3 = ConvLSTMCell(input_channels=hidden2, hidden_channels=hidden3, kernel_size=5)
        self.pool3 = nn.MaxPool2d(2,2)
        self.lstm = nn.LSTMCell(int(input_size/8)**2*hidden3, hidden_lstm)
        self.fc = nn.Linear(hidden_lstm, n_classes)
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

        self.convlstm1_h0 = Parameter(torch.Tensor(hidden1, input_size, input_size))
        self.convlstm1_c0 = Parameter(torch.Tensor(hidden1, input_size, input_size))

        self.convlstm2_h0 = Parameter(torch.Tensor(hidden2, int(input_size/2), int(input_size/2)))
        self.convlstm2_c0 = Parameter(torch.Tensor(hidden2, int(input_size/2), int(input_size/2)))

        self.convlstm3_h0 = Parameter(torch.Tensor(hidden3, int(input_size/4), int(input_size/4)))
        self.convlstm3_c0 = Parameter(torch.Tensor(hidden3, int(input_size/4), int(input_size/4)))

        self.lstm_h0 = Parameter(torch.Tensor(hidden_lstm))
        self.lstm_c0 = Parameter(torch.Tensor(hidden_lstm))

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
        init.normal(self.lstm.weight_hh, 0.0, 1.0/math.sqrt(self.hidden_lstm))
        init.constant(self.lstm.bias_ih, 0.0)
        # We initialize the forget gate bias to 1.0. Note that we need to modify only one of bias_ih or bias_hh.
        init.constant(self.lstm.bias_ih[self.hidden_lstm:2*self.hidden_lstm], 1.0)
        init.constant(self.lstm.bias_hh, 0.0)

        init.normal(self.fc.weight, 0.0, 1.0/math.sqrt(self.hidden_lstm))
        init.constant(self.fc.bias, 0.0)

    def cell(self, predict, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0):
        # When hc0, cc0 are the initial states of ROOM, we need to take into account that they have
        # shapes (hidden_size,), not (batch_size, hidden_size). So, we just copy them along the batch dimension.
        convlstm1_h1, convlstm1_c1 = self.convlstm1(x, convlstm1_hidden0)
        h = self.relu(self.pool1(convlstm1_h1))
        convlstm2_h1, convlstm2_c1 = self.convlstm2(h, convlstm2_hidden0)
        h = self.relu(self.pool2(convlstm2_h1))
        convlstm3_h1, convlstm3_c1 = self.convlstm3(h, convlstm3_hidden0)
        h = self.relu(self.pool3(convlstm3_h1)).view(-1, int(self.input_size/8)**2*self.hidden3)
        h = self.dropout(h)
        lstm_h1, lstm_c1 = self.lstm(h, lstm_hidden0)
        if predict:
            h = self.dropout(lstm_h1)
            scores = self.fc(h)
            return scores, (convlstm1_h1, convlstm1_c1), (convlstm2_h1, convlstm2_c1), (convlstm3_h1, convlstm3_c1), (lstm_h1, lstm_c1)
        else:
            return (convlstm1_h1, convlstm1_c1), (convlstm2_h1, convlstm2_c1), (convlstm3_h1, convlstm3_c1), (lstm_h1, lstm_c1)

    def forward(self, x):
        N = x.size(0)
        convlstm1_hidden0 = (self.convlstm1_h0.expand(N, -1, -1, -1), self.convlstm1_c0.expand(N, -1, -1, -1))
        convlstm2_hidden0 = (self.convlstm2_h0.expand(N, -1, -1, -1), self.convlstm2_c0.expand(N, -1, -1, -1))
        convlstm3_hidden0 = (self.convlstm3_h0.expand(N, -1, -1, -1), self.convlstm3_c0.expand(N, -1, -1, -1))
        lstm_hidden0 = (self.lstm_h0.expand(N, -1), self.lstm_c0.expand(N, -1))

        for i in range(self.n_iter):
            scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0) 
            break

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