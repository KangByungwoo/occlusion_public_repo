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

class FDMultiObjectClassifier(nn.Module):
    def __init__(self, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden4=1024, n_classes=10, dropout_p=0.5):
        super(FDMultiObjectClassifier, self).__init__()
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
        self.fc21 = nn.Linear(hidden4, n_classes)
        self.fc22 = nn.Linear(hidden4, n_classes)
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

        init.normal(self.fc21.weight, 0.0, 1.0/math.sqrt(self.hidden4))
        init.constant(self.fc21.bias, 0.0)

        init.normal(self.fc22.weight, 0.0, 1.0/math.sqrt(self.hidden4))
        init.constant(self.fc22.bias, 0.0)
    
    def forward(self, x):
        h = self.relu(self.pool1(self.conv1(x)))
        h = self.relu(self.pool2(self.conv2(h)))
        h = self.relu(self.pool3(self.conv3(h))).view(-1, int(self.input_size/8)**2*self.hidden3)
        h = self.dropout(h)
        h = self.relu(self.fc1(h))
        h = self.dropout(h)
        scores1 = self.fc21(h)
        scores2 = self.fc22(h)

        return scores1, scores2



class FDSeqMultiObjectClassifier(nn.Module):
    def __init__(self, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden4=1024, n_classes=10, dropout_p=0.5):
        super(FDSeqMultiObjectClassifier, self).__init__()
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
        self.br1fc1 = nn.Linear(int(input_size/4)**2*hidden2, hidden4)
        self.br1fc2 = nn.Linear(hidden4, n_classes)
        self.br2fc1 = nn.Linear(int(input_size/8)**2*hidden3, hidden4)
        self.br2fc2 = nn.Linear(hidden4, n_classes)
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

        init.kaiming_normal(self.br1fc1.weight)
        init.constant(self.br1fc1.bias, 0.0)

        init.kaiming_normal(self.br2fc1.weight)
        init.constant(self.br2fc1.bias, 0.0)

        init.normal(self.br1fc2.weight, 0.0, 1.0/math.sqrt(self.hidden4))
        init.constant(self.br1fc2.bias, 0.0)

        init.normal(self.br2fc2.weight, 0.0, 1.0/math.sqrt(self.hidden4))
        init.constant(self.br2fc2.bias, 0.0)
    
    def forward(self, x):
        h = self.relu(self.pool1(self.conv1(x)))
        h = self.relu(self.pool2(self.conv2(h)))
        
        # branch1 for fg obj
        br1 = h.view(-1, int(self.input_size/4)**2*self.hidden2)
        br1 = self.dropout(br1)
        br1 = self.relu(self.br1fc1(br1))
        br1 = self.dropout(br1)
        scores1 = self.br1fc2(br1)

        # branch2 for bg obj
        br2 = self.relu(self.pool3(self.conv3(h))).view(-1, int(self.input_size/8)**2*self.hidden3)
        br2 = self.dropout(br2)
        br2 = self.relu(self.br2fc1(br2))
        br2 = self.dropout(br2)
        scores2 = self.br2fc2(br2)

        return scores1, scores2



class FDSeqReversed4MultiObjectClassifier(nn.Module):
    def __init__(self, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden4=64, hidden5=1024, n_classes=10, dropout_p=0.5):
        super(FDSeqReversed4MultiObjectClassifier, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.hidden5 = hidden5
        self.n_classes = n_classes
        self.dropout_p = dropout_p
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden1, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=hidden1, out_channels=hidden2, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(in_channels=hidden2, out_channels=hidden3, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=hidden3, out_channels=hidden4, kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool2d(2,2)

        self.br1fc1 = nn.Linear(int(input_size/8)**2*hidden3, hidden5)
        self.br1fc2 = nn.Linear(hidden5, n_classes)
        self.br2fc1 = nn.Linear(int(input_size/16)**2*hidden4, hidden5)
        self.br2fc2 = nn.Linear(hidden5, n_classes)
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

        init.kaiming_normal(self.conv4.weight)
        init.constant(self.conv4.bias, 0.0)

        init.kaiming_normal(self.br1fc1.weight)
        init.constant(self.br1fc1.bias, 0.0)

        init.kaiming_normal(self.br2fc1.weight)
        init.constant(self.br2fc1.bias, 0.0)

        init.normal(self.br1fc2.weight, 0.0, 1.0/math.sqrt(self.hidden5))
        init.constant(self.br1fc2.bias, 0.0)

        init.normal(self.br2fc2.weight, 0.0, 1.0/math.sqrt(self.hidden5))
        init.constant(self.br2fc2.bias, 0.0)
    
    def forward(self, x):
        h = self.relu(self.pool1(self.conv1(x)))
        h = self.relu(self.pool2(self.conv2(h)))
        h = self.relu(self.pool3(self.conv3(h)))
        # branch1 for bg obj
        br1 = h.view(-1, int(self.input_size/8)**2*self.hidden3)
        br1 = self.dropout(br1)
        br1 = self.relu(self.br1fc1(br1))
        br1 = self.dropout(br1)
        scores2 = self.br1fc2(br1)

        # branch2 for fg obj
        br2 = self.relu(self.pool4(self.conv4(h))).view(-1, int(self.input_size/16)**2*self.hidden4)
        br2 = self.dropout(br2)
        br2 = self.relu(self.br2fc1(br2))
        br2 = self.dropout(br2)
        scores1 = self.br2fc2(br2)


        return scores1, scores2




class FDSeqCompDepthRCMultiObjectClassifier(nn.Module):
    def __init__(self, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden4=64, hidden_fc=512, n_classes=10, dropout_p=0.5, with_pool4=False):
        super(FDSeqCompDepthRCMultiObjectClassifier, self).__init__()
        '''
        Difference from FDSeqReversed4MultiObjectClassifier:
        1. Removed the pool4.
        2. Set default hidden5 to 512.
        Both of these changes to be consistent with RC. And of course, the readout ordering is different.
        '''

        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.hidden_fc = hidden_fc
        self.n_classes = n_classes
        self.dropout_p = dropout_p
        self.with_pool4 = with_pool4
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden1, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=hidden1, out_channels=hidden2, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(in_channels=hidden2, out_channels=hidden3, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=hidden3, out_channels=hidden4, kernel_size=5, stride=1, padding=2)
        if self.with_pool4:
            self.pool4 = nn.MaxPool2d(2,2)

        self.br1fc1 = nn.Linear(int(input_size/8)**2*hidden3, hidden_fc)
        self.br1fc2 = nn.Linear(hidden_fc, n_classes)

        if self.with_pool4:
            self.br2fc1 = nn.Linear(int(input_size/16)**2*hidden4, hidden_fc)
        else:
            self.br2fc1 = nn.Linear(int(input_size/8)**2*hidden4, hidden_fc)
        self.br2fc2 = nn.Linear(hidden_fc, n_classes)
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

        init.kaiming_normal(self.conv4.weight)
        init.constant(self.conv4.bias, 0.0)

        init.kaiming_normal(self.br1fc1.weight)
        init.constant(self.br1fc1.bias, 0.0)

        init.kaiming_normal(self.br2fc1.weight)
        init.constant(self.br2fc1.bias, 0.0)

        init.normal(self.br1fc2.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.br1fc2.bias, 0.0)

        init.normal(self.br2fc2.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.br2fc2.bias, 0.0)
    
    def forward(self, x):
        h = self.relu(self.pool1(self.conv1(x)))
        h = self.relu(self.pool2(self.conv2(h)))
        h = self.relu(self.pool3(self.conv3(h)))
        # branch1 for fg obj
        br1 = h.view(-1, int(self.input_size/8)**2*self.hidden3)
        br1 = self.dropout(br1)
        br1 = self.relu(self.br1fc1(br1))
        br1 = self.dropout(br1)
        scores1 = self.br1fc2(br1)

        # branch2 for bg obj
        if self.with_pool4:
            br2 = self.relu(self.pool4(self.conv4(h))).view(-1, int(self.input_size/16)**2*self.hidden4)
        else:
            br2 = self.relu(self.conv4(h)).view(-1, int(self.input_size/8)**2*self.hidden4)
        br2 = self.dropout(br2)
        br2 = self.relu(self.br2fc1(br2))
        br2 = self.dropout(br2)
        scores2 = self.br2fc2(br2)


        return scores1, scores2




class FDSeqCompDepthRCReversed3MultiObjectClassifier(nn.Module):
    def __init__(self, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden4=64, hidden5=64, hidden_fc=512, n_classes=10, dropout_p=0.5, with_pool4_and_5=False):
        super(FDSeqCompDepthRCReversed3MultiObjectClassifier, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.hidden5 = hidden5
        self.hidden_fc = hidden_fc
        self.n_classes = n_classes
        self.dropout_p = dropout_p
        self.with_pool4_and_5 = with_pool4_and_5
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden1, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=hidden1, out_channels=hidden2, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(in_channels=hidden2, out_channels=hidden3, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=hidden3, out_channels=hidden4, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(in_channels=hidden4, out_channels=hidden5, kernel_size=5, stride=1, padding=2)
        if self.with_pool4_and_5:
            self.pool4 = nn.MaxPool2d(2,2)
            self.pool5 = nn.MaxPool2d(2,2)

        if self.with_pool4_and_5:
            self.br1fc1 = nn.Linear(int(input_size/16)**2*hidden3, hidden_fc)
        else:
            self.br1fc1 = nn.Linear(int(input_size/8)**2*hidden4, hidden_fc)
        self.br1fc2 = nn.Linear(hidden_fc, n_classes)
        if self.with_pool4_and_5:
            self.br2fc1 = nn.Linear(int(input_size/32)**2*hidden4, hidden_fc)
        else:
            self.br2fc1 = nn.Linear(int(input_size/8)**2*hidden5, hidden_fc)
        self.br2fc2 = nn.Linear(hidden_fc, n_classes)
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

        init.kaiming_normal(self.conv4.weight)
        init.constant(self.conv4.bias, 0.0)

        init.kaiming_normal(self.conv5.weight)
        init.constant(self.conv5.bias, 0.0)

        init.kaiming_normal(self.br1fc1.weight)
        init.constant(self.br1fc1.bias, 0.0)

        init.kaiming_normal(self.br2fc1.weight)
        init.constant(self.br2fc1.bias, 0.0)

        init.normal(self.br1fc2.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.br1fc2.bias, 0.0)

        init.normal(self.br2fc2.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.br2fc2.bias, 0.0)
    
    def forward(self, x):
        h = self.relu(self.pool1(self.conv1(x)))
        h = self.relu(self.pool2(self.conv2(h)))
        h = self.relu(self.pool3(self.conv3(h)))
        if self.with_pool4_and_5:
            h = self.relu(self.pool4(self.conv4(h)))
        else:
            h = self.relu(self.conv4(h))

        # branch1 for fg obj
        if self.with_pool4_and_5:
            br1 = h.view(-1, int(self.input_size/16)**2*self.hidden3)
        else:
            br1 = h.view(-1, int(self.input_size/8)**2*self.hidden4)
        br1 = self.dropout(br1)
        br1 = self.relu(self.br1fc1(br1))
        br1 = self.dropout(br1)
        scores2 = self.br1fc2(br1)

        # branch2 for bg obj
        if self.with_pool4_and_5:
            br2 = self.relu(self.pool5(self.conv5(h))).view(-1, int(self.input_size/32)**2*self.hidden4)
        else:
            br2 = self.relu(self.conv5(h)).view(-1, int(self.input_size/8)**2*self.hidden5)
        br2 = self.dropout(br2)
        br2 = self.relu(self.br2fc1(br2))
        br2 = self.dropout(br2)
        scores1 = self.br2fc2(br2)


        return scores1, scores2






class FDTallMultiObjectClassifier(nn.Module):
    def __init__(self, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden4=64, hidden5=64, hidden6=64, hidden7=1024, n_classes=10, dropout_p=0.5):
        super(FDTallMultiObjectClassifier, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.hidden5 = hidden5
        self.hidden6 = hidden6
        self.hidden7 = hidden7
        self.n_classes = n_classes
        self.dropout_p = dropout_p
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden1, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=hidden1, out_channels=hidden2, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(2,2)
        
        self.conv3 = nn.Conv2d(in_channels=hidden2, out_channels=hidden3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=hidden3, out_channels=hidden4, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(2,2)
        
        self.conv5 = nn.Conv2d(in_channels=hidden4, out_channels=hidden5, kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(in_channels=hidden5, out_channels=hidden6, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(int(input_size/8)**2*hidden6, hidden7)
        self.fc21 = nn.Linear(hidden7, n_classes)
        self.fc22 = nn.Linear(hidden7, n_classes)
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

        init.kaiming_normal(self.conv4.weight)
        init.constant(self.conv4.bias, 0.0)

        init.kaiming_normal(self.conv5.weight)
        init.constant(self.conv5.bias, 0.0)

        init.kaiming_normal(self.conv6.weight)
        init.constant(self.conv6.bias, 0.0)

        init.kaiming_normal(self.fc1.weight)
        init.constant(self.fc1.bias, 0.0)

        init.normal(self.fc21.weight, 0.0, 1.0/math.sqrt(self.hidden7))
        init.constant(self.fc21.bias, 0.0)

        init.normal(self.fc22.weight, 0.0, 1.0/math.sqrt(self.hidden7))
        init.constant(self.fc22.bias, 0.0)
    
    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.pool1(self.conv2(h)))

        h = self.relu(self.conv3(h))
        h = self.relu(self.pool2(self.conv4(h)))

        h = self.relu(self.conv5(h))
        h = self.relu(self.pool3(self.conv6(h))).view(-1, int(self.input_size/8)**2*self.hidden6)
        h = self.dropout(h)
        h = self.relu(self.fc1(h))
        h = self.dropout(h)
        scores1 = self.fc21(h)
        scores2 = self.fc22(h)

        return scores1, scores2


class FDTallerMultiObjectClassifier(nn.Module):
    def __init__(self, input_size=50, n_layers=12, input_channels=1, hidden=64, hidden_fc=1024, n_classes=10, dropout_p=0.5):
        super(FDTallerMultiObjectClassifier, self).__init__()
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
        self.fc21 = nn.Linear(hidden_fc, n_classes)
        self.fc22 = nn.Linear(hidden_fc, n_classes)
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

        self.init_parameters()

    def init_parameters(self):
        for i in range(self.n_layers):
            init.kaiming_normal(getattr(self, 'conv{}'.format(i+1)).weight)
            init.constant(getattr(self, 'conv{}'.format(i+1)).bias, 0.0)

        init.kaiming_normal(self.fc1.weight)
        init.constant(self.fc1.bias, 0.0)

        init.normal(self.fc21.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.fc21.bias, 0.0)

        init.normal(self.fc22.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.fc22.bias, 0.0)
    
    def forward(self, x):
        for i in range(self.n_layers):
            x = self.relu(getattr(self, 'conv{}'.format(i+1))(x))
            if (i+1) % (self.n_layers//3) == 0:
                x = self.pool(x)

        x = self.dropout(x.view(-1, int(self.input_size/8)**2*self.hidden))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc21(x), self.fc22(x)



class FDTallerSeqMultiObjectClassifier(nn.Module):
    def __init__(self, input_size=50, n_layers=12, input_channels=1, hidden=64, hidden_fc=1024, n_classes=10, dropout_p=0.5):
        super(FDTallerSeqMultiObjectClassifier, self).__init__()
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

        self.br1fc1 = nn.Linear(int(input_size/4)**2*hidden, hidden_fc)
        self.br1fc2 = nn.Linear(hidden_fc, n_classes)

        self.br2fc1 = nn.Linear(int(input_size/8)**2*hidden, hidden_fc)
        self.br2fc2 = nn.Linear(hidden_fc, n_classes)
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

        self.init_parameters()

    def init_parameters(self):
        for i in range(self.n_layers):
            init.kaiming_normal(getattr(self, 'conv{}'.format(i+1)).weight)
            init.constant(getattr(self, 'conv{}'.format(i+1)).bias, 0.0)

        init.kaiming_normal(self.br1fc1.weight)
        init.constant(self.br1fc1.bias, 0.0)

        init.kaiming_normal(self.br2fc1.weight)
        init.constant(self.br2fc1.bias, 0.0)

        init.normal(self.br1fc2.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.br1fc2.bias, 0.0)

        init.normal(self.br2fc2.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.br2fc2.bias, 0.0)
    
    def forward(self, x):
        for i in range(self.n_layers):
            x = self.relu(getattr(self, 'conv{}'.format(i+1))(x))
            if (i+1) % (self.n_layers//3) == 0:
                x = self.pool(x)
            if (i+1) == (self.n_layers//3)*2:
                br1 = self.dropout(x.view(-1, int(self.input_size/4)**2*self.hidden))
                br1 = self.relu(self.br1fc1(br1))
                br1 = self.dropout(br1)
                fg_score = self.br1fc2(br1)
                

        br2 = self.dropout(x.view(-1, int(self.input_size/8)**2*self.hidden))
        br2 = self.relu(self.br2fc1(br2))
        br2 = self.dropout(br2)
        bg_score = self.br2fc2(br2)

        return fg_score, bg_score



class FDTallerSeqReversed4MultiObjectClassifier(nn.Module):
    def __init__(self, input_size=50, n_layers=16, input_channels=1, hidden=64, hidden_fc=1024, n_classes=10, dropout_p=0.5):
        super(FDTallerSeqReversed4MultiObjectClassifier, self).__init__()
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

        self.br1fc1 = nn.Linear(int(input_size/8)**2*hidden, hidden_fc)
        self.br1fc2 = nn.Linear(hidden_fc, n_classes)

        self.br2fc1 = nn.Linear(int(input_size/16)**2*hidden, hidden_fc)
        self.br2fc2 = nn.Linear(hidden_fc, n_classes)
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

        self.init_parameters()

    def init_parameters(self):
        for i in range(self.n_layers):
            init.kaiming_normal(getattr(self, 'conv{}'.format(i+1)).weight)
            init.constant(getattr(self, 'conv{}'.format(i+1)).bias, 0.0)

        init.kaiming_normal(self.br1fc1.weight)
        init.constant(self.br1fc1.bias, 0.0)

        init.kaiming_normal(self.br2fc1.weight)
        init.constant(self.br2fc1.bias, 0.0)

        init.normal(self.br1fc2.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.br1fc2.bias, 0.0)

        init.normal(self.br2fc2.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.br2fc2.bias, 0.0)
    
    def forward(self, x):
        for i in range(self.n_layers):
            x = self.relu(getattr(self, 'conv{}'.format(i+1))(x))
            if (i+1) % (self.n_layers//4) == 0:
                x = self.pool(x)
            if (i+1) == (self.n_layers//4)*3:
                br1 = self.dropout(x.view(-1, int(self.input_size/8)**2*self.hidden))
                br1 = self.relu(self.br1fc1(br1))
                br1 = self.dropout(br1)
                bg_score = self.br1fc2(br1)
                

        br2 = self.dropout(x.view(-1, int(self.input_size/16)**2*self.hidden))
        br2 = self.relu(self.br2fc1(br2))
        br2 = self.dropout(br2)
        fg_score = self.br2fc2(br2)

        return fg_score, bg_score



class WeightSharedFDTallerMultiObjectClassifier(nn.Module):
    def __init__(self, input_size=50, n_layers=12, input_channels=1, hidden=64, hidden_fc=1024, n_classes=10, dropout_p=0.5):
        super(WeightSharedFDTallerMultiObjectClassifier, self).__init__()
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
        self.fc21 = nn.Linear(hidden_fc, n_classes)
        self.fc22 = nn.Linear(hidden_fc, n_classes)
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

        init.normal(self.fc21.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.fc21.bias, 0.0)

        init.normal(self.fc22.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.fc22.bias, 0.0)
    
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
        return self.fc21(x), self.fc22(x)



class WeightSharedFDTallerSeqMultiObjectClassifier(nn.Module):
    def __init__(self, input_size=50, n_layers=12, input_channels=1, hidden=64, hidden_fc=1024, n_classes=10, dropout_p=0.5):
        super(WeightSharedFDTallerSeqMultiObjectClassifier, self).__init__()
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

        self.br1fc1 = nn.Linear(int(input_size/4)**2*hidden, hidden_fc)
        self.br1fc2 = nn.Linear(hidden_fc, n_classes)

        self.br2fc1 = nn.Linear(int(input_size/8)**2*hidden, hidden_fc)
        self.br2fc2 = nn.Linear(hidden_fc, n_classes)
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

        self.init_parameters()

    def init_parameters(self):
        for i in range(self.n_layers):
            if (i+1) == 1 or (i+1) % (self.n_layers//3) == 0:
                init.kaiming_normal(getattr(self, 'conv{}'.format(i+1)).weight)
                init.constant(getattr(self, 'conv{}'.format(i+1)).bias, 0.0)

        init.kaiming_normal(self.br1fc1.weight)
        init.constant(self.br1fc1.bias, 0.0)

        init.kaiming_normal(self.br2fc1.weight)
        init.constant(self.br2fc1.bias, 0.0)

        init.normal(self.br1fc2.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.br1fc2.bias, 0.0)

        init.normal(self.br2fc2.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.br2fc2.bias, 0.0)
    
    def forward(self, x):
        for i in range(self.n_layers):
            if (i+1) == 1:
                x = self.relu(getattr(self, 'conv{}'.format(i+1))(x))
            elif (i+1) % (self.n_layers//3) == 0:
                x = self.relu(getattr(self, 'conv{}'.format(i+1))(x))
                x = self.pool(x)
                if (i+1) == (self.n_layers//3)*2:
                    br1 = self.dropout(x.view(-1, int(self.input_size/4)**2*self.hidden))
                    br1 = self.relu(self.br1fc1(br1))
                    br1 = self.dropout(br1)
                    fg_score = self.br1fc2(br1)
            else:
                k = ((i+1)//(self.n_layers//3) + 1)*(self.n_layers//3)
                x = self.relu(getattr(self, 'conv{}'.format(k))(x))

        br2 = self.dropout(x.view(-1, int(self.input_size/8)**2*self.hidden))
        br2 = self.relu(self.br2fc1(br2))
        br2 = self.dropout(br2)
        bg_score = self.br2fc2(br2)

        return fg_score, bg_score



class WeightSharedFDTallerSeqReversed4MultiObjectClassifier(nn.Module):
    def __init__(self, input_size=50, n_layers=16, input_channels=1, hidden=64, hidden_fc=1024, n_classes=10, dropout_p=0.5):
        super(WeightSharedFDTallerSeqReversed4MultiObjectClassifier, self).__init__()
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
            elif (i+1) % (n_layers//4) == 0:
                setattr(self, 'conv{}'.format(i+1), nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=5, stride=1, padding=2))                

        self.pool = nn.MaxPool2d(2,2)

        self.br1fc1 = nn.Linear(int(input_size/8)**2*hidden, hidden_fc)
        self.br1fc2 = nn.Linear(hidden_fc, n_classes)

        self.br2fc1 = nn.Linear(int(input_size/16)**2*hidden, hidden_fc)
        self.br2fc2 = nn.Linear(hidden_fc, n_classes)
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

        self.init_parameters()

    def init_parameters(self):
        for i in range(self.n_layers):
            if (i+1) == 1 or (i+1) % (self.n_layers//4) == 0:
                init.kaiming_normal(getattr(self, 'conv{}'.format(i+1)).weight)
                init.constant(getattr(self, 'conv{}'.format(i+1)).bias, 0.0)

        init.kaiming_normal(self.br1fc1.weight)
        init.constant(self.br1fc1.bias, 0.0)

        init.kaiming_normal(self.br2fc1.weight)
        init.constant(self.br2fc1.bias, 0.0)

        init.normal(self.br1fc2.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.br1fc2.bias, 0.0)

        init.normal(self.br2fc2.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.br2fc2.bias, 0.0)
    
    
    def forward(self, x):
        for i in range(self.n_layers):
            if (i+1) == 1:
                x = self.relu(getattr(self, 'conv{}'.format(i+1))(x))
            elif (i+1) % (self.n_layers//4) == 0:
                x = self.relu(getattr(self, 'conv{}'.format(i+1))(x))
                x = self.pool(x)
                if (i+1) == (self.n_layers//4)*3:
                    br1 = self.dropout(x.view(-1, int(self.input_size/8)**2*self.hidden))
                    br1 = self.relu(self.br1fc1(br1))
                    br1 = self.dropout(br1)
                    bg_score = self.br1fc2(br1)
            else:
                k = ((i+1)//(self.n_layers//4) + 1)*(self.n_layers//4)
                x = self.relu(getattr(self, 'conv{}'.format(k))(x))

        br2 = self.dropout(x.view(-1, int(self.input_size/16)**2*self.hidden))
        br2 = self.relu(self.br2fc1(br2))
        br2 = self.dropout(br2)
        fg_score = self.br2fc2(br2)

        return fg_score, bg_score



class FDTallerBNMultiObjectClassifier(nn.Module):
    def __init__(self, input_size=50, n_layers=24, input_channels=1, hidden=64, hidden_fc=1024, n_classes=10, dropout_p=0.5):
        super(FDTallerBNMultiObjectClassifier, self).__init__()
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
            setattr(self, 'conv{}_bn'.format(i+1), nn.BatchNorm2d(self.hidden))

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(int(input_size/8)**2*hidden, hidden_fc)
        self.fc21 = nn.Linear(hidden_fc, n_classes)
        self.fc22 = nn.Linear(hidden_fc, n_classes)
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

        self.init_parameters()

    def init_parameters(self):
        for i in range(self.n_layers):
            init.kaiming_normal(getattr(self, 'conv{}'.format(i+1)).weight)
            init.constant(getattr(self, 'conv{}'.format(i+1)).bias, 0.0)

        init.kaiming_normal(self.fc1.weight)
        init.constant(self.fc1.bias, 0.0)

        init.normal(self.fc21.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.fc21.bias, 0.0)

        init.normal(self.fc22.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.fc22.bias, 0.0)
    
    def forward(self, x):
        for i in range(self.n_layers):
            x = self.relu(getattr(self, 'conv{}_bn'.format(i+1))(getattr(self, 'conv{}'.format(i+1))(x)))
            if (i+1) % (self.n_layers//3) == 0:
                x = self.pool(x)

        x = self.dropout(x.view(-1, int(self.input_size/8)**2*self.hidden))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc21(x), self.fc22(x)

class FDTallerResNetMultiObjectClassifier(nn.Module):
    def __init__(self, input_size=50, n_layers=24, input_channels=1, hidden=64, hidden_fc=1024, n_classes=10, dropout_p=0.5):
        super(FDTallerResNetMultiObjectClassifier, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden = hidden
        self.hidden_fc = hidden_fc

        self.n_classes = n_classes
        self.dropout_p = dropout_p
        
        self.n_layers = n_layers
        assert self.n_layers % 6 == 0

        for i in range(n_layers):
            if i == 0:
                setattr(self, 'conv{}'.format(i+1), nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden, kernel_size=5, stride=1, padding=2))
            else:
                setattr(self, 'conv{}'.format(i+1), nn.Conv2d(in_channels=self.hidden, out_channels=self.hidden, kernel_size=5, stride=1, padding=2))
            setattr(self, 'conv{}_bn'.format(i+1), nn.BatchNorm2d(self.hidden))

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(int(input_size/8)**2*hidden, hidden_fc)
        self.fc21 = nn.Linear(hidden_fc, n_classes)
        self.fc22 = nn.Linear(hidden_fc, n_classes)
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

        self.init_parameters()

    def init_parameters(self):
        for i in range(self.n_layers):
            init.kaiming_normal(getattr(self, 'conv{}'.format(i+1)).weight)
            init.constant(getattr(self, 'conv{}'.format(i+1)).bias, 0.0)

        init.kaiming_normal(self.fc1.weight)
        init.constant(self.fc1.bias, 0.0)

        init.normal(self.fc21.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.fc21.bias, 0.0)

        init.normal(self.fc22.weight, 0.0, 1.0/math.sqrt(self.hidden_fc))
        init.constant(self.fc22.bias, 0.0)
    
    def forward(self, x):
        for i in range(self.n_layers//2):
            h = self.relu(getattr(self, 'conv{}_bn'.format(2*(i+1)-1))(getattr(self, 'conv{}'.format(2*(i+1)-1))(x)))
            h = getattr(self, 'conv{}_bn'.format(2*(i+1)))(getattr(self, 'conv{}'.format(2*(i+1)))(h))
            x = self.relu(x+h)
            if 2*(i+1) % (self.n_layers//3) == 0:
                x = self.pool(x)

        x = self.dropout(x.view(-1, int(self.input_size/8)**2*self.hidden))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc21(x), self.fc22(x)



class RCMultiObjectClassifier(nn.Module):
    def __init__(self, reverse=False, n_iter=2, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden_lstm=512, n_classes=10, dropout_p=0.5):
        super(RCMultiObjectClassifier, self).__init__()
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

        # This added for rep similarity.
        self.last_lstm_h = None

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
            self.last_lstm_h = h
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
            if i == self.n_iter - 2:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)
                if not self.reverse:
                    scores1 = scores
                else:
                    scores2 = scores
            elif i == self.n_iter - 1:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)
                if not self.reverse:
                    scores2 = scores 
                else:
                    scores1 = scores 
            else:
                convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(False, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)


        return scores1, scores2


class RCTwoReadoutMultiObjectClassifier(nn.Module):
    def __init__(self, reverse=False, n_iter=2, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden_lstm=512, n_classes=10, dropout_p=0.5):
        super(RCTwoReadoutMultiObjectClassifier, self).__init__()
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

        self.convlstm1 = ConvLSTMCell(input_channels=input_channels, hidden_channels=hidden1, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.convlstm2 = ConvLSTMCell(input_channels=hidden1, hidden_channels=hidden2, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.convlstm3 = ConvLSTMCell(input_channels=hidden2, hidden_channels=hidden3, kernel_size=5)
        self.pool3 = nn.MaxPool2d(2,2)
        self.lstm = nn.LSTMCell(int(input_size/8)**2*hidden3, hidden_lstm)
        self.fc1 = nn.Linear(hidden_lstm, n_classes)
        self.fc2 = nn.Linear(hidden_lstm, n_classes)
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

        init.normal(self.fc1.weight, 0.0, 1.0/math.sqrt(self.hidden_lstm))
        init.constant(self.fc1.bias, 0.0)

        init.normal(self.fc2.weight, 0.0, 1.0/math.sqrt(self.hidden_lstm))
        init.constant(self.fc2.bias, 0.0)


    def cell(self, predict, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0, readout):
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
            scores = readout(h)
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
            if i == self.n_iter - 2:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0, self.fc1)
                if not self.reverse:
                    scores1 = scores
                else:
                    scores2 = scores
            elif i == self.n_iter - 1:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0, self.fc2)
                if not self.reverse:
                    scores2 = scores 
                else:
                    scores1 = scores 
            else:
                convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(False, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)


        return scores1, scores2


class RCControlMultiObjectClassifier(nn.Module):
    def __init__(self, n_iter=2, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden_lstm=512, n_classes=10, dropout_p=0.5):
        super(RCControlMultiObjectClassifier, self).__init__()
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
        self.fc1 = nn.Linear(hidden_lstm, n_classes)
        self.fc2 = nn.Linear(hidden_lstm, n_classes)
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

        init.normal(self.fc1.weight, 0.0, 1.0/math.sqrt(self.hidden_lstm))
        init.constant(self.fc1.bias, 0.0)

        init.normal(self.fc2.weight, 0.0, 1.0/math.sqrt(self.hidden_lstm))
        init.constant(self.fc2.bias, 0.0)


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
            scores1 = self.fc1(h)
            scores2 = self.fc2(h)
            return scores1, scores2, (convlstm1_h1, convlstm1_c1), (convlstm2_h1, convlstm2_c1), (convlstm3_h1, convlstm3_c1), (lstm_h1, lstm_c1)
        else:
            return (convlstm1_h1, convlstm1_c1), (convlstm2_h1, convlstm2_c1), (convlstm3_h1, convlstm3_c1), (lstm_h1, lstm_c1)


    def forward(self, x):
        N = x.size(0)
        convlstm1_hidden0 = (self.convlstm1_h0.expand(N, -1, -1, -1), self.convlstm1_c0.expand(N, -1, -1, -1))
        convlstm2_hidden0 = (self.convlstm2_h0.expand(N, -1, -1, -1), self.convlstm2_c0.expand(N, -1, -1, -1))
        convlstm3_hidden0 = (self.convlstm3_h0.expand(N, -1, -1, -1), self.convlstm3_c0.expand(N, -1, -1, -1))
        lstm_hidden0 = (self.lstm_h0.expand(N, -1), self.lstm_c0.expand(N, -1))


        for i in range(self.n_iter):
            scores1, scores2, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0) 
            break

        return scores1, scores2


class RCLastMultiObjectClassifier(nn.Module):
    def __init__(self, n_iter=2, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden_lstm=512, n_classes=10, dropout_p=0.5, init_method='normal'):
        super(RCLastMultiObjectClassifier, self).__init__()
        self.n_iter = n_iter
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden_lstm = hidden_lstm
        self.n_classes = n_classes
        self.dropout_p = dropout_p
        self.init_method = init_method

        self.convlstm1 = ConvLSTMCell(input_channels=input_channels, hidden_channels=hidden1, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.convlstm2 = ConvLSTMCell(input_channels=hidden1, hidden_channels=hidden2, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.convlstm3 = ConvLSTMCell(input_channels=hidden2, hidden_channels=hidden3, kernel_size=5)
        self.pool3 = nn.MaxPool2d(2,2)
        self.lstm = nn.LSTMCell(int(input_size/8)**2*hidden3, hidden_lstm)
        self.fc1 = nn.Linear(hidden_lstm, n_classes)
        self.fc2 = nn.Linear(hidden_lstm, n_classes)
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
        if self.init_method == 'normal':
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

            init.normal(self.fc1.weight, 0.0, 1.0/math.sqrt(self.hidden_lstm))
            init.constant(self.fc1.bias, 0.0)

            init.normal(self.fc2.weight, 0.0, 1.0/math.sqrt(self.hidden_lstm))
            init.constant(self.fc2.bias, 0.0)
        elif self.init_method == 'uniform':
            init.uniform(self.convlstm1.conv.weight[:,:self.input_channels], -math.sqrt(3.0/(self.input_channels*(5**2))), math.sqrt(3.0/(self.input_channels*(5**2)))) # 5**2 is (kernel_size)**2.
            init.uniform(self.convlstm1.conv.weight[:,self.input_channels:], -math.sqrt(3.0/self.hidden1*(5**2)), math.sqrt(3.0/self.hidden1*(5**2))) # 5**2 is (kernel_size)**2.
            init.constant(self.convlstm1.conv.bias, 0.0)
            # We initialize the forget gate bias to 1.0.
            init.constant(self.convlstm1.conv.bias[self.hidden1:2*self.hidden1], 1.0)

            init.uniform(self.convlstm2.conv.weight[:,:self.hidden1], -math.sqrt(3.0/(self.hidden1*(5**2))), math.sqrt(3.0/(self.hidden1*(5**2)))) # 5**2 is (kernel_size)**2.
            init.uniform(self.convlstm2.conv.weight[:,self.hidden1:], -math.sqrt(3.0/(self.hidden2*(5**2))), math.sqrt(3.0/(self.hidden2*(5**2)))) # 5**2 is (kernel_size)**2.
            init.constant(self.convlstm2.conv.bias, 0.0)
            # We initialize the forget gate bias to 1.0.
            init.constant(self.convlstm2.conv.bias[self.hidden2:2*self.hidden2], 1.0)

            init.uniform(self.convlstm3.conv.weight[:,:self.hidden2], -math.sqrt(3.0/(self.hidden2*(5**2))), math.sqrt(3.0/(self.hidden2*(5**2)))) # 5**2 is (kernel_size)**2.
            init.uniform(self.convlstm3.conv.weight[:,self.hidden2:], -math.sqrt(3.0/(self.hidden3*(5**2))), math.sqrt(3.0/(self.hidden3*(5**2)))) # 5**2 is (kernel_size)**2.
            init.constant(self.convlstm3.conv.bias, 0.0)
            # We initialize the forget gate bias to 1.0.
            init.constant(self.convlstm3.conv.bias[self.hidden3:2*self.hidden3], 1.0)

            init.uniform(self.lstm.weight_ih, -math.sqrt(3.0/(int(self.input_size/8)**2*self.hidden3)), math.sqrt(3.0/(int(self.input_size/8)**2*self.hidden3)))
            init.uniform(self.lstm.weight_hh, -math.sqrt(3.0/self.hidden_lstm), math.sqrt(3.0/self.hidden_lstm))
            init.constant(self.lstm.bias_ih, 0.0)
            # We initialize the forget gate bias to 1.0. Note that we need to modify only one of bias_ih or bias_hh.
            init.constant(self.lstm.bias_ih[self.hidden_lstm:2*self.hidden_lstm], 1.0)
            init.constant(self.lstm.bias_hh, 0.0)

            init.uniform(self.fc1.weight, -math.sqrt(3.0/self.hidden_lstm), math.sqrt(3.0/self.hidden_lstm))
            init.constant(self.fc1.bias, 0.0)

            init.uniform(self.fc2.weight, -math.sqrt(3.0/self.hidden_lstm), math.sqrt(3.0/self.hidden_lstm))
            init.constant(self.fc2.bias, 0.0)




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
            scores1 = self.fc1(h)
            scores2 = self.fc2(h)
            return scores1, scores2, (convlstm1_h1, convlstm1_c1), (convlstm2_h1, convlstm2_c1), (convlstm3_h1, convlstm3_c1), (lstm_h1, lstm_c1)
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
                scores1, scores2, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0) 
        
        return scores1, scores2




class TDRCMultiObjectClassifierA(nn.Module):
    '''
    TDRCMultiObjectClassifierA is the same as TDRCMultiObjectClassifier except that we remove the top-down feedback from the fully-connected lstm.
    '''
    def __init__(self, reverse=False, n_iter=2, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden_lstm=512, n_classes=10, dropout_p=0.5):
        super(TDRCMultiObjectClassifierA, self).__init__()
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
        self.convlstm2 = ConvLSTMCell(input_channels=hidden1*2, hidden_channels=hidden2, kernel_size=5)
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
        self.td_convlstm3 = nn.Conv2d(in_channels=hidden3, out_channels=hidden1, kernel_size=1, stride=1, padding=0)
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

        # TD layers. Since we are not applying relu, we use the normal distribution initialization, without the factor of sqrt(2).
        init.normal(self.td_convlstm3.weight, 1.0/math.sqrt(self.hidden3))
        init.constant(self.td_convlstm3.bias, 0.0)

        init.normal(self.td_convlstm2.weight, 1.0/math.sqrt(self.hidden2))
        init.constant(self.td_convlstm2.bias, 0.0)


    def cell(self, predict, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0):
        # TD feedbacks.
        td_convlstm3 = self.td_convlstm3(F.upsample(convlstm3_hidden0[0], scale_factor=2, mode='nearest')) # The linear size goes from 12 to 24.
        # We need to pad td_convlstm3 so that it has the linear size 25, not 24.
        td_convlstm3 = F.pad(td_convlstm3, (0,1,0,1), mode='constant', value=0)

        td_convlstm2 = self.td_convlstm2(F.upsample(convlstm2_hidden0[0], scale_factor=2, mode='nearest')) # The linear size goes from 25 to 50.

        convlstm1_h1, convlstm1_c1 = self.convlstm1(torch.cat([x, td_convlstm2], 1), convlstm1_hidden0) # input x: C=input_channels, H,W = input_size = 50.
        h = self.relu(self.pool1(convlstm1_h1))
        convlstm2_h1, convlstm2_c1 = self.convlstm2(torch.cat([h, td_convlstm3], 1), convlstm2_hidden0) # input h: C=hidden1, H,W = int(input_size/2) = 25.
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
            if i == self.n_iter - 2:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)
                if not self.reverse:
                    scores1 = scores
                else:
                    scores2 = scores
            elif i == self.n_iter - 1:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)
                if not self.reverse:
                    scores2 = scores 
                else:
                    scores1 = scores 
            else:
                convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(False, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)


        return scores1, scores2




class TDRCMultiObjectClassifierB(nn.Module):
    '''
    TDRCMultiObjectClassifierB is the same as TDRCMultiObjectClassifier except that we remove the top-down feedback from the fully-connected lstm and from the convlstm2.
    '''
    def __init__(self, reverse=False, n_iter=2, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden_lstm=512, n_classes=10, dropout_p=0.5):
        super(TDRCMultiObjectClassifierB, self).__init__()
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
        self.convlstm1 = ConvLSTMCell(input_channels=input_channels, hidden_channels=hidden1, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.convlstm2 = ConvLSTMCell(input_channels=hidden1*2, hidden_channels=hidden2, kernel_size=5)
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
        self.td_convlstm3 = nn.Conv2d(in_channels=hidden3, out_channels=hidden1, kernel_size=1, stride=1, padding=0)

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

        # TD layers. Since we are not applying relu, we use the normal distribution initialization, without the factor of sqrt(2).
        init.normal(self.td_convlstm3.weight, 1.0/math.sqrt(self.hidden3))
        init.constant(self.td_convlstm3.bias, 0.0)


    def cell(self, predict, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0):
        # TD feedbacks.
        td_convlstm3 = self.td_convlstm3(F.upsample(convlstm3_hidden0[0], scale_factor=2, mode='nearest')) # The linear size goes from 12 to 24.
        # We need to pad td_convlstm3 so that it has the linear size 25, not 24.
        td_convlstm3 = F.pad(td_convlstm3, (0,1,0,1), mode='constant', value=0)

        convlstm1_h1, convlstm1_c1 = self.convlstm1(x, convlstm1_hidden0) # input x: C=input_channels, H,W = input_size = 50.
        h = self.relu(self.pool1(convlstm1_h1))
        convlstm2_h1, convlstm2_c1 = self.convlstm2(torch.cat([h, td_convlstm3], 1), convlstm2_hidden0) # input h: C=hidden1, H,W = int(input_size/2) = 25.
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
            if i == self.n_iter - 2:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)
                if not self.reverse:
                    scores1 = scores
                else:
                    scores2 = scores
            elif i == self.n_iter - 1:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)
                if not self.reverse:
                    scores2 = scores 
                else:
                    scores1 = scores 
            else:
                convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(False, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)


        return scores1, scores2





class TDRCMultiObjectClassifierC(nn.Module):
    '''
    TDRCMultiObjectClassifierC is the same as TDRCMultiObjectClassifier except that we remove the top-down feedback from the fully-connected lstm and from the convlstm3.
    '''
    def __init__(self, reverse=False, n_iter=2, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden_lstm=512, n_classes=10, dropout_p=0.5):
        super(TDRCMultiObjectClassifierC, self).__init__()
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
            if i == self.n_iter - 2:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)
                if not self.reverse:
                    scores1 = scores
                else:
                    scores2 = scores
            elif i == self.n_iter - 1:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)
                if not self.reverse:
                    scores2 = scores 
                else:
                    scores1 = scores 
            else:
                convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(False, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)


        return scores1, scores2





class TDRCMultiObjectClassifierD(nn.Module):
    def __init__(self, reverse=False, n_iter=2, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden_lstm=512, n_classes=10, dropout_p=0.5):
        super(TDRCMultiObjectClassifierD, self).__init__()
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
            if i == self.n_iter - 2:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)
                if not self.reverse:
                    scores1 = scores
                else:
                    scores2 = scores
            elif i == self.n_iter - 1:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)
                if not self.reverse:
                    scores2 = scores 
                else:
                    scores1 = scores 
            else:
                convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(False, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)


        return scores1, scores2




class TDRCMultiObjectClassifier(nn.Module):
    def __init__(self, reverse=False, n_iter=2, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden_lstm=512, n_classes=10, dropout_p=0.5):
        super(TDRCMultiObjectClassifier, self).__init__()
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
        self.convlstm2 = ConvLSTMCell(input_channels=hidden1*2, hidden_channels=hidden2, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.convlstm3 = ConvLSTMCell(input_channels=hidden2*2, hidden_channels=hidden3, kernel_size=5)
        self.pool3 = nn.MaxPool2d(2,2)
        self.lstm = nn.LSTMCell(int(input_size/8)**2*hidden3, hidden_lstm)
        self.fc = nn.Linear(hidden_lstm, n_classes)
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

        '''
        TD layers are labeled by the origin of the top-down feedback.
        '''
        self.td_lstm = nn.Linear(hidden_lstm, int(input_size/4)**2*hidden2)
        self.td_convlstm3 = nn.Conv2d(in_channels=hidden3, out_channels=hidden1, kernel_size=1, stride=1, padding=0)
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

        # TD layers. Since we are not applying relu, we use the normal distribution initialization, without the factor of sqrt(2).
        init.normal(self.td_lstm.weight, 0.0, 1.0/math.sqrt(self.hidden_lstm))
        init.constant(self.td_lstm.bias, 0.0)

        init.normal(self.td_convlstm3.weight, 1.0/math.sqrt(self.hidden3))
        init.constant(self.td_convlstm3.bias, 0.0)

        init.normal(self.td_convlstm2.weight, 1.0/math.sqrt(self.hidden2))
        init.constant(self.td_convlstm2.bias, 0.0)


    def cell(self, predict, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0):
        # TD feedbacks.
        td_lstm = self.td_lstm(lstm_hidden0[0]).view(-1, self.hidden2, int(self.input_size/4), int(self.input_size/4))
        td_convlstm3 = self.td_convlstm3(F.upsample(convlstm3_hidden0[0], scale_factor=2, mode='nearest')) # The linear size goes from 12 to 24.
        # We need to pad td_convlstm3 so that it has the linear size 25, not 24.
        td_convlstm3 = F.pad(td_convlstm3, (0,1,0,1), mode='constant', value=0)

        td_convlstm2 = self.td_convlstm2(F.upsample(convlstm2_hidden0[0], scale_factor=2, mode='nearest')) # The linear size goes from 25 to 50.

        convlstm1_h1, convlstm1_c1 = self.convlstm1(torch.cat([x, td_convlstm2], 1), convlstm1_hidden0) # input x: C=input_channels, H,W = input_size = 50.
        h = self.relu(self.pool1(convlstm1_h1))
        convlstm2_h1, convlstm2_c1 = self.convlstm2(torch.cat([h, td_convlstm3], 1), convlstm2_hidden0) # input h: C=hidden1, H,W = int(input_size/2) = 25.
        h = self.relu(self.pool2(convlstm2_h1))
        convlstm3_h1, convlstm3_c1 = self.convlstm3(torch.cat([h, td_lstm], 1), convlstm3_hidden0) # input h: C=hidden2, H,W = int(input_size/4) = 12.
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
            if i == self.n_iter - 2:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)
                if not self.reverse:
                    scores1 = scores
                else:
                    scores2 = scores
            elif i == self.n_iter - 1:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)
                if not self.reverse:
                    scores2 = scores 
                else:
                    scores1 = scores 
            else:
                convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(False, x, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)


        return scores1, scores2



class TDRCMultiObjectClassifierOld(nn.Module):
    def __init__(self, reverse=False, n_iter=2, input_size=50, input_channels=1, hidden1=64, hidden2=64, hidden3=64, hidden_lstm=512, n_classes=10, dropout_p=0.5):
        super(TDRCMultiObjectClassifierOld, self).__init__()
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

        # TD layers
        self.td_fc1 = nn.Linear(hidden_lstm, int(input_size/8)**2*hidden3)
        self.td_conv1 = nn.ConvTranspose2d(in_channels=hidden3, out_channels=hidden2, kernel_size=4, stride=2, padding=1)
        self.td_conv2 = nn.ConvTranspose2d(in_channels=hidden2, out_channels=hidden1, kernel_size=4, stride=2, padding=1, output_padding=1) # output_padding=1 to increase the linear size from 12*2 to 25.
        self.td_conv3 = nn.ConvTranspose2d(in_channels=hidden1, out_channels=input_channels, kernel_size=4, stride=2, padding=1)

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

        # TD layers. Kaiming normal. But, due to the non-unit stride, I had to do this by hand.
        init.normal(self.td_fc1.weight, 0.0, math.sqrt(2.0/self.hidden_lstm))
        init.constant(self.td_fc1.bias, 0.0)

        init.normal(self.td_conv1.weight, math.sqrt(2.0/(self.hidden3*((4/2)**2))))
        init.constant(self.td_conv1.bias, 0.0)

        init.normal(self.td_conv2.weight, math.sqrt(2.0/(self.hidden2*((4/2)**2))))
        init.constant(self.td_conv2.bias, 0.0)

        init.normal(self.td_conv3.weight, math.sqrt(2.0/(self.hidden1*((4/2)**2))))
        init.constant(self.td_conv3.bias, 0.0)



    def cell(self, predict, x, z, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0):
        # z is the top-level representation (i.e. lstm_h1) from the previous iteration, with size hidden_lstm.
        # z is None for the initial iteration.
        if z is not None:
            m3 = self.relu(self.td_fc1(z))
            m2 = self.relu(self.td_conv1(m3.view(-1, self.hidden3, int(self.input_size/8), int(self.input_size/8)))) # m2: C=hidden2, H, W = 12.
            m1 = self.relu(self.td_conv2(m2))
            m0 = self.relu(self.td_conv3(m1))
            convlstm1_h1, convlstm1_c1 = self.convlstm1(x*m0, convlstm1_hidden0) # input x: C=input_channels, H,W = input_size = 50.
            h = self.relu(self.pool1(convlstm1_h1))
            convlstm2_h1, convlstm2_c1 = self.convlstm2(h*m1, convlstm2_hidden0) # input h: C=hidden1, H,W = int(input_size/2) = 25.
            h = self.relu(self.pool2(convlstm2_h1))
            convlstm3_h1, convlstm3_c1 = self.convlstm3(h*m2, convlstm3_hidden0) # input h: C=hidden2, H,W = int(input_size/4) = 12.
            h = self.relu(self.pool3(convlstm3_h1)).view(-1, int(self.input_size/8)**2*self.hidden3)
            h = self.dropout(h*m3)
        else:
            convlstm1_h1, convlstm1_c1 = self.convlstm1(x, convlstm1_hidden0) # input x: C=input_channels, H,W = input_size = 50.
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

        z = None
        for i in range(self.n_iter):
            if i == self.n_iter - 2:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, z, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)
                if not self.reverse:
                    scores1 = scores
                else:
                    scores2 = scores
            elif i == self.n_iter - 1:
                scores, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(True, x, z, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)
                if not self.reverse:
                    scores2 = scores 
                else:
                    scores1 = scores 
            else:
                convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0 = self.cell(False, x, z, convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0)
            z = lstm_hidden0[0]


        return scores1, scores2


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
        (ai, af, ag, ao) = torch.chunk(A, 4, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        g = torch.tanh(ag)
        o = torch.sigmoid(ao)

        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c