from torch import nn
import torch.nn.functional as F

import matplotlib.pylab as plt
import os
import json
import torch,torchvision
from torchvision import transforms
from torch import nn,optim
from torch.autograd import Variable
import itertools


class Classifier(nn.Module):
    def __init__(self,latentDimension=1,ngpu=1,batch_size=128):
        super(Classifier, self).__init__()
        self.ngpu , self.batch_size= ngpu,batch_size
        self.latentDimension=latentDimension

        self.model=nn.Sequential(
            nn.Linear(self.latentDimension,self.latentDimension),
            nn.ReLU(),
            nn.Linear(self.latentDimension, 10),
            nn.Softmax()
        )

    def forward(self, input):
        return self.model(input)


class ConvEncoder(nn.Module):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self,latentDimension=1,channels=1):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, latentDimension)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        #return F.softmax(x)


class ConvDecoder(nn.Module):
    def __init__(self,latentDimension=1,channels=1):
        super(ConvDecoder, self).__init__()
        self.latentDimension,self.channels=latentDimension,channels
        ngf=1
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(1024, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 3, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            #nn.ConvTranspose2d(    ngf,outChannels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.fc1=nn.Sequential(
            nn.Linear(self.latentDimension,512),
            nn.LeakyReLU(0.01,inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.01, inplace=True)
        )


    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        #dimension is [batch size, hight, width]

        return self.model(self.fc1(input.view(input.shape[0],-1)).view(input.shape[0],1024,1,1)).view(input.shape[0],self.channels,28,28)


class Discriminator(nn.Module):
    def __init__(self,latentDimension=1,ngpu=1,batch_size=128):
        super(Discriminator, self).__init__()
        self.ngpu , self.batch_size= ngpu,batch_size
        self.latentDimension=latentDimension


        self.model=nn.Sequential(
            nn.Linear(self.latentDimension,self.latentDimension),
            nn.Linear(self.latentDimension, self.latentDimension//4),
            nn.Linear(self.latentDimension//4, 1),
            nn.Sigmoid()
        )


    def forward(self, input):
        return self.model(input)
