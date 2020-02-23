import torch
from torch import nn,optim
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import itertools

class ConvDecoder(nn.Module):
    def __init__(self, in_channels=1,out_channels=1,latent_dim=1,ngpu=1,batch_size=128):
        super(ConvDecoder, self).__init__()
        self.ngpu , self.batch_size= ngpu,batch_size
        ngf,nc,nz=out_channels,in_channels,latent_dim
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
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
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        return self.model(input)
    

    

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=1,out_channels=1,ngpu=1,batch_size=128):
        super(ConvEncoder, self).__init__()
        self.ngpu,self.batch_size= ngpu, batch_size
        nc,ndf=in_channels,out_channels
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 4, 2, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 1, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.model(input).view(-1, 1).squeeze(1)
    
class encoder(nn.Module):
    def __init__(self,dim=2,k=2,batch_size=128):
        super(encoder, self).__init__()
        #define some variables
        self.k, self.dim,self.batch_size = k, dim, batch_size

        #define the model
        self.encode=nn.Sequential(
            nn.Linear(self.dim, 4),
            nn.ReLU(),
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        self.mu=nn.Linear(8,1)
        self.logvar = nn.Linear(8, 1)

    def forward(self,x):
        h=self.encode(x)
        mu=self.mu(h)
        logvar=self.logvar(h)
        z=mu+torch.exp(logvar)*torch.randn((self.batch_size,1))
        return  mu, logvar,z

    
class decoder(nn.Module):
    def __init__(self,dim=2,k=2,batch_size=128):
        super(decoder, self).__init__()
        #define some variables
        self.k, self.dim,self.batch_size = k, dim, batch_size

        #define the model
        self.decode=nn.Sequential(
        nn.Linear(1, 4),
        nn.ReLU(),
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
        )

    def forward(self,z):
        x_hat=self.decode(z)
        return x_hat

class discriminator(nn.Module):
    def __init__(self,dim=2,k=2,batch_size=128):
        #This is vanilla VAE
        super(discriminator, self).__init__()
        self.k, self.dim,self.batch_size = k, dim, batch_size

        #discriminator model
        self.discriminate=nn.Sequential(
        nn.Linear(1, 4),
        nn.ReLU(),
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid())

    def forward(self,z):
        return self.discriminate(z)



## classes for mixtures
#
# class mixture_encoder(nn.Module):
#     def __init__(self,dim=2,k=2,batch_size=128):
#         super(encoder, self).__init__()
#         #define some variables
#         self.k, self.dim,self.batch_size = k, dim, batch_size
#
#         #define the model
#         self.encode=nn.Sequential(
#         nn.Linear(self.dim, 4),
#         nn.ReLU(),
#         nn.Linear(4, 8,bias=True),
#         nn.ReLU()
#         )
#
#         self.mu=nn.Linear(8,1)
#         self.logvar = nn.Linear(8, 1)
#         self.pi = nn.Softmax(nn.Linear(8, 2))
#
#     def forward(self,x):
#         h=self.encode(x)
#         mu=self.mu(h)
#         logvar=self.logvar(h)
#         z=mu+torch.exp(logvar)*torch.randn((self.batch_size,1))
#         pi=self.pi(h)
#         return  mu, logvar,z,pi
#
#
# class decoder(nn.Module):
#     def __init__(self,dim=2,k=2,batch_size=128):
#         super(decoder, self).__init__()
#         #define some variables
#         self.k, self.dim,self.batch_size = k, dim, batch_size
#
#         #define the model
#         self.decode=nn.Sequential(
#         nn.Linear(1, 4),
#         nn.ReLU(),
#         nn.Linear(4, 8),
#         nn.ReLU(),
#         nn.Linear(8, 2)
#         )
#
#     def forward(self,z):
#         x_hat=self.decode(z)
#         return x_hat
#
# class discriminator(nn.Module):
#     def __init__(self,dim=2,k=2,batch_size=128):
#         #This is vanilla VAE
#         super(discriminator, self).__init__()
#         self.k, self.dim,self.batch_size = k, dim, batch_size
#
#         #discriminator model
#         self.discriminate=nn.Sequential(
#         nn.Linear(1, 8,bias=True),
#         nn.ReLU(),
#         nn.Linear(8, 16, bias=True),
#         nn.ReLU(),
#         nn.Linear(16, 1,bias=True),
#         nn.Sigmoid())
#
#     def forward(self,z):
#         return self.discriminate(z)




#classes for loss functions

class loss_functions:
    def __init__(self):
        return

    def reconstruction_error_l2(self,x,x_hat):
        celoss=F.mse_loss(x, x_hat)
        return celoss

    def reconstruction_error_l1(self,x,x_hat):
        celoss=F.l1_loss(x, x_hat)
        return celoss

    def categorical_loss_function(qy, device, categorical_dim=2):
        # first try without a KL divergence term
        log_qy = torch.log(qy + 1e-20)
        g = Variable(torch.log(torch.Tensor([1.0 / categorical_dim]))).to(device)
        catloss = torch.sum(qy * (log_qy - g), dim=-1).mean()
        return catloss

    def KL_Gaussian(self,mu_p, logvar_p, mu_q, logvar_q, batch_size=128):
        first_term = -0.5 + ((logvar_p.exp() + torch.mul(mu_p - mu_q, mu_p - mu_q)) / (2 * logvar_q.exp()));
        second_term = torch.log(torch.sqrt(logvar_q.exp() / logvar_p.exp()));
        KLD_mus = torch.mean(first_term + second_term) ;

        return KLD_mus;

# class reparametrizations:
#     def __init__(self):
#         return
#
#
#     def gumbel_softmax(self,logits, temperature,categorical_dim=2,latent_dim=1):
#         """
#         ST-gumple-softmax
#         input: [*, n_class]
#         return: flatten --> [*, n_class] an one-hot vector
#         """
#         y = self.gumbel_softmax_sample(self,logits, temperature)
#         shape = y.size()
#         _, ind = y.max(dim=-1)
#         y_hard = torch.zeros_like(y).view(-1, shape[-1])
#         y_hard.scatter_(1, ind.view(-1, 1), 1)
#         y_hard = y_hard.view(*shape)
#         y_hard = (y_hard - y).detach() + y
#         return y_hard.view(-1, latent_dim * categorical_dim)
#
#     def gumbel_softmax_sample(self,logits, temperature):
#         y = logits + self.sample_gumbel(logits.size())
#         return F.softmax(y / temperature, dim=-1)
#
#     def sample_gumbel(self,shape, eps=1e-20):
#         U = torch.rand(shape).to(self.device)
#         return -Variable(torch.log(-torch.log(U + eps) + eps))
