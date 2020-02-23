import torch
from torch import nn,optim
from torch.autograd import Variable
import itertools
from models import encoder,decoder,discriminator,loss_functions
import helpers

import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# This is a simple mixture model with two means that we are optimizing together with
# with KL divergences
#
#
#%% setting up parameters

batch_size,dim=200,2
Enc=encoder(dim=dim,k=2,batch_size=batch_size)
Dec=decoder(dim=dim,k=2,batch_size=batch_size)
Disc=discriminator(dim=dim,k=2,batch_size=batch_size)
losses_=loss_functions()
dataHandler=helpers.data_and_plotting(batch_size,encoder=Enc,decoder=Dec,discriminator=Disc,mixture=True,\
                            semi_circle=False)


#%% mixture parameters
# self.z = Variable(torch.FloatTensor(torch.rand(1, self.k)), requires_grad=True);

mu_s=list([])
logvar_s=list([])

mu1 = Variable(torch.randn(1), requires_grad=True);
logvar1 = Variable(torch.FloatTensor([0]), requires_grad=False);

mu2 = Variable(torch.randn(1), requires_grad=True);
logvar2 = Variable(torch.FloatTensor([0]), requires_grad=False);


print(mu1,mu2)
#%% setting up the optimizers

#generator optimizer
optimizerED = optim.Adam(itertools.chain(Enc.parameters(),Dec.parameters()), lr=5e-4)

#discriminator optimizer
optimizerMixture = optim.Adam([mu1,mu2], lr=5e-4)

#%% training loop

for i in range(5000):


    #autoencoder training
    optimizerED.zero_grad()


    #take a sample
    x=dataHandler.get_data_sample()
    mu, logvar,z= Enc.forward(x)
    x_hat=Dec(z)

    #reconstruction optimization
    recon_loss=losses_.reconstruction_error_l1(x,x_hat)
    KL1 = 0.5 * (losses_.KL_Gaussian(mu, logvar, mu1, logvar1,batch_size=batch_size) \
                 + losses_.KL_Gaussian(mu, logvar, mu2, logvar2,batch_size=batch_size))
    tot_loss=recon_loss+KL1;
    enc_loss=tot_loss.item()

    tot_loss.backward()
    optimizerED.step()

    #prior optimization
    #calculate the KL with every mixture
    optimizerMixture.zero_grad()

    #take a sample
    x = dataHandler.get_data_sample()
    mu, logvar, z = Enc.forward(x)

    KL1=0.5*(losses_.KL_Gaussian(mu,logvar,mu1,logvar1,batch_size=batch_size)+losses_.KL_Gaussian(mu,logvar,mu2,logvar2,batch_size=batch_size))
    print("i {} ==> {} mixture loss, {} encoder loss ".format(i,KL1.item(),enc_loss))
    KL1.backward()

    #optimizerMixture.step()



 #%% sample plot
dataHandler.plot_sample_raw()
#%%
dataHandler.plot_reconstruction(generative=True)
dataHandler.plot_reconstruction(generative=False)
#%%
dataHandler.plot_embhist()



