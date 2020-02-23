import torch
from torch import nn,optim
from torch.autograd import Variable
import itertools
from models import encoder,decoder,discriminator,loss_functions
import helpers

import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F

#%% setting up parameters

batch_size,dim=750,2

Enc=encoder(dim=dim,k=2,batch_size=batch_size)
Dec=decoder(dim=dim,k=2,batch_size=batch_size)
Disc=discriminator(dim=dim,k=2,batch_size=batch_size)
losses_=loss_functions()
dataHandler=helpers.data_and_plotting(batch_size,encoder=Enc,decoder=Dec,discriminator=Disc,mixture=False,\
                            semi_circle=True)


#%% setting up the optimizers

#generator optimizer
optimizerE = optim.Adam(itertools.chain(Enc.parameters(),Dec.parameters()), lr=5e-4)

#discriminator optimizer
optimizerD = optim.Adam(itertools.chain(Disc.parameters()), lr=5e-4)

ones = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
zeros = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

adverserial_loss=nn.BCELoss()

g_loss=[]
d_loss=[]
#%% training loop
err=1000
for i in range(2000):


    #autoencoder training
    optimizerE.zero_grad()
    x=dataHandler.get_data_sample()
    mu, logvar,z= Enc.forward(x)
    x_hat=Dec(z)

    #computer loss and optimization step
    encoder_loss=losses_.reconstruction_error_l2(x,x_hat)+adverserial_loss(Disc(z),ones)
    encoder_loss.backward()
    optimizerE.step()


    #D training
    optimizerD.zero_grad()

    real_sample=Disc(Variable(torch.randn((batch_size,1)),requires_grad=False))

    #computer loss and optimization step
    mu, logvar, z = Enc.forward(x)
    D_loss=0.5*(adverserial_loss(real_sample,ones)+adverserial_loss(Disc(z),zeros))
    D_loss.backward()
    #if i>2000:
    optimizerD.step()


    print("iter {} G , D error is {} , {}".format(i,encoder_loss.item(),D_loss.item()))
    g_loss.append(encoder_loss.item())
    d_loss.append(D_loss.item())



#%% sample plot
dataHandler.plot_sample_raw()
#%%
dataHandler.plot_reconstruction(generative=True)
dataHandler.plot_reconstruction(generative=False)
#%%
dataHandler.plot_embhist()
#%%
plt.figure()
plt.plot(g_loss)
plt.show()


#%%
plt.figure()
plt.plot(d_loss)
plt.show()



