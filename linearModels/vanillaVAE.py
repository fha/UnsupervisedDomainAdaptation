# This is the vanilla VAE
import torch
from torch import nn,optim
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from models import encoder,decoder,discriminator,loss_functions


#%% get the required components
batchSize=100
encode=encoder(batch_size=batchZ)
decode=decoder(batch_size=batchZ)
losses=loss_functions()
#%%
optimizerEnc = optim.Adam(encoder.parameters(), lr=1e-4)
optimizerDec = optim.Adam(encoder.parameters(), lr=1e-4)

for i in range(50000):
    optimizerEnc.zero_grad()
    mu, logvar,z,x_hat = encoder(x)
    KL_loss=losses.KL_Gaussian( mu, logvar,torch.zeros(batchSize),torch.zeros(batchSize),batch_size=batchSize)
    mse_error=losses.reconstruction_error(x,x_hat)
    print(KL_loss.item(),mse_error.item())
    mse_error.backward()
    optimizerEnc.step()


#%% constructing semi-circle data
plt.figure()
x=np.random.uniform(-1,1,1000)
y=np.sqrt(1-x*x)+np.random.normal(0,0.1,len(x))

plt.plot(x,y,'*')
plt.show()
