import torch
from torch import nn,optim
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

class simpleVAE(nn.Module):
    def __init__(self,device,dim=2,k=2,batch_size=128):
        #super
        super(simpleVAE, self).__init__()
        self.device=device



        # initialize the mu
        self.k = 2;
        self.dim = 2;
        self.batch_size=batch_size;

        #self.z = Variable(torch.FloatTensor(torch.rand(1, self.k)), requires_grad=True);
        self.mu1 = torch.nn.Parameter(torch.FloatTensor(torch.randn(1)), requires_grad=True).to(self.device);
        self.logvar1 = torch.nn.Parameter(torch.FloatTensor([0]), requires_grad=True).to(self.device);

        self.mu2 = torch.nn.Parameter(torch.FloatTensor(torch.randn(1)), requires_grad=True).to(self.device);
        self.logvar2 = torch.nn.Parameter(torch.FloatTensor([0]), requires_grad=True).to(self.device);

        #encoder layers
        self.encoder=nn.Sequential(
        nn.Linear(self.dim, 2),
        nn.ReLU(),
        nn.Linear(2, 2),
        nn.Softmax()
        )

        #decoder layers
        self.decoder=nn.Sequential(
        nn.Linear(1, 10),
        nn.ReLU(),
        nn.Linear(10, 2)
        )


    def forward(self,x,temp):
        qy=self.encoder(x)
        #print(self.mu1)
        #print(self.logvar1)
        #print(self.mu2)
        #print(self.logvar2)
        #print(qy)
        #VAE_model.mu1 + VAE_model.logvar1.exp() * torch.randn(100)
        sample1=self.mu1+self.logvar1.exp()*torch.randn(self.batch_size).to(self.device)
        sample2=self.mu2+self.logvar2.exp()*torch.randn(self.batch_size).to(self.device)
        mus=torch.cat((sample1.view(self.batch_size,-1),sample2.view(self.batch_size,-1)),1).to(self.device)
        z = self.gumbel_softmax(qy, temp).to(device)
        combined_sample=torch.sum(z*mus,1).view(self.batch_size,-1)
        #print("size {} {}".format(qy.size(),mus.size()))
        x_hat=self.decoder(combined_sample)
        return  sample1,sample2,x_hat,qy

    

    def gumbel_softmax(self,logits, temperature,categorical_dim=2,latent_dim=1):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(-1, latent_dim * categorical_dim)

    def gumbel_softmax_sample(self,logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def sample_gumbel(self,shape, eps=1e-20):
        U = torch.rand(shape).to(self.device)
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def freeze_mixture(self):
        self.mu1.requires_grad=False
        self.mu2.requires_grad = False
        self.logvar1.requires_grad = False
        self.logvar2.requires_grad = False

    def unfreeze_mixture(self):
        self.mu1.requires_grad=True
        self.mu2.requires_grad = True
        self.logvar1.requires_grad = True
        self.logvar2.requires_grad = True


def KL_Gaussian(mu_p, logvar_p, mu_q, logvar_q, batch_size=128):
    first_term = -0.5 + ((logvar_p.exp() + torch.mul(mu_p - mu_q, mu_p - mu_q)) / (2 * logvar_q.exp()));
    second_term = torch.log(torch.sqrt(logvar_q.exp() / logvar_p.exp()));
    KLD_mus = torch.sum(first_term + second_term) / (batch_size );

    return KLD_mus;

def reconstruction_error(x,x_hat):
    #cross-entropy loss
    #celoss=F.l1_loss(x.view(-1,), x.view(-1, ))
    #mse loss
    celoss=F.mse_loss(x, x_hat)
    return celoss


def categorical_loss_function(qy,device,categorical_dim=2):
    #first try without a KL divergence term
    log_qy = torch.log(qy+1e-20)
    g = Variable(torch.log(torch.Tensor([1.0/categorical_dim]))).to(device)
    catloss = torch.sum(qy*(log_qy - g),dim=-1).mean()
    return catloss



def get_multivariate_normal_samples(mu,cov,batch):
    x = torch.distributions.multivariate_normal.MultivariateNormal(mu, cov).sample([batch])
    return x




#def get_mixture_mutlivariate_normal_samples(mu1,mu2,)

#%% select cpu or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#%%
mixture=True
#sample training data
#get a normally random sample
batch,dim,epochs=10000,2,10000
#correlated covariance matrix

#multivariate x
if mixture:
    mu1 = torch.FloatTensor([3, 3])
    cov1 = torch.FloatTensor([[1, 0.5], [0.5, 1]])
    logvar1 = torch.log(cov1)

    mu2 = torch.FloatTensor([20, 20])
    cov2 = torch.FloatTensor([[2, -1.5], [-1.5, 2]])
    logvar2 = torch.log(cov1)

    x=get_multivariate_normal_samples(mu1,cov1,batch//2)
    y=get_multivariate_normal_samples(mu2,cov2,batch//2)
    x=torch.cat((x,y),0)

else:
    mu = torch.FloatTensor([10, 10])
    cov = torch.FloatTensor([[2, 0], [0, 5]])
    logvar = torch.log(cov)
    x=get_multivariate_normal_samples(mu,logvar.exp(),batch)

x=x[torch.randperm(x.size()[0]),:]
x=x.to(device)
#x[:,1]=x[:,0]
## defining the model and generating some random data
VAE_model=simpleVAE(device,batch_size=10000,dim=dim,k=2).to(device)


#%% plotting the random  data
plt.figure()
y=x.to("cpu").numpy()

plt.plot(y[:,0],y[:,1],'*')
plt.axis('equal')
plt.show()
#%% plotting the encoded data
sample1,sample2,x_hat,qy=VAE_model(x,temp=1)
y=x.to("cpu").numpy()
plt.plot(y[:,0],y[:,1],'.r')
y=x_hat.cpu().data.numpy()
plt.plot(y[:,0],y[:,1],'*')
plt.show()

#%% plot the distribution of the embeddings
emb=qy.cpu().data.numpy()
print(emb)
plt.hist(emb[:,1])
plt.hist(emb[:,0])
plt.show()

#%% this is for a model test
plt.plot(sorted(emb[:,0]))
plt.show()
plt.plot(sorted(emb[:,1]))
plt.show()


#%%
optimizer = optim.Adam(VAE_model.parameters(), lr=1e-3)
tmp=0.99
for i in range(10000):
    if i>500:
        tmp=tmp-0.001
        if tmp<0.1:
            tmp=0.1
    optimizer.zero_grad()
    sample1,sample2,x_hat,qy = VAE_model.forward(x,temp=tmp)

    #KL_loss=KL_Gaussian( mu, logvar,torch.zeros(batch),torch.zeros(batch),batch_size=batch)
    #if (i%2)==0:
    #    VAE_model.freeze_encoder()
    #    VAE_model.unfreeze_mixture()
    #else:
    #    VAE_model.freeze_mixture()
    #    VAE_model.unfreeze_encoder()

    #cat_loss=categorical_loss_function(qy,device)
    mse_error=reconstruction_error(x,x_hat)
    print("at iteration {} and losses are {} {}".format(i,0,mse_error.item()))
    #print(mse_error.item())

    loss=mse_error#+cat_loss
    loss.backward()
    optimizer.step()

#%%
for i in VAE_model.parameters():
    print(i.name,i)
