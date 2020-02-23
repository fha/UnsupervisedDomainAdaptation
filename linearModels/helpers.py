#this is a class for generating data and plotting
import torch
import matplotlib.pyplot as plt
import numpy as np


from torch import nn,optim
from torch.nn import functional as F
from torch.autograd import Variable

class data_generator:
    def __init__(self,batch_size=1000,n_components=1,encoder=None,decoder=None,discriminator=None,distribution='Mixture'):
        super(data_generator,self)
        self.batch_size=batch_size
        self.encoder=encoder
        self.decoder=decoder
        # self.mixture=mixture
        # self.semi_circle_data=semi_circle
        self.data_distrib=distribution
        self.n_components=n_components
        self.mu,self.cov=[],[]
        if self.data_distrib=="Mixture":
            for i in range(n_components):
                self.mu.append(torch.FloatTensor([np.random.randn(), np.random.randn()]) * 4)
                cov11, cov22 = np.abs(np.random.uniform(-0.5, 0.5)), np.abs(np.random.uniform(-0.5, 0.5))
                cov12 = np.random.uniform(-min(cov11, cov22) / 2, min(cov11, cov22) / 2)
                self.cov.append(torch.FloatTensor([[np.abs(cov11), cov12], [cov12, np.abs(cov22)]]))

    def get_data_sample(self):
        # sample training data

        # correlated gaussian
        if self.data_distrib=="Fixed-mixture":
            mu1 = torch.FloatTensor([0, 0])
            cov1 = torch.FloatTensor([[4, -3], [-3, 4]])
            logvar1 = torch.log(cov1)

            mu2 = torch.FloatTensor([20, 20])
            cov2 = torch.FloatTensor([[4, 3], [3, 4]])
            logvar2 = torch.log(cov1)

            x = self.get_multivariate_normal_samples(mu1, cov1, self.batch_size // 2)
            y = self.get_multivariate_normal_samples(mu2, cov2, self.batch_size // 2)
            x = torch.cat((torch.FloatTensor(x), torch.FloatTensor(y)), 0)

        elif self.data_distrib == "Mixture":
            first = True;
            for i in range(0, self.n_components):
                mu_ = self.mu[i]
                #cov11, cov22 = np.abs(np.random.uniform(-0.5, 0.5)), np.abs(np.random.uniform(-0.5, 0.5))
                #cov12 = np.random.uniform(-min(cov11, cov22) / 2, min(cov11, cov22) / 2)
                cov_ = self.cov[i]

                multvariate_sample = self.get_multivariate_normal_samples(mu_, cov_, self.batch_size // self.n_components)
                mv_s = multvariate_sample.data.numpy()
                if first:
                    x = multvariate_sample
                    first = False
                else:
                    x = torch.cat((x, multvariate_sample), 0)
        # semi-circle
        elif self.data_distrib=="Semi-circle":
            x_ = np.random.uniform(-1, 1, self.batch_size)
            y_ = np.sqrt(1 - x_ * x_) + np.random.normal(0, 0.1, len(x_))
            x = torch.cat((torch.FloatTensor(x_).view(-1, 1), torch.FloatTensor(y_).view(-1, 1)), 1)

        # circle
        elif self.data_distrib=="Circle":
            eps1 = np.random.normal(0, 0.1, self.batch_size)
            eps2 = np.random.normal(0, 0.1, self.batch_size)

            thetas = np.random.uniform(0, 2 * np.pi, self.batch_size)
            x = torch.cat((torch.FloatTensor(np.cos(thetas) + eps1).view(-1, 1),
                           torch.FloatTensor(np.sin(thetas) + eps2).view(-1, 1)), 1)  # .view(-1,2)


        # single correlated gaussian
        elif self.data_distrib=="Gaussian":
            mu = torch.FloatTensor([0, 0])
            cov = torch.FloatTensor([[2, 1], [1, 2]])
            logvar = torch.log(cov)
            x = self.get_multivariate_normal_samples(mu, logvar.exp(), self.batch_size)

        else:
            print("couldn't read a distribution type")
            return

        return x



    def get_multivariate_normal_samples(self,mu,cov,batch):
        x = torch.distributions.multivariate_normal.MultivariateNormal(mu, cov).sample([batch])
        return x


    def plot_sample_raw(self,show=True):
        x=self.get_data_sample()
        plot=plt.figure()
        y=x.data.numpy()
        plt.plot(y[:,0],y[:,1],'*')
        plt.axis('equal')
        if show:
            plt.show()


    def plot_porjection(self):
        x=self.get_data_sample()
        mu, logvar, z = self.encoder(x)
        x_hat = self.decoder(z)
        y = x.data.numpy()
        plt.plot(y[:, 0], y[:, 1], '.r')
        y = x_hat.data.numpy()
        plt.plot(y[:, 0], y[:, 1], '*')
        plt.show()

    def plot_z_hist(self):
        x=self.get_data_sample()
        mu, logvar, z = self.encoder(x)
        emb = z.data.numpy()
        print(emb)
        plt.hist(emb)
        plt.show()

    def plot_reconstruction(self,z,over_sample=True):
        if over_sample:
            self.plot_sample_raw(show=False)

        x_hat=self.decoder(z);
        x_hat=x_hat.data.numpy()
        plt.plot(x_hat[:, 0], x_hat[:, 1], '.r')
        plt.show()



