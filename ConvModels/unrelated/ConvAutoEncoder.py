import torch,torchvision
from torchvision import transforms
from torch import nn,optim
import itertools
from torch import nn
#from torch.nn.functional import mse_loss
import matplotlib.pylab as plt
from torch.nn import functional as F

#%% model experimentation part
class ConvEncoder(nn.Module):
    def __init__(self,latentDimension=1,channels=1):
        super(ConvEncoder, self).__init__()
        self.latentDimension= latentDimension
        ndf=2
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
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
            # nn.Conv2d(ndf * 4, ndf * 8, 1, 1, 0, bias=False),
            #nn.Linear(2,1),
            #nn.BatchNorm2d(5),
            #nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc1=nn.Sequential(
            nn.Linear(32, 512),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(512,latentDimension),
            nn.LeakyReLU(0.01,inplace=True)
        )

    def forward(self, input):
        return self.fc1(self.model(input).view(input.shape[0],-1)).view(input.shape[0],self.latentDimension,1,1)


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
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 3, bias=False),
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



#%%
latentD=32
batchSize=128
torch.backends.cudnn.enabled=False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#define the models
cEnc=ConvEncoder(latentDimension=latentD).to(device)
cDec=ConvDecoder(latentDimension=latentD).to(device)

#define optimizer
optimizer = optim.Adam(itertools.chain(cEnc.parameters(),cDec.parameters()), lr=5e-3)
lossData=[]

#%%

#loop and train
i=0
MSELoss=nn.MSELoss().to(device)

for passes in range(3):
    # defining the data loaders
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=False,
                                                                          transform=transforms.Compose(
                                                                              [transforms.ToTensor()])),
                                               batch_size=batchSize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=False,
                                                                         transform=transforms.Compose(
                                                                             [transforms.ToTensor()])),
                                              batch_size=batchSize, shuffle=True)

    for batchIdx,(batchData,batchLabels) in enumerate(train_loader):
        i+=1
        #encode and decode
        batchData=batchData.to(device)
        batchLabels=batchLabels.to(device)

        encodedDataBatch=cEnc(batchData)
        #print(encodedDataBatch.shape)
        decodedData=cDec(encodedDataBatch)
        #
        #optimize parameters: squared loss for now
        optimizer.zero_grad()
        loss=F.l1_loss(decodedData,batchData) #MSELoss(decodedData,batchData)
        loss.backward()
        optimizer.step()

        lossData.append(loss.data.item())


    print("finished pass {} ...".format(passes))

#%% plot the loss
plt.plot(lossData)
plt.show()

#%% plot sample reconstructions
j=10
for i in range(j,j+20):
    plt.figure()
    plt.subplot(121)
    plt.title("original")
    plt.imshow(batchData[i].view(28,28).cpu().numpy())
    #plt.show()
    #generated=cEnc(cDec(decodedData[i]))
    plt.subplot(122)
    plt.title("generated")
    plt.imshow(decodedData[i].cpu().view(28,28).detach().numpy())

    plt.show()



