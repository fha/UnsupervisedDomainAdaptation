import torch,torchvision
from torchvision import transforms
from torch import nn,optim
from torch.autograd import Variable
import itertools
from torch import nn
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


class Discriminator(nn.Module):
    def __init__(self,latentDimension=1,ngpu=1,batch_size=128):
        super(Discriminator, self).__init__()
        self.ngpu , self.batch_size= ngpu,batch_size
        self.latentDimension=latentDimension

        self.model=nn.Sequential(
            nn.Linear(self.latentDimension,512),
            nn.LeakyReLU(0.01,inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )


    def forward(self, input):
        return self.model(input)

#%%
latentD=32
batchSize=64
torch.backends.cudnn.enabled=False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#define the models
cEnc=ConvEncoder(latentDimension=latentD).to(device)
cDec=ConvDecoder(latentDimension=latentD).to(device)
disc=Discriminator(latentDimension=latentD).to(device)


#define optimizer
optimizer_a = optim.Adam(itertools.chain(cEnc.parameters(),cDec.parameters()), lr=5e-4)
optimizer_b = optim.Adam(itertools.chain(disc.parameters()), lr=1e-5)
lossData_a,lossData_b=[],[]

#%%
#loop and train

MSELoss=nn.MSELoss().to(device)
BCELoss=nn.BCELoss().to(device)

for passes in range(3):
    # defining the data loaders
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('~/data/UDA_related', train=True, download=False,
                                                                          transform=transforms.Compose(
                                                                              [transforms.ToTensor()])),
                                               batch_size=batchSize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('~/data/UDA_related', train=False, download=False,
                                                                         transform=transforms.Compose(
                                                                             [transforms.ToTensor()])),
                                              batch_size=batchSize, shuffle=True)

    _ones = torch.ones([batchSize,1]).to(device)
    _zeros = torch.zeros([batchSize,1]).to(device)

    ones = Variable(torch.FloatTensor(batchSize, 1).fill_(1.0), requires_grad=False).to(device)
    zeros = Variable(torch.FloatTensor(batchSize, 1).fill_(0.0), requires_grad=False).to(device)
    for batchIdx,(batchData,batchLabels) in enumerate(train_loader):

        print(batchIdx)

        #data setup
        _batchSize=batchData.shape[0]
        batchData=batchData.to(device)
        batchLabels=batchLabels.to(device)



        #convAutoencoder Pass
        encodedDataBatch=cEnc(batchData)
        decodedData=cDec(encodedDataBatch)

        #optimize parameters for encoder/decoder: squared loss for now
        optimizer_a.zero_grad()
        loss_a=F.l1_loss(decodedData,batchData) \
               + 0.001*BCELoss(disc(encodedDataBatch.view([_batchSize,-1])).view([_batchSize,1]),_ones[:_batchSize])
        loss_a.backward()
        optimizer_a.step()
        lossData_a.append(loss_a.data.item())


        #train the discriminator, 1/0 is real/fake data
        #sample=getPriorSamples(_batchSize)

        optimizer_b.zero_grad()

        latentSpaceSample=torch.randn([_batchSize,latentD]).to(device)
        discDataInput=Variable(encodedDataBatch.view(_batchSize,latentD).cpu().data,requires_grad=False).to(device)
        loss_b=BCELoss(disc(discDataInput).view([_batchSize,-1]),_zeros[:_batchSize])+\
             BCELoss(disc(latentSpaceSample),_ones[:_batchSize])

        loss_b.backward()
        optimizer_b.step()
        lossData_b.append(loss_b.data.item())

    print("finished pass {} ...".format(passes))

#%% plot the loss_a
plt.plot(lossData_a)
plt.show()

#%% plot the loss_b
plt.plot(lossData_b)
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



