import torch,torchvision
from torchvision import transforms
from torch import nn,optim
from torch.autograd import Variable
import itertools
from torch import nn
import matplotlib.pylab as plt
import torch.nn.functional as F
import os
import json
from multiprocessing import Pool,set_start_method
import time
import torch.multiprocessing as mp


#%% model experimentation part

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

def validateModel(epochNum,options,models=[]):
    device =options["device"]
    totalCorrect,totalReal, total = 0, 0, 0
    _batchSize = 200
    #test_dataset = torchvision.datasets.MNIST(options['dataPath'], train=False, download=False,
    #                                          transform=transforms.Compose([
    #                                              transforms.ToTensor(),
    #                                              transforms.Normalize((0.1307,), (0.3081,))]))
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_batchSize)
    test_loader=options['testLoader']
    for batchIdx, (batchData, batchLabels) in enumerate(test_loader):
        batchData = batchData.to(device)
        batchLabels = batchLabels.to(device)

        encodedDataBatch = models[0](batchData)
        classesPrediction =models[1](encodedDataBatch)
        realOrFake=models[2](encodedDataBatch)

        predictedLabeles = classesPrediction.argmax(1)

        totalCorrect += float(torch.sum(predictedLabeles == batchLabels).data.cpu().numpy())
        totalReal+=float(torch.sum(realOrFake>=0.5).data.cpu().numpy())
        total += float(batchLabels.shape[0])

    ratio = totalCorrect / total
    line="{}, {}, {}, {}, {}, {}, {}, {}/{},{} \r\n".format(
        options['lrA'],options['lrB'],options["lrA_DiscCoeff"],options['latentD'],options['batchSize'],epochNum, ratio, totalCorrect, total,totalReal)
    print(line)
    return line

#%%
def getSampleFromLatentSpace(size,options):
    device = options['device']
    components=options['components']
    if components==1:
        return torch.randn([size[0], size[1]]).to(device)
    else:
        batchSize, latentD = size[0], size[1]
        mixDim = latentD // components
        sample = torch.zeros([batchSize, latentD])
        classes = torch.tensor([0]*batchSize)

        sample[:, :mixDim] = torch.randn([batchSize, mixDim]) * 2 + 10

        batchPerComponent = batchSize // components
        for i in range(batchSize // batchPerComponent):
            classes[i * batchPerComponent:(i + 1) * batchPerComponent] = i
            sample[i * batchPerComponent:(i + 1) * batchPerComponent, i * mixDim:(i + 1) * mixDim] = \
                sample[i * batchPerComponent:(i + 1) * batchPerComponent, 0:mixDim]
            if i != 0:
                sample[i * batchPerComponent:(i + 1) * batchPerComponent, 0:mixDim] = torch.zeros(
                    [batchPerComponent, mixDim])

        # shuffling..
        shuffle_idx = torch.randperm(sample.shape[0])
        sample = sample[shuffle_idx, :].to(device)
        classes = classes[shuffle_idx].to(device)

        return sample,classes

def getDataLoader(parameters):
    if parameters['dataName']=="mnist":
        if parameters['train']:
            train_dataset = torchvision.datasets.MNIST(parameters['dataPath'], train=True, download=False,
                                                       transform=transforms.Compose(
                                                           [transforms.ToTensor(),
                                                            transforms.Normalize((0.1307,), (0.3081,))]
                                                       ))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=parameters['batchSize'], shuffle=True)
            return train_loader
        else:
            test_dataset = torchvision.datasets.MNIST(parameters['dataPath'], train=False, download=False,
                                                      transform=transforms.Compose([
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.1307,), (0.3081,))]))
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=parameters['batchSize'])
            return test_loader





#%%

def TrainSourceModel(options):
    #defining the models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    options['device']=device
    cEnc = ConvEncoder(latentDimension=options['latentD']).to(device)
    #cDec = ConvDecoder(latentDimension=opts['latentD']).to(device)
    disc = Discriminator(latentDimension=options['latentD']).to(device)
    classify = Classifier(latentDimension=options['latentD']).to(device)

    # defining loss functions
    MSELoss = nn.MSELoss().to(device)  # mean squared error loss
    BCELoss = nn.BCELoss().to(device)  # binary cross-entropy
    NLLLoss = nn.NLLLoss().to(device)  # negative-log likelihood
    CELoss = nn.CrossEntropyLoss().to(device)  # cross entropy loss

    # loss log data
    lossData_a, lossData_b = [], []

    optimizer_a = optim.Adam(itertools.chain(cEnc.parameters(), classify.parameters()), lr=options['lrA'])
    optimizer_b = optim.Adam(itertools.chain(disc.parameters()), lr=options['lrB'])

    logText=""
    train_loader = options['trainLoader']

    for epochIdx in range(options['epochs']):

        # defining the data loaders
        #_ones = torch.ones([options['batchSize'], 1]).to(device)
        _ones = Variable(torch.FloatTensor(options['batchSize'], 1).fill_(1.0), requires_grad=False).to(device)

        #_zeros = torch.zeros([options['batchSize'], 1]).to(device)
        _zeros = Variable(torch.FloatTensor(options['batchSize'], 1).fill_(0.0), requires_grad=False).to(device)

        for batchIdx, (batchData, batchLabels) in enumerate(train_loader):

            # data setup
            _batchSize = batchData.shape[0]
            batchData = batchData.to(device)
            batchLabels = batchLabels.to(device)
            latentSpaceSample_A,latentSpaceClasses_A=getSampleFromLatentSpace([_batchSize,options['latentD']],options)

            # convAutoencoder Pass
            encodedDataBatch = cEnc(batchData)
            classesPrediction = classify(encodedDataBatch)

            #sample pass
            sampleClassesPrediction = classify(latentSpaceSample_A)

            # optimzation step I
            optimizer_a.zero_grad()
            #--- first loss function
            loss_a = CELoss(classesPrediction, batchLabels)+\
                     options["lrA_DiscCoeff"]*BCELoss(disc(encodedDataBatch),_ones[:_batchSize])+\
                     CELoss(sampleClassesPrediction,latentSpaceClasses_A)

            loss_a.backward()
            optimizer_a.step()
            lossData_a.append(loss_a.data.item())


            # optimization step II
            #---train the discriminator, 1/0 is real/fake data
            optimizer_b.zero_grad()
            latentSpaceSample_B,latentSpaceClasses_B=getSampleFromLatentSpace([_batchSize,options['latentD']],options)
            discDataInput=Variable(encodedDataBatch.view(_batchSize,options['latentD']).cpu().data,requires_grad=False).to(device)

            #---second loss function
            loss_b=BCELoss(disc(discDataInput),_zeros[:_batchSize])+\
                   BCELoss(disc(latentSpaceSample_B),_ones[:_batchSize])
            loss_b.backward()
            optimizer_b.step()
            lossData_b.append(loss_b.data.item())


            # print running accuracy on this batchData
            #predictedLabeles = classesPrediction.argmax(1)
            # print("{}/{} accuracy, and loss={}".format(float(torch.sum(predictedLabeles == batchLabels).data.cpu().numpy()),_batchSize,loss_a.data.item()))

        ####
        #### End of an epoch
        ####

        logText+=validateModel(epochIdx,options,models=[cEnc,classify,disc])
        # end of an epoch - CHECK ACCURACY ON TEST SET

    outputs={
        'lossA':lossData_a,
        'lossB':lossData_b,
        'encoder':cEnc,
        'disc':disc,
        'classifier':classify,
        'logText':logText
    }
    return outputs



#%%
def savePlots(optsInstance,outputs):
    plt.figure()
    plt.plot(outputs['lossA'])
    plt.savefig('{}/{}__lossA_lrA{}_lrB{}_latentD{}_discCoeff{}.png'.format(
        logPath['plots'],optsInstance["experimentID"],optsInstance['lrA'], optsInstance['lrB'], optsInstance['latentD'],optsInstance["lrA_DiscCoeff"]))
    plt.figure()
    plt.plot(outputs['lossB'])
    plt.savefig('{}/{}__lossB_lrA{}_lrB{}_latentD{}__discCoeff{}.png'.format(
        logPath['plots'],optsInstance["experimentID"],optsInstance['lrA'], optsInstance['lrB'], optsInstance['latentD'],optsInstance["lrA_DiscCoeff"]))


def saveModels(outputs,cfig):
    modelName="{}/{}_lrA{}_lrB{}_latentD{}_discCoeff{}".format(
        cfig["modelsPath"],cfig["experimentID"],cfig['lrA'], cfig['lrB'], cfig['latentD'],cfig["lrA_DiscCoeff"])
    torch.save(outputs["encoder"],"{}__encoder".format(modelName))
    torch.save(outputs["disc"],"{}__disc".format(modelName))
    torch.save(outputs["classifier"],"{}__classifier".format(modelName))

#%% if a log file is needed
#init run parameters


logfile=True
experimentID="checkRun.json"

logPath={"plots":os.path.expanduser("~/Dropbox (MIT)/temp_git/VAE_experiments/logfiles/plots"),
         "txt":os.path.expanduser("~/Dropbox (MIT)/temp_git/VAE_experiments/logfiles/txt")}

experimentConfigPath=os.path.expanduser("~/Dropbox (MIT)/temp_git/VAE_experiments/configFiles/"+experimentID)
if logfile:
    log=open('{}/{}__logfile.txt'.format(logPath['txt'],experimentID),"w")
with open(experimentConfigPath,'r') as tempFile:
    print(experimentConfigPath)
    opts = json.load(tempFile)

torch.backends.cudnn.enabled=False
# removed the line below for parallelization
#deviceVal = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#%% execute runs

startTime=time.time()
runsConfigsList=[]
if opts['search'] == 'gridSearch':
    for j in range(len(opts['batchSize'])):
        trainLoader = getDataLoader(
            parameters={'dataName': 'mnist', 'train': True, 'dataPath': opts['dataPath'],
                        'batchSize': opts['batchSize'][j]})
        testLoader = getDataLoader(
            parameters={'dataName': 'mnist', 'train': False, 'dataPath': opts['dataPath'],
                        'batchSize': opts['batchSize'][j]})
        for a in range(len(opts['lrA'])):
            for b in range(len(opts['lrB'])):
                for c in range(len(opts['lrA_DiscCoeff'])):
                        for k in range(len(opts['latentD'])):
                            optsInstance={"lrA":opts['lrA'][a],
                                          "lrB": opts['lrB'][b],
                                          'epochs':opts['epochs'],
                                          'batchSize': opts['batchSize'][j],
                                          'latentD': opts['latentD'][k],
                                          'dataPath':opts['dataPath'],
                                          "lrA_DiscCoeff":opts["lrA_DiscCoeff"][c],
                                          "components":10,
                                          "trainLoader":trainLoader,
                                          "testLoader":testLoader
                                          }

                            optsInstance["experimentID"] = experimentID
                            runsConfigsList.append(optsInstance)

elif opts['search'] == 'listSetup':
    for i in range(len(opts['configList'])):
        optsInstance = {"lrA": opts['configList'][i]['lrA'],
                        "lrB": opts['configList'][i]['lrB'],
                        'epochs': opts['configList'][i]['epochs'],
                        'batchSize': opts['configList'][i]['batchSize'],
                        'latentD': opts['configList'][i]['latentD'],
                        'dataPath': opts['configList'][i]['dataPath'],
                        "lrA_DiscCoeff": opts['configList'][i]['lrA_DiscCoeff'],
                        "components": 10,
                        "trainLoader": trainLoader,
                        "testLoader": testLoader
                        }
        
        optsInstance["experimentID"] = experimentID
        runsConfigsList.append(optsInstance)

#%%

p=mp.Pool(5)
outputs= p.map(TrainSourceModel,runsConfigsList)
p.close()
p.join()

log.writelines("lrA,lrB,lrA_discCoeff,latentD,batchSize,epoch,accuracy,ration,totalReal \r\n")
for i in range(len(outputs)):
    savePlots(runsConfigsList[i],outputs[i])
    log.writelines(outputs[i]['logText'])
    log.flush()

endTime=time.time()
print("code took {} seconds".format(endTime-startTime))

#%%
#
#
#
####################################
#
#
#
#        validating models PART
#
#
####################################
#
#
#
#%%
for i in len(outputs):
    saveModels(outputs[i],runsConfigsList[i])
#%%
####################################
#
#
#
#        testing transferability
#
#
####################################
#
#
#%%

#%%
#
#
#
####################################
#
#
#
#       DEBUGGING PART
#
#
####################################
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# #
# #%% plot the loss_a
# plt.plot(outputs['lossA'])
# plt.show()
#
# #%% plot the loss_b
# plt.plot(outputs['lossB'])
# plt.show()
#
# #%%
# loaderParameters={}
# train_loader=getDataLoader(parameters={'dataName': 'mnist', 'train': True, 'dataPath': '~/data/UDA_related', 'batchSize':100})
#
# encoder = outputs['encoder']
# for batchIndex,(batchData,batchLabel) in enumerate(train_loader):
#     print(batchIndex)
#     encodedBatch=encoder(batchData)
#     break
#
# #%% plot sample reconstructions
# j=10
# for i in range(j,j+20):
#     plt.figure()
#     plt.subplot(121)
#     plt.title("original")
#     plt.imshow(batchData[i].view(28,28).cpu().numpy())
#     #plt.show()
#     #generated=cEnc(cDec(decodedData[i]))
#     plt.subplot(122)
#     plt.title("generated")
#     plt.imshow(decodedData[i].cpu().view(28,28).detach().numpy())
#     plt.show()
#
# #%% sampling multiple components of gaussian distribution
# def getSampleFromLatentSpace(size, components=1,device):
#     if components == 1:
#         return torch.randn([size[0], size[1]]).to(device)
#     else:
#         componentsD = 2
#         batchSize = size[0]
#         latentD = size[1]
#
#         row,col= 0,0
#         sample = torch.zeros([batchSize, latentD])
#         sample[:,:components]=torch.randn([batchSize, componentsD]) * 2 + 10
#         increment=batchSize//components
#
#         for i in range(0, increment):
#             sample[row:row+increment,col:col+componentsD]
#         return sample.to(device)
#
# getSampleFromLatentSpace([32,10], components=10)
#
#
# #%%
# parameters={'dataName':'mnist','train':True,'dataPath':os.path.expanduser('~/data/UDA_related'),'batchSize':20}
# someLoader=getDataLoader(parameters)
# for batchX,(bData,bLabel) in enumerate(someLoader):
#     print(bLabel)
#     break
#
# #%%
# batchSize, latentD = 20,6
# components=6
#
# #batchSize, latentD = size[0], size[1]
# mixDim = latentD // components
# sample = torch.zeros([batchSize, latentD])
# classes=torch.zeros(batchSize)
#
# sample[:, :mixDim] = torch.randn([batchSize, mixDim]) * 2 + 10
#
#
# batchPerComponent = batchSize // components
# for i in range(batchSize // batchPerComponent):
#     classes[i * batchPerComponent:(i + 1) * batchPerComponent]=i
#     sample[i * batchPerComponent:(i + 1) * batchPerComponent, i * mixDim:(i + 1) * mixDim] = \
#         sample[i * batchPerComponent:(i + 1) * batchPerComponent, 0:mixDim]
#     if i != 0:
#         sample[i * batchPerComponent:(i + 1) * batchPerComponent, 0:mixDim] = torch.zeros(
#             [batchPerComponent, mixDim])
#
# #shuffling..
# shuffle_idx=torch.randperm(sample.shape[0])
# sample=sample[shuffle_idx,:]
# classes=classes[shuffle_idx]
# print(sample)
# print(classes.type(torch.IntTensor))
#
# #%%
# import numpy as np
# import torchvision
# import os
# import torch
# from torchvision import transforms
# from matplotlib import pylab as plt
#
# #%%
# def someParallelLoaderParser(loader_):
#     i=1
#     for batchX,(bData,bLabel) in enumerate(loader_["loader"]):
#         #plt.imshow(np.transpose(bData[0],[1,2,0]))
#         #plt.show()
#         i+=1
#         print(i)
#
# def printa(a):
#     print(a)
#
#
# #%%
#
# import torchvision,os,torchvision.transforms,torch
#
# deviceVal = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# someD=torchvision.datasets.MNIST(root=os.path.expanduser("~/data/UDA_related"),transform=torchvision.transforms.Compose([transforms.ToTensor()]),download=True)
# loader=torch.utils.data.DataLoader(someD,batch_size=10)
# from multiprocessing import Pool
# p=Pool(10)
# loaders=list([{"loader":loader}]*10)
# print(loaders)
# p.map(someParallelLoaderParser,loaders)
# p.close()
# p.join()
#
# #%%
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# cEnc = ConvEncoder(latentDimension=50).to(device)
# # cDec = ConvDecoder(latentDimension=opts['latentD']).to(device)
# disc = Discriminator(latentDimension=50).to(device)
# classify = Classifier(latentDimension=50).to(device)

