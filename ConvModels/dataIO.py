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
from multiprocessing import Pool
import time
from memory_profiler import profile





################################################
#
#       SAMPLING FROM LATENT SPACE
#
################################################
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

        sample[:, :mixDim] = torch.randn([batchSize, mixDim]) *5  + 20

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


########################
#
#       LOGGERS
#
########################
def savePlots(optsInstance,outputs):
    plt.figure()
    plt.plot(outputs['lossA'])
    plt.savefig('{}/{}__lossA_lrA{}_lrB{}_latentD{}_discCoeff{}.png'.format(
        optsInstance['logPathPlot'],optsInstance["experimentID"],optsInstance['lrA'], optsInstance['lrB'], optsInstance['latentD'],optsInstance["lrA_DiscCoeff"]))
    plt.figure()
    plt.plot(outputs['lossB'])
    plt.savefig('{}/{}__lossB_lrA{}_lrB{}_latentD{}__discCoeff{}.png'.format(
        optsInstance['logPathPlot'],optsInstance["experimentID"],optsInstance['lrA'], optsInstance['lrB'], optsInstance['latentD'],optsInstance["lrA_DiscCoeff"]))


def saveModels(outputs,cfig):
    modelName="{}/{}_lrA{}_lrB{}_latentD{}_discCoeff{}".format(
        cfig["modelsPath"],cfig["experimentID"],cfig['lrA'], cfig['lrB'], cfig['latentD'],cfig["lrA_DiscCoeff"])
    torch.save(outputs["encoder"],"{}__encoder".format(modelName))
    torch.save(outputs["disc"],"{}__disc".format(modelName))
    torch.save(outputs["classifier"],"{}__classifier".format(modelName))


########################
#
#       DATA LOADERS
#
########################
def getDataLoader(parameters):
    if parameters['dataName']=="mnist":
        if parameters['train']:
            train_dataset = torchvision.datasets.MNIST(parameters['dataPath'], train=True, download=False,
                                                       transform=transforms.Compose(
                                                           [transforms.RandomRotation(20,fill=(0,)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.1307,), (0.3081,))
                                                            ]
                                                       ))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=parameters['batchSize'], shuffle=True)
            return train_loader
        else:
            test_dataset = torchvision.datasets.MNIST(parameters['dataPath'], train=False, download=False,
                                                      transform=transforms.Compose([
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.1307,), (0.3081,))]))
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=parameters['batchSize'],shuffle=True)
            return test_loader

    elif parameters['dataName']=="usps":
        train_dataset = torchvision.datasets.USPS(parameters['dataPath'], train=True, download=False,
                                                   transform=transforms.Compose(
                                                       [transforms.Resize([28, 28]),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.1307,), (0.3081,))
                                                        ]
                                                   ))
        if parameters['train']:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=parameters['batchSize'], shuffle=True)
            return train_loader
        else:
            test_dataset = torchvision.datasets.USPS(parameters['dataPath'], train=False, download=False,
                                                      transform=transforms.Compose([
                                                          transforms.Resize([28, 28]),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.1307,), (0.3081,))]))
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=parameters['batchSize'], shuffle=True)
            return test_loader


#%%

#
# trainLoader=getDataLoader({'dataPath':os.path.expanduser("~/data/UDA_related"),'dataName':'mnist',
#                          'train':True,"batchSize":32})
#
# testLoader=getDataLoader({'dataPath':os.path.expanduser("~/data/UDA_related"),'dataName':'mnist',
#                          'train':False,"batchSize":32})
#
# for (a,b),(c,d) in zip(trainLoader,testLoader):
#     print(a.shape,b.shape,c.shape,d.shape)
#
# # for a,(b,c) in enumerate(mstLoader):
# #     print(a)
