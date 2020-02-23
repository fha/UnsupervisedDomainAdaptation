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

from models import ConvEncoder,Discriminator,Classifier
from dataIO import getSampleFromLatentSpace


################################################
#
#       TRAINING SOURCE MODELS
#
################################################
def TrainSourceTargetModel_exp(options):
    #defining the models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    options['device']=device
    cEnc = ConvEncoder(latentDimension=options['latentD']).to(device)
    disc = Discriminator(latentDimension=options['latentD']).to(device)
    classify = Classifier(latentDimension=options['latentD']).to(device)

    # defining loss functions
    MSELoss = nn.MSELoss().to(device)  # mean squared error loss
    BCELoss = nn.BCELoss().to(device)  # binary cross-entropy
    NLLLoss = nn.NLLLoss().to(device)  # negative-log likelihood
    CELoss = nn.CrossEntropyLoss().to(device)  # cross entropy loss

    # loss log data
    lossData_a, lossData_b = [], []

    optimizer_classification = optim.Adam(itertools.chain(classify.parameters()), lr=options['lrA'])
    optimizer_encoder = optim.Adam(itertools.chain(cEnc.parameters()), lr=options['lrA'])
    optimizer_discriminator = optim.Adam(itertools.chain(disc.parameters()), lr=options['lrB'])

    sourceLogText,targetLogText="",""
    sourceTrainLoader = options['sourceTrainLoader']
    targetTrainLoader = options['targetTrainLoader']

    for epochIdx in range(options['epochs']):

        # defining the data loaders
        _ones = Variable(torch.FloatTensor(options['batchSize'], 1).fill_(1.0), requires_grad=False).to(device)
        _zeros = Variable(torch.FloatTensor(options['batchSize'], 1).fill_(0.0), requires_grad=False).to(device)

        for (batchData, batchLabels),(targetBatchData, targetBatchLabels)  in zip(sourceTrainLoader,targetTrainLoader):

            #SOURCE DATA

            # data setup
            _batchSize = batchData.shape[0]
            _targetBatchSize=targetBatchData.shape[0]
            #source
            batchData = batchData.to(device)
            batchLabels = batchLabels.to(device)
            #target
            targetBatchData=targetBatchData.to(device)
            targetBatchLabels=targetBatchLabels.to(device)

            #generate synthetic sample from the feature space
            latentSpaceSample_A,latentSpaceClasses_A=getSampleFromLatentSpace([_batchSize,options['latentD']],options)
            latentSpaceSample_B,latentSpaceClasses_B=getSampleFromLatentSpace([_batchSize,options['latentD']],options)


            # cEnc pass
            encodedDataBatch = cEnc(batchData)
            targetEncodedDataBatch=cEnc(targetBatchData)

            #classification pass
            classesPrediction = classify(encodedDataBatch)
            sampleClassesPrediction = classify(latentSpaceSample_A)

            #discriminator pass
            sourceDiscOutput=disc(encodedDataBatch)
            targetDiscOutput=disc(targetEncodedDataBatch)

            # optimzation step I -- classification
            optimizer_classification.zero_grad()
            #--- first loss function
            loss_a = CELoss(sampleClassesPrediction,latentSpaceClasses_A)
            loss_a.backward()
            optimizer_classification.step()

            #optimization step II -- encoder
            optimizer_encoder.zero_grad()
            loss_b=CELoss(classesPrediction, batchLabels)
            loss_b.backward()
            optimizer_encoder.step()

            # loss_c=BCELoss(targetDiscDataInput,_zeros[:_targetBatchSize])+\
            #        BCELoss(sampleDiscOutputB,_ones[:_batchSize])

            #
            # #discriminator pass
            # discDataInput=Variable(encodedDataBatch.view(_batchSize,options['latentD']).cpu().data,requires_grad=False).to(device)
            # discDataOutput=disc(discDataInput)
            # targetDiscDataInput=Variable(targetEncodedDataBatch.view(_targetBatchSize,options['latentD']).cpu().data,requires_grad=False).to(device)
            # targetDiscDataInput=disc(targetDiscDataInput)
            # sampleDiscOutputB=disc(latentSpaceSample_B)
            # # optimization step II
            # #---train the discriminator, 1/0 is real/fake data
            # optimizer_b.zero_grad()

            #---second loss function
            # loss_b=BCELoss(discDataOutput,_zeros[:_batchSize])+\
            #        BCELoss(targetDiscDataInput,_zeros[:_targetBatchSize])+\
            #        BCELoss(sampleDiscOutputB,_ones[:_batchSize])
            # loss_b.backward()
            # optimizer_b.step()
            # lossData_b.append(loss_b.data.item())

        ####
        #### End of an epoch
        ####

        sourceLogText+=validateModel(epochIdx,options,models=[cEnc,classify,disc])
        targetLogText+=validateModel(epochIdx,options,source=False,models=[cEnc,classify,disc])

        # end of an epoch - CHECK ACCURACY ON TEST SET

    outputs={
        'lossA':lossData_a,
        'lossB':lossData_b,
        'encoder':cEnc,
        'disc':disc,
        'classifier':classify,
        'sourceLogText':sourceLogText,
        'targetLogText': targetLogText,
    }
    return outputs



def TrainSourceTargetModel(options):
    #defining the models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    options['device']=device
    cEnc = ConvEncoder(latentDimension=options['latentD']).to(device)
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

    sourceLogText,targetLogText="",""
    sourceTrainLoader = options['sourceTrainLoader']
    targetTrainLoader = options['targetTrainLoader']

    for epochIdx in range(options['epochs']):

        # defining the data loaders
        _ones = Variable(torch.FloatTensor(options['batchSize'], 1).fill_(1.0), requires_grad=False).to(device)
        _zeros = Variable(torch.FloatTensor(options['batchSize'], 1).fill_(0.0), requires_grad=False).to(device)

        for (batchData, batchLabels),(targetBatchData, targetBatchLabels)  in zip(sourceTrainLoader,targetTrainLoader):

            #SOURCE DATA

            # data setup
            _batchSize = batchData.shape[0]
            _targetBatchSize=targetBatchData.shape[0]
            #source
            batchData = batchData.to(device)
            batchLabels = batchLabels.to(device)
            #target
            targetBatchData=targetBatchData.to(device)
            targetBatchLabels=targetBatchLabels.to(device)

            #generate synthetic sample from the feature space
            latentSpaceSample_A,latentSpaceClasses_A=getSampleFromLatentSpace([_batchSize,options['latentD']],options)
            latentSpaceSample_B,latentSpaceClasses_B=getSampleFromLatentSpace([_batchSize,options['latentD']],options)


            # cEnc pass
            encodedDataBatch = cEnc(batchData)
            targetEncodedDataBatch=cEnc(targetBatchData)

            #classification pass
            classesPrediction = classify(encodedDataBatch)
            sampleClassesPrediction = classify(latentSpaceSample_A)

            #discriminator pass
            sourceDiscOutput=disc(encodedDataBatch)
            targetDiscOutput=disc(targetEncodedDataBatch)

            # optimzation step I
            optimizer_a.zero_grad()
            #--- first loss function
            loss_a = CELoss(classesPrediction, batchLabels)+\
                     options["lrA_DiscCoeff"]*BCELoss(sourceDiscOutput,_ones[:_batchSize])+ \
                     options["lrA_DiscCoeff"] * BCELoss(targetDiscOutput, _ones[:_targetBatchSize]) + \
                     CELoss(sampleClassesPrediction,latentSpaceClasses_A)

            loss_a.backward()
            optimizer_a.step()
            lossData_a.append(loss_a.data.item())

            #discriminator pass
            discDataInput=Variable(encodedDataBatch.view(_batchSize,options['latentD']).cpu().data,requires_grad=False).to(device)
            discDataOutput=disc(discDataInput)
            targetDiscDataInput=Variable(targetEncodedDataBatch.view(_targetBatchSize,options['latentD']).cpu().data,requires_grad=False).to(device)
            targetDiscDataInput=disc(targetDiscDataInput)
            sampleDiscOutputB=disc(latentSpaceSample_B)
            # optimization step II
            #---train the discriminator, 1/0 is real/fake data
            optimizer_b.zero_grad()

            #---second loss function
            loss_b=BCELoss(discDataOutput,_zeros[:_batchSize])+\
                   BCELoss(targetDiscDataInput,_zeros[:_targetBatchSize])+\
                   BCELoss(sampleDiscOutputB,_ones[:_batchSize])
            loss_b.backward()
            optimizer_b.step()
            lossData_b.append(loss_b.data.item())

        ####
        #### End of an epoch
        ####

        sourceLogText+=validateModel(epochIdx,options,models=[cEnc,classify,disc])
        targetLogText+=validateModel(epochIdx,options,source=False,models=[cEnc,classify,disc])

        # end of an epoch - CHECK ACCURACY ON TEST SET

    outputs={
        'lossA':lossData_a,
        'lossB':lossData_b,
        'encoder':cEnc,
        'disc':disc,
        'classifier':classify,
        'sourceLogText':sourceLogText,
        'targetLogText': targetLogText,
    }
    return outputs


def TrainSourceModel(options):
    #defining the models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    options['device']=device
    cEnc = ConvEncoder(latentDimension=options['latentD']).to(device)
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
        _ones = Variable(torch.FloatTensor(options['batchSize'], 1).fill_(1.0), requires_grad=False).to(device)
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



################################################
#
#       TESTING AND VALIDATION OF TRAINED MODELS
#
################################################
def validateModel(epochNum,options,source=True,models=[]):
    device =options["device"]
    totalCorrect,totalReal, total = 0, 0, 0
    _batchSize = 200
    if source:
        localLoader=options['sourceTestLoader']
    else:
        localLoader= options['targetTestLoader']

    for batchIdx, (batchData, batchLabels) in enumerate(localLoader):
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
        epochNum,options['lrA'],options['lrB'],options["lrA_DiscCoeff"],options['latentD'],options['batchSize'], ratio, totalCorrect, total,totalReal)
    if options['verbose']:
        if source:
            print("source==>"+line)
        else:
            print("target==>" + line)
    return line

#%%
