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
from torch import multiprocessing as mp
import time
from memory_profiler import profile

from train import TrainSourceModel,TrainSourceTargetModel,TrainSourceTargetModel_exp
from dataIO import getDataLoader,savePlots



#reading configurations file
logfile=True
configPath="~/Dropbox (MIT)/temp_git/VAE_experiments/configFiles/"
experimentID="8.json"
experimentConfigPath=os.path.expanduser(configPath+experimentID)
with open(experimentConfigPath,'r') as tempFile:
    opts = json.load(tempFile)

if logfile:
    sourceLog=open('{}/{}__source_logfile.txt'.format(os.path.expanduser(opts["logPathTxt"]),experimentID),"w")
    if opts['trainOnTarget']:
        targetLog = open('{}/{}__target_logfile.txt'.format(os.path.expanduser(opts["logPathTxt"]), experimentID), "w")

#%%

torch.backends.cudnn.enabled=False
# removed the line below for parallelization
#deviceVal = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

startTime = time.time()
runsConfigsList = []
if opts['search'] == 'gridSearch':
    for j in range(len(opts['batchSize'])):
        sourceTrainLoader = getDataLoader(
            parameters={'dataName': opts["sourceDataName"], 'train': True, 'dataPath': opts['dataPath'],
                        'batchSize': opts['batchSize'][j]})
        sourceTestLoader = getDataLoader(
            parameters={'dataName': opts["sourceDataName"], 'train': False, 'dataPath': opts['dataPath'],
                        'batchSize': opts['batchSize'][j]})

        if opts['trainOnTarget']:
            targetTrainLoader = getDataLoader(
                parameters={'dataName': opts["targetDataName"], 'train': True, 'dataPath': opts['dataPath'],
                            'batchSize': opts['batchSize'][j]})
            targetTestLoader = getDataLoader(
                parameters={'dataName': opts["targetDataName"], 'train': False, 'dataPath': opts['dataPath'],
                            'batchSize': opts['batchSize'][j]})

        for a in range(len(opts['lrA'])):
            for b in range(len(opts['lrB'])):
                for c in range(len(opts['lrA_DiscCoeff'])):
                    for k in range(len(opts['latentD'])):
                        optsInstance = {"lrA": opts['lrA'][a],
                                        "lrB": opts['lrB'][b],
                                        'epochs': opts['epochs'],
                                        'batchSize': opts['batchSize'][j],
                                        'latentD': opts['latentD'][k],
                                        'dataPath': opts['dataPath'],
                                        "lrA_DiscCoeff": opts["lrA_DiscCoeff"][c],
                                        "components": 10,
                                        "sourceTrainLoader": sourceTrainLoader,
                                        "sourceTestLoader": sourceTestLoader
                                        }
                        if opts["trainOnTarget"]:
                            optsInstance['targetTrainLoader']=targetTrainLoader
                            optsInstance['targetTestLoader']=targetTestLoader

                        optsInstance["logPathTxt"]=os.path.expanduser(opts["logPathTxt"])
                        optsInstance["logPathPlot"]=os.path.expanduser(opts["logPathPlot"])
                        optsInstance["experimentID"] = experimentID
                        optsInstance["verbose"]=True
                        runsConfigsList.append(optsInstance)

# elif opts['search'] == 'listSetup':
#     for i in range(len(opts['configList'])):
#         optsInstance = {"lrA": opts['configList'][i]['lrA'],
#                         "lrB": opts['configList'][i]['lrB'],
#                         'epochs': opts['configList'][i]['epochs'],
#                         'batchSize': opts['configList'][i]['batchSize'],
#                         'latentD': opts['configList'][i]['latentD'],
#                         'dataPath': opts['configList'][i]['dataPath'],
#                         "lrA_DiscCoeff": opts['configList'][i]['lrA_DiscCoeff'],
#                         "components": 10,
#                         "trainLoader": trainLoader,
#                         "testLoader": testLoader
#                         }
#
#         optsInstance["experimentID"] = experimentID
#         runsConfigsList.append(optsInstance)
# #
#%% TRAINING RUN
#
p=mp.Pool(8)
outputs= p.map(TrainSourceTargetModel_exp,runsConfigsList)
p.close()
p.join()

#
#% LOG SOURCE KEEPING
#
sourceLog.writelines("lrA,lrB,lrA_discCoeff,latentD,batchSize,epoch,accuracy,ration,totalReal \r\n")
targetLog.writelines("lrA,lrB,lrA_discCoeff,latentD,batchSize,epoch,accuracy,ration,totalReal \r\n")
for i in range(len(outputs)):
    savePlots(runsConfigsList[i], outputs[i])
    sourceLog.writelines(outputs[i]['sourceLogText'])
    sourceLog.flush()
    targetLog.writelines(outputs[i]['targetLogText'])
    targetLog.flush()

endTime = time.time()
print("code took {} seconds".format(endTime - startTime))

#%%

print(outputs[0]['sourceLogText'])
print(outputs[0]['targetLogText'])


#%%
# import numpy as np
# import matplotlib.pylab as plt
#
# alpha=0.1
# beta=0.1
# sample=[]
# for i in range(500):
#     for j in range(500):
#         z=alpha*np.random.uniform()+beta*np.random.uniform()
#         sample.append(z)
# a=plt.hist(sample,np.linspace(min(sample),max(sample)))
# a_normed=[i/np.sum(a[0]) for i in a[0]]
# plt.figure()
# plt.plot(a[1][1:],a_normed)
# plt.show()
#
#
# #%% testing the loaders
# for (a, b), (c, d) in zip(enumerate(sourceTrainLoader), enumerate(targetTrainLoader)):
#     print(b.shape)
#
# #%%
# a=zip(enumerate(sourceTrainLoader),enumerate(targetTrainLoader))
# for i in a:
#     print(a)
