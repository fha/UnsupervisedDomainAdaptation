import torch,torchvision
from torchvision import transforms,datasets
import matplotlib.pylab as plt
import os
import numpy as np
import json
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
from dataIO import getTestDataLoader

#%%
# loading and testing model on ups data
modelsPath="/home/fahad/data/UDA_related/models/";
encoderFile="6.json_lrA0.0001_lrB0.1_latentD10_discCoeff0.001__encoder"
discriminatorFile="6.json_lrA0.0001_lrB0.1_latentD10_discCoeff0.001__disc"
classifierFile="6.json_lrA0.0001_lrB0.1_latentD10_discCoeff0.001__classifier"

#%%
torch.backends.cudnn.enabled=False
convEncoder=torch.load(modelsPath+encoderFile)
classifier=torch.load(modelsPath+classifierFile)
discriminator=torch.load(modelsPath+discriminatorFile)

#%%
parameters={"path":os.path.expanduser("~/data/UDA_related")}
test_loader=getTestDataLoader("usps")
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
totalCorrectPredictions,totalReals,total=0,0,0
for batchIdx,(batchData,batchLabel) in enumerate(test_loader):
    batchData,batchLabel=batchData.to(device),batchLabel.to(device)

    encoded_data=convEncoder(batchData)
    classPrediction=torch.argmax(classifier(encoded_data),1)
    realsPrediction=discriminator(encoded_data)
    totalCorrectPredictions+=torch.sum(classPrediction==batchLabel).item()
    totalReals+=torch.sum(realsPrediction>=0.5).item()
    total+=len(realsPrediction)
print(totalCorrectPredictions,totalCorrectPredictions/totalReals,totalReals,total)


#%%
b=encoded_data.data.cpu().numpy()
covar=np.cov(b.T)
plt.imshow(covar)
plt.show()
