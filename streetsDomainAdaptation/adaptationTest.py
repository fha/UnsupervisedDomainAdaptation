import torch
import torchvision
import numpy as np
import gc
import torch.nn as nn
from torchvision import datasets,transforms
from matplotlib import pylab as plt


modelPath="/media/hdd/data/streetContext/streetImages/Boston/modelRelated/models/inceptionv3_1/inceptionv3_model.999"
#modelPath="/media/hdd/data/streetContext/streetImages/SF/modelRelated/models/inceptionv3_poi/SF_POI_inceptionv3.1000"

#model_dataset="/media/hdd/data/streetContext/streetImages/SF/modelRelated/data/resnetPOI/train"
model_dataset="/media/hdd/data/streetContext/streetImages/Boston/modelRelated/data/resnetPOI/train/"

#target_dataset="/media/hdd/data/streetContext/streetImages/SF/modelRelated/data/resnetPOI/test"
target_dataset="/media/hdd/data/streetContext/streetImages/Boston/modelRelated/data/resnetPOI/test/"

#%% defining helper functions

def getConfusionMatrix(labels, predictions, nClasses=10):
    confusionMatrix = np.zeros([nClasses, nClasses])
    correct = (labels == predictions).sum().item()
    wrong = len(labels) - correct;
    outOfRange = 0;
    for i in range(len(labels)):
        if predictions[i] < nClasses:
            confusionMatrix[labels[i], predictions[i]] += 1;
        else:
            outOfRange += 1;
    return wrong, correct, outOfRange, confusionMatrix


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/(1+float(count[i]))
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

def validation(val_loader,model,device,classMapper,nClasses):
    setSize=loss=i=0;
    criterion = nn.CrossEntropyLoss().to(device)

    for batch_num, (val_inputs, val_labels) in enumerate(val_loader, 1):
        gc.collect()
        val_inputs = val_inputs.to(device)
        val_labels = val_labels.numpy()
        val_labels_mapped=torch.tensor([classMapper[cc] for cc in val_labels]).to(device)
        val_outputs = model(val_inputs)
        val_predictions = torch.argmax(val_outputs, 1)
        loss += criterion(val_outputs, val_labels_mapped).item()
        setSize+=len(val_labels_mapped)

        if i==0:
            wrong, correct, outOfRange, confusionMatrix=getConfusionMatrix(val_labels_mapped, val_predictions, nClasses)
        else:
            a1,a2,a3,a4 = getConfusionMatrix(val_labels_mapped, val_predictions, nClasses)
            wrong=wrong+a1
            correct=correct+a2
            outOfRange=outOfRange+a3
            confusionMatrix=confusionMatrix+a4
        i+=1;

    print("-------------------------------- with {} wrong and {} correct and {} outOfRange \n--Classification accuracy {}".format(wrong, correct,
                                                                                                   outOfRange,correct/(wrong+correct+outOfRange)))
    return confusionMatrix,correct/(wrong+correct+outOfRange)

#%% parameters go here
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
validationTransform = transforms.Compose([transforms.ToTensor()])

#%% get the class idx mapper from Boston to SF
model_dataset = datasets.ImageFolder(root=model_dataset)
target_dataset = datasets.ImageFolder(root=target_dataset, transform=validationTransform)

# load the target dataset
weights = make_weights_for_balanced_classes(target_dataset.imgs, len(target_dataset.classes))
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
val_loader = torch.utils.data.DataLoader(target_dataset, batch_size=15, shuffle=False,sampler = sampler,
                                         num_workers=16,pin_memory=True)
target_to_model={}
for class_ in model_dataset.class_to_idx:
    if class_ in model_dataset.class_to_idx and class_ in target_dataset.class_to_idx:
        target_to_model[target_dataset.class_to_idx[class_]]=model_dataset.class_to_idx[class_]

#%% run this only if needed, it takes time

#define an inception model and load it:
model_ = torchvision.models.inception_v3(num_classes=11,aux_logits=False)
model_.load_state_dict(torch.load(modelPath))
model_.eval()
model_.to(device)

##%%loading the model

#%% get the validation accuracy and confusion matrix
confmat_numpy,accuracy=validation(val_loader, model_, device,target_to_model, nClasses=11)



#%% plotting conf matrix
fig,ax=plt.subplots(figsize=[8,8])
im=ax.imshow(confmat_numpy)
labels_=list(model_dataset.class_to_idx)
ax.set_xticks(np.arange(len(labels_)))
ax.set_yticks(np.arange(len(labels_)))

ax.set_xticklabels(labels_)
ax.set_yticklabels(labels_)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

plt.show()


