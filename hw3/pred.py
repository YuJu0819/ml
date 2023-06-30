# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image
# from autoaugment import ImageNetPolicy
import torchvision.models as models
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
# This is for the progress bar.
from tqdm.auto import tqdm
import random

# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
test_set = FoodDataset("/kaggle/input/ml2023spring-hw3/test", tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loaders = []
for i in range(5):
    test_set_i = FoodDataset("/kaggle/input/ml2023spring-hw3/test", tfm=test_tfm)
    test_loader_i = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loaders.append(test_loader_i)
# model_best = Classifier().to(device)
model_best = models.resnet34(weights = None).to(device)
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
model_best.eval()
preds = [[],[],[],[],[],[]]
prediction = []
with torch.no_grad():
    for data,_ in tqdm(test_loader):
        test_pred = model_best(data.to(device)).cpu().data.numpy()
        preds[0].extend(test_pred)
#         test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
#         prediction += test_label.squeeze().tolist()
    
    for i, loader in test_loaders:
        for data,_ in tqdm(test_loader):
            test_pred = model_best(data.to(device)).cpu().data.numpy()
            preds[i+1].extend(test_pred)
    #         test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
    #         prediction += test_label.squeeze().tolist()

pred_np = np.array(preds, dtype = object)
tmp = 0.6*pred_np[0]
for i in range(1, 6):
    tmp+=0.1*pred_np[i]

prediction = np.argmax(tmp, axis = 1)

#create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(len(test_set))]
df["Category"] = prediction
df.to_csv("submission.csv",index = False)