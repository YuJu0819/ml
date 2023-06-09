# -*- coding: utf-8 -*-
"""ml2023spring-hw2-6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10OBedxvLBGHALoTArRCcvaAVigUUjWj7

# **Homework 2: Phoneme Classification**

Objectives:
* Solve a classification problem with deep neural networks (DNNs).
* Understand recursive neural networks (RNNs).

If you have any questions, please contact the TAs via TA hours, NTU COOL, or email to mlta-2023-spring@googlegroups.com

# Some Utility Functions
**Fixes random number generator seeds for reproducibility.**
"""

import numpy as np
import torch
import random
import torch.optim.lr_scheduler as lr_scheduler
def same_seeds(seed):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

"""**Helper functions to pre-process the training data from raw MFCC features of each utterance.**

A phoneme may span several frames and is dependent to past and future frames. \
Hence we concatenate neighboring phonemes for training to achieve higher accuracy. The **concat_feat** function concatenates past and future k frames (total 2k+1 = n frames), and we predict the center frame.

Feel free to modify the data preprocess functions, but **do not drop any frame** (if you modify the functions, remember to check that the number of frames are the same as mentioned in the slides)
"""

import os
import torch
from tqdm import tqdm

def load_feat(path):
    feat = torch.load(path)
    return feat

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]
    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)

def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n) 
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2)

def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, random_seed=1213):
    class_num = 41 # NOTE: pre-computed, should not need change

    if split == 'train' or split == 'val':
        mode = 'train'
    elif split == 'test':
        mode = 'test'
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    label_dict = {}
    if mode == 'train':
        for line in open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines():
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]
        
        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(random_seed)
        random.shuffle(usage_list)
        train_len = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:train_len] if split == 'train' else usage_list[train_len:]

    elif mode == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len,concat_nframes,39)
    if mode == 'train':
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode == 'train':
          label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode == 'train':
          y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode == 'train':
      y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode == 'train':
      print(y.shape)
      return X, y
    else:
      return X

"""# Dataset"""

import torch
from torch.utils.data import Dataset

class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

"""# Model
Feel free to modify the structure of the model.
"""

import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
#             nn.LeakyReLU(0.1),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
#             nn.Linear(2048, output_dim),
#             nn.BatchNorm1d(output_dim),
#             nn.LeakyReLU(),
#             nn.Dropout(0.28),
#             nn.Linear(2048, 1024),
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(1024),
#             nn.Dropout(0.35),
#             nn.Linear(1024, 512),
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(512),
#             nn.Dropout(0.35),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(0.35),
#             nn.Linear(256, output_dim),
        )

    def forward(self, x):
        x = self.block(x)
        return x


# class Classifier(nn.Module):
#     def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
#         super(Classifier, self).__init__()
#         self.hidden_dim = hidden_dim
# #         self.rnn = nn.RNN(input_size, hidden_size)
#         self.fc = nn.Sequential(
#             BasicBlock(input_dim, hidden_dim),
#             *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, x):
#         x = self.fc(x)
#         return x

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=3, hidden_dim=2048):
        super(LSTM, self).__init__()
#         print(input_dim, output_dim, hidden_layers, hidden_dim)
        # TODO: apply batch normalization and dropout for strong baseline.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html (batch normalization)
        #       https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html (dropout)
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.lstm = nn.LSTM(input_size = 39, hidden_size = hidden_dim, num_layers = 4 , batch_first=True, dropout=0.4, bidirectional = True)
#         self.batchnorm = nn.BatchNorm1d(hidden_dim)
#         self.dropout = nn.Dropout(0.45)
        self.fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim*2),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim*2, output_dim),
            # nn.LeakyReLU(0.1),
            # nn.BatchNorm1d(128),
            # nn.Dropout(0.4),
            # nn.Linear(128, output_dim),
        )

    def forward(self, x):
#         h0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_dim).to(device)
#         c0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_dim).to(device)
#         x = x.view(len(x), 1, -1)
#         print(x.shape)
        out, _ = self.lstm(x)
        
        out2 = out[:, 30, :]
        
#         out = self.batchnorm(out)
# #         out = self.dropout(out)
        out = self.fc(out2)
        return out

"""# Hyper-parameters"""

concat_nframes = 61   # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.98   # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 819     # random seed
batch_size = 512       # batch size
num_epoch = 15   # the number of training epoch
learning_rate = 2e-3      # learning rate
model_path = './model.ckpt'  # the path where the checkpoint will be saved

# model parameters
# TODO: change the value of "hidden_layers" or "hidden_dim" for medium baseline
# input_dim = 39 * concat_nframes # the input dim of the model, you should not change the value
input_dim = 39
hidden_layers = 3   # the number of hidden layers
hidden_dim = 256  # the hidden dim

"""# Dataloader"""

from torch.utils.data import DataLoader
import gc

same_seeds(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

# preprocess data
train_X, train_y = preprocess_data(split='train', feat_dir='/kaggle/input/ml2023spring-hw2/libriphone/feat', phone_path='/kaggle/input/ml2023spring-hw2/libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)
val_X, val_y = preprocess_data(split='val', feat_dir='/kaggle/input/ml2023spring-hw2/libriphone/feat', phone_path='/kaggle/input/ml2023spring-hw2/libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)

# get dataset
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# remove raw feature to save memory
del train_X, train_y, val_X, val_y
gc.collect()

# get dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

"""# Training"""

model = LSTM(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = 1e-3)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = 8,T_mult = 2)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max = 6)
best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    
    # training
    model.train() # set the model to training mode
    for i, batch in enumerate(tqdm(train_loader)):
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad() 
        outputs = model(features) 
        
        loss = criterion(outputs, labels)
        loss.backward() 
        scheduler.step(epoch+i/len(train_loader))
        optimizer.step() 
#         scheduler.step()
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        train_acc += (train_pred.detach() == labels.detach()).sum().item()
        train_loss += loss.item()
    
    # validation
    model.eval() # set the model to evaluation mode
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            
            loss = criterion(outputs, labels) 
            
            _, val_pred = torch.max(outputs, 1) 
            val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
            val_loss += loss.item()

    print(f'[{epoch+1:03d}/{num_epoch:03d}] Train Acc: {train_acc/len(train_set):3.5f} Loss: {train_loss/len(train_loader):3.5f} | Val Acc: {val_acc/len(val_set):3.5f} loss: {val_loss/len(val_loader):3.5f}')

    # if the model improves, save a checkpoint at this epoch
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print(f'saving model with acc {best_acc/len(val_set):.5f}')

del train_set, val_set
del train_loader, val_loader
gc.collect()

del train_set, val_set
del train_loader, val_loader
gc.collect()

"""# Testing
Create a testing dataset, and load model from the saved checkpoint.
"""

# load data
test_X = preprocess_data(split='test', feat_dir='/kaggle/input/ml2023spring-hw2/libriphone/feat', phone_path='/kaggle/input/ml2023spring-hw2/libriphone', concat_nframes=concat_nframes)
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# load model
model = LSTM(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
model.load_state_dict(torch.load(model_path))

"""Make prediction."""

pred = np.array([], dtype=np.int32)

model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = model(features)

        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

"""Write prediction to a CSV file.

After finish running this block, download the file `prediction.csv` from the files section on the left-hand side and submit it to Kaggle.
"""

with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))