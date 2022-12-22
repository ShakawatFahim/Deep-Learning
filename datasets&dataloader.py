import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math


class WineDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('./wine.csv',delimiter=',',dtype=np.float32,skiprows=1)
        self.x = torch.from_numpy(xy[: ,1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]
    def __len__(self):
        return self.n_samples

dataset = WineDataset()
#data=dataset[0]
dataloader = DataLoader(dataset=dataset,batch_size=4,shuffle=True,num_workers=0)
dataiter = iter(dataloader)
data = next(dataiter)
features,labels = data
print(features,labels)

#training loop
num_epoch = 2
total_samples = len(dataset)
n_iter = math.ceil(total_samples/4)
print(total_samples,n_iter)

for epoch in range(num_epoch):
    for i,(inputs,labels) in enumerate(dataloader):
        if (i+1) % 5 == 0:
            print(f'epoch{epoch+1}/{num_epoch},step{i+1}/{n_iter},input {inputs.shape }')
