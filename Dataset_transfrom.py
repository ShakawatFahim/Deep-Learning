import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

class Winedatasets (Dataset):

    def __init__(self, transform = None):
        xy = np.loadtxt("wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        #we don't convert tensor here
        self.x = xy[:,1:]
        self.y = xy[:,[0]]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self,sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


#if we use transfrom = None we get the ndarrery
dataset = Winedatasets(transform=ToTensor())
first_data = dataset[0]
features , labels = first_data
print(type(features),type(labels))

