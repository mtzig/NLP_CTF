
import torch
from torch.utils.data import Dataset



class SimpleDataset(Dataset):
    '''
    A simple dataset
    '''

    def __init__(self, features, labels):
        '''
        Inputs:
        features: a tensor of the features of the dataset
        labels: a tensor of same shape, of the corresponding labels
        '''
        self.features = features
        self.labels = labels


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = self.features[idx]
        y = self.labels[idx].clone() #clone is needed as metal has some bug where returne value gets collapsed to 1

        return X, y
