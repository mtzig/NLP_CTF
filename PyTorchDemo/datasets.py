
import torch
from torch.utils.data import Dataset



class SimpleDataset(Dataset):
    """
    A simple dataset
    """

    def __init__(self, features, labels):
        '''
        INPUTS:
        args: of the form X, y, c
              where X is the model inputs
              y is the labels
              and c is the subclass labels
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