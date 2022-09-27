
'''
    Code to create infinite dataloaders
    adapted from code for domainbeds paper
'''

import torch
import math


class _InfiniteSampler(torch.utils.data.Sampler):
    '''Wraps another Sampler to yield an infinite stream'''

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    '''
    Uses an infinite sampler to create a dataloader that never becomes empty
    '''
    def __init__(self, dataset, batch_size, num_workers=0):
        super().__init__()

        sampler = torch.utils.data.RandomSampler(
                dataset,
                replacement=False)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=False)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __next__(self):
        return next(self._infinite_iterator)

    def __len__(self):
        raise ValueError

    def batches_per_epoch(self):
        return math.ceil(len(self.dataset) / self.batch_size)
