from sklearn.utils import shuffle
import torch

class DistributedDataLoader:
    def __init__(self, dataset, *args, shuffle=False, **kwargs):
        self.sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        self.loader = torch.utils.data.DataLoader(dataset, shuffle=False, sampler=self.sampler)
        self.epoch = 0
        self.shuffle= shuffle

    def __iter__(self):
        if shuffle:
            self.epoch += 1
            self.sampler.set_epoch(self.epoch)
        return self.loader.__iter__()

    def __len__(self):
        return len(self.loader)
    