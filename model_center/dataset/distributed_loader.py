import torch.utils.data as data
import bmtrain as bmt

class DistributedDataLoader:
    def __init__(self, dataset, shuffle=False, **kwargs):
        self.sampler = data.distributed.DistributedSampler(dataset, shuffle=shuffle, rank=bmt.rank(), num_replicas=bmt.world_size())
        self.loader = data.DataLoader(dataset, shuffle=False, sampler=self.sampler, **kwargs)
        self.epoch = 0
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.epoch += 1
        self.sampler.set_epoch(self.epoch)
        return self.loader.__iter__()

    def __len__(self):
        return len(self.loader)
    