import torchvision.transforms as T
import torch
import random


class RandomNoise(object):
    def __init__(self, mean=0., std=(0., 1.)):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        cur_std = random.uniform(self.std[0], self.std[1])
        return tensor + torch.randn(tensor.size()) * cur_std + self.mean

class BatchRandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, *args, **kwargs):
        super(BatchRandomHorizontalFlip, self).__init__(*args, **kwargs)

    def forward(self, images):
        saved_p = self.p
        self.p = 1 if torch.rand(1) < self.p else 0

        if self.p == 0:
            self.p = saved_p
            return images

        res = torch.zeros(images.shape)
        for i in range(len(images)):
            res[i] = super().forward(images[i])

        self.p = saved_p
        return res