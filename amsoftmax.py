import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSoftmaxLoss(nn.Module):

    def __init__(self, scale:float=30, margin:float=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.s = scale
        self.m = margin
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y):
        index_y = torch.arange(len(y))

        cos_theta = x

        # cos_theta_m = cos_theta - self.m
        # cos_theta[index_y, y] = cos_theta_m[index_y, y]
        cos_theta[index_y, y] = cos_theta[index_y, y] - self.m

        logits = cos_theta * self.s
        loss = self.criterion(logits, y)
        return loss