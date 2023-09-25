import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):

    def __init__(self, scale:float=30, margin:float=0.4, input_as_cos_theta:bool=True):
        super(ArcFaceLoss, self).__init__()
        self.s = scale
        self.m = margin
        self.input_as_cos_theta = input_as_cos_theta
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y):
        index_y = torch.arange(len(y))

        if self.input_as_cos_theta:
            cos_theta = x
            sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))

            cos_m = math.cos(self.m)
            sin_m = math.sin(self.m)

            cos_theta_m = (cos_theta * cos_m) - (sin_theta * sin_m)
            cos_theta[index_y, y] = cos_theta_m[index_y, y]

            x = cos_theta

        logits = x * self.scale
        loss = self.criterion(logits, y)
        return loss