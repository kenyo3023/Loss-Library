import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SphereLoss(nn.Module):

    def __init__(self, scale:float=1, margin:float=0.4, input_as_cos_theta:bool=True):
        super(SphereLoss, self).__init__()
        self.s = scale
        self.m = margin
        self.input_as_cos_theta = input_as_cos_theta
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y):

        if self.input_as_cos_theta:
            index_y = torch.arange(len(y))

            cos_theta = x

            theta = torch.arccos(cos_theta)
            theta_m = theta * self.m

            cos_theta_m = torch.cos(theta_m)

            k = (theta_m / math.pi).floor()
            phi_theta_m = torch.pow(-1, k) * cos_theta_m - 2. * k
            cos_theta[index_y, y] = phi_theta_m[index_y, y]

            x = cos_theta

        logits = x * self.s
        loss = self.criterion(logits, y)
        return loss
