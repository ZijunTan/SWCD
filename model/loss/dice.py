import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
from kornia.losses import dice_loss


class dice_loss_v1(nn.Module):
    def __init__(self):
        super(dice_loss_v1, self).__init__()

    def forward(self, input, target):
        target = target.squeeze(1).to(torch.int64)
        loss = dice_loss(input, target)

        return loss

def dice_loss_v2(inputs, targets):
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return 1 - dice