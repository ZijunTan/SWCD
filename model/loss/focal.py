"""
https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
"""

import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from einops import rearrange


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=4.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        if isinstance(alpha, (float, int)):
            self.alpha = torch.as_tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.as_tensor(alpha)

    def forward(self, input, target):
        N, C, H, W = input.size()
        assert C == 2
        # input = input.view(N, C, -1)
        # input = input.transpose(1, 2)
        # input = input.contiguous().view(-1, C)
        input = rearrange(input, 'b c h w -> (b h w) c')
        # input = input.contiguous().view(-1)

        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        loss = -1 * (1-pt)**self.gamma * logpt

        return loss.mean()


def cross_entropy(input, target, weight=None, reduction='mean', ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """

    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

def simloss(x1, x2, label):
    sim = F.cosine_similarity(x1, x2, dim=1)
    sim = torch.sigmoid(sim)
    sim = 1 - sim.unsqueeze(dim=1)
    # dist = F.pairwise_distance(x1, x2, keepdim=True)
    # sim = 1 - 1 / (1+dist)
    b, c, h, w = x1.shape
    # patch_size = h // 8
    label_new = F.adaptive_max_pool2d(label.float().unsqueeze(dim=1), (h, w))   ##8 16 32 64
    loss = F.binary_cross_entropy(sim, label_new)
    return loss

