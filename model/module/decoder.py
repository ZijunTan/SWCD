import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.sim_agg import Feature_Enhancement_Module

class Similarity_Fusion_Module(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Similarity_Fusion_Module, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp1 = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(channel // reduction, channel, 1, bias=False))

        self.mlp2 = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(channel // reduction, channel, 1, bias=False))

        self.conv = nn.Sequential(nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(channel),
                                  nn.ReLU(inplace=True))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, sim):
        max_self = self.mlp1(self.max_pool(x))
        avg_self = self.mlp1(self.avg_pool(x))
        channel_self = self.sigmoid(max_self + avg_self)
        x_self = channel_self * x

        x_sim = sim * x

        out = self.conv(torch.cat([x_self, x_sim], dim=1))
        max_out = self.mlp2(self.max_pool(out))
        avg_out = self.mlp2(self.avg_pool(out))
        channel_out = self.sigmoid(max_out + avg_out)
        out = channel_out * out

        return out


class Interaction_Module(nn.Module):
    def __init__(self, channels, num_paths=2):
        super(Interaction_Module, self).__init__()
        self.num_paths = num_paths
        attn_channels = channels // 16
        attn_channels = max(attn_channels, 8)

        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x1, x2):

        x = torch.stack([x1, x2], dim=1)
        attn = x.sum(1).mean((2, 3), keepdim=True)
        attn = self.fc_reduce(attn)
        attn = self.act(attn)
        attn = self.fc_select(attn)
        B, C, H, W = attn.shape
        attn1, attn2 = attn.reshape(B, self.num_paths, C // self.num_paths, H, W).transpose(0, 1)
        attn1 = torch.sigmoid(attn1)
        attn2 = torch.sigmoid(attn2)

        return x1 * attn1, x2 * attn2



class Decoder(nn.Module):
    def __init__(self, in_d, out_d):
        super(Decoder, self).__init__()
        self.in_d = in_d
        self.out_d = out_d

        self.conv5 = nn.Sequential(nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_d),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_d),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_d),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_d),
                                   nn.ReLU(inplace=True))

        self.cls = nn.Conv2d(self.in_d, self.out_d, kernel_size=1)

        self.SFM5 = Similarity_Fusion_Module(in_d)
        self.SFM4 = Similarity_Fusion_Module(in_d)
        self.SFM3 = Similarity_Fusion_Module(in_d)
        self.SFM2 = Similarity_Fusion_Module(in_d)

    def forward(self, d5, d4, d3, d2, sim5, sim4, sim3, sim2):

        d5 = self.conv5(d5)
        d5 = self.SFM5(d5, sim5)
        d5 = F.interpolate(d5, d4.size()[2:], mode='bilinear')

        d4 = self.conv4(d4 + d5)
        d4 = self.SFM4(d4, sim4)
        d4 = F.interpolate(d4, d3.size()[2:], mode='bilinear')

        d3 = self.conv3(d3 + d4)
        d3 = self.SFM3(d3, sim3)
        d3 = F.interpolate(d3, d2.size()[2:], mode='bilinear')

        d2 = self.conv2(d2 + d3)
        d2 = self.SFM2(d2, sim2)

        mask = self.cls(d2)

        return mask


class Decoder_sim(nn.Module):
    def __init__(self, in_d):
        super(Decoder_sim, self).__init__()
        self.in_d = in_d
        self.conv4 = nn.Sequential(nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_d),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_d),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_d),
                                   nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_d),
                                   nn.ReLU(inplace=True))
        self.IM4 = Interaction_Module(channels=in_d)
        self.IM3 = Interaction_Module(channels=in_d)
        self.IM2 = Interaction_Module(channels=in_d)
        self.IM1 = Interaction_Module(channels=in_d)
        self.FEM = Feature_Enhancement_Module()

    def cal_sim(self, x1, x2):
        sim = F.cosine_similarity(x1, x2, dim=1)
        sim = torch.sigmoid(sim)
        sim = 1 - sim.unsqueeze(dim=1)
        return sim

    def forward(self, x1_1, x1_2, x1_3, x1_4, x2_1, x2_2, x2_3, x2_4):

        x1_1, x1_2, x1_3, x1_4 = self.FEM(x1_1, x1_2, x1_3, x1_4)
        x2_1, x2_2, x2_3, x2_4 = self.FEM(x2_1, x2_2, x2_3, x2_4)

        x1_4 = self.conv4(x1_4)
        x2_4 = self.conv4(x2_4)
        x1_4, x2_4 = self.IM4(x1_4, x2_4)
        sim4 = self.cal_sim(x1_4, x2_4)
        x1_4 = F.interpolate(x1_4, x1_3.size()[2:], mode='bilinear')
        x2_4 = F.interpolate(x2_4, x2_3.size()[2:], mode='bilinear')

        x1_3 = self.conv3(x1_4 + x1_3)
        x2_3 = self.conv3(x2_4 + x2_3)
        x1_3, x2_3 = self.IM3(x1_3, x2_3)
        sim3 = self.cal_sim(x1_3, x2_3)
        x1_3 = F.interpolate(x1_3, x1_2.size()[2:], mode='bilinear')
        x2_3 = F.interpolate(x2_3, x2_2.size()[2:], mode='bilinear')

        x1_2 = self.conv2(x1_3 + x1_2)
        x2_2 = self.conv2(x2_3 + x2_2)
        x1_2, x2_2 = self.IM2(x1_2, x2_2)
        sim2 = self.cal_sim(x1_2, x2_2)
        x1_2 = F.interpolate(x1_2, x1_1.size()[2:], mode='bilinear')
        x2_2 = F.interpolate(x2_2, x2_1.size()[2:], mode='bilinear')

        x1_1 = self.conv1(x1_2 + x1_1)
        x2_1 = self.conv1(x2_2 + x2_1)
        x1_1, x2_1 = self.IM1(x1_1, x2_1)
        sim1 = self.cal_sim(x1_1, x2_1)

        return sim4, sim3, sim2, sim1


if __name__ == '__main__':
    model = decoder(256, 2)
    x = torch.randn((1, 256, 8, 8))
    y1 = torch.randn((1, 77, 512))
    out = model(x, y1)
    print(out.shape)