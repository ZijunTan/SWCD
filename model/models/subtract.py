import torch
import torch.nn as nn
import torch.nn.functional as F


class Bitemporal_Fusion_Module(nn.Module):
    def __init__(self, in_d, out_d):
        super(Bitemporal_Fusion_Module, self).__init__()
        self.in_d = in_d
        self.out_d = out_d

        self.conv_sub = nn.Sequential(nn.Conv2d(self.in_d, self.in_d, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(self.in_d),
                                      nn.ReLU(inplace=True))
        self.conv_sub_out = nn.Sequential(nn.Conv2d(self.in_d * 2, self.out_d, 3, padding=1, bias=False),
                                          nn.BatchNorm2d(self.out_d),
                                          nn.ReLU(inplace=True))
        self.conv_sim_out = nn.Sequential(nn.Conv2d(self.in_d * 2, self.out_d, 3, padding=1, bias=False),
                                          nn.BatchNorm2d(self.out_d),
                                          nn.ReLU(inplace=True))

    def forward(self, x1, x2):

        x_sub = torch.abs(x1 - x2)
        x_sub_att = torch.sigmoid(self.conv_sub(x_sub))
        x1_sub = x1 * x_sub_att + x1
        x2_sub = x2 * x_sub_att + x2
        x_sub_out = self.conv_sub_out(torch.cat([x1_sub, x2_sub], dim=1))

        sim = F.cosine_similarity(x1, x2, dim=1)
        sim = torch.sigmoid(sim)
        x_sim_att = 1 - sim.unsqueeze(dim=1)
        x1_sim = x1 * x_sim_att + x1
        x2_sim = x2 * x_sim_att + x2
        x_sim_out = self.conv_sim_out(torch.cat([x1_sim, x2_sim], dim=1))

        return x_sub_out + x_sim_out


class BFM(nn.Module):
    def __init__(self, in_d, out_d):
        super(BFM, self).__init__()
        self.subtract4 = Bitemporal_Fusion_Module(in_d * 8, out_d)
        self.subtract3 = Bitemporal_Fusion_Module(in_d * 4, out_d)
        self.subtract2 = Bitemporal_Fusion_Module(in_d * 2, out_d)
        self.subtract1 = Bitemporal_Fusion_Module(in_d, out_d)

    def forward(self, x1_4, x1_3, x1_2, x1_1, x2_4, x2_3, x2_2, x2_1):
        d4 = self.subtract4(x1_4, x2_4)
        d3 = self.subtract3(x1_3, x2_3)
        d2 = self.subtract2(x1_2, x2_2)
        d1 = self.subtract1(x1_1, x2_1)
        return d4, d3, d2, d1


if __name__ == '__main__':
    x1 = torch.randn((32, 512, 8, 8))
    x2 = torch.randn((32, 512, 8, 8))
    model = SKNet(512, 512)
    out = model(x1)
    print(out.shape)