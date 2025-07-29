import torch
import torch.nn as nn
import torch.nn.functional as F


class GLobal_Aggregation_Module(nn.Module):
    def __init__(self, dim):
        super(GLobal_Aggregation_Module, self).__init__()
        self.conv_dr = nn.Sequential(nn.Conv2d(dim * 4, dim, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(dim),
                                     nn.ReLU(inplace=True))

        self.pools_sizes = [2, 4, 8]
        self.conv_pool1 = nn.Sequential(nn.AvgPool2d(kernel_size=self.pools_sizes[0], stride=self.pools_sizes[0]),
                                        nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(dim),
                                        nn.ReLU(inplace=True))

        self.conv_pool2 = nn.Sequential(nn.AvgPool2d(kernel_size=self.pools_sizes[1], stride=self.pools_sizes[1]),
                                        nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(dim),
                                        nn.ReLU(inplace=True))

        self.conv_pool3 = nn.Sequential(nn.AvgPool2d(kernel_size=self.pools_sizes[2], stride=self.pools_sizes[2]),
                                        nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(dim),
                                        nn.ReLU(inplace=True))

    def forward(self, d4, d3, d2, d1):
        d4 = F.interpolate(d4, d1.size()[2:], mode='bilinear')
        d3 = F.interpolate(d3, d1.size()[2:], mode='bilinear')
        d2 = F.interpolate(d2, d1.size()[2:], mode='bilinear')

        x = torch.cat([d4, d3, d2, d1], dim=1)
        x = self.conv_dr(x)

        d1 = x
        d2 = self.conv_pool1(x)
        d3 = self.conv_pool2(x)
        d4 = self.conv_pool3(x)

        return d4, d3, d2, d1

class Local_Aggregation_Module(nn.Module):
    def __init__(self, dim):
        super(Local_Aggregation_Module, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=5, padding=2, bias=False),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=7, padding=3, bias=False),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))
        self.spatial_se = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=3, kernel_size=7, padding=3),
                                        nn.Sigmoid())
        self.conv4 = nn.Sequential(nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))

    def forward_conv(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        conv_out = torch.cat([x1, x2, x3], dim=1)
        avg_att = torch.mean(conv_out, dim=1, keepdim=True)
        max_att, _ = torch.max(conv_out, dim=1, keepdim=True)
        att = torch.cat([avg_att, max_att], dim=1)
        att = self.spatial_se(att)
        out = torch.cat([x1 * att[:, 0, :, :].unsqueeze(1), x2 * att[:, 1, :, :].unsqueeze(1), x3 * att[:, 2, :, :].unsqueeze(1)], dim=1)
        out = self.conv4(out)
        return out

    def forward(self, x):
        conv_out = self.forward_conv(x)
        return conv_out

class Adaptive_Aggregation_Module(nn.Module):
    def __init__(self, dim):
        super(Adaptive_Aggregation_Module, self).__init__()
        self.LAM1 = Local_Aggregation_Module(dim)
        self.LAM2 = Local_Aggregation_Module(dim)
        self.LAM3 = Local_Aggregation_Module(dim)
        self.LAM4 = Local_Aggregation_Module(dim)

        self.GAM = GLobal_Aggregation_Module(dim)

    def forward(self, d4, d3, d2, d1):

        g4, g3, g2, g1 = self.GAM(d4, d3, d2, d1)

        l4 = self.LAM4(d4)
        l3 = self.LAM3(d3 + F.interpolate(d4, d3.size()[2:], mode='bilinear'))
        l2 = self.LAM2(d2 + F.interpolate(d3, d2.size()[2:], mode='bilinear'))
        l1 = self.LAM1(d1 + F.interpolate(d2, d1.size()[2:], mode='bilinear'))

        d4 = l4 + g4
        d3 = l3 + g3
        d2 = l2 + g2
        d1 = l1 + g1

        return d4, d3, d2, d1

if __name__ == '__main__':
    torch.cuda.set_device(1)
    x1 = torch.randn((1, 64, 64, 64)).cuda()
    x2 = torch.randn((1, 64, 32, 32)).cuda()
    x3 = torch.randn((1, 64, 16, 16)).cuda()
    x4 = torch.randn((1, 64, 8, 8)).cuda()
    model = Adaptive_Aggregation_Module(64).cuda()
    out = model(x4, x3, x2, x1)
    for i in out:
        print(i.shape)



