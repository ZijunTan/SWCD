import torch
import torch.nn as nn
import torch.nn.functional as F

class Spatial_Attention(nn.Module):
    def __init__(self, spatial_kernel=7):
        super(Spatial_Attention, self).__init__()
        self.mlp = nn.Sequential(nn.Conv2d(3, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False),
                                 nn.Sigmoid())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        merge = avg_out + max_out
        return x * self.mlp(torch.concat([merge, avg_out, max_out], dim=1))


class Channel_Attention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Channel_Attention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(channel // reduction, channel, 1, bias=False),
                                 nn.Sigmoid())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        p = self.pool(x)
        return x * self.mlp(p)


class FeatureFusionModule(nn.Module):
    def __init__(self, dim):
        super(FeatureFusionModule, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(dim, dim, 7, padding=6, groups=dim, dilation=2, bias=False),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(dim, dim, 7, padding=9, groups=dim, dilation=3, bias=False),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(dim, dim, 7, padding=12, groups=dim, dilation=4, bias=False),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))

        self.conv_dr1 = nn.Sequential(nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(dim),
                                      nn.ReLU(inplace=True))
        self.conv_dr2 = nn.Sequential(nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(dim),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(dim),
                                      nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))

        self.ca = Channel_Attention(dim)
        self.sa = Spatial_Attention()

    def forward(self, x):
        x = self.conv_dr1(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        att = torch.sigmoid(self.conv4(x1 + x2 + x3))
        out1 = att * x + x
        out2 = self.ca(x)
        out3 = self.sa(x)
        out = out1 + out2 + out3
        return out

class Feature_Enhancement_Module(nn.Module):
    def __init__(self, in_d=None, out_d=64):
        super(Feature_Enhancement_Module, self).__init__()
        if in_d is None:
            in_d = [64, 128, 256, 512]
        self.in_d = in_d
        self.mid_d = in_d[0]
        self.out_d = out_d

        self.conv_scale1_c1 = nn.Sequential(nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True))
        self.conv_scale2_c1 = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),
                                            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True))
        self.conv_scale3_c1 = nn.Sequential(nn.AvgPool2d(kernel_size=4, stride=4),
                                            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True))
        self.conv_scale4_c1 = nn.Sequential(nn.AvgPool2d(kernel_size=8, stride=8),
                                            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True))

        # scale 2
        self.conv_scale1_c2 = nn.Sequential(nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True))
        self.conv_scale2_c2 = nn.Sequential(nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True))
        self.conv_scale3_c2 = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),
                                            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True))
        self.conv_scale4_c2 = nn.Sequential(nn.AvgPool2d(kernel_size=4, stride=4),
                                            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True))

        # scale 3
        self.conv_scale1_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale2_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale3_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale4_c3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )

        # scale 4
        self.conv_scale1_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale2_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale3_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale4_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

        # fusion
        self.conv_aggregation_s1 = FeatureFusionModule(self.mid_d)
        self.conv_aggregation_s2 = FeatureFusionModule(self.mid_d)
        self.conv_aggregation_s3 = FeatureFusionModule(self.mid_d)
        self.conv_aggregation_s4 = FeatureFusionModule(self.mid_d)

    def forward(self, c1, c2, c3, c4):
        # scale 1
        c1_s1 = self.conv_scale1_c1(c1)
        c1_s2 = self.conv_scale2_c1(c1)
        c1_s3 = self.conv_scale3_c1(c1)
        c1_s4 = self.conv_scale4_c1(c1)

        # scale 2
        c2_s1 = F.interpolate(self.conv_scale1_c2(c2), scale_factor=(2, 2), mode='bilinear')
        c2_s2 = self.conv_scale2_c2(c2)
        c2_s3 = self.conv_scale3_c2(c2)
        c2_s4 = self.conv_scale4_c2(c2)

        # scale 3
        c3_s1 = F.interpolate(self.conv_scale1_c3(c3), scale_factor=(4, 4), mode='bilinear')
        c3_s2 = F.interpolate(self.conv_scale2_c3(c3), scale_factor=(2, 2), mode='bilinear')
        c3_s3 = self.conv_scale3_c3(c3)
        c3_s4 = self.conv_scale4_c3(c3)

        # scale 4
        c4_s1 = F.interpolate(self.conv_scale1_c4(c4), scale_factor=(8, 8), mode='bilinear')
        c4_s2 = F.interpolate(self.conv_scale2_c4(c4), scale_factor=(4, 4), mode='bilinear')
        c4_s3 = F.interpolate(self.conv_scale3_c4(c4), scale_factor=(2, 2), mode='bilinear')
        c4_s4 = self.conv_scale4_c4(c4)

        s1 = self.conv_aggregation_s1(torch.cat([c1_s1, c2_s1, c3_s1, c4_s1], dim=1))
        s2 = self.conv_aggregation_s2(torch.cat([c1_s2, c2_s2, c3_s2, c4_s2], dim=1))
        s3 = self.conv_aggregation_s3(torch.cat([c1_s3, c2_s3, c3_s3, c4_s3], dim=1))
        s4 = self.conv_aggregation_s4(torch.cat([c1_s4, c2_s4, c3_s4, c4_s4], dim=1))

        return s1, s2, s3, s4

if __name__ == '__main__':
    x1 = torch.randn((1, 64, 64, 64))
    x2 = torch.randn((1, 128, 32, 32))
    x3 = torch.randn((1, 256, 16, 16))
    x4 = torch.randn((1, 512, 8, 8))
    model = FeatureReinforcementModule()
    out = model(x1, x2, x3, x4)
    for i in out:
        print(i.shape)