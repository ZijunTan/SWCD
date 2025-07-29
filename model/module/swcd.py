import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.resnet import resnet18
from model.module.subtract import BFM
from model.module.decoder import Decoder, Decoder_sim
from model.module.neck_att import Adaptive_Aggregation_Module

class BaseNet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(BaseNet, self).__init__()
        self.res = resnet18(pretrained=True)
        self.res_mid_d = 64
        self.BFM = BFM(self.res_mid_d, self.res_mid_d)
        self.AAM = Adaptive_Aggregation_Module(self.res_mid_d)
        self.decoder = Decoder(self.res_mid_d, 2)
        self.decoder_sim = Decoder_sim(self.res_mid_d)

    def forward(self, x1, x2):
        xr1_0, xr1_1, xr1_2, xr1_3, xr1_4 = self.res.base_forward(x1)
        xr2_0, xr2_1, xr2_2, xr2_3, xr2_4 = self.res.base_forward(x2)

        dr4, dr3, dr2, dr1 = self.BFM(xr1_4, xr1_3, xr1_2, xr1_1, xr2_4, xr2_3, xr2_2, xr2_1)
        dr4, dr3, dr2, dr1 = self.AAM(dr4, dr3, dr2, dr1)

        sim4, sim3, sim2, sim1 = self.decoder_sim(xr1_1, xr1_2, xr1_3, xr1_4, xr2_1, xr2_2, xr2_3, xr2_4)
        sim_mask = F.interpolate(sim1, x1.size()[2:], mode='bilinear')

        mask = self.decoder(dr4, dr3, dr2, dr1, sim4, sim3, sim2, sim1)
        mask = F.interpolate(mask, x1.size()[2:], mode='bilinear')

        return mask, sim_mask

if __name__ == '__main__':
    torch.cuda.set_device(1)
    x1 = torch.randn((1, 3, 256, 256)).cuda()
    x2 = torch.randn((1, 3, 256, 256)).cuda()
    model = BaseNet(3, 2).cuda()
    out = model(x1, x2)
    for i in out:
        print(i.shape)




