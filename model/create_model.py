import torch
from torch import nn
import torch.nn.functional as F
import os
import torch.optim as optim
from .schedular import get_cosine_schedule_with_warmup
from .loss.focal import cross_entropy
from model.module.swcd import BaseNet



class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.device = torch.device("cuda:%s" % opt.gpu_ids[0] if torch.cuda.is_available() else "cpu")
        self.opt = opt
        self.base_lr = opt.lr
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.detector = BaseNet(3, 2)

        self.bce = cross_entropy
        self.weight_dir = opt.weight_dir
        self.optimizer = optim.AdamW(self.detector.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

        self.schedular = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=625 * opt.warmup_epochs,
                                                         num_training_steps=625 * opt.num_epochs)
        if opt.load_pretrain:
            self.load_ckpt(self.detector)
        self.detector.cuda()

    def forward(self, x1, x2, label, label_weak, label_flag):
        pred, sim = self.detector(x1, x2)
        loss_sup = 0.0
        if True in label_flag:
            true_indices = torch.nonzero(label_flag, as_tuple=False)
            loss_sup = self.bce(pred[true_indices].squeeze(1), label[true_indices].squeeze(1))
        loss_weak = F.binary_cross_entropy(sim, label_weak.unsqueeze(1).float())
        loss = loss_sup + loss_weak
        return loss


    def inference(self, x1, x2):
        with torch.no_grad():
            pred, sim = self.detector(x1, x2)
        return pred

    def load_ckpt(self, network):
        if not os.path.isfile(self.weight_dir):
            print("%s not exists yet!" % self.weight_dir)
            raise ("%s must exist!" % self.weight_dir)
        else:
            checkpoint = torch.load(self.weight_dir, map_location='cpu')
            network.load_state_dict(checkpoint['network'], strict=False)

    def save_ckpt(self, network, optimizer):
        save_filename = 'best.pth'
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.exists(save_path):
            os.remove(save_path)
        torch.save({'network': network.cpu().state_dict(),
                    'optimizer': optimizer.state_dict()},
                   save_path)
        if torch.cuda.is_available():
            network.cuda()

    def save(self):
        self.save_ckpt(self.detector, self.optimizer)

    def name(self):
        return self.opt.name


def create_model(opt):
    model = Model(opt)
    print("model [%s] was created" % model.name())

    return model.cuda()

