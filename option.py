import argparse
import torch
import logging

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('--gpu_ids', type=str, default='2', help='gpu ids: e.g. 0. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='CLCD', help='project name')
        self.parser.add_argument('--dataroot', type=str, default="/10T/students/doctor/2024/tanzj/dataset")
        self.parser.add_argument('--dataset', type=str, default='CLCD_256')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--weight_dir', type=str,
                                 default="/10T/students/doctor/2024/tanzj/PyCharm-Remote/SWCD_Github/checkpoints/CLCD/best.pth", help='models are saved here')

        self.parser.add_argument('--label_rate', type=str, default='10', help='10, 30, 50, None')
        self.parser.add_argument('--result_dir', type=str, default='./results', help='results are saved here')
        self.parser.add_argument('--load_pretrain', type=bool, default=False)
        self.parser.add_argument('--phase', type=str, default='train')
        self.parser.add_argument('--input_size', type=int, default=256)
        self.parser.add_argument('--batch_size', type=int, default=16)
        self.parser.add_argument('--num_epochs', type=int, default=200)
        self.parser.add_argument('--warmup_epochs', type=int, default=20)
        self.parser.add_argument('--num_workers', type=int, default=8, help='#threads for loading data')
        self.parser.add_argument('--lr', type=float, default=3e-4)
        self.parser.add_argument('--weight_decay', type=float, default=5e-4)

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        return self.opt
