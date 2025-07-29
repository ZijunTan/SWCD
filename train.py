import time
import torch
from option import Options
from data.cd_dataset import DataLoader
from model.create_model import create_model
from tqdm import tqdm
import math
from util.metric_tool import ConfuseMatrixMeter
import os
import numpy as np
import random
import logging
import datetime
logging.getLogger('PIL').setLevel(logging.WARNING)

def init_logging(filedir: str):
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    logger = logging.getLogger()
    fh = logging.FileHandler(filename=filedir + '/log_' + get_date_str() + '.txt')
    sh = logging.StreamHandler()
    formatter_fh = logging.Formatter('%(asctime)s %(message)s')
    formatter_sh = logging.Formatter('%(message)s')
    # formatter_sh = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')
    fh.setFormatter(formatter_fh)
    sh.setFormatter(formatter_sh)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(10)
    fh.setLevel(10)
    sh.setLevel(10)
    return logging

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  #!
    torch.backends.cudnn.benchmark = True       #!
    torch.backends.cudnn.enabled = True         #! for accelerating training

class Trainval(object):
    def __init__(self, opt):
        self.opt = opt

        train_loader = DataLoader(opt)
        self.train_data = train_loader.load_data()
        train_size = len(train_loader)
        logging.info("#training images = %d" % train_size)
        opt.phase = 'test'
        opt.batch_size = 1
        val_loader = DataLoader(opt)
        self.val_data = val_loader.load_data()
        val_size = len(val_loader)
        logging.info("#validation images = %d" % val_size)
        opt.phase = 'train'

        self.model = create_model(opt)
        self.optimizer = self.model.optimizer
        self.schedular = self.model.schedular
        self.best_epoch = 0

        self.iters = 0
        self.total_iters = math.ceil(train_size / opt.batch_size) * opt.num_epochs
        self.previous_best = 0.0
        self.running_metric = ConfuseMatrixMeter(n_class=2)

    def train(self):
        tbar = tqdm(self.train_data)
        opt.phase = 'train'
        _loss = 0.0
        for i, data in enumerate(tbar):
            self.model.detector.train()
            loss = self.model(data['img1'].cuda(), data['img2'].cuda(), data['label'].cuda(), data['label_weak'].cuda(), data['label_flag'].cuda())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.schedular.step()
            _loss += loss.item()

            tbar.set_description("Loss: %.4f, LR: %.6f" % (_loss / (i + 1), self.optimizer.param_groups[0]['lr']))

        logging.info('Train_loss: %.4f' % (_loss/len(self.train_data)))

    def val(self):
        tbar = tqdm(self.val_data, ncols=80)
        self.running_metric.clear()
        opt.phase = 'test'
        self.model.eval()

        with torch.no_grad():
            for i, _data in enumerate(tbar):
                val_pred = self.model.inference(_data['img1'].cuda(), _data['img2'].cuda())
                # update metric
                val_target = _data['label'].detach()
                val_pred = torch.argmax(val_pred.detach(), dim=1)
                _ = self.running_metric.update_cm(pr=val_pred.cpu().numpy(), gt=val_target.cpu().numpy())
            val_scores = self.running_metric.get_scores()
            message = '(phase: %s) ' % (self.opt.phase)
            for k, v in val_scores.items():
                message += '%s: %.3f ' % (k, v * 100)
            logging.info(message)

        if val_scores['F1_1'] >= self.previous_best:
            self.model.save()
            self.previous_best = val_scores['F1_1']
            self.best_epoch = epoch

if __name__ == "__main__":
    opt = Options().parse()
    if not os.path.exists(opt.checkpoint_dir + '/' + opt.name):
        os.makedirs(opt.checkpoint_dir + '/' + opt.name)
    logging = init_logging(opt.checkpoint_dir + '/' + opt.name)
    logging.info(f"opt: {opt}\t")
    trainval = Trainval(opt)
    setup_seed(seed=1)
    for epoch in range(1, opt.num_epochs + 1):
        logging.info("\n==> Name %s, Epoch %i, previous best = %.4f in epoch %i" % (opt.name, epoch, trainval.previous_best * 100, trainval.best_epoch))
        trainval.train()
        trainval.val()



