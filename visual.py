import os

from util.metric_tool import ConfuseMatrixMeter
import torch
from option import Options
from data.cd_dataset import DataLoader
from model.create_model import create_model
from tqdm import tqdm
import numpy as np
from PIL import Image

if __name__ == '__main__':
    opt = Options().parse()
    opt.phase = 'test'
    opt.batch_size = 1
    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()
    test_size = len(test_loader)
    print("#testing images = %d" % test_size)

    opt.load_pretrain = True
    model = create_model(opt)

    vis_dir = opt.result_dir
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    model.eval()
    with torch.no_grad():
        for i, _data in enumerate(test_data):
            print('{}/{},{}'.format(i+1, test_size, _data['name'][0]))
            val_pred = model.inference(_data['img1'].cuda(), _data['img2'].cuda())
            val_target = _data['label'].detach()
            val_pred = torch.argmax(val_pred.detach(), dim=1)
            pr = val_pred[0].cpu().numpy()
            gt = val_target[0].cpu().numpy()
            index_tp = np.where(np.logical_and(pr == 1, gt == 1))
            index_fp = np.where(np.logical_and(pr == 1, gt == 0))
            index_tn = np.where(np.logical_and(pr == 0, gt == 0))
            index_fn = np.where(np.logical_and(pr == 0, gt == 1))

            map = np.zeros([gt.shape[0], gt.shape[1], 3])
            map[index_tp] = [255, 255, 255]
            map[index_fp] = [255, 0, 0]
            map[index_tn] = [0, 0, 0]
            map[index_fn] = [0, 255, 255]
            change_map = Image.fromarray(np.array(map, dtype=np.uint8))
            path = os.path.join(vis_dir, _data['name'][0])
            change_map.save(path)


