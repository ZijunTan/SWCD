from data.transform import Transforms
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from option import Options


class Load_Dataset(Dataset):
    def __init__(self, opt):
        super(Load_Dataset, self).__init__()
        self.opt = opt
        self.label_rate = opt.label_rate
        self.phase = opt.phase

        file_root = opt.dataroot + '/' + opt.dataset

        self.img_names = open(file_root + '/' + opt.phase + '/list/' + opt.phase + '.txt').read().splitlines()
        self.t1_paths = [file_root + '/' + opt.phase + '/A/' + x for x in self.img_names]
        self.t2_paths = [file_root + '/' + opt.phase + '/B/' + x for x in self.img_names]
        self.label_paths = [file_root + '/' + opt.phase + '/label/' + x for x in self.img_names]
        self.label_weak_paths = [file_root + '/' + opt.phase + '/label_weak/' + x for x in self.img_names]

        if self.phase == 'train':
            if opt.label_rate is not None:
                self.label_img_names = open(file_root + '/' + opt.phase + '/list/' + opt.phase + '_semi_' + opt.label_rate +'.txt').read().splitlines()
                self.label_t1_paths = [file_root + '/' + opt.phase + '/A/' + x for x in self.label_img_names]
                self.label_t2_paths = [file_root + '/' + opt.phase + '/B/' + x for x in self.label_img_names]
                self.label_label_paths = [file_root + '/' + opt.phase + '/label/' + x for x in self.label_img_names]

        self.normalize = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.transform = transforms.Compose([Transforms()])
        self.to_tensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.t1_paths)

    def __getitem__(self, index):

        t1_path = self.t1_paths[index]
        t2_path = self.t2_paths[index]
        label_path = self.label_paths[index]
        label_weak_path = self.label_weak_paths[index]

        name = self.img_names[index]

        if self.phase == 'train':
            if self.label_rate is not None:
                if t1_path in self.label_t1_paths:
                    with_label = True
                else:
                    with_label = False
            else:
                with_label = True
        else:
            with_label = True

        img1 = Image.open(t1_path)
        img2 = Image.open(t2_path)
        label = np.array(Image.open(label_path)) // 255
        label = Image.fromarray(label)
        label_weak = np.array(Image.open(label_weak_path)) // 255
        label_weak = Image.fromarray(label_weak)

        if self.opt.phase == 'train':
            data = self.transform({'img1': img1, 'img2': img2, 'label': label, 'label_weak': label_weak})
            img1, img2, label, label_weak = data['img1'], data['img2'], data['label'], data['label_weak']

        img1 = self.to_tensor(img1)
        img2 = self.to_tensor(img2)
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        label = torch.from_numpy(np.array(label))
        label_weak = torch.from_numpy(np.array(label_weak))

        input_dict = {'img1': img1, 'img2': img2, 'label': label, 'label_weak': label_weak, 'label_flag': with_label, 'name': name}

        return input_dict


class DataLoader(torch.utils.data.Dataset):

    def __init__(self, opt):
        self.dataset = Load_Dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=opt.batch_size,
                                                      shuffle=opt.phase=='train',
                                                      pin_memory=True,
                                                      drop_last=opt.phase=='train',
                                                      num_workers=int(opt.num_workers))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    opt = Options().parse()
    train_loader = DataLoader(opt).load_data()
    for i, data in enumerate(train_loader):
        img1, img2, label, label_weak, label_flag, name = data['img1'], data['img2'], data['label'], data['label_weak'], data['label_flag'], data['fname']
        # print(img1.shape)
        # print(img2.shape)
        # print(label.shape)
        # print(label_weak.shape)
        # print(label_flag)
        if True in label_flag:
            true_indices = torch.nonzero(data['label_flag'], as_tuple=False)
            print(data['label_flag'])
            print(true_indices)
            print(true_indices.shape)
            print(data['img1'][true_indices].squeeze(1).shape)
            print(data['label'][true_indices].squeeze(1).shape)

        print('**************')

    # dataset = DataLoader(opt).dataset
    # for i in range(len(dataset)):
    #     daa = dataset.__getitem__(i)


