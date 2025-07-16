import random
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class Transforms(object):
    def __call__(self, _data):
        img1, img2, label, label_weak = _data['img1'], _data['img2'], _data['label'], _data['label_weak']

        if random.random() < 0.5:
            img1_ = img1
            img1 = img2
            img2 = img1_

        if random.random() < 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            label = TF.hflip(label)
            label_weak = TF.hflip(label_weak)

        if random.random() < 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            label = TF.vflip(label)
            label_weak = TF.vflip(label_weak)

        if random.random() < 0.5:
            angles = [90, 180, 270]
            angle = random.choice(angles)
            img1 = TF.rotate(img1, angle)
            img2 = TF.rotate(img2, angle)
            label = TF.rotate(label, angle)
            label_weak = TF.rotate(label_weak, angle)

        ### We didnt use colorjitters for the SV dataset.
        """
        if random.random() < 0.5:
            colorjitters = []
            brightness_factor = random.uniform(0.5, 1.5)
            colorjitters.append(Lambda(lambda img: TF.adjust_brightness(img, brightness_factor)))
            contrast_factor = random.uniform(0.5, 1.5)
            colorjitters.append(Lambda(lambda img: TF.adjust_contrast(img, contrast_factor)))
            saturation_factor = random.uniform(0.5, 1.5)
            colorjitters.append(Lambda(lambda img: TF.adjust_saturation(img, saturation_factor)))
            random.shuffle(colorjitters)
            colorjitter = Compose(colorjitters)
            img1 = colorjitter(img1)
            img2 = colorjitter(img2)
        """
        if random.random() < 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=(256, 256)).get_params(img=img1, scale=[0.333, 1.0],ratio=[0.75, 1.333])
            img1 = TF.resized_crop(img1, i, j, h, w, size=(256, 256), interpolation=InterpolationMode.BILINEAR)
            img2 = TF.resized_crop(img2, i, j, h, w, size=(256, 256), interpolation=InterpolationMode.BILINEAR)
            label = TF.resized_crop(label, i, j, h, w, size=(256, 256), interpolation=InterpolationMode.NEAREST)
            label_weak = TF.resized_crop(label_weak, i, j, h, w, size=(256, 256), interpolation=InterpolationMode.NEAREST)

        return {'img1': img1, 'img2': img2, 'label': label, 'label_weak': label_weak}


class Lambda(object):
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string





