# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
import random
from torchvision.transforms import functional as F
import torchvision.transforms as T
from utils.bounding_box import BoxList


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_dict):
        for t in self.transforms:
            input_dict = t(input_dict)
        return input_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ColorJitter(object):
    def __init__(self,brightness=0,contrast=0,saturation=0,hue=0):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, input_dict):
        if random.random() < 0.8:
            frames = input_dict['frames']
            frames = self.color_jitter(frames)
            input_dict['frames'] = frames
        
        return input_dict


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, input_dict):
        if random.random() < self.prob:
            frames = input_dict['frames']
            boxs = input_dict['boxs']
            text = input_dict['text']

            frames = F.hflip(frames)
            boxs = boxs.transpose(0)
            text = text.replace('right','*&^special^&*').replace('left','right').replace('*&^special^&*','left')

            input_dict['frames'] = frames
            input_dict['boxs'] = boxs
            input_dict['text'] = text

        return input_dict


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, input_dict):
        # if random.random() < self.p:
        #     return self.transforms1(input_dict)
        return self.transforms1(input_dict)


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, image_size):
        h, w = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, input_dict):
        frames = input_dict['frames']
        boxs = input_dict['boxs']
        img_size = (frames.shape[2],frames.shape[3])
        size = (frames.size(2), frames.size(3))  # self.get_size(img_size)

        frames = F.resize(frames, size)
        boxs = boxs.resize((size[1],size[0]))
        input_dict['frames'] = frames
        input_dict['boxs'] = boxs
        
        return input_dict


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, max_try: int=50):
        self.min_size = min_size
        self.max_size = max_size
        self.max_try  = max_try
    
    def __call__(self, input_dict):
        frames = input_dict['frames']
        boxs = input_dict['boxs']

        for _ in range(self.max_try):
            h = frames.shape[2]
            w = frames.shape[3]
            tw = random.randint(self.min_size, min(w, self.max_size))
            th = random.randint(self.min_size, min(h, self.max_size))
            
            region = T.RandomCrop.get_params(frames, [th, tw]) # [i, j, th, tw]
            if boxs.check_crop_valid(region):
                frames = F.crop(frames, *region)
                boxs = boxs.crop(region)
                input_dict['frames'] = frames
                input_dict['boxs'] = boxs
                return input_dict

        return input_dict


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, input_dict):
        frames = input_dict['frames']
        boxs = input_dict['boxs']
        frames = F.normalize(frames, mean=self.mean, std=self.std)
        assert boxs.size == (frames.shape[3],frames.shape[2])  # (w, h)
        boxs = boxs.normalize()
        input_dict['frames'] = frames
        input_dict['boxs'] = boxs
        return input_dict


class NormalizeAndPad(object):
    def __init__(self, mean, std, size, aug_translate=False):
        self.mean = mean
        self.std = std
        self.size = size
        self.aug_translate = aug_translate
    
    def __call__(self, input_dict):
        frames = input_dict['frames']
        frames = F.normalize(frames, mean=self.mean, std=self.std)
        
        t, _, h, w = frames.shape
        dw = self.size - w
        dh = self.size - h

        if self.aug_translate:
            top = random.randint(0, dh)
            left = random.randint(0, dw)
        else:
            top = round(dh / 2.0 - 0.1)
            left = round(dw / 2.0 - 0.1)

        out_frames = torch.zeros((t,3,self.size,self.size)).float()
        out_mask = torch.ones((self.size, self.size)).int()

        out_frames[:, :, top:top+h, left:left+w] = frames
        out_mask[top:top+h, left:left+w] = 0

        input_dict['frames'] = out_frames
        input_dict['mask'] = out_mask

        if 'boxs' in input_dict.keys():
            boxs = input_dict['boxs']
            boxs = boxs.shift((self.size,self.size),left,top)
            input_dict['boxs'] = boxs

        return input_dict