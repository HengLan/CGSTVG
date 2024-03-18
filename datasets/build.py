import os
import math
import bisect
import copy


import torch
import torch.utils.data
from utils.comm import get_world_size

from torch.utils.data import DistributedSampler

from . import samplers
from . import transforms as T
from .vidstg import VidSTGDataset
from .hcstvg import HCSTVGDataset
from .collate_batch import collate_fn


def build_transforms(cfg, is_train=True):
    imsize = cfg.INPUT.RESOLUTION
    max_size = 720
    if is_train:
        flip_horizontal_prob = cfg.INPUT.FLIP_PROB_TRAIN
            
        scales = []
        if cfg.INPUT.AUG_SCALE:
            for i in range(4):
                scales.append(imsize - 32 * i)
        else:
            scales = [imsize]
            
        transform = T.Compose(
            [
                T.RandomHorizontalFlip(flip_horizontal_prob),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=max_size),
                        ]
                    ),
                ),
                T.Normalize(
                    mean=cfg.INPUT.PIXEL_MEAN,
                    std=cfg.INPUT.PIXEL_STD
                ),
            ]
        )

    else:
        transform = T.Compose(
            [
                T.RandomResize(imsize, max_size=max_size),
                T.Normalize(
                    mean=cfg.INPUT.PIXEL_MEAN,
                    std=cfg.INPUT.PIXEL_STD
                ),
            ]
        )

    return transform


def build_dataset(cfg, split, transforms):
    dataset_name = cfg.DATASET.NAME
    if dataset_name == 'VidSTG':
        return VidSTGDataset(
                    cfg,
                    split,
                    transforms
                )
    elif dataset_name == 'HC-STVG':
        return HCSTVGDataset(
            cfg,
            split,
            transforms
        )
    else:
        raise ValueError("{} is not Supported".format(dataset_name))


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        video_info = dataset.get_video_info(i)
        aspect_ratio = float(video_info["height"]) / float(video_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _count_frame_size(dataset):
    img_sizes = dict()
    for i in range(len(dataset)):
        video_info = dataset.get_video_info(i)
        img_sizes.setdefault((video_info['width'],video_info['height']),0)
        img_sizes[(video_info['width'],video_info['height'])] += 1


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, batch_size, num_iters=None, start_iter=0, is_train=True
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, batch_size, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last=True if is_train else False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, mode='train', is_distributed=False, start_iter=0):
    assert mode in {'train', 'val', 'test'}
    num_gpus = get_world_size()
    is_train = mode == 'train'

    transforms = build_transforms(cfg, is_train)
    dataset = build_dataset(cfg, mode, transforms)
    
    if is_train:
        videos_per_batch = cfg.SOLVER.BATCH_SIZE * num_gpus
        assert cfg.SOLVER.BATCH_SIZE == 1, "Each GPU should only take 1 video."
        videos_per_gpu = cfg.SOLVER.BATCH_SIZE
        shuffle = True
        num_epochs = cfg.SOLVER.MAX_EPOCH 
        num_iters = num_epochs * math.ceil(len(dataset) / videos_per_batch)
    else:
        assert cfg.SOLVER.BATCH_SIZE == 1, "Each GPU should only take 1 video."
        videos_per_gpu = cfg.SOLVER.BATCH_SIZE
        shuffle = False
        num_iters = None
        start_iter = 0

    # group videos which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        dataset, sampler, aspect_grouping, videos_per_gpu, num_iters, start_iter, is_train=is_train
    )
    num_workers = cfg.DATALOADER.NUM_WORKERS

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
    )

    return data_loader