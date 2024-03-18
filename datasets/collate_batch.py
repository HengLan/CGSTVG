import torch
import numpy as np
from utils.misc import NestedTensor


def collate_fn(batch):
    transposed_batch = list(zip(*batch))
    videos = transposed_batch[0]
    texts = transposed_batch[1]
    targets  = transposed_batch[2]
    
    batch_dict = {}
    batch_dict['durations'] = [video.shape[0] for video in videos]
    batch_dict['videos']  = NestedTensor.from_tensor_list(videos)
    batch_dict['texts'] = [text for text in texts]
    batch_dict['targets'] = [target for target in targets]

    return batch_dict
    