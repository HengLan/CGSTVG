import torch
import random
import math
import re
import numpy as np
from copy import copy
from pytorch_pretrained_bert.tokenization import BertTokenizer

from .gaussion_hm import gaussian_radius, draw_umich_gaussian


SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


def crop_for_2d_map(cfg, video_data):
    p = random.random()
    if p < 1 - cfg.INPUT.TEMP_CROP_PROB:
        return video_data

    if len(video_data['frame_ids']) <= cfg.MODEL.TEMPFORMER.MAX_MAP_SIZE:
        return video_data
    
    if cfg.DATASET.NAME == 'VidSTG':
        data_item = {
            'item_id' : video_data['item_id'],
            'vid' : video_data['vid'],
            'width' : video_data['width'],
            'height' : video_data['height'],
            'qtype' : video_data['qtype'],
            'description' : video_data['description'],
            'object' : video_data['object'],
            'bboxs' :  video_data['bboxs'],
            'gt_temp_bound' : video_data['gt_temp_bound'],
            'segment_bound' : video_data['segment_bound']
        }
    elif cfg.DATASET.NAME == 'HC-STVG':
        data_item = {
            'item_id' : video_data['item_id'],
            'vid' : video_data['vid'],
            'width' : video_data['width'],
            'height' : video_data['height'],
            'description' : video_data['description'],
            'object' : video_data['object'],
            'bboxs' :  video_data['bboxs'],
            'gt_temp_bound' : video_data['gt_temp_bound']
        }

    video_frames = copy(video_data['frame_ids'])
    gt_mask = video_data['actioness'].copy()
    start_heatmap = video_data['start_heatmap'].copy()
    end_heatmap = video_data['end_heatmap'].copy()
    
    action_span = np.where(gt_mask)[0]
    starts_list = [i for i in range(len(video_frames)) if i < action_span[0]]
    ends_list = [i for i in range(len(video_frames)) if i > action_span[-1]]

    max_try = 30
    for _ in range(max_try):
        if starts_list:
            start_idx = random.choice(starts_list)
        else:
            start_idx = 0
        if ends_list:
            end_idx = random.choice(ends_list)
        else:
            end_idx = len(video_frames) - 1

        if end_idx - start_idx + 1 >= cfg.MODEL.TEMPFORMER.MAX_MAP_SIZE:
            sample_slice = slice(start_idx,end_idx+1)
            data_item.update({
                'frame_ids' : video_frames[sample_slice], 
                'actioness' : gt_mask[sample_slice],
                'start_heatmap' : start_heatmap[sample_slice], 
                'end_heatmap' : end_heatmap[sample_slice]}
            )
            return data_item
    
    return video_data


def make_hcstvg_input_clip(cfg, split, video_data):    
    input_fps = cfg.INPUT.SAMPLE_FPS
    if split == 'test':
        input_fps = input_fps * 2
    crop = True
    if split == "train":
        p = random.random()
        if p < 1 - cfg.INPUT.TEMP_CROP_PROB:
            crop = False
    else:
        crop = False
   
    data_item = {
        'item_id' : video_data['item_id'],
        'vid' : video_data['vid'],
        'width' : video_data['width'],
        'height' : video_data['height'],
        'description' : video_data['description'],
        'object' : video_data['object'],
        'bboxs' :  video_data['bboxs'],
        'gt_temp_bound' : video_data['gt_temp_bound'],
    }
    
    video_frames = copy(video_data['frame_ids'])
    gt_mask = video_data['actioness'].copy()
    start_heatmap = video_data['start_heatmap'].copy()
    end_heatmap = video_data['end_heatmap'].copy()
    frame_count = video_data['frame_count']
    
    video_fps = frame_count / 20   # the duration of hc-stvg videos is 20s
    sampling_rate = input_fps / video_fps
    sample_slice = [0]
    
    for idx in range(0, len(video_frames)):
        frame_id = video_frames[idx]
        if int(video_frames[sample_slice[-1]] * sampling_rate) < int(frame_id * sampling_rate):
            sample_slice.append(idx)
                
    if sample_slice[-1] != len(video_frames) - 1:
        sample_slice.append(len(video_frames) - 1)
    
    video_frames = [video_frames[idx] for idx in sample_slice]
    gt_mask = gt_mask[sample_slice]
    start_heatmap = start_heatmap[sample_slice]
    end_heatmap = end_heatmap[sample_slice]
    
    if crop:
        # Perform the temporal crop
        action_span = np.where(gt_mask)[0]
        starts_list = [i for i in range(len(video_frames)) if i < action_span[0]]
        ends_list = [i for i in range(len(video_frames)) if i > action_span[-1]]
    
        if starts_list:
            start_idx = random.choice(starts_list)
        else:
            start_idx = 0
        if ends_list:
            end_idx = random.choice(ends_list)
        else:
            end_idx = len(video_frames) - 1

        sample_slice = slice(start_idx,end_idx+1)
        video_frames = video_frames[sample_slice]
        gt_mask = gt_mask[sample_slice]
        start_heatmap = start_heatmap[sample_slice]
        end_heatmap = end_heatmap[sample_slice]
        
    data_item.update({
        'frame_ids' : video_frames, 
        'actioness' : gt_mask,
        'start_heatmap' : start_heatmap, 
        'end_heatmap' : end_heatmap}
    )
    
    return data_item


def make_vidstg_input_clip(cfg, split, video_data):
    if split == 'train':
        input_frame_num = cfg.INPUT.TRAIN_SAMPLE_NUM
    else:
        input_frame_num = cfg.INPUT.TRAIN_SAMPLE_NUM * 2
    
    crop = False
    if split == "train":
        p = random.random()
        if p < cfg.INPUT.TEMP_CROP_PROB:
            crop = True
    
    data_item = {
        'item_id' : video_data['item_id'],
        'vid' : video_data['vid'],
        'width' : video_data['width'],
        'height' : video_data['height'],
        'qtype' : video_data['qtype'],
        'description' : video_data['description'],
        'object' : video_data['object'],
        'bboxs' :  video_data['bboxs'],
        'gt_temp_bound' : video_data['gt_temp_bound'],
        'segment_bound' : video_data['segment_bound']
    }
    
    video_frames = copy(video_data['frame_ids'])
    gt_mask = video_data['actioness'].copy()
    start_heatmap = video_data['start_heatmap'].copy()
    end_heatmap = video_data['end_heatmap'].copy()
    
    if crop:
        # Perform the temporal crop
        action_span = np.where(gt_mask)[0]
        starts_list = [i for i in range(len(video_frames)) if i < action_span[0]]
        ends_list = [i for i in range(len(video_frames)) if i > action_span[-1]]
    
        if starts_list:
            start_idx = random.choice(starts_list)
        else:
            start_idx = 0
        if ends_list:
            end_idx = random.choice(ends_list)
        else:
            end_idx = len(video_frames) - 1

        sample_slice = list(range(start_idx, end_idx + 1))  
    
    else:
        sample_slice = list(range(0, len(video_frames))) 

    # Need to Sample the input frames to satisfy
    if len(sample_slice) > input_frame_num:
        sample_slice_idx = np.linspace(0, len(sample_slice)-1, num=input_frame_num)
        sample_slice_idx = [int(idx) for idx in sample_slice_idx]
        assert len(set(sample_slice_idx)) == len(sample_slice_idx)
        sample_slice = [sample_slice[idx] for idx in sample_slice_idx]
    
    data_item.update({
        'frame_ids' : [video_frames[idx] for idx in sample_slice], 
        'actioness' : gt_mask[sample_slice],
        'start_heatmap' : start_heatmap[sample_slice], 
        'end_heatmap' : end_heatmap[sample_slice]}
    )
    
    return data_item


def crop_clip(cfg, video_data):
    """
    Usage:
        random crop a video clip while preserve its groundtruth
    Args:
        cfg: config file
        video_data : a groundtruth data item (Down FPS sampled) 
    """
    p = random.random()
    if p < 1 - cfg.INPUT.TEMP_CROP_PROB:
        return video_data

    if cfg.DATASET.NAME == 'VidSTG':
        data_item = {
            'item_id' : video_data['item_id'],
            'vid' : video_data['vid'],
            'width' : video_data['width'],
            'height' : video_data['height'],
            'qtype' : video_data['qtype'],
            'description' : video_data['description'],
            'object' : video_data['object'],
            'bboxs' :  video_data['bboxs'],
            'gt_temp_bound' : video_data['gt_temp_bound'],
            'segment_bound' : video_data['segment_bound']
        }
    elif cfg.DATASET.NAME == 'HC-STVG':
        data_item = {
            'item_id' : video_data['item_id'],
            'vid' : video_data['vid'],
            'width' : video_data['width'],
            'height' : video_data['height'],
            'description' : video_data['description'],
            'object' : video_data['object'],
            'bboxs' :  video_data['bboxs'],
            'gt_temp_bound' : video_data['gt_temp_bound']
        }

    video_frames = copy(video_data['frame_ids'])
    gt_mask = video_data['actioness'].copy()
    start_heatmap = video_data['start_heatmap'].copy()
    end_heatmap = video_data['end_heatmap'].copy()
    
    action_span = np.where(gt_mask)[0]
    starts_list = [i for i in range(len(video_frames)) if i < action_span[0]]
    ends_list = [i for i in range(len(video_frames)) if i > action_span[-1]]

    if starts_list:
        start_idx = random.choice(starts_list)
    else:
        start_idx = 0
    if ends_list:
        end_idx = random.choice(ends_list)
    else:
        end_idx = len(video_frames) - 1

    sample_slice = slice(start_idx,end_idx+1)
    data_item.update({
        'frame_ids' : video_frames[sample_slice], 
        'actioness' : gt_mask[sample_slice],
        'start_heatmap' : start_heatmap[sample_slice], 
        'end_heatmap' : end_heatmap[sample_slice]}
    )
    
    return data_item


def iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = torch.tensor([gt[0]]).float(), torch.tensor([gt[1]]).float()
    # print(s.dtype, start.dtype)
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union


def score2d_to_moments_scores(score2d, num_clips, duration):
    grids = score2d.nonzero()
    scores = score2d[grids[:,0], grids[:,1]]
    grids[:, 1] += 1
    moments = grids * duration / num_clips
    return moments, scores


def make_2dmap(cfg, video_data):
    num_clips = cfg.MODEL.TEMPFORMER.MAX_MAP_SIZE
    iou2d = torch.ones(num_clips, num_clips)
    # the Input segment frames num
    duration = video_data['frame_ids'][-1] - video_data['frame_ids'][0]  + 1 
    candidates, _ = score2d_to_moments_scores(iou2d, num_clips, duration)    
    moment = video_data['gt_temp_bound']
    iou2d = iou(candidates, moment).reshape(num_clips, num_clips)
    return iou2d, candidates


def sample_clip(cfg, video_data):
    """
    Usage:
        sample a samll video clip and its groundtruth
    Args:
        cfg: config file
        video_data : a groundtruth data item (Down FPS sampled) 
    """
    data_item = {
        'gt_file' : video_data['gt_file'],
        'width' : video_data['width'],
        'height' : video_data['height'],
        'description' : video_data['description'],
        'object' : video_data['object']
    }
    boxs = video_data['bboxs'].copy()
    video_frames = copy(video_data['frame_names'])
    gt_mask = video_data['actioness'].copy()
    start_heatmap = video_data['start_heatmap'].copy()
    end_heatmap = video_data['end_heatmap'].copy()

    gt_temp_length = boxs.shape[0]
    clip_length = cfg.DATASET.NUM_CLIP_FRAMES
    min_gt_num = min(cfg.DATASET.MIN_GT_FRAME,gt_temp_length)
    
    video_length = len(video_frames)
    assert gt_mask.shape[0] == video_length

    action_span = np.where(gt_mask)[0]
    min_start_idx = max(0, action_span[0] + min_gt_num - clip_length)
    max_start_idx = min(max(0,video_length - clip_length), action_span[-1] - min_gt_num + 1)     

    start_idx = random.choice(list(range(min_start_idx,max_start_idx+1))) 
    sample_slice = slice(start_idx,start_idx + clip_length)
    bbox_slice = slice(max(0,start_idx - action_span[0]),start_idx + clip_length - action_span[0])
    data_item.update({
        'frame_names' : video_frames[sample_slice], 
        'actioness' : gt_mask[sample_slice],
        'start_heatmap' : start_heatmap[sample_slice], 
        'end_heatmap' : end_heatmap[sample_slice],
        'bboxs' : boxs[bbox_slice]}
    )
    assert np.where(data_item['actioness'])[0].shape[0] == data_item['bboxs'].shape[0]

    return data_item
    

def make_heatmap(cfg,input_dict):
    """
    Usage:
        Generate the Gaussion hetmap for the bounding box
    Args:
        cfg: config file
        input_dict : images and its bounding box 
    """
    video_clip = input_dict['frames']
    bboxs = input_dict['boxs'].bbox
    gt_mask = input_dict['actioness']
    action_span = np.where(gt_mask)[0]

    input_t = video_clip.shape[0]
    input_h = video_clip.shape[-2]
    input_w = video_clip.shape[-1]
    output_h = input_h // cfg.MODEL.DOWN_RATIO
    output_w = input_w // cfg.MODEL.DOWN_RATIO
    
    hm = np.zeros((input_t, output_h, output_w), dtype=np.float32)
    wh = np.zeros((input_t, 2), dtype=np.float32)
    offset = np.zeros((input_t, 2), dtype=np.float32)


    for box_idx in range(len(bboxs)):
        bbox = bboxs[box_idx].numpy()
        bbox[0] = bbox[0] * (output_w / input_w)
        bbox[1] = bbox[1] * (output_h / input_h)
        bbox[2] = bbox[2] * (output_w / input_w)
        bbox[3] = bbox[3] * (output_h / input_h)

        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        
        assert h > 0 and w > 0

        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)

        assert 0 <= ct_int[0] and ct_int[0] <= output_w and 0 <= ct_int[1] and ct_int[1] <= output_h

        frame_idx = action_span[box_idx]
        draw_umich_gaussian(hm[frame_idx], ct_int, radius)
        wh[frame_idx] = 1. * w, 1. * h
        offset[frame_idx] = ct - ct_int

    return hm, wh, offset


## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    line = input_line 
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    
    examples.append(InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    return examples


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            raise NotImplementedError
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def make_word_tokens(cfg,sentence,index,vocab=None):
    max_query_len = cfg.INPUT.MAX_QUERY_LEN

    if cfg.MODEL.USE_LSTM:
        # words = SENTENCE_SPLIT_REGEX.split(sentence.strip())
        # words = [w.lower() for w in words if (len(w) > 0 and w!=' ')]   ## do not include space as a token
        # if words[-1] == '.' or words[-1] == '?' or words[-1] == '!':
        #     words = words[:-1]
        words = sentence.strip().split()
        word_idx = torch.tensor([vocab.stoi.get(w.lower(), 400000) for w in words], dtype=torch.long)
        padded_word_idx = torch.zeros(max_query_len,dtype=torch.long)
        padded_word_idx[:word_idx.shape[0]] = word_idx
        word_mask = torch.zeros(max_query_len,dtype=torch.long)
        word_mask[:word_idx.shape[0]] = 1
        word_idx = padded_word_idx

    else:
        bert_model = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        ## encode phrase to bert input
        examples = read_examples(sentence, index)
        features = convert_examples_to_features(
            examples=examples, seq_length=max_query_len, tokenizer=tokenizer)
        word_idx = torch.tensor(features[0].input_ids,dtype=torch.long) 
        word_mask = torch.tensor(features[0].input_mask,dtype=torch.long) 

    return word_idx, word_mask