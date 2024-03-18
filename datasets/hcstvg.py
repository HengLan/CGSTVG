import os
import json
from copy import deepcopy
import torch
import random

from tqdm import tqdm
import torch.utils.data as data
import numpy as np
from PIL import Image
import ffmpeg

from torchvision.transforms import ToTensor, ToPILImage, Resize
from utils.bounding_box import BoxList
from .data_utils import make_hcstvg_input_clip


import cv2
import torch
import numpy as np

class HCSTVGDataset(data.Dataset):

    def __init__(self, cfg, split, transforms=None) -> None:
        super(HCSTVGDataset,self).__init__()
        assert split in ['train', 'test']
        self.cfg = cfg.clone()
        self.split = split
        self.transforms = transforms

        self.data_dir = cfg.DATA_DIR
        self.anno_dir = os.path.join(self.data_dir,'annos/hcstvg_v2' if 'hc-stvg2' in self.data_dir else 'annos/hcstvg_v1')
        self.sent_file = os.path.join(self.anno_dir, f'{split}.json')  # split
        self.epsilon = 1e-10

        self.all_gt_data = self.load_data()
        self.clean_miss()
        self.vocab = None
        
        if cfg.DATA_TRUNK is not None:
            self.all_gt_data = self.all_gt_data[:cfg.DATA_TRUNK]
    
    def clean_miss(self):
        miss_name = '10__Gvp-cj3bmIY.mp4'
        for item in self.all_gt_data:
            if item['vid'] == miss_name:
                self.all_gt_data.remove(item)
                break
        
        miss_name = '1_aMYcLyh9OhU.mkv'
        for item in self.all_gt_data:
            if item['vid'] == miss_name:
                self.all_gt_data.remove(item)
                break
        
    def get_video_info(self,index):
        video_info = {}
        data_item = self.all_gt_data[index]
        video_info['height'] = data_item['height']
        video_info['width'] = data_item['width']
        return video_info

    def load_frames(self, data_item, load_video=True):
        video_name = data_item['vid']
        frame_ids = data_item['frame_ids']
        patience = 20
        max_rate = 1.4

        if load_video:
            video_path = os.path.join(self.data_dir,'v2_video' if 'hc-stvg2' in self.data_dir else 'v1_video', video_name)
            h, w = data_item['height'], data_item['width']
            succ_flag = False
            for _ in range(patience):
                try:
                    out, _ = (
                        ffmpeg
                        .input(video_path)
                        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                        .run(capture_stdout=True, quiet=True)
                    )
                    frames = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
                    succ_flag = True
                    if succ_flag:
                        break
                except Exception:
                    # print(video_name)
                    aa = 0

            if not succ_flag:
                print("video load wrong", video_path)
                frames = np.ones((1000, self.cfg.INPUT.RESOLUTION, int(self.cfg.INPUT.RESOLUTION*max_rate), 3), dtype=np.uint8)
                # raise RuntimeError("Load Video Error")
            try:
                frames = frames[frame_ids]
            except:
                print("frame_ids wrong", video_path)
                frames = np.ones((1000, self.cfg.INPUT.RESOLUTION, int(self.cfg.INPUT.RESOLUTION*max_rate), 3), dtype=np.uint8)
                frames = frames[frame_ids]

            rate = frames.shape[2] / frames.shape[1]
            frames = [Resize((self.cfg.INPUT.RESOLUTION, min(int(self.cfg.INPUT.RESOLUTION*rate), int(self.cfg.INPUT.RESOLUTION*max_rate))), antialias=True)(ToTensor()(frame)) for frame in frames]
            
            frames = torch.stack(frames)
        else:
            raise NotImplementedError("Not Implement load from frames")

        return frames

    def __getitem__(self, index: int):
        """
        Usage:
            In training, sample a random clip from video
            In testing, chunk the video to a set of clips
        """
        video_data = deepcopy(self.all_gt_data[index]) 

        data_item = make_hcstvg_input_clip(self.cfg, self.split, video_data)
                
        frames = self.load_frames(data_item)   # T * C * H * W

        # load the sampled gt bounding box
        frame_ids = data_item['frame_ids']
        temp_gt = data_item['gt_temp_bound']
        action_idx = np.where(data_item['actioness'])[0]
        start_idx, end_idx = action_idx[0], action_idx[-1]
        bbox_idx = [frame_ids[idx] - temp_gt[0] for idx in range(start_idx,end_idx + 1)]
        bboxs = torch.from_numpy(data_item['bboxs'][bbox_idx]).reshape(-1, 4)
        assert bboxs.shape[0] == len(action_idx)

        w, h = data_item['width'], data_item['height']
        bboxs = BoxList(bboxs, (w, h), 'xyxy')
        
        sentence = data_item['description']
        sentence = sentence.lower()
        input_dict = {'frames': frames, 'boxs': bboxs, 'text': sentence, \
                'actioness' : data_item['actioness']}

        if self.transforms is not None:
            input_dict = self.transforms(input_dict)
        
        targets = {
            'item_id' : data_item['item_id'],
            'frame_ids' : data_item['frame_ids'],
            'actioness' : torch.from_numpy(data_item['actioness']) ,
            'start_heatmap' : torch.from_numpy(data_item['start_heatmap']),
            'end_heatmap' : torch.from_numpy(data_item['end_heatmap']),
            'boxs' : input_dict['boxs'],
            'img_size' : input_dict['frames'].shape[2:],
            'ori_size' : (h, w)
        }

        return input_dict['frames'], sentence, targets

    def __len__(self) -> int:
        return len(self.all_gt_data)

    def load_data(self):
        """
        Prepare the Input Data Cache and the evaluation data groundtruth
        """
        cache_dir = os.path.join(self.data_dir,'data_cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
         # Used for Model Input
        dataset_cache = os.path.join(cache_dir, f'hcstvg-{self.split}-input.cache')
        # Used For Evaluateion
        gt_anno_cache = os.path.join(cache_dir, f'hcstvg-{self.split}-anno.cache')
        
        if os.path.exists(dataset_cache):
            data = torch.load(dataset_cache)
            return data
        
        gt_data, gt_anno = [], []
        vstg_anno = self.preprocess(self.sent_file)
        
        for anno_id in tqdm(vstg_anno):  
            gt_file = vstg_anno[anno_id]
            frame_nums = gt_file['frame_count']
            video_name = gt_file['vid']
        
            start_fid = 0
            end_fid = frame_nums - 1
            temp_gt_begin = max(0, gt_file['tube_start_frame'])
            temp_gt_end = min(gt_file['tube_end_frame'], end_fid)

            assert len(gt_file['target_bboxs']) == temp_gt_end - temp_gt_begin + 1
            
            frame_ids = []
            for frame_id in range(start_fid, end_fid):
                frame_ids.append(frame_id)
                    
            actioness = np.array([int(fid <= temp_gt_end and fid >= temp_gt_begin) for fid in frame_ids]) 
            
            # prepare the temporal heatmap
            action_idx = np.where(actioness)[0]
            start_idx, end_idx = action_idx[0], action_idx[-1]
            
            start_heatmap = np.ones(actioness.shape) * self.epsilon
            pesudo_prob = (1 - (start_heatmap.shape[0] - 3) * self.epsilon - 0.5) / 2
            
            start_heatmap[start_idx] = 0.5
            if start_idx > 0:
                start_heatmap[start_idx-1] = pesudo_prob
            if start_idx < actioness.shape[0] - 1:
                start_heatmap[start_idx+1] = pesudo_prob

            end_heatmap = np.ones(actioness.shape) * self.epsilon
            end_heatmap[end_idx] = 0.5
            if end_idx > 0:
                end_heatmap[end_idx-1] = pesudo_prob
            if end_idx < actioness.shape[0] - 1:
                end_heatmap[end_idx+1] = pesudo_prob

            bbox_array = []
            for idx in range(len(gt_file['target_bboxs'])):
                bbox = gt_file['target_bboxs'][idx]
                x1, y1, w, h = bbox
                bbox_array.append(np.array([x1,y1,min(x1+w, gt_file['width']), min(y1+h, gt_file['height'])]))
                assert x1 <= gt_file['width'] and x1 + w <= gt_file['width']
                assert y1 <= gt_file['height'] and y1 + h <= gt_file['height']
            
            bbox_array = np.array(bbox_array)
            assert bbox_array.shape[0] == temp_gt_end - temp_gt_begin + 1
            
            gt_bbox_dict = {fid : bbox_array[fid - temp_gt_begin].tolist() \
                    for fid in range(temp_gt_begin, temp_gt_end + 1)}
            
            gt_item = {
                'item_id' : gt_file['id'],
                'vid' : video_name,
                'bboxs' : gt_bbox_dict,
                'description' : gt_file['sentence'],
                'gt_temp_bound' : [temp_gt_begin, temp_gt_end],
                'frame_count' : gt_file['frame_count']
            }
            
            item = {
                'item_id' : gt_file['id'],
                'vid' : video_name,
                'frame_ids' : frame_ids,
                'width' : gt_file['width'],
                'height' : gt_file['height'],
                'start_heatmap': start_heatmap,
                'end_heatmap': end_heatmap,
                'actioness': actioness,
                'bboxs' : bbox_array,
                'gt_temp_bound' : [temp_gt_begin, temp_gt_end],
                'description' : gt_file['sentence'],
                'object' : 'person',
                'frame_count' : gt_file['frame_count']
            }
            
            gt_data.append(item)
            gt_anno.append(gt_item)
        
        random.shuffle(gt_data)
        torch.save(gt_data, dataset_cache)
        torch.save(gt_anno, gt_anno_cache)
        return gt_data

    def preprocess(self,anno_file):
        """
        preoprocess from the original annotation
        """
        pair_cnt = 0
        print(f"Prepare {self.split} Data")
        
        with open(anno_file, 'r') as fr:
            hcstvg_anno = json.load(fr)

        proc_hcstvg_anno = {}
        for vid in tqdm(hcstvg_anno):
            anno = hcstvg_anno[vid]
            data_pairs = {}
            data_pairs['vid'] = vid
            data_pairs['width'] = anno['width']
            data_pairs['height'] = anno['height']
            data_pairs['frame_count'] = anno['img_num']
            data_pairs['tube_start_frame'] = anno['st_frame'] - 1
            data_pairs['tube_end_frame'] = data_pairs['tube_start_frame'] + len(anno['bbox']) - 1
            data_pairs['tube_start_time'] = anno['st_time']
            data_pairs['tube_end_time'] = anno['ed_time']
            data_pairs['id'] = pair_cnt
            data_pairs['sentence'] = anno['caption']
            data_pairs['target_bboxs'] = anno['bbox']
            proc_hcstvg_anno[pair_cnt] = data_pairs
            pair_cnt += 1
        
        print(f'{self.split} pair number : {pair_cnt}')
        return proc_hcstvg_anno
