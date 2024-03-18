import os
from typing import Dict, List

import numpy as np
from utils.comm import is_main_process, all_gather

import torch
from functools import reduce
from utils.box_utils import np_box_iou
import json

def save_json(path, data):
    with open(path, "w") as f:
        return json.dump(data, f)

class HCSTVGiouEvaluator:
    def __init__(
        self,
        vidstg_path: str,
        subset: str = "test",
        iou_thresholds: list = None,
    ):
        """
        :param vidstg_path: path to VidSTG annotations
        :param subset: train, val or test
        :param iou_thresholds: IoU thresholds for the vIoU metrics
        """
        assert subset in ["train", "test"], f"Wrong HCSTVG subset {subset}"
    
        gt_data = []
        cache_dir = os.path.join(vidstg_path, 'data_cache')
        dataset_cache = os.path.join(cache_dir, f'hcstvg-{subset}-anno.cache')
        gt_data = torch.load(dataset_cache) 

        self.vid2steds = {}  # map video_id to [start, end] of the GT tube
        self.vid2box = {}  # map video to bbox
        self.vid2names = {}
        self.vid2sents = {}
        
        for data_item in gt_data:
            video_id = data_item['item_id']
            temp_gt = data_item['gt_temp_bound']
            self.vid2names[video_id] = data_item['vid']
            self.vid2sents[video_id] = data_item['description']
            box_dict = data_item['bboxs']
            self.vid2box[video_id]={key : [box_dict[key]] for key in box_dict}
            self.vid2steds[video_id] = temp_gt

        self.iou_thresholds = iou_thresholds

    def evaluate(self, predictions: List[Dict], video_predictions: List[Dict]):
        vid_metrics = {}
        for video_id, video_pred in video_predictions.items():
            if video_id in vid_metrics:
                print(f"Warning, multiple predictions found for video {video_id}")
                continue
            
            gt_sted = self.vid2steds[video_id]
            pred_sted = video_pred["sted"]

            # compute temporal iou
            max_start = max(gt_sted[0], pred_sted[0])
            min_end = min(gt_sted[1], pred_sted[1])
            min_start = min(gt_sted[0], pred_sted[0])
            max_end = max(gt_sted[1], pred_sted[1])
            if min_end <= max_start:
                tiou = 0
            else:
                intersection = min_end - max_start
                gt_span = gt_sted[1] - gt_sted[0]
                pred_span = pred_sted[1] - pred_sted[0]
                union = gt_span + pred_span - intersection
                tiou = intersection / union

            # compute viou and gt_viou
            vid_metrics[video_id] = {
                "gt_sted": gt_sted,
                "pred_sted": pred_sted,
                "tiou": tiou,
                "img_metrics": {},
            }

            union_predgt = set([
                frame_id for frame_id in range(min_start, max_end)
            ])
            inter_predgt = set(
                [frame_id for frame_id in range(max_start, min_end)]
            )

            viou = 0
            gt_viou = 0
            prediction = predictions[video_id]

            for fid in self.vid2box[video_id].keys():  # iterate on all frames of the annotated moment to update GT metrics
                if fid not in prediction:
                    # raise RuntimeError(f"No prediction for frame {fid}")
                    # print(self.vid2box[video_id].keys(), fid)
                    continue
                pred_boxes = prediction[fid]
                gt_boxes = self.vid2box[video_id][fid]
                iou = np_box_iou(np.array(pred_boxes), np.array(gt_boxes))[0][0]
                if fid in inter_predgt:
                    viou += iou
                gt_viou += iou

            viou = viou / max(len(union_predgt), 1)
            vid_metrics[video_id]["viou"] = viou
            recalls = {thresh: 0 for thresh in self.iou_thresholds}
            for thresh in self.iou_thresholds:
                if viou > thresh:
                    recalls[thresh] += 1
            vid_metrics[video_id].update(
                {
                    f"viou@{thresh}": recalls[thresh]
                    for thresh in self.iou_thresholds
                }
            )

            # compute gt_viou@R
            gt_viou = gt_viou / max(len(self.vid2box[video_id]), 1)
            vid_metrics[video_id]["gt_viou"] = gt_viou
            gt_recalls = {thresh: 0 for thresh in self.iou_thresholds}
            for thresh in self.iou_thresholds:
                if gt_viou > thresh:
                    gt_recalls[thresh] += 1
            vid_metrics[video_id].update(
                {
                    f"gt_viou@{thresh}": gt_recalls[thresh]
                    for thresh in self.iou_thresholds
                }
            )

        return vid_metrics, self.vid2names, self.vid2sents


class HCSTVGEvaluator(object):
    def __init__(
        self,
        logger,
        vidstg_path,
        subset,
        iou_thresholds,
        save_pred=False,
        save_dir=None
    ):
        """
        :param vidstg_path: path to VidSTG annotations
        :param subset: train, val or test
        :param iou_thresholds: IoU thresholds for the vIoU metrics
        :param save_pred: whether to save predictions in the output of summarize
        """
        self.evaluator = HCSTVGiouEvaluator(
            vidstg_path,
            subset=subset,
            iou_thresholds=iou_thresholds,
        )
        self.predictions = {}
        self.video_predictions = {}
        self.video_cross_attn = {}
        self.results = None
        self.iou_thresholds = iou_thresholds
        self.save_pred = save_pred
        self.save_dir = save_dir
        self.logger = logger
        
        self.tsa_weights = {}
        self.text_weights = {}
        self.spatial_weights = {}
        self.pred_sted = {}

    def accumulate(self):
        pass

    def update(self, predictions):
        self.predictions.update(predictions)

    def update_cross_attn(self, cross_weights):
        self.video_cross_attn.update(cross_weights)

    def video_update(self, video_predictions):
        self.video_predictions.update(video_predictions)

    def synchronize_between_processes(self):
        all_predictions = all_gather(self.predictions)
        self.predictions = reduce(lambda a, b: a.update(b) or a, all_predictions, {})
        all_video_predictions = all_gather(self.video_predictions)
        self.video_predictions = reduce(lambda a, b: a.update(b) or a, all_video_predictions, {})

    def summarize(self):
        if is_main_process():
            self.logger.info("#######  Start Calculating the metrics  ########")
            self.results, vid2names, vid2sents = self.evaluator.evaluate(
                self.predictions, self.video_predictions
            )
            
            metrics = {"gt_viou": 0}
            metrics.update({"tiou": 0, "viou": 0})
            for thresh in self.iou_thresholds:  # init metrics
                metrics[f"viou@{thresh}"] = 0
                metrics[f"gt_viou@{thresh}"] = 0
            counter = 0
            result_str = ''
            result_str += '\n' + '=' * 100 + '\n'
            for x in self.results.values():  # sum results
                metrics["tiou"] += x["tiou"]
                metrics["viou"] += x["viou"]
                metrics["gt_viou"] += x["gt_viou"]
                for thresh in self.iou_thresholds: 
                    metrics[f"viou@{thresh}"] += x[f"viou@{thresh}"]
                    metrics[f"gt_viou@{thresh}"] += x[f"gt_viou@{thresh}"]
                counter += 1
                
            for key in metrics:  # average results
                metrics[key] = metrics[key] / counter
                result_str += f"{key}: {metrics[key]:.4f}" + '\n'

            result_str += '=' * 100 + '\n'
            self.logger.info(result_str)
            
            out = {f"{name}": metrics[name] for name in metrics}
            
            if self.save_pred:
                out["predictions"] = self.predictions
                out["gt"] = self.evaluator.vid2box
                out["video_predictions"] = self.video_predictions
                out["vid_metrics"] = self.results
                out['vid2names'] = vid2names
                out['vid2sents'] = vid2sents
                res_path = os.path.join(self.save_dir, 'test_results.json')
                save_json(res_path, out)

            return out

        return None
