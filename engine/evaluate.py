import torch
import torch.nn
from typing import Dict

from utils.misc import to_device
from utils.comm import synchronize, is_main_process
from tqdm import tqdm


@torch.no_grad()
def linear_interp(bbox_dict):
    frame_ids = sorted([fid for fid in bbox_dict])
    if len(frame_ids) < 2:
        return bbox_dict
    for idx in range(0, len(frame_ids) - 1):
        left_fid = frame_ids[idx]
        right_fid = frame_ids[idx + 1]
        if right_fid - left_fid > 1:
            interval = right_fid - left_fid
            delta_x1 = (bbox_dict[right_fid][0][0] - bbox_dict[left_fid][0][0]) / interval
            delta_y1 = (bbox_dict[right_fid][0][1] - bbox_dict[left_fid][0][1]) / interval
            delta_x2 = (bbox_dict[right_fid][0][2] - bbox_dict[left_fid][0][2]) / interval
            delta_y2 = (bbox_dict[right_fid][0][3] - bbox_dict[left_fid][0][3]) / interval
            for step in range(1, interval):
                bbox_dict[left_fid + step] = [[
                  bbox_dict[left_fid][0][0] + step * delta_x1, 
                  bbox_dict[left_fid][0][1] + step * delta_y1, 
                  bbox_dict[left_fid][0][2] + step * delta_x2, 
                  bbox_dict[left_fid][0][3] + step * delta_y2, 
                ]]
    
    frame_ids = sorted([fid for fid in bbox_dict])
    assert max(frame_ids) - min(frame_ids) + 1 == len(frame_ids) 
    return {fid : bbox_dict[fid] for fid in frame_ids}

@torch.no_grad()
def single_forward(cfg, model, videos, texts, targets, device, postprocessor):
    durations = videos.durations
    targets[0]["durations"] = durations
    outputs = model(videos, texts, targets)
    
    b = len(durations)
    t = max(durations)
    batch_img_size = [list(target['ori_size']) for target in targets]
    orig_target_sizes = [img_size for img_size in batch_img_size for _ in range(t)]
    orig_target_sizes = torch.tensor(orig_target_sizes,device=device)
    assert orig_target_sizes.shape[0] == outputs['pred_boxes'].shape[0]
    
    frames_ids = [target['frame_ids'] for target in targets] 
    pred_boxs, pred_steds = postprocessor(outputs, orig_target_sizes, frames_ids, durations)
    pred_boxs = pred_boxs.view(b, t, 4)
  
    vids = [target['item_id'] for target in targets]
    bbox_pred, temp_pred = {}, {}
    
    for i_b in range(b):
        frames_id = frames_ids[i_b]
        bbox_pred[vids[i_b]] = {}
        assert durations[i_b] == len(frames_id)
        for idx in range(durations[i_b]):
            bbox_pred[vids[i_b]][frames_id[idx]] = [pred_boxs[i_b][idx].detach().cpu().tolist()]
          
    if cfg.DATASET.NAME == 'VidSTG':
        qtypes = [target['qtype'] for target in targets]
        assert len(pred_steds) == len(qtypes)
        for i_b in range(b):
            temp_pred[vids[i_b]] = {
                "sted": pred_steds[i_b],
                "qtype": qtypes[i_b],
            }
    else:
        for i_b in range(b):
            temp_pred[vids[i_b]] = {
                "sted": pred_steds[i_b]
            }
            
    return bbox_pred, temp_pred
    

@torch.no_grad()
def do_eval(cfg, mode, logger, model, postprocessor, data_loader, evaluator, device):
    """
    Video Spatial-Temporal Grounding Evaluation
    """
    model.eval()
    logger.info("Start evaluation on the {} split of {} dataset".format(mode, cfg.DATASET.NAME))

    for _, batch_dict in enumerate(tqdm(data_loader)):
        videos = batch_dict['videos'].to(device)
        texts = batch_dict['texts']
        targets = to_device(batch_dict["targets"], device) 
        
        for i in range(len(targets)):
            if 'qtype' not in targets[i]:
                targets[i]['qtype'] = 'none'
        
        videos1 = videos.subsample(2, start_idx=0)
        targets1 = [{'item_id': target['item_id'], 'ori_size': target['ori_size'],
                     'qtype': target['qtype'], 'frame_ids': target['frame_ids'][0::2], "boxs":target['boxs'].bbox.clone(), 'actioness':target['actioness'][0::2], "eval":True} for target in targets]

        videos2 = videos.subsample(2, start_idx=1)
        targets2 = [{'item_id': target['item_id'], 'ori_size': target['ori_size'],
                     'qtype': target['qtype'], 'frame_ids': target['frame_ids'][1::2], "boxs":target['boxs'].bbox.clone(), 'actioness':target['actioness'][1::2], "eval":True} for target in targets]

        if torch.where(targets[0]["actioness"])[0][0] % 2 == 0:
            targets1[0]['boxs'] = targets1[0]['boxs'][0::2]
            targets2[0]['boxs'] = targets2[0]['boxs'][1::2]
        else:
            targets1[0]['boxs'] = targets1[0]['boxs'][1::2]
            targets2[0]['boxs'] = targets2[0]['boxs'][0::2]

        bbox_pred1, temp_pred1 = single_forward(cfg, model, videos1, texts,
                                targets1, device, postprocessor)
        bbox_pred2, temp_pred2 = single_forward(cfg, model, videos2, texts,
                                targets2, device, postprocessor)
        
        bbox_pred, temp_pred = {}, {}
        for vid in bbox_pred1:
            bbox_pred1[vid].update(bbox_pred2[vid])
            bbox_pred[vid] = linear_interp(bbox_pred1[vid])
            temp_pred[vid] = {'sted' : [min(temp_pred1[vid]['sted'][0], temp_pred2[vid]['sted'][0]),
                              max(temp_pred1[vid]['sted'][1], temp_pred2[vid]['sted'][1])]}
            if 'qtype' in temp_pred1[vid]:
                temp_pred[vid]['qtype'] = temp_pred1[vid]['qtype']

        evaluator.update(bbox_pred)
        evaluator.video_update(temp_pred)
    
    synchronize()
    evaluator.synchronize_between_processes()
    if is_main_process():
        logger.info(f"Complete the inference on {mode} split of {cfg.DATASET.NAME}")
    
    res = evaluator.summarize()
    return res