from time import time
import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

from utils.box_utils import generalized_box_iou, box_cxcywh_to_xyxy, box_iou
from utils.comm import is_dist_avail_and_initialized, get_world_size


class VideoSTGLoss(nn.Module):
    """This class computes the loss for VideoSTG Model
    The process happens in two steps:
        1) compute ground truth boxes and the outputs of the model
        2) compute ground truth temporal segment and the outputs sted of model
    """

    def __init__(self, cfg, losses):
        """Create the criterion.
        """
        super().__init__()
        self.cfg = cfg
        self.losses = losses
        self.eos_coef = cfg.SOLVER.EOS_COEF
    
    def loss_boxes(self, outputs, targets, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        
        src_boxes = outputs["pred_boxes"]
        target_boxes = torch.cat([target["boxs"].bbox for target in targets], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / max(num_boxes, 1)

        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / max(num_boxes, 1)
        return losses

    def loss_conf(self, outputs, targets, num_boxes, gt_index):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "boxes_conf" in outputs
        losses = {}
        src_boxes = outputs["pred_boxes"]
        target_boxes = torch.cat([target["boxs"].bbox for target in targets], dim=0)
        iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        iou = torch.diag(iou)
        conf = outputs['boxes_conf'][gt_index]
        # v1 and v2
        losses["loss_conf"] = nn.BCEWithLogitsLoss()(conf, iou)
        # vid
        # losses["loss_conf"] = F.smooth_l1_loss(conf, iou, reduction="none").sum() / max(num_boxes, 1)
        return losses
    
    def loss_actioness(self, outputs, targets, gt_temp_bound, time_mask=None):
        assert "pred_actioness" in outputs
        losses = {}
        pred_actioness = outputs['pred_actioness'].squeeze(-1)
        target_actioness = torch.stack([target["actioness"] for target in targets], dim=0).float()
        weight = torch.full(pred_actioness.shape, self.eos_coef, device=pred_actioness.device)
        
        for i_b in range(len(weight)):
            temp_bound = gt_temp_bound[i_b]
            weight[i_b][temp_bound[0] : temp_bound[1] + 1] = 1
    
        loss_actioness = F.binary_cross_entropy_with_logits(pred_actioness, \
                target_actioness, weight=weight, reduction='none')
        
        loss_actioness = loss_actioness * time_mask
        losses["loss_actioness"] = loss_actioness.mean()
        return losses

    def loss_sted(self, outputs, num_boxes, gt_temp_bound, positive_map, time_mask=None):
        assert "pred_sted" in outputs
        sted = outputs["pred_sted"]
        losses = {}
        
        target_start = torch.tensor([x[0] for x in gt_temp_bound], dtype=torch.long).to(sted.device)
        target_end = torch.tensor([x[1] for x in gt_temp_bound], dtype=torch.long).to(sted.device)
        sted = sted.masked_fill(~time_mask[:, :, None], -1e32)  # put very low probability on the padded positions before softmax
        eps = 1e-6
        
        sigma = self.cfg.SOLVER.SIGMA
        start_distrib = (
            -(
                (
                    torch.arange(sted.shape[1])[None, :].to(sted.device)
                    - target_start[:, None]
                )
                ** 2
            )
            / (2 * sigma ** 2)
        ).exp()  # gaussian target
        start_distrib = F.normalize(start_distrib + eps, p=1, dim=1)
        pred_start_prob = (sted[:, :, 0]).softmax(1)
        loss_start = (
            pred_start_prob * ((pred_start_prob + eps) / start_distrib).log()
        )
        loss_start = loss_start * time_mask
        end_distrib = (
            -(
                (
                    torch.arange(sted.shape[1])[None, :].to(sted.device)
                    - target_end[:, None]
                )
                ** 2
            )
            / (2 * sigma ** 2)
        ).exp()  # gaussian target
        end_distrib = F.normalize(end_distrib + eps, p=1, dim=1)
        pred_end_prob = (sted[:, :, 1]).softmax(1)
        loss_end = (
            pred_end_prob * ((pred_end_prob + eps) / end_distrib).log()
        )
        loss_end = loss_end * time_mask
        loss_sted = loss_start + loss_end
        losses["loss_sted"] = loss_sted.mean()
        return losses

    def loss_guided_attn(
        self, outputs, num_boxes, gt_temp_bound, positive_map, time_mask=None
    ):
        """Compute guided attention loss
        targets dicts must contain the key "weights" containing a tensor of attention matrices of dim [B, T, T]
        """
        weights = outputs["weights"]  # BxTxT
        
        positive_map = positive_map + (~time_mask)  # the padded positions also have to be taken out
        eps = 1e-6  # avoid log(0) and division by 0

        loss = -(1 - weights + eps).log()
        loss = loss.masked_fill(positive_map[:, :, None], 0)
        nb_neg = (~positive_map).sum(1) + eps
        loss = loss.sum(2) / nb_neg[:, None]  # sum on the column
        loss = loss.sum(1)  # mean on the line normalized by the number of negatives
        loss = loss.mean()  # mean on the batch
        
        losses = {"loss_guided_attn": loss}
        return losses

    def get_loss(
        self, loss, outputs, targets, num_boxes, gt_temp_bound, positive_map, time_mask, gt_bbox_slice, **kwargs,
    ):
        loss_map = {
            "boxes": self.loss_boxes,
            "sted": self.loss_sted,
            "guided_attn": self.loss_guided_attn,
            "actioness": self.loss_actioness,
            "conf": self.loss_conf
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        if loss in ["sted", "guided_attn"]:
            return loss_map[loss](
                outputs, num_boxes, gt_temp_bound, positive_map, time_mask, **kwargs
            )
        if loss == "actioness":
            return loss_map[loss](outputs, targets, gt_temp_bound, time_mask, **kwargs)
        if loss == "conf":
            return loss_map[loss](outputs, targets, num_boxes, gt_bbox_slice)

        return loss_map[loss](outputs, targets, num_boxes, **kwargs)

    def forward(self, outputs, targets, durations):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        max_duration = max(durations)
        device = outputs["pred_boxes"].device
        gt_bbox_slice, gt_temp_bound = [], []
        
        for i_dur, (duration, target) in enumerate(zip(durations, targets)):
            inter = torch.where(target['actioness'])[0].cpu().numpy().tolist()
            gt_temp_bound.append([inter[0],inter[-1]])
            gt_bbox_slice.extend(list(range(i_dur * max_duration + inter[0], i_dur * max_duration + inter[-1] + 1)))
            
        gt_bbox_slice = torch.LongTensor(gt_bbox_slice).to(device)
        outputs["pred_boxes"] = outputs["pred_boxes"][gt_bbox_slice]
       
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(target['boxs']) for target in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # computer the temporal mask, used for guided-attn
        b = len(durations)
        time_mask = torch.zeros(b, max(durations)).bool().to(device)
        for i_dur, duration in enumerate(durations):
            time_mask[i_dur, :duration] = True
    
        positive_map = torch.zeros(time_mask.shape, dtype=torch.bool)
        for k, idx in enumerate(gt_temp_bound):
            if idx[0] < 0:  # empty intersection
                continue
            positive_map[k][idx[0] : idx[1] + 1].fill_(True)

        positive_map = positive_map.to(time_mask.device)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, num_boxes, gt_temp_bound, positive_map, time_mask, gt_bbox_slice))
        
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i_aux in range(len(outputs["aux_outputs"])):
                outputs["aux_outputs"][i_aux]["pred_boxes"] = outputs["aux_outputs"][i_aux]["pred_boxes"][gt_bbox_slice]
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, num_boxes, gt_temp_bound, positive_map, time_mask, gt_bbox_slice, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        return losses