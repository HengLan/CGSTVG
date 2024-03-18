import numpy as np
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from models.net_utils import MLP, gen_sineembed_for_position, greater_than_indices, generalized_box_iou, box_cxcywh_to_xyxy
import math
from .position_encoding import SeqEmbeddingLearned, SeqEmbeddingSine
from .attention import MultiheadAttention
from ..bert_model.bert_module import BertLayerNorm, BertLayer_Cross
from easydict import EasyDict as EDict


class QueryDecoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.MODEL.CG.HIDDEN
        nhead = cfg.MODEL.CG.HEADS
        num_layers = cfg.MODEL.CG.DEC_LAYERS

        self.d_model = d_model
        self.query_pos_dim = cfg.MODEL.CG.QUERY_DIM
        self.nhead = nhead
        self.video_max_len = cfg.INPUT.MAX_VIDEO_LEN
        self.return_weights = cfg.SOLVER.USE_ATTN
        return_intermediate_dec = True

        self.template_generator = TemplateGenerator(cfg)

        self.decoder = PosDecoder(
            cfg,
            num_layers,
            return_intermediate=return_intermediate_dec,
            return_weights=self.return_weights,
            d_model=d_model,
            query_dim=self.query_pos_dim
        )

        self.time_decoder = TimeDecoder(
            cfg,
            num_layers,
            return_intermediate=return_intermediate_dec,
            return_weights=True,
            d_model=d_model
        )

        # The position embedding of global tokens
        if cfg.MODEL.CG.USE_LEARN_TIME_EMBED:
            self.time_embed = SeqEmbeddingLearned(self.video_max_len + 1, d_model)
        else:
            self.time_embed = SeqEmbeddingSine(self.video_max_len + 1, d_model)

        self.pos_fc = nn.Sequential(
            BertLayerNorm(256, eps=1e-12),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
            nn.ReLU(True),
            BertLayerNorm(4, eps=1e-12),
        )

        self.time_fc = nn.Sequential(
            BertLayerNorm(256, eps=1e-12),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(True),
            BertLayerNorm(256, eps=1e-12),
        )
        self.time_embed2 = None
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, encoded_info, vis_pos=None, targets=None):
        encoded_feature = encoded_info["encoded_feature"]  # len, n_frame, d_model
        encoded_mask = encoded_info["encoded_mask"]  # n_frame, len
        n_vis_tokens = encoded_info["fea_map_size"][0] * encoded_info["fea_map_size"][1]
        encoded_pos = vis_pos.flatten(2).permute(2, 0, 1)
        encoded_pos = torch.cat([encoded_pos, torch.zeros_like(encoded_feature[n_vis_tokens:])], dim=0)
        # the contextual feature to generate dynamic learnable anchors
        frames_cls = encoded_info["frames_cls"]  # [n_frames, d_model]
        videos_cls = encoded_info["videos_cls"]  # the video-level gloabl contextual token, b x d_model

        b = len(encoded_info["durations"])
        t = max(encoded_info["durations"])
        device = encoded_feature.device

        # pos_query, content_query = self.template_generator(frames_cls, videos_cls)  # STCAT
        pos_query, content_query = self.pos_fc(frames_cls), self.time_fc(videos_cls)

        pos_query = pos_query.sigmoid().unsqueeze(1)  # [n_frames, bs, 4]
        content_query = content_query.expand(t, content_query.size(-1)).unsqueeze(1)  # [n_frames, bs, d_model]

        query_mask = torch.zeros(b, t).bool().to(device)
        query_time_embed = self.time_embed(t).repeat(1, b, 1)  # [n_frames, bs, d_model]

        tgt = torch.zeros(t, b, self.d_model).to(device)

        outputs_time = self.time_decoder(
            query_tgt=tgt,
            query_content=content_query,  # n_queriesx(b*t)xF
            query_time=query_time_embed,
            query_mask=query_mask,
            encoded_feature=encoded_feature,
            encoded_pos=encoded_pos,  # n_tokensx(b*t)xF
            encoded_mask=encoded_mask
        )

        outputs_pos = self.decoder(
            query_tgt=tgt,  # t x b x c
            pred_boxes=pos_query,  # n_queriesx(b*t)xF
            query_time=query_time_embed,
            query_mask=query_mask,  # bx(t*n_queries)
            encoded_feature=encoded_feature,  # n_tokens x n_frames x c
            encoded_pos=encoded_pos,  # n_tokens x n_frames x c
            encoded_mask=encoded_mask,  # n_frames * n_tokens
            feature_size=(encoded_info["fea_map_size"][0], encoded_info["fea_map_size"][1], encoded_info["videos"]),
            targets=targets,
            iteration_rate=encoded_info["iteration_rate"],
            time_info=self.time_embed2(outputs_time[-1]).squeeze()
        )

        return outputs_pos, outputs_time


class PosDecoder(nn.Module):
    def __init__(self, cfg, num_layers, return_intermediate=False, return_weights=False, d_model=256, query_dim=4):
        super().__init__()
        self.layers = nn.ModuleList([PosDecoderLayer(cfg) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate
        self.return_weights = False
        self.query_dim = query_dim
        self.d_model = d_model

        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.conf = MLP(d_model, d_model, 1, 2, dropout=0.3)
        self.conf2 = MLP(d_model, d_model, 1, 2, dropout=0.3)

        self.gf_mlp = MLP(d_model, d_model, d_model, 2)
        self.gf_mlp2 = MLP(d_model, d_model, d_model, 2)
        self.fuse_linear = nn.Linear(d_model*2, d_model)
        for layer_id in range(num_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None
        self.norm2 = nn.LayerNorm(d_model)

        self.theta_t = cfg.MODEL.CG.TEMP_THETA
        self.theta_s_gt = cfg.MODEL.CG.SPAT_GT_THETA
        self.theta_s = cfg.MODEL.CG.SPAT_THETA

    def gt_info(self, targets):
        if "eval" in targets[0].keys():
            target_boxes = torch.cat([target["boxs"] for target in targets], dim=0)
        else:
            target_boxes = torch.cat([target["boxs"].bbox for target in targets], dim=0)

        gt_bbox_slice, gt_temp_bound = [], []
        durations = targets[0]["durations"]
        max_duration = max(durations)
        for i_dur, (duration, target) in enumerate(zip(durations, targets)):
            inter = torch.where(target['actioness'])[0].cpu().numpy().tolist()
            gt_temp_bound.append([inter[0], inter[-1]])
            gt_bbox_slice.extend(list(range(i_dur * max_duration + inter[0], i_dur * max_duration + inter[-1] + 1)))
        gt_bbox_slice = torch.LongTensor(gt_bbox_slice)
        return target_boxes, gt_bbox_slice

    def get_context_index_by_gt(self, pred_boxes, target_boxes, gt_bbox_slice, conf):
        conf_list = torch.zeros(pred_boxes.shape[0]).to(pred_boxes.device)
        if len(gt_bbox_slice) > 1:
            pred_boxes = pred_boxes[gt_bbox_slice].squeeze()
        else:
            pred_boxes = pred_boxes[gt_bbox_slice][0]

        iou = generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(target_boxes))

        conf_list[gt_bbox_slice[0]:gt_bbox_slice[-1] + 1] = torch.diag(iou)

        context_index = greater_than_indices(conf_list, conf).reshape(-1)

        return context_index

    def get_context_index_by_our(self, conf_list, conf=0.7):
        context_index = greater_than_indices(conf_list, conf).reshape(-1)
        return context_index

    def generate_roi_feature(self, all_feature, v_size, bboxs, type):
        if type == "2d":
            feature_map = all_feature.permute(1, 0, 2)[:, :v_size[0] * v_size[1]]  # [b, H*W, C]
        else:
            feature_map = all_feature.permute(1, 0, 2)[:, -v_size[0] * v_size[1]:]

        feature_map = feature_map.reshape(-1, v_size[0], v_size[1], all_feature.size(-1))  # [b, H, W, C]
        bboxs = box_cxcywh_to_xyxy(bboxs).clamp(min=0).squeeze() * torch.Tensor([v_size[1], v_size[0], v_size[1], v_size[0]]).to(bboxs.device)
        bboxs = torch.stack([(bboxs[:, 0]).round(), (bboxs[:, 1]).round(), (bboxs[:, 2]).ceil(), (bboxs[:, 3]).round()], dim=-1).long()  # torch.round(bboxs).int()
        roi_feature = []
        for i in range(len(bboxs)):
            f = feature_map[i].clone()
            x1, y1, x2, y2 = bboxs[i]
            x2 = min(max(x2, 1), f.size(1))
            x1 = min(max(x1, 0), x2 - 1)
            y2 = min(max(y2, 1), f.size(0))
            y1 = min(max(y1, 0), y2 - 1)
            try:
                r = f[y1:y2, x1:x2].clone().reshape(-1, all_feature.size(-1))
                pooling_r = torch.mean(r, dim=0)
            except:
                pooling_r = torch.zeros(256).to(bboxs.device)
            roi_feature.append(pooling_r)
        return torch.stack(roi_feature)

    def generate_context(self, roi_2d, roi_3d, index=None):
        context_2d = self.gf_mlp(roi_2d[index])
        context_3d = self.gf_mlp2(roi_3d[index])
        context = torch.cat((context_2d, context_3d), dim=0)
        return context

    def forward(
            self,
            query_tgt: Optional[Tensor] = None,
            pred_boxes: Optional[Tensor] = None, 
            query_time: Optional[Tensor] = None, 
            query_mask: Optional[Tensor] = None, 
            encoded_feature: Optional[Tensor] = None,
            encoded_pos: Optional[Tensor] = None,
            encoded_mask: Optional[Tensor] = None,
            feature_size=None,
            targets=None,
            iteration_rate=None,
            time_info=None
    ):
        conf_list = []
        intermediate = []
        intermediate_weights = []
        ref_anchors = []  # the query pos is like t x b x 4
        context = None

        for layer_id, layer in enumerate(self.layers):
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(pred_boxes)
            query_pos = self.ref_point_head(query_sine_embed)  # generated the position embedding

            # For the first decoder layer, we do not apply transformation over p_s
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(query_tgt)

            # apply transformation
            query_sine_embed = query_sine_embed[..., :self.d_model] * pos_transformation

            query_tgt, temp_weights = layer(
                query_tgt=query_tgt, query_pos=query_pos,
                query_time_embed=query_time, query_sine_embed=query_sine_embed, query_mask=query_mask,
                encoded_feature=encoded_feature, encoded_pos=encoded_pos, encoded_mask=encoded_mask,
                context=context, is_first=(layer_id == 0))

            # iter update
            if self.bbox_embed is not None:
                tmp = self.bbox_embed(query_tgt)
                new_pred_boxes = tmp.sigmoid()
                ref_anchors.append(new_pred_boxes)
                pred_boxes = new_pred_boxes.detach()

            target_boxes, gt_bbox_slice = self.gt_info(targets)

            roi_2d = self.generate_roi_feature(encoded_feature, feature_size, pred_boxes, "2d")
            roi_3d = self.generate_roi_feature(encoded_feature, feature_size, pred_boxes, "3d")

            conf_2d = self.conf(roi_2d).sigmoid().squeeze()
            conf_3d = self.conf2(roi_3d).sigmoid().squeeze()
            conf_list.append(conf_2d + conf_3d)

            if iteration_rate >= 0:
                context_indexs = self.get_context_index_by_gt(pred_boxes, target_boxes, gt_bbox_slice, self.theta_s_gt)
            else:
                context_indexs = self.get_context_index_by_our(conf_list[-1], self.theta_s)

                pred_time = torch.nonzero(time_info > self.theta_t).squeeze(-1).tolist()
                # if isinstance(pred_time, int):
                #    pred_time = [pred_time]

                context_indexs = [int(i) for i in context_indexs if i in pred_time]

            context = self.generate_context(roi_2d, roi_3d, context_indexs) if len(context_indexs) > 0 else None

            if self.return_intermediate:
                intermediate.append(self.norm(query_tgt))
                if self.return_weights:
                    intermediate_weights.append(temp_weights)

        if self.norm is not None:
            query_tgt = self.norm(query_tgt)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query_tgt)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                outputs = [
                    torch.stack(ref_anchors).transpose(1, 2),
                    torch.stack(conf_list),
                    torch.zeros((ref_anchors[0].size(0), 4)).to(ref_anchors[0].device)
                ]
            else:
                outputs = [
                    torch.stack(intermediate).transpose(1, 2),
                    pred_boxes.unsqueeze(0).transpose(1, 2)
                ]

        if self.return_weights:
            return outputs, torch.stack(intermediate_weights)
        else:
            return outputs


class PosDecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Decoder Self-Attention
        d_model = cfg.MODEL.CG.HIDDEN
        nhead = cfg.MODEL.CG.HEADS
        dim_feedforward = cfg.MODEL.CG.FFN_DIM
        dropout = cfg.MODEL.CG.DROPOUT
        activation = "relu"
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_qtime_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_ktime_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_qtime_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)

        self.from_scratch_cross_attn = cfg.MODEL.CG.FROM_SCRATCH
        self.cross_attn_image = None
        self.cross_attn = None
        self.tgt_proj = None

        if self.from_scratch_cross_attn:
            self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)
        else:
            self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        bert_config = EDict(
            num_attention_heads=8,
            hidden_size=256,
            attention_head_size=256,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            intermediate_size=256
        )
        #self.ca = BertLayer_Cross(bert_config)
        self.ca2 = BertLayer_Cross(bert_config)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
            self,
            query_tgt: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None,
            query_time_embed=None,
            query_sine_embed=None,
            query_mask: Optional[Tensor] = None,
            encoded_feature: Optional[Tensor] = None,
            encoded_pos: Optional[Tensor] = None,
            encoded_mask: Optional[Tensor] = None,
            context=None,
            is_first=False,
    ):
        # Apply projections here
        # shape: num_queries x batch_size x 256
        # ========== Begin of Self-Attention =============
        q_content = self.sa_qcontent_proj(query_tgt)  # target is the input of the first decoder layer. zero by default.
        q_time = self.sa_qtime_proj(query_time_embed)
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(query_tgt)
        k_time = self.sa_ktime_proj(query_time_embed)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(query_tgt)

        q = q_content + q_time + q_pos
        k = k_content + k_time + k_pos

        # Temporal Self attention
        tgt2, weights = self.self_attn(q, k, value=v)
        tgt = query_tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ========== End of Self-Attention =============

        if context != None:
            tgt = self.ca2(tgt, context.unsqueeze(0).expand(query_mask.size(-1),context.size(0),context.size(1)))

        # ========== Begin of Cross-Attention =============
        # Time Aligned Cross attention
        t, b, c = tgt.shape    # b is the video number
        n_tokens, bs, f = encoded_feature.shape   # bs is the total frames in a batch
        assert f == c   # all the token dim should be same

        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(encoded_feature)
        v = self.ca_v_proj(encoded_feature)

        k_pos = self.ca_kpos_proj(encoded_pos)

        if is_first:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(t, b, self.nhead, c // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(t, b, self.nhead, c // self.nhead)

        if self.from_scratch_cross_attn:
            q = torch.cat([q, query_sine_embed], dim=3).view(t, b, c * 2)
        else:
            q = (q + query_sine_embed).view(t, b, c)
            q = q + self.ca_qtime_proj(query_time_embed)

        k = k.view(n_tokens, bs, self.nhead, f//self.nhead)
        k_pos = k_pos.view(n_tokens, bs, self.nhead, f//self.nhead)

        if self.from_scratch_cross_attn:
            k = torch.cat([k, k_pos], dim=3).view(n_tokens, bs, f * 2)
        else:
            k = (k + k_pos).view(n_tokens, bs, f)

        # extract the actual video length query
        device = tgt.device
        if self.from_scratch_cross_attn:
            q_cross = torch.zeros(1,bs,2 * c).to(device)
        else:
            q_cross = torch.zeros(1,bs,c).to(device)

        q_clip = q[:,0,:]   # t x f
        q_cross[0,:] = q_clip[:]

        if self.from_scratch_cross_attn:
            tgt2, _ = self.cross_attn(
                query=q_cross,
                key=k,
                value=v
            )
        else:
            tgt2, _ = self.cross_attn_image(
                query=q_cross,
                key=k,
                value=v
            )

        # reshape to the batched query
        tgt2_pad = torch.zeros(1,t*b,c).to(device)

        tgt2_pad[0, :] = tgt2[0, :]

        tgt2 = tgt2_pad
        tgt2 = tgt2.view(b, t, f).transpose(0, 1)  # 1x(b*t)xf -> bxtxf -> txbxf

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt, weights


class TemplateGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.MODEL.CG.HIDDEN
        self.pos_query_dim = cfg.MODEL.CG.QUERY_DIM
        self.content_proj = nn.Linear(self.d_model, self.d_model)
        self.gamma_proj = nn.Linear(self.d_model, self.d_model)
        self.beta_proj = nn.Linear(self.d_model, self.d_model)
        self.anchor_proj = nn.Linear(self.d_model, self.pos_query_dim)

    def forward(self, frames_cls=None, videos_cls=None):
        gamma_vec = torch.tanh(self.gamma_proj(videos_cls))
        beta_vec = torch.tanh(self.beta_proj(videos_cls))
        pos_query = self.anchor_proj(gamma_vec * frames_cls + beta_vec)  # [n_frame, 4]
        content_query = self.content_proj(videos_cls)  # [b, d_model]
        content_query = content_query.expand(content_query.size(0) * frames_cls.size(0), content_query.size(1))

        return pos_query, content_query  # [n_frame, 4]  [n_frame, d_model]


class TimeDecoder(nn.Module):
    def __init__(self, cfg, num_layers, return_intermediate=False, return_weights=False, d_model=256):
        super().__init__()
        self.layers = nn.ModuleList([TimeDecoderLayer(cfg) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate
        self.return_weights = return_weights

    def forward(
            self,
            query_tgt: Optional[Tensor] = None,
            query_content: Optional[Tensor] = None,
            query_time: Optional[Tensor] = None,
            query_mask: Optional[Tensor] = None,
            encoded_feature: Optional[Tensor] = None,
            encoded_pos: Optional[Tensor] = None,
            encoded_mask: Optional[Tensor] = None
    ):
        intermediate = []
        intermediate_weights = []

        for _, layer in enumerate(self.layers):
            query_tgt, weights = layer(
                query_tgt=query_tgt,
                query_content=query_content,
                query_time=query_time,
                query_mask=query_mask,
                encoded_feature=encoded_feature,
                encoded_pos=encoded_pos,
                encoded_mask=encoded_mask
            )
            if self.return_intermediate:
                intermediate.append(self.norm(query_tgt))
                if self.return_weights:
                    intermediate_weights.append(weights)

        if self.norm is not None:
            query_tgt = self.norm(query_tgt)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query_tgt)

        if self.return_intermediate:
            return torch.stack(intermediate).transpose(1, 2)

class TimeDecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.MODEL.CG.HIDDEN
        nhead = cfg.MODEL.CG.HEADS
        dim_feedforward = cfg.MODEL.CG.FFN_DIM
        dropout = cfg.MODEL.CG.DROPOUT
        activation = "relu"

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
            self,
            query_tgt: Optional[Tensor] = None,
            query_content: Optional[Tensor] = None,
            query_time: Optional[Tensor] = None,
            query_mask: Optional[Tensor] = None,
            encoded_feature: Optional[Tensor] = None,
            encoded_pos: Optional[Tensor] = None,
            encoded_mask: Optional[Tensor] = None
    ):
        q = k = self.with_pos_embed(query_tgt, query_time)

        # Temporal Self attention
        query_tgt2, weights = self.self_attn(q, k, value=query_tgt, key_padding_mask=query_mask)
        query_tgt = self.norm1(query_tgt + self.dropout1(query_tgt2))

        query_tgt2, _ = self.cross_attn_image(
            query=query_tgt.permute(1, 0, 2),
            key=self.with_pos_embed(encoded_feature, encoded_pos),
            value=encoded_feature,
            key_padding_mask=encoded_mask,
        )

        query_tgt2 = query_tgt2.transpose(0, 1)  # 1x(b*t)xf -> bxtxf -> txbxf
        query_tgt = self.norm3(query_tgt + self.dropout3(query_tgt2))

        # FFN
        query_tgt2 = self.linear2(self.dropout(self.activation(self.linear1(query_tgt))))
        query_tgt = query_tgt + self.dropout4(query_tgt2)
        query_tgt = self.norm4(query_tgt)
        return query_tgt, weights


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
