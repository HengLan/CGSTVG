import copy
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List, Optional, Tuple

from utils.misc import NestedTensor
from .position_encoding import SeqEmbeddingLearned, SeqEmbeddingSine


class CrossModalEncoder(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        # attention configuration
        d_model = cfg.MODEL.CG.HIDDEN
        nhead = cfg.MODEL.CG.HEADS
        dim_feedforward = cfg.MODEL.CG.FFN_DIM
        dropout = cfg.MODEL.CG.DROPOUT
        activation = "relu"
        num_layers = cfg.MODEL.CG.ENC_LAYERS
        self.d_model = d_model
        
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        encoder_norm = None
        self.encoder = SpatialTemporalEncoder(cfg, encoder_layer, num_layers, encoder_norm)
        self.fusion = nn.Linear(d_model, d_model)
        
        # The position embedding for feature map
        # self.spatial_embed = PositionEmbeddingLearned(d_model // 2)
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, videos: NestedTensor = None, vis_pos=None, texts: Tuple = None, vid_features=None):
        vis_features, vis_mask, vis_durations = videos.decompose()
        assert vis_pos.shape[0] == sum(vis_durations), "{} != {}".format(vis_pos.shape[0], sum(vis_durations))
    
        vis_mask[:, 0, 0] = False  # avoid empty masks

        _, _, H, W = vis_features.shape
        # n_frames x c x h x w => hw x n_frames x c
        vis_features = vis_features.flatten(2).permute(2, 0, 1)  # torch.Size([156, 64, 256])
        vid_features = vid_features.flatten(2).permute(2, 0, 1)
        vis_pos = vis_pos.flatten(2).permute(2, 0, 1)
        vis_mask = vis_mask.flatten(1)

        # prepare the text encodings
        text_mask, text_features, _ = texts

        # expand the attention mask and text token from [b, len] to [n_frames, len]
        frame_length = vis_durations[0]
        text_mask = text_mask.expand(frame_length, text_mask.size(-1))
        text_features = text_features.expand(text_features.size(0), frame_length, text_features.size(-1))   # [text_len, n_frames, d_model]

        # concat visual and text features and Pad the vis_pos with 0 for the text tokens
        features = torch.cat([vis_features, text_features, vid_features], dim=0)
        mask = torch.cat([vis_mask, text_mask, vis_mask], dim=1)
        vis_pos = torch.cat([vis_pos, torch.zeros_like(text_features), vis_pos], dim=0)

        # perfrom cross-modality interaction
        encoded_feature, frames_cls, videos_cls = self.encoder(
            features, 
            src_key_padding_mask=mask,
            pos=vis_pos,
        )
        
        memory_cache = {
            "encoded_feature": encoded_feature,  #
            "encoded_mask": mask,  # batch first
            "frames_cls" : frames_cls,  # n_frame, d_model
            "videos_cls" : videos_cls, # b , d_model
            "durations": vis_durations,
            "fea_map_size": (H, W)
        }
        
        return memory_cache
        

class SpatialTemporalEncoder(nn.Module):
    def __init__(self, cfg, encoder_layer, num_layers, norm=None, return_weights=False):
        super().__init__()
        self.spatial_layers = _get_clones(encoder_layer, num_layers)
        self.temporal_layers = _get_clones(encoder_layer, num_layers)
        video_max_len = cfg.INPUT.MAX_VIDEO_LEN
        d_model = cfg.MODEL.CG.HIDDEN 
        self.d_model = d_model
        
        # The position embedding of global tokens
        if cfg.MODEL.CG.USE_LEARN_TIME_EMBED:
            self.time_embed = SeqEmbeddingLearned(video_max_len + 1 , d_model)
        else:
            self.time_embed = SeqEmbeddingSine(video_max_len + 1, d_model) 
    
        # The position embedding of local frame tokens
        self.local_pos_embed = nn.Embedding(1, d_model) # the learned pos embed for frame cls token
        
        # The learnd local and global embedding
        self.frame_cls = nn.Embedding(1, d_model)  # the frame level local cls token
        self.video_cls = nn.Embedding(1, d_model)  # the video level global cls token
        
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_weights = return_weights

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for i_layer, layer in enumerate(self.spatial_layers):
            # spatial interaction on each single frame
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )


        if self.norm is not None:
            output = self.norm(output)

        frame_src = torch.mean(output, dim=0)
        video_src = torch.mean(frame_src, dim=0)
        return output, frame_src, video_src


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
