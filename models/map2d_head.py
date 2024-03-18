import re
import copy
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Gen2DMap(nn.Module):
    def __init__(self, cfg):
        super().__init__()        
        N = cfg.MODEL.TEMPFORMER.MAX_MAP_SIZE
        pooling_counts = cfg.MODEL.TEMPFORMER.POOLING_COUNTS
        self.map_size = N
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            for _ in range(c): 
                # fill a diagonal line 
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2
        
        poolers = [nn.MaxPool1d(2,1) for _ in range(pooling_counts[0])]
        for c in pooling_counts[1:]:
            poolers.extend(
                [nn.MaxPool1d(3,2)] + [nn.MaxPool1d(2,1) for _ in range(c - 1)]
            )

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij
        self.poolers = poolers
        
    def forward(self, x):
        """
        input : 
            tensor : [batch, n_frames, dim]
        output : 
            2D map : [batch, n_frames, n_frames, dim]
            mask :   [batch, n_frames, n_frames, 1]
        """
        x = x.permute(0,2,1)
        b, d_model, n_frames = x.shape
        
        if n_frames > self.map_size:
            x = F.adaptive_avg_pool1d(x, self.map_size)
        
        x = F.adaptive_max_pool1d(x, self.map_size)
        
        N = self.map_size
        map2d = x.new_zeros(b, d_model, N, N)
        map2d[:, :, range(N), range(N)] = x
        for pooler, (i, j) in zip(self.poolers, self.maskij):
            x = pooler(x)
            map2d[:, :, i, j] = x
            
        return map2d
        
        
class TempPredictionHead(nn.Module):
    """The Temporal Interaction Head"""

    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.MODEL.TEMPFORMER.HIDDEN
        nhead = cfg.MODEL.TEMPFORMER.HEADS
        dim_feedforward = cfg.MODEL.TEMPFORMER.FFN_DIM
        dropout = cfg.MODEL.TEMPFORMER.DROPOUT
        num_layers = cfg.MODEL.TEMPFORMER.TEMP_PRED_LAYERS
        activation = "relu"
        self.temp_head = cfg.MODEL.TEMPFORMER.TEMP_HEAD
        self.map_maker = Gen2DMap(cfg)
        self.mask_2d = self.map_maker.mask2d
        
        self.encoder = None
        if self.temp_head == 'attn':
            encoder_layer = TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation, self.mask_2d
            )
            encoder_norm = None
            self.encoder = TransformerEncoder(
                encoder_layer, num_layers, encoder_norm
            )
        else:
            kernel_size = cfg.MODEL.TEMPFORMER.KERNAL_SIZE
            num_conv_layers = cfg.MODEL.TEMPFORMER.CONV_LAYERS
            self.encoder = TempConvInteraction(
                d_model, kernel_size, num_conv_layers, self.mask_2d
            )
        
        self._reset_parameters()    
        self.predictor = nn.Conv2d(d_model, 1, 1)
        
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x):
        """
        x : [layers, b, len, d_model]
        """
        n_layers, b , t, d_model = x.shape
        x = x.view(-1, t, d_model)
        map2d = self.map_maker(x) # n_layers * b, d_model, t, t
        
        # the segment level interaction
        if self.temp_head == 'attn':
            for i_layer in range(len(map2d)):
                map2d[i_layer] =  self.encoder(map2d[i_layer])
        else:
            map2d = self.encoder(map2d)
        
        scores2d = self.predictor(map2d).squeeze_() # n_layers * b, t, t
        _, N, N = scores2d.shape
        scores2d = scores2d.view(n_layers, b, N, N)
        
        if self.training:
            return scores2d
        else:
            return scores2d.sigmoid_() * self.mask_2d
        

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos=None):
        output = src
        for layer in self.layers:
            output = layer(
                output,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output
    

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", mask2d=None
    ):
        super().__init__()
        self.self_attn_row = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_col = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.mask2d = mask2d
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        pos: Optional[Tensor] = None,
    ):
        # from d_modelxNxN to NxNxd_model
        src = src.permute(1,2,0)
        mask2d = self.mask2d
        # row self attention
        q = k = self.with_pos_embed(src, pos)
        src2, _ = self.self_attn_row(
            q, k, value=src, attn_mask=None, key_padding_mask=mask2d
        )
        
        # column self attention
        src2 = src2.permute(1,0,2)
        mask2d = mask2d.permute(1,0)
        q = k = self.with_pos_embed(src2, pos)
        src2, _ = self.self_attn_col(
            q, k, value=src2, attn_mask=None, key_padding_mask=mask2d
        )
        src2 = src2.permute(1,0,2)
        mask2d = mask2d.permute(1,0)
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        src = src.permute(2,0,1)
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

def mask2weight(mask2d, mask_kernel, padding=0):
    # from the feat2d.py,we can know the mask2d is 4-d
    weight = torch.conv2d(mask2d[None,None,:,:].float(),
        mask_kernel, padding=padding)[0, 0]
    weight[weight > 0] = 1 / weight[weight > 0]
    return weight

class TempConvInteraction(nn.Module):
    def __init__(self, hidden_size, k, num_stack_layers, mask2d): 
        super(TempConvInteraction, self).__init__()
        
        # Padding to ensure the dimension of the output map2d
        mask_kernel = torch.ones(1,1,k,k).to(mask2d.device) 
        first_padding = (k - 1) * num_stack_layers // 2

        self.weights = [
            mask2weight(mask2d, mask_kernel, padding=first_padding) 
        ]
        self.convs = nn.ModuleList(
            [nn.Conv2d(hidden_size, hidden_size, k, padding=first_padding)]
        )
 
        for _ in range(num_stack_layers - 1):
            self.weights.append(mask2weight(self.weights[-1] > 0, mask_kernel))
            self.convs.append(nn.Conv2d(hidden_size, hidden_size, k))
        
    def forward(self, x):
        for conv, weight in zip(self.convs, self.weights):
            x = conv(x).relu() * weight
        return x        


if __name__ == "__main__":
    model = Gen2DMap(64, [15, 8, 8])