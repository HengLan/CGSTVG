from torch import nn
from .net_utils import MLP
from .vision_model import build_vis_encoder
from .language_model import build_text_encoder
from .grounding_model import build_encoder, build_decoder
from utils.misc import NestedTensor
from .vidswin.video_swin_transformer import vidswin_model

class CGSTVG(nn.Module):
    def __init__(self, cfg):
        super(CGSTVG, self).__init__()
        self.cfg = cfg.clone()
        self.max_video_len = cfg.INPUT.MAX_VIDEO_LEN
        self.use_attn = cfg.SOLVER.USE_ATTN
        
        self.use_aux_loss = cfg.SOLVER.USE_AUX_LOSS  # use the output of each transformer layer
        self.use_actioness = cfg.MODEL.CG.USE_ACTION
        self.query_dim = cfg.MODEL.CG.QUERY_DIM

        self.vis_encoder = build_vis_encoder(cfg)
        vis_fea_dim = self.vis_encoder.num_channels
      
        self.text_encoder = build_text_encoder(cfg)
        
        self.ground_encoder = build_encoder(cfg)
        self.ground_decoder = build_decoder(cfg)
        
        hidden_dim = cfg.MODEL.CG.HIDDEN
        self.input_proj = nn.Conv2d(vis_fea_dim, hidden_dim, kernel_size=1)
        self.temp_embed = MLP(hidden_dim, hidden_dim, 2, 2, dropout=0.3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.vid = vidswin_model("video_swin_t_p4w7", "video_swin_t_p4w7_k400_1k")
        self.input_proj2 = nn.Conv2d(768, hidden_dim, kernel_size=1)
        for param in self.vid.parameters():
            param.requires_grad = False

        self.action_embed = None
        if self.use_actioness:
            self.action_embed = MLP(hidden_dim, hidden_dim, 1, 2, dropout=0.3)

        self.ground_decoder.time_embed2 = self.action_embed

        # add the iteration anchor update
        self.ground_decoder.decoder.bbox_embed = self.bbox_embed

    def forward(self, videos, texts, targets, iteration_rate=-1):
        # Visual Feature
        vis_outputs, vis_pos_embed = self.vis_encoder(videos)
        vis_features, vis_mask, vis_durations = vis_outputs.decompose()
        vis_features = self.input_proj(vis_features) 
        vis_outputs = NestedTensor(vis_features, vis_mask, vis_durations)

        vid_features = self.vid(videos.tensors, len(videos.tensors))
        vid_features = self.input_proj2(vid_features['3'])
     
        # Textual Feature
        device = vis_features.device
        text_outputs, _ = self.text_encoder(texts, device)

        # Multimodal Feature Encoding
        encoded_info = self.ground_encoder(videos=vis_outputs, vis_pos=vis_pos_embed, texts=text_outputs, vid_features=vid_features)
        encoded_info["iteration_rate"] = iteration_rate
        encoded_info["videos"] = videos
        # Query-based Decoding
        outputs_pos, outputs_time = self.ground_decoder(encoded_info=encoded_info, vis_pos=vis_pos_embed, targets=targets)

        out = {}

        # the final decoder embeddings and the refer anchors
        ###############  predict bounding box ###############
        refer_anchors, anchors_conf, fake_anchors = outputs_pos
        outputs_coord = refer_anchors.flatten(1,2)  # [num_layers, T, 4]
        out.update({"pred_boxes": outputs_coord[-1]})
        out.update({"boxes_conf": anchors_conf[-1]})
        out.update({"fake_boxes": fake_anchors})
        ######################################################

        #######  predict the start and end probability #######
        time_hiden_state = outputs_time
        outputs_time = self.temp_embed(time_hiden_state)  # [num_layers, b, T, 2]
        out.update({"pred_sted": outputs_time[-1]})
        #######################################################

        if self.use_actioness:
            outputs_actioness = self.action_embed(time_hiden_state)  # [num_layers, b, T, 1]
            out.update({"pred_actioness": outputs_actioness[-1]})

        if self.use_aux_loss:
            out["aux_outputs"] = [
                {
                    "pred_sted": a,
                    "pred_boxes": b,
                    "boxes_conf": c
                }
                for a, b, c in zip(outputs_time[:-1], outputs_coord[:-1], anchors_conf[:-1])
            ]
            for i_aux in range(len(out["aux_outputs"])):
                if self.use_actioness:
                    out["aux_outputs"][i_aux]["pred_actioness"] = outputs_actioness[i_aux]

        return out