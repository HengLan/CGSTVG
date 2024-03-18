from .position_encoding import build_position_encoding
from .backbone import GroupNormBackbone, Backbone, Joiner


def build_vis_encoder(cfg):
    position_embedding = build_position_encoding(cfg)
    train_backbone = cfg.SOLVER.VIS_BACKBONE_LR  > 0
    backbone_name = cfg.MODEL.VISION_BACKBONE.NAME
    if backbone_name in ("resnet50-gn", "resnet101-gn"):
        backbone = GroupNormBackbone(
            backbone_name, 
            train_backbone, 
            False, 
            cfg.MODEL.VISION_BACKBONE.DILATION
        )
    else:
        backbone = Backbone(
            backbone_name, 
            train_backbone, 
            False, 
            cfg.MODEL.VISION_BACKBONE.DILATION
        )
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model