from .pipeline import CGSTVG
from .criterion import VideoSTGLoss
from .post_processor import PostProcess

def build_model(cfg):
    """
    Prepare the model architecture and 
    """
    model = CGSTVG(cfg)
    
    weight_dict = {
        "loss_bbox": cfg.SOLVER.BBOX_COEF,
        "loss_giou": cfg.SOLVER.GIOU_COEF,
        "loss_sted": cfg.SOLVER.TEMP_COEF,
        "loss_conf": cfg.SOLVER.CONF_COEF
    }
        
    if cfg.MODEL.CG.USE_ACTION:
        weight_dict["loss_actioness"] = cfg.SOLVER.ACTIONESS_COEF
        
    if cfg.SOLVER.USE_ATTN:
        weight_dict["loss_guided_attn"] = cfg.SOLVER.ATTN_COEF

    if cfg.SOLVER.USE_AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.MODEL.CG.DEC_LAYERS - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    losses = ["boxes", "sted", "conf"]
    if cfg.SOLVER.USE_ATTN:
        losses += ["guided_attn"]
    if cfg.MODEL.CG.USE_ACTION:
        losses += ["actioness"]
    
    loss_model = VideoSTGLoss(cfg, losses)
    
    return model, loss_model, weight_dict


def build_postprocessors():
    return PostProcess()