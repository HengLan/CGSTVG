import torch
from .lr_scheduler import WarmupMultiStepLR, WarmupReduceLROnPlateau, WarmupPolyLR


def update_ema(model, model_ema, decay):
    """Apply exponential moving average update.

    The  weights are updated in-place as follow:
    w_ema = w_ema * decay + (1 - decay) * w
    Args:
        model: active model that is being optimized
        model_ema: running average model
        decay: exponential decay parameter
    """
    with torch.no_grad():
        if hasattr(model, "module"):
            # unwrapping DDP
            model = model.module
        msd = model.state_dict()
        for k, ema_v in model_ema.state_dict().items():
            model_v = msd[k].detach()
            ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v)


def make_optimizer(cfg, model, logger):
    vis_enc_param = [p for n, p in model.named_parameters() \
                            if (("vis_encoder" in n) and p.requires_grad)]
    text_enc_param = [p for n, p in model.named_parameters() \
                            if (("text_encoder" in n) and p.requires_grad)]
    temp_dec_param = [p for n, p in model.named_parameters() \
                            if (("ground_decoder.time_decoder" in n) and p.requires_grad)]
    rest_param = [p for n, p in model.named_parameters() if(('vis_encoder' not in n) and \
                ('text_encoder' not in n) and ("ground_decoder.time_decoder" not in n) and p.requires_grad)]

    base_lr = cfg.SOLVER.BASE_LR
    optim_type = cfg.SOLVER.OPTIMIZER
    weight_decay = cfg.SOLVER.WEIGHT_DECAY

    param_list = [
        {"params" : rest_param},
        {"params" : vis_enc_param, "lr" : cfg.SOLVER.VIS_BACKBONE_LR},
        {"params" : text_enc_param, "lr" : cfg.SOLVER.TEXT_LR},  
        {"params" : temp_dec_param, "lr" : cfg.SOLVER.TEMP_LR},  
    ]

    # using RMSProp or AdamW
    if optim_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_list, lr=base_lr, weight_decay=weight_decay)
    elif optim_type == 'adamw':
        optimizer = torch.optim.AdamW(param_list, lr=base_lr, weight_decay=weight_decay)
    elif optim_type == 'adam':
        optimizer = torch.optim.Adam(param_list, lr=base_lr, weight_decay=weight_decay)
    elif optim_type== 'sgd':
        optimizer = torch.optim.SGD(param_list, lr=base_lr, weight_decay=weight_decay, momentum=cfg.SOLVER.MOMENTUM)
    else:
        raise ValueError('Lr scheduler type not supportted ')

    return optimizer


def make_lr_scheduler(cfg, optimizer, logger=None):
    if cfg.SOLVER.SCHEDULE.TYPE == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    
    elif cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
        return WarmupReduceLROnPlateau(
            optimizer,
            cfg.SOLVER.SCHEDULE.FACTOR,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            patience=cfg.SOLVER.SCHEDULE.PATIENCE,
            threshold=cfg.SOLVER.SCHEDULE.THRESHOLD,
            cooldown=cfg.SOLVER.SCHEDULE.COOLDOWN,
            logger=logger,
        )
    elif cfg.SOLVER.SCHEDULE.TYPE == "WarmupPolyLR":
        return WarmupPolyLR(
            optimizer,
            cfg.SOLVER.POWER,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    else:
        raise ValueError("Invalid Schedule Type")
