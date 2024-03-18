from .vidstg_eval import VidSTGEvaluator
from .hcstvg_eval import HCSTVGEvaluator

def build_evaluator(cfg, logger, mode):
    if cfg.DATASET.NAME == 'VidSTG':
        return VidSTGEvaluator(
            logger,
            cfg.DATA_DIR,
            mode,
            iou_thresholds=[0.3, 0.5],
            save_pred=(mode=='test'),
            save_dir=cfg.OUTPUT_DIR,
        )
    elif cfg.DATASET.NAME == 'HC-STVG':
        return HCSTVGEvaluator(
            logger,
            cfg.DATA_DIR,
            mode,
            iou_thresholds=[0.3, 0.5],
            save_pred=(mode=='test'),
            save_dir=cfg.OUTPUT_DIR,
        )
    else:
        raise NotImplementedError