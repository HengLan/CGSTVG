import argparse
import os
import time
import datetime
from copy import deepcopy
import torch
import torch.backends.cudnn as cudnn
from config import cfg
from utils.comm import synchronize, get_rank, is_main_process, reduce_loss_dict
from utils.logger import setup_logger
from utils.misc import mkdir, save_config, set_seed, to_device
from utils.checkpoint import VSTGCheckpointer
from datasets import make_data_loader, build_evaluator, build_dataset
from models import build_model, build_postprocessors
from engine import make_optimizer, adjust_learning_rate, update_ema, do_eval
from utils.metric_logger import MetricLogger
from torch.utils.tensorboard import SummaryWriter


def train(cfg, local_rank, distributed, logger):
    model, criteria, weight_dict = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    criteria.to(device)

    optimizer = make_optimizer(cfg, model, logger)
    model_ema = deepcopy(model) if cfg.MODEL.EMA else None
    model_without_ddp = model
    
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True
        )
        model_without_ddp = model.module
    
    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = local_rank == 0
    checkpointer = VSTGCheckpointer(
        cfg, model_without_ddp, model_ema, optimizer, output_dir, save_to_disk, logger, is_train=True
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)
    
    verbose_loss = set(["loss_bbox", "loss_giou", "loss_sted", "loss_conf"])
    
    if cfg.SOLVER.USE_ATTN:
        verbose_loss.add("loss_guided_attn")
    
    if cfg.MODEL.CG.USE_ACTION:
        verbose_loss.add("loss_actioness")
    
    # Prepare the dataset cache
    if local_rank == 0:
        split = ['train', 'test']
        if cfg.DATASET.NAME == "VidSTG":
            split += ['val']
        for mode in split:
            _ = build_dataset(cfg, split=mode, transforms=None)
       
    synchronize()

    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    val_data_loader = make_data_loader(
        cfg,
        mode='val' if cfg.DATASET.NAME == "VidSTG" else "test",
        is_distributed=distributed,
    )

    if cfg.TENSORBOARD_DIR and is_main_process():
        writer = SummaryWriter(cfg.TENSORBOARD_DIR)
    else:
        writer = None
    
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    logger.info("Start training")

    if cfg.SOLVER.PRE_VAL:
        logger.info("Validating before training")
        run_eval(cfg, model, model_ema, logger, val_data_loader, device)
    
    metric_logger = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    for iteration, batch_dict in enumerate(train_data_loader, start_iter):
        model.train()
        criteria.train()

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        videos = batch_dict['videos'].to(device)
        texts = batch_dict['texts']
        durations = batch_dict['durations']
        targets = to_device(batch_dict["targets"], device) 
        targets[0]["durations"] = durations
        outputs = model(videos, texts, targets, iteration/max_iter)

        # compute loss
        loss_dict = criteria(outputs, targets, durations)

        # loss used for update param
        # assert set(weight_dict.keys()) == set(loss_dict.keys())
        losses = sum(loss_dict[k] * weight_dict[k] for k in \
                            loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled" : v \
                        for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k : v * weight_dict[k] for k, v in loss_dict_reduced.items()\
                 if k in weight_dict and k in verbose_loss
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # filter unrelated loss
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)

        optimizer.zero_grad()
        losses.backward()
        if cfg.SOLVER.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.MAX_GRAD_NORM)
        optimizer.step()

        adjust_learning_rate(cfg, optimizer, iteration, max_iter)

        if model_ema is not None:
            update_ema(model, model_ema, cfg.MODEL.EMA_DECAY)

        batch_time = time.time() - end
        end = time.time()
        metric_logger.update(time=batch_time, data=data_time)

        eta_seconds = metric_logger.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if writer is not None and is_main_process() and iteration % 50 == 0:
            for k in loss_dict_reduced_scaled:
                writer.add_scalar(f"{k}", metric_logger.meters[k].avg, iteration)
        
        if iteration % 50 == 0 or iteration == max_iter:
            logger.info(
                metric_logger.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter} / {max_iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "lr_vis_encoder: {lr_vis:.6f}",
                        "lr_text_encoder: {lr_text:.6f}",
                        "lr_temp_decoder: {lr_temp:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    max_iter = max_iter,
                    meters=str(metric_logger),
                    lr=optimizer.param_groups[0]["lr"],
                    lr_vis=optimizer.param_groups[1]["lr"],
                    lr_text=optimizer.param_groups[2]["lr"],
                    lr_temp=optimizer.param_groups[3]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)
            
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
            run_eval(cfg, model, model_ema, logger, val_data_loader, device)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    if writer is not None:
        writer.close()

    return model, model_ema


def run_eval(cfg, model, model_ema, logger, val_data_loader, device):
    logger.info("Start validating")
    test_model = model_ema if model_ema is not None else model
    evaluator = build_evaluator(cfg, logger, mode='val' \
        if cfg.DATASET.NAME == "VidSTG" else "test",)   # mode = ['val','test']
    postprocessor = build_postprocessors()
    torch.cuda.empty_cache()
    do_eval(
        cfg,
        mode='val',
        logger=logger,
        model=test_model,
        postprocessor=postprocessor,
        data_loader=val_data_loader,
        evaluator=evaluator,
        device=device
    )
    synchronize()


def run_test(cfg, model, model_ema, logger, distributed):
    logger.info("Start Testing")
    test_model = model_ema if model_ema is not None else model
    torch.cuda.empty_cache()

    evaluator = build_evaluator(cfg, logger, mode='test')   # mode = ['val','test']
    postprocessor = build_postprocessors()
    val_data_loader = make_data_loader(cfg, mode='test', is_distributed=distributed)
    do_eval(
        cfg,
        mode='test',
        logger=logger,
        model=test_model,
        postprocessor=postprocessor,
        data_loader=val_data_loader,
        evaluator=evaluator,
        device=torch.device(cfg.MODEL.DEVICE)
    )
    synchronize()


def main():
    parser = argparse.ArgumentParser(description="Spatio-Temporal Grounding Training")
    parser.add_argument(
        "--config-file",
        default="./experiments/vidstg.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--use-seed",
        dest="use_seed",
        help="If use the random seed",
        default=True
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
        
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.use_seed:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args.seed + get_rank())
    
    synchronize()
    
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("Video Grounding", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    
    if args.config_file:
        logger.info("Loaded configuration file {}".format(args.config_file))
    
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model, model_ema = train(cfg, args.local_rank, args.distributed, logger)
    
    if not args.skip_test:
        run_test(cfg, model, model_ema, logger, args.distributed)


if __name__ == "__main__":
    main()
