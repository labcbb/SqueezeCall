import os, sys, random
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
import yaml
import torch
import datetime
import json
import math
import logging
import numpy as np
from glob import glob
from transformers import SchedulerType, get_scheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate import DistributedDataParallelKwargs
from torch.utils.data import DataLoader

logger = get_logger(__name__)

sys.path.append('/home/share/huadjyin/home/cyclone_ops/users/ryl/code_repo/basenet/squeeze_call')
from squeeze_call.utils.init_model import init_model
from squeeze_call.utils.dataset import get_npz_dataloader

def get_data_loaders(args):
    train_npz_files = glob(f"{args.data_dir}/train*.npz")
    train_loader = get_npz_dataloader(train_npz_files, batch_size=args.batch_size, cycle=args.epochs)
    # valid_loader = get_npz_dataloader([os.path.join(args.data_dir, 'validation.npz')], batch_size=1024)
    return train_loader


def get_grad_norm(params, scale=1):
    """Compute grad norm given a gradient scale."""
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def main(args):
    args.data_dir = "/home/share/huadjyin/home/cyclone_ops/users/ryl/code_repo/basenet/squeeze_call/data"
    args.config_path = "/home/share/huadjyin/home/cyclone_ops/users/ryl/code_repo/basenet/squeeze_call/config/base.yaml"

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )  

    accelerator = Accelerator()
    logger.info(accelerator.state, main_process_only=False)

    with open(args.config_path, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    
    model = init_model(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader = get_data_loaders(args)

    
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch  = math.ceil(math.ceil(len(train_loader) / accelerator.num_processes) / args.gradient_accumulation_steps)
    if args.max_train_steps < 0:
        args.max_train_steps = args.epochs * num_update_steps_per_epoch


    logger.info(f"{args.warmup_steps=} {args.max_train_steps=} {args.gradient_accumulation_steps=}")

    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps*args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps*args.gradient_accumulation_steps,
    )

    model, optimizer = accelerator.prepare(model, optimizer)

    for epoch in range(int(args.epochs)):
        # train_loader.sampler.set_epoch(epoch)
        #######################
        # train one epoch
        #######################
        model.train()
        for step, batch in enumerate(train_loader, start=1):

            #######################
            # train one step
            #######################
            batch['feats'] = batch['feats'].to(accelerator.device)
            batch['feats_lengths'] = batch['feats_lengths'].to(accelerator.device)
            batch['target'] = batch['target'].to(accelerator.device)
            batch['target_lengths'] = batch['target_lengths'].to(accelerator.device)
            out = model(batch)
            losses =  out['loss'] / args.gradient_accumulation_steps
            losses =  losses / args.gradient_accumulation_steps
            accelerator.backward(losses)
            
            # update step
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                # compute grad norm for monitoring
                scale = (
                    accelerator.scaler._scale.item()
                    if hasattr(accelerator, "scaler") and accelerator.scaler is not None
                    else 1
                )
                if accelerator.state.num_processes > 1:
                    grad_norm = get_grad_norm(model.module.parameters(), scale)
                else:
                    grad_norm = get_grad_norm(model.parameters(), scale)   

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)         

                optimizer.step()
                optimizer.zero_grad()

                lr = lr_scheduler.get_last_lr()[0]
                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                elif accelerator.is_main_process:
                    logger.info(
                        f"Gradients have overflown - skipping update step... Updating gradient scale to {scale}..."
                    )
 
            if accelerator.is_main_process:
                # log traing info per step
                msg = (
                    f"Epoch: {epoch}, step: {step}, loss: {losses:.6f}, grad_norm: {grad_norm:.6f}, lr: {lr:.6f}"
                )
                logger.info(msg)

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, add_help=False
    )
    # parser.add_argument("training_directory")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument("--lr", default=0.0005)
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--data_dir", required=False, type=str)
    parser.add_argument("--config_path", required=False, type=str)
    parser.add_argument("--warmup_steps", default=100, type=int)  # 该值不能为0，否则初始阶段loss不收敛！    
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--max_train_steps", type=int, default=-1)
   
    return parser


if __name__ == "__main__":
    args = argparser().parse_args()
    main(args)

