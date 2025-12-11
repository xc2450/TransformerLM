import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')
            
            if wandb: wandb.log({"loss": current_loss, "lr": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            del state_dict

        del X, Y, loss_mask, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="Model save directory")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="Prefix name for saved weights")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Mixed precision type")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loading threads")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--log_interval", type=int, default=100, help="Log printing interval")
    parser.add_argument("--save_interval", type=int, default=100, help="Model save interval")
    parser.add_argument('--hidden_size', default=512, type=int, help="Hidden layer dimension")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Number of hidden layers")
    parser.add_argument('--max_seq_len', default=512, type=int, help="Maximum truncation length for training")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Whether to use MoE architecture (0=no, 1=yes)")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="Training data path")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="Which weight to train from, use 'none' to start from scratch")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="Whether to automatically detect & resume training (0=no, 1=yes)")
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb project name")
    args = parser.parse_args()

    # ========== 1. Initialize environment and random seed ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. Configure directory, model parameters, check ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. Set mixed precision ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. Configure wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. Define model, data, optimizer ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. Restore state from ckp ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. Wrap model with DDP ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. Start training ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0: # First epoch and checkpoint exists
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: Skipping first {start_step} steps, starting from step {start_step + 1}')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
        else: # Default: start from beginning
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), 0, wandb)
