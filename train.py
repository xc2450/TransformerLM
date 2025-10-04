import os
import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math
import numpy as np
from tqdm import tqdm, trange
from cs336_basic2 import transformer
import argparse

def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss with numerical stability.
    
    This function computes the cross-entropy loss between predicted logits and true labels,
    using log-sum-exp trick for numerical stability.
    
    Args:
        logits: torch.Tensor - Predicted logits of shape (batch_size, num_classes)
        labels: torch.Tensor - True class labels of shape (batch_size,)
        
    Returns:
        torch.Tensor - Scalar loss value
    """
    logits = logits-torch.max(logits, dim=-1, keepdim=True).values
    numerator = logits[torch.arange(logits.shape[0]), labels]
    denominator = torch.sum(torch.exp(logits), dim=-1)
    loss = -torch.mean(numerator-torch.log(denominator))
    return loss

class SGD(torch.optim.Optimizer):
    """
    Stochastic Gradient Descent optimizer with adaptive learning rate.
    
    This optimizer implements SGD with a learning rate that decreases as 1/sqrt(t),
    where t is the number of steps taken.
    """
    def __init__(self, params, lr=1e-3):
        """
        Initialize the SGD optimizer.
        
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups
            lr: float = 1e-3 - Learning rate
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] 
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p] 
            t = state.get("t", 0) 
            grad = p.grad.data 
            p.data -= lr / math.sqrt(t + 1) * grad 
            state["t"] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer with decoupled weight decay.
    
    AdamW is an extension of Adam that decouples weight decay from gradient updates,
    often leading to better generalization.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """
        Initialize the AdamW optimizer.
        
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups
            lr: float = 1e-3 - Learning rate
            betas: tuple = (0.9, 0.999) - Coefficients for computing running averages of gradient and its square
            eps: float = 1e-8 - Term added to denominator to improve numerical stability
            weight_decay: float = 0.01 - Weight decay coefficient (L2 penalty)
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            grad = p.grad.data
            state["exp_avg"] = betas[0] * state.get("exp_avg", torch.zeros_like(p.data)) + (1 - betas[0]) * grad
            state["exp_avg_sq"] = betas[1] * state.get("exp_avg_sq", torch.zeros_like(p.data)) + (1 - betas[1]) * grad**2
            state["t"] = state.get("t", 0) + 1
            alpha_t = lr * math.sqrt(1 - betas[1]**state["t"]) / (1 - betas[0]**state["t"])
            p.data -= alpha_t * state["exp_avg"] / (torch.sqrt(state["exp_avg_sq"]) + eps)
            p.data -= p.data * lr * weight_decay
        return loss

def learning_rate_schedule(t: int, a_max: float, a_min: float, T_w: int, T_c: int) -> float:
    """
    Compute learning rate using warmup + cosine annealing schedule.
    
    This function implements a learning rate schedule that:
    1. Linearly increases from 0 to a_max during warmup (0 <= t < T_w)
    2. Cosine anneals from a_max to a_min during training (T_w <= t <= T_c)
    3. Stays at a_min after training completes (t > T_c)
    
    Args:
        t: int - Current training step
        a_max: float - Maximum learning rate
        a_min: float - Minimum learning rate
        T_w: int - Number of warmup steps
        T_c: int - Total number of training steps
        
    Returns:
        float - Current learning rate
    """
    if t < T_w:
        return t/T_w * a_max
    elif T_w <= t and t <= T_c:
        return a_min+1/2*(a_max-a_min)*(1+math.cos(math.pi*(t-T_w)/(T_c-T_w)))
    return a_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """
    Clip gradients to prevent exploding gradients.
    
    This function clips all gradients to have a maximum L2 norm. If the total
    L2 norm of all gradients exceeds max_l2_norm, all gradients are scaled down
    proportionally to maintain the same direction but with the maximum allowed norm.
    
    Args:
        parameters: Iterable[torch.nn.Parameter] - Model parameters to clip
        max_l2_norm: float - Maximum allowed L2 norm for all gradients combined
        eps: float = 1e-6 - Small value to prevent division by zero
        
    Note:
        This function modifies gradients in-place.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    l2_norm = 0.0
    for g in grads:
        l2_norm += torch.sum(g**2)
    l2_norm = torch.sqrt(l2_norm)
    clip_coef = min(1, max_l2_norm / (l2_norm + eps))
    for g in grads:
        g *= clip_coef

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language modeling input sequences and their corresponding next-token targets.
    
    Args:
        x: 1D numpy array of integer token IDs
        batch_size: Number of sequences to sample
        context_length: Length of each sequence
        device: PyTorch device string ('cpu', 'cuda:0', etc.)
        
    Returns:
        Tuple of (input_sequences, target_sequences) both with shape (batch_size, context_length)
    """
    max_start_idx = len(x) - context_length - 1
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    input_sequences = np.array([
        x[start:start + context_length] for start in start_indices
    ])
    target_sequences = np.array([
        x[start + 1:start + context_length + 1] for start in start_indices
    ])
    input_tensor = torch.from_numpy(input_sequences).long().to(device)
    target_tensor = torch.from_numpy(target_sequences).long().to(device)
    return input_tensor, target_tensor

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out):
    """
    Save a checkpoint containing model weights, optimizer state, and iteration number.
    
    Args:
        model: PyTorch model to save
        optimizer: PyTorch optimizer to save
        iteration: Current iteration number
        out: Output file path or file-like object
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """
    Load a checkpoint and restore model weights, optimizer state, and iteration number.
    
    Args:
        src: Source file path or file-like object
        model: PyTorch model to restore
        optimizer: PyTorch optimizer to restore
        
    Returns:
        The iteration number that was saved in the checkpoint
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load datasets using memory mapping
    train_data = np.load(args.train_data_path, mmap_mode='r')
    val_data = np.load(args.val_data_path, mmap_mode='r')
    model = transformer.TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Load checkpoint if it exists
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        iteration = load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"Loaded checkpoint from {args.checkpoint_path} at iteration {iteration}")
    else:
        iteration = 0
    progress_bar = trange(args.start_iter, args.total_iters, desc="Training")
    for t in progress_bar:
        lr = learning_rate_schedule(t, args.lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Train step
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()
        
        # Logging to progress bar
        if t % args.log_interval == 0:
            progress_bar.set_postfix(loss=loss.item(), lr=lr)
        
        # Evaluate
        if t % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                x_eval, y_eval = get_batch(val_data, args.batch_size, args.context_length, device)
                val_logits = model(x_eval)
                val_loss = cross_entropy_loss(val_logits.view(-1, val_logits.size(-1)), y_eval.view(-1))
                tqdm.write(f"Validation loss: {val_loss.item():.4f} at iteration {t}")
            model.train()

        # Save checkpoint
        if args.checkpoint_path and t % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{t}.pt")
            save_checkpoint(model, optimizer, t, checkpoint_path)
            tqdm.write(f"Saved checkpoint to {checkpoint_path} at iteration {t}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train a transformer language model")
    
    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feed-forward network dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    
    # Learning rate schedule
    parser.add_argument("--lr", type=float, default=1e-4, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="Number of warmup iterations")
    parser.add_argument("--cosine_cycle_iters", type=int, default=100000, help="Total training iterations")
    
    # Training loop
    parser.add_argument("--start_iter", type=int, default=0, help="Starting iteration")
    parser.add_argument("--total_iters", type=int, default=100000, help="Total number of iterations")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Evaluation interval")
    
    # Data and checkpointing
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--save_interval", type=int, default=10000, help="Checkpoint saving interval")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training
    train(args)