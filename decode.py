import torch
import torch.nn.functional as F

from cs336_basics.BPE_tokenizer import Tokenizer

@torch.no_grad()
def decode(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    dtype: torch.dtype = torch.float32,
) -> str:
    model.to(device)
    model.eval()
    # Encode the prompt 
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=dtype, device=device)

    for _ in range(max_tokens):
        logits = model(input_ids)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Set the cutoff to the tokens that are not in the top p
        cutoff = cumulative_probs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        # Set the probabilities of the tokens that are not in the top p to 0
        sorted_probs[cutoff] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(sorted_probs, num_samples=1)
        # Get the token from the sorted indices
        next_token = sorted_indices.gather(-1, next_token)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == tokenizer.encode("<|endoftext|>")[0]:
            break
    return tokenizer.decode(input_ids[0].tolist())