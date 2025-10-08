# LLM from Scratch

A complete implementation of a Large Language Model (LLM) built from scratch, including transformer architecture, training pipeline, tokenization, and text generation.

## 🎯 Overview

This project implements a full-stack language model training and inference system, featuring:
- Custom transformer architecture with modern optimizations (RoPE, RMSNorm, SwiGLU)
- Byte Pair Encoding (BPE) tokenizer
- Complete training pipeline with AdamW optimizer and learning rate scheduling
- Multiple decoding strategies for text generation

---

## 📁 Project Structure

```
LLM from scratch/
├── transformer.py      # Neural network architecture
├── train.py           # Training utilities and main loop
├── BPE_tokenizer.py   # Tokenization implementation
├── decode.py          # Text generation strategies
```

---

## 🏗️ Architecture (`transformer.py`)

### Core Components

The transformer implementation includes modern architectural improvements:

#### **Building Blocks**
- **Linear**: Bias-free linear transformation with truncated normal initialization
- **Embedding**: Token embedding lookup layer
- **RMSNorm**: Root Mean Square normalization (more efficient than LayerNorm)
- **SwiGLU**: Swish-Gated Linear Unit activation function
- **RotaryPositionEmbedding (RoPE)**: Rotary position embeddings for better length extrapolation

#### **Attention Mechanism**
- **scaled_dot_product_attention**: Efficient scaled dot-product attention
- **MultiHeadSelfAttention**: Multi-head attention with optional RoPE

#### **High-Level Modules**
- **TransformerBlock**: Complete transformer layer with pre-norm architecture
- **TransformerLM**: Full language model with causal masking

### Example Usage

```python
from transformer import TransformerLM
import torch

# Initialize a GPT-2 style model
model = TransformerLM(
    vocab_size=50257,
    context_length=1024,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=3072,
    rope_theta=10000.0
)

# Forward pass
token_ids = torch.randint(0, 50257, (4, 512))  # (batch_size, seq_len)
logits = model(token_ids)  # (batch_size, seq_len, vocab_size)
```

### Architecture Highlights

✅ **Pre-norm architecture** - More stable training than post-norm  
✅ **RMSNorm** - Faster and more efficient than LayerNorm  
✅ **SwiGLU activation** - Better performance than ReLU/GELU  
✅ **Rotary Position Embeddings** - Superior position encoding for transformers  
✅ **Causal masking** - Autoregressive generation support  

---

## 🎓 Training (`train.py`)

### Optimizers

#### **SGD with Adaptive Learning Rate**
Learning rate decays as 1/√t where t is the step number.

```python
from train import SGD

optimizer = SGD(model.parameters(), lr=1e-3)
```

#### **AdamW (Recommended)**
Adam with decoupled weight decay for better generalization.

```python
from train import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.1
)
```

### Training Utilities

#### **Loss Function**
```python
from train import cross_entropy_loss

loss = cross_entropy_loss(logits, labels)  # Numerically stable
```

#### **Learning Rate Schedule**
Combines warmup and cosine annealing:

```python
from train import learning_rate_schedule

# Warmup for first 1000 steps, then cosine decay
lr = learning_rate_schedule(
    t=current_step,
    a_max=1e-4,      # max learning rate
    a_min=1e-5,      # min learning rate
    T_w=1000,        # warmup steps
    T_c=100000       # total steps
)
```

#### **Gradient Clipping**
Prevent exploding gradients:

```python
from train import gradient_clipping

gradient_clipping(model.parameters(), max_l2_norm=1.0)
```

#### **Checkpointing**
Save and restore training state:

```python
from train import save_checkpoint, load_checkpoint

# Save
save_checkpoint(model, optimizer, iteration, "checkpoint.pt")

# Load
iteration = load_checkpoint("checkpoint.pt", model, optimizer)
```

### Running Training

```bash
python train.py \
    --train_data_path data/train.npy \
    --val_data_path data/val.npy \
    --vocab_size 50257 \
    --context_length 1024 \
    --d_model 768 \
    --num_layers 12 \
    --num_heads 12 \
    --d_ff 3072 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --warmup_iters 1000 \
    --total_iters 100000 \
    --output_dir ./checkpoints
```

### Training Features

✅ **AdamW optimizer** - State-of-the-art optimization  
✅ **Learning rate warmup** - Stable training start  
✅ **Cosine annealing** - Smooth learning rate decay  
✅ **Gradient clipping** - Prevent gradient explosions  
✅ **Checkpointing** - Resume training anytime  
✅ **Progress tracking** - Real-time training metrics  
✅ **Validation** - Monitor overfitting  

---

## 🔤 Tokenization (`BPE_tokenizer.py`)

### Byte Pair Encoding (BPE)

A byte-level BPE tokenizer compatible with GPT-2 style tokenization.

#### **Training a Tokenizer**

```python
from BPE_tokenizer import train_bpe, Tokenizer

# Train on your corpus
vocab, merges = train_bpe(
    input_path="corpus.txt",
    vocab_size=50000,
    special_tokens=["<|endoftext|>"],
    num_processes=8  # parallel processing
)

# Create tokenizer instance
tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
```

#### **Using the Tokenizer**

```python
# Encode text
text = "Hello, world! How are you today?"
token_ids = tokenizer.encode(text)

# Decode back
decoded = tokenizer.decode(token_ids)

# Save for later use
tokenizer.save("my_tokenizer")

# Load
tokenizer = Tokenizer.load("my_tokenizer")
```

#### **Preparing Training Data**

```python
import numpy as np

# Encode entire dataset
with open("train.txt", "r") as f:
    text = f.read()

tokens = tokenizer.encode(text)

# Save as uint16 for memory efficiency
token_array = np.array(tokens, dtype=np.uint16)
np.save("train_encoded.npy", token_array)
```

### Tokenization Features

✅ **Byte-level BPE** - Handles any UTF-8 text  
✅ **Special tokens** - Support for `<|endoftext|>` and custom tokens  
✅ **Parallel training** - Fast tokenizer training  
✅ **GPT-2 compatible** - Pre-tokenization regex patterns  
✅ **Memory efficient** - uint16 encoding for large datasets  

---

## 📝 Text Generation (`decode.py`)

### Decoding Strategies

Generate text using various sampling methods:

```python
from decode import decode
from BPE_tokenizer import Tokenizer

# Load your trained model and tokenizer
model = ...  # Load your trained model
tokenizer = Tokenizer.load("my_tokenizer")

# Generate text
output = decode(
    model=model,
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_tokens=100,
    temperature=0.8,
    top_p=0.95,
    device='cuda'
)

print(output)
```

### Sampling Parameters

#### **Temperature** (0.0 to 2.0)
Controls randomness:
- **Low (0.3-0.5)**: More deterministic, focused
- **Medium (0.7-0.9)**: Balanced creativity
- **High (1.0-2.0)**: More random, creative

#### **Top-P (Nucleus Sampling)** (0.0 to 1.0)
Sample from the smallest set of tokens with cumulative probability > p:
- **0.9-0.95**: Recommended for most cases
- **0.99**: More diverse
- **0.5**: More focused

### Generation Examples

```python
# Creative story generation
story = decode(
    model, tokenizer,
    prompt="In a distant galaxy",
    max_tokens=200,
    temperature=0.9,
    top_p=0.95
)

# Code generation (more deterministic)
code = decode(
    model, tokenizer,
    prompt="def fibonacci(n):",
    max_tokens=50,
    temperature=0.3,
    top_p=0.9
)

# Question answering (balanced)
answer = decode(
    model, tokenizer,
    prompt="Q: What is machine learning?\nA:",
    max_tokens=100,
    temperature=0.7,
    top_p=0.9
)
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies

```bash
pip install torch numpy einops regex tqdm
```

### 2. Prepare Your Data

```python
# Organize your text data
data/
├── train.txt  # Training corpus
└── val.txt    # Validation corpus
```

### 3. Train Tokenizer

```python
from BPE_tokenizer import train_bpe, Tokenizer
import numpy as np

# Train tokenizer
vocab, merges = train_bpe(
    input_path="data/train.txt",
    vocab_size=32000,
    special_tokens=["<|endoftext|>"]
)

tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
tokenizer.save("tokenizer")

# Encode datasets
for split in ["train", "val"]:
    with open(f"data/{split}.txt", "r") as f:
        text = f.read()
    
    tokens = tokenizer.encode(text)
    np.save(f"data/{split}_encoded.npy", np.array(tokens, dtype=np.uint16))
```

### 4. Train Model

```bash
python train.py \
    --train_data_path data/train_encoded.npy \
    --val_data_path data/val_encoded.npy \
    --vocab_size 32000 \
    --context_length 512 \
    --d_model 512 \
    --num_layers 8 \
    --num_heads 8 \
    --d_ff 2048 \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --total_iters 50000
```

### 5. Generate Text

```python
from transformer import TransformerLM
from decode import decode
from BPE_tokenizer import Tokenizer
import torch

# Load tokenizer
tokenizer = Tokenizer.load("tokenizer")

# Load model
model = TransformerLM(
    vocab_size=32000,
    context_length=512,
    d_model=512,
    num_layers=8,
    num_heads=8,
    d_ff=2048,
    rope_theta=10000.0
)

# Load trained weights
checkpoint = torch.load("checkpoints/checkpoint_50000.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Generate!
output = decode(
    model, tokenizer,
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.8,
    top_p=0.95
)

print(output)
```

---

## 📊 Pre-configured Model Sizes

### Small Model (~50M parameters)
```python
model = TransformerLM(
    vocab_size=32000,
    context_length=512,
    d_model=512,
    num_layers=8,
    num_heads=8,
    d_ff=2048,
    rope_theta=10000.0
)
```

### Medium Model (~117M parameters - GPT-2 Small)
```python
model = TransformerLM(
    vocab_size=50257,
    context_length=1024,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=3072,
    rope_theta=10000.0
)
```

### Large Model (~345M parameters - GPT-2 Medium)
```python
model = TransformerLM(
    vocab_size=50257,
    context_length=1024,
    d_model=1024,
    num_layers=24,
    num_heads=16,
    d_ff=4096,
    rope_theta=10000.0
)
```

---

## 💡 Key Features

### Modern Architecture
✅ Rotary Position Embeddings (RoPE)  
✅ RMSNorm for stable training  
✅ SwiGLU activation  
✅ Pre-norm transformer architecture  
✅ Efficient attention implementation  

### Training Infrastructure
✅ AdamW optimizer with weight decay  
✅ Learning rate warmup and cosine scheduling  
✅ Gradient clipping  
✅ Checkpoint management  
✅ Progress tracking with tqdm  
✅ Memory-efficient data loading  

### Tokenization
✅ Byte-level BPE  
✅ Parallel training support  
✅ Special token handling  
✅ GPT-2 compatible  

### Text Generation
✅ Temperature sampling  
✅ Top-p (nucleus) sampling  
✅ Efficient generation loop  
✅ Batch generation support  

---

## 🎯 Use Cases

### 1. **Language Model Pre-training**
Train a general-purpose language model on large text corpora (books, Wikipedia, web text).

### 2. **Domain-Specific Models**
Fine-tune on specific domains:
- Code generation
- Scientific writing
- Creative writing
- Dialogue systems

### 3. **Research and Experimentation**
- Test new architectures
- Experiment with different tokenization strategies
- Study scaling laws
- Analyze model behavior

### 4. **Educational Purposes**
- Learn how transformers work
- Understand training dynamics
- Explore generation strategies

---

## 📖 Technical Details

### Why These Design Choices?

#### **RoPE vs Absolute Positional Encoding**
- Better length extrapolation
- No trainable position embeddings
- More efficient

#### **RMSNorm vs LayerNorm**
- Simpler computation
- No bias term needed
- Slightly faster

#### **SwiGLU vs GELU/ReLU**
- Better empirical performance
- Gating mechanism helps with gradient flow

#### **AdamW vs Adam**
- Decoupled weight decay
- Better generalization
- More stable training

#### **Pre-norm vs Post-norm**
- More stable gradient flow
- Easier to train deep models
- Better convergence

---

## 🔬 Performance Tips

### Training
1. **Use gradient accumulation** for larger effective batch sizes
2. **Start with small learning rate** during warmup
3. **Monitor gradient norms** to detect instabilities
4. **Use mixed precision (fp16)** for faster training
5. **Save frequent checkpoints** for fault tolerance

### Generation
1. **Adjust temperature** based on task (lower for factual, higher for creative)
2. **Use top-p sampling** for better quality than top-k
3. **Batch generation** for efficiency
4. **Cache key-value pairs** for faster generation (not yet implemented)

### Memory Optimization
1. **Use gradient checkpointing** for very deep models
2. **Memory-map large datasets** with numpy
3. **Use uint16 for token IDs** to save memory
4. **Clear cache regularly** during long training runs

---

## 📚 References

### Papers
- **Attention Is All You Need** - Original Transformer paper
- **Language Models are Unsupervised Multitask Learners** - GPT-2
- **RoFormer** - Rotary Position Embeddings
- **Root Mean Square Layer Normalization** - RMSNorm
- **GLU Variants Improve Transformer** - SwiGLU
- **Decoupled Weight Decay Regularization** - AdamW

### Resources
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

---

## 🤝 Contributing

This is an educational implementation built from scratch. Contributions, suggestions, and feedback are welcome!

---

## 📄 License

See LICENSE file for details.

---

## 🙏 Acknowledgments

Built as part of learning transformer architectures and language modeling from first principles.
Inspired by GPT-2, GPT-3, and modern LLM research.

---

**Happy Training! 🚀**

