from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.init as init

class Linear(nn.Module):
    """
    Linear transformation module without bias.
    
    This module performs a linear transformation: y = xW^T
    where x is the input tensor and W is the weight matrix.
    """
    
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Construct a linear transformation module.
        
        Args:
            in_features: int - final dimension of the input
            out_features: int - final dimension of the output  
            device: torch.device | None = None - Device to store the parameters on
            dtype: torch.dtype | None = None - Data type of the parameters
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation to input.
        
        Args:
            x: torch.Tensor - Input tensor of shape (..., in_features)
            
        Returns:
            torch.Tensor - Output tensor of shape (..., out_features)
        """
        return x @ self.weight.T

class Embedding(nn.Module):
    """
    Embedding lookup module.
    
    This module performs embedding lookup: given token IDs, returns the corresponding
    embedding vectors from the embedding matrix.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Construct an embedding module.
        
        Args:
            num_embeddings: int - Size of the vocabulary
            embedding_dim: int - Dimension of the embedding vectors (d_model)
            device: torch.device | None = None - Device to store the parameters on
            dtype: torch.dtype | None = None - Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for token IDs.
        
        Args:
            token_ids: torch.Tensor - Token IDs of shape (..., seq_len)
            
        Returns:
            torch.Tensor - Embedding vectors of shape (..., seq_len, embedding_dim)
        """
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm normalizes inputs by dividing by the root mean square of the inputs.
    Unlike LayerNorm, RMSNorm doesn't use a learnable bias term.
    """
  
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct an RMSNorm module.
        
        Args:
            d_model: int - Dimension of the input features
            eps: float = 1e-5 - Small value added to denominator for numerical stability
            device: torch.device | None = None - Device to store the parameters on
            dtype: torch.dtype | None = None - Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to input.
        
        Args:
            x: torch.Tensor - Input tensor of shape (..., d_model)
            
        Returns:
            torch.Tensor - Normalized tensor of shape (..., d_model)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = x*torch.rsqrt(torch.mean(x**2+self.eps, dim=-1, keepdim=True))*self.weight
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    
    SwiGLU is a variant of the GLU activation function that uses a Swish-like activation function.
    """
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        Construct a SwiGLU module.
        
        Args:
            d_model: int - Input and output dimension
            d_ff: int - Hidden dimension of the feed-forward network
            device: torch.device | None = None - Device to store the parameters on
            dtype: torch.dtype | None = None - Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Swish activation function: x * sigmoid(x).
        
        Args:
            x: torch.Tensor - Input tensor
            
        Returns:
            torch.Tensor - Swish-activated tensor
        """
        return x * torch.sigmoid(x)
    
    def _glu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gated Linear Unit: SwiGLU(x) = Swish(W1(x)) * W3(x).
        
        Args:
            x: torch.Tensor - Input tensor of shape (..., d_model)
            
        Returns:
            torch.Tensor - GLU output of shape (..., d_ff)
        """
        return self._silu(self.w1(x)) * self.w3(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation.
        
        Args:
            x: torch.Tensor - Input tensor of shape (..., d_model)
            
        Returns:
            torch.Tensor - Output tensor of shape (..., d_model)
        """
        return self.w2(self._glu(x))

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding.
    
    Rotary Position Embedding is a positional encoding technique that uses a sinusoidal
    function to encode the position of the tokens in the sequence.
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        """
        Construct a rotary position embedding module.
        
        Args:
            theta: float - Base frequency for the rotary embedding
            d_k: int - Dimension of the key vectors
            max_seq_len: int - Maximum sequence length to support
            device: torch.device | None = None - Device to store the parameters on
            dtype: torch.dtype | None = None - Data type of the parameters
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        pos = torch.arange(self.max_seq_len, device=self.device)
        theta = 1 / (self.theta ** (torch.arange(0, self.d_k, 2, device=self.device) / self.d_k))
        pos_emb = torch.outer(pos, theta)
        cos_pos = torch.cos(pos_emb)
        sin_pos = torch.sin(pos_emb)
        self.register_buffer("cos_pos", cos_pos, persistent=False)
        self.register_buffer("sin_pos", sin_pos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embedding to input.
        
        Args:
            x: torch.Tensor - Input tensor of shape (..., seq_len, d_k)
            token_positions: torch.Tensor - Position indices of shape (seq_len,)
            
        Returns:
            torch.Tensor - Position-encoded tensor of shape (..., seq_len, d_k)
        """
        cos_pos = self.cos_pos[token_positions]
        sin_pos = self.sin_pos[token_positions]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        even = cos_pos * x_even - sin_pos * x_odd
        odd = sin_pos * x_even + cos_pos * x_odd
        return torch.stack([even, odd], dim=-1).reshape(x.shape)

def softmax(x: torch.Tensor, dimension: int) -> torch.Tensor:
    """
    Compute softmax function with numerical stability.
    
    Args:
        x: torch.Tensor - Input tensor
        dimension: int - Dimension along which to compute softmax
        
    Returns:
        torch.Tensor - Softmax output with same shape as input
    """
    x_max = x.max(dim=dimension, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dimension, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query tensor of shape (batch_size, ..., seq_len, d_k)
        K: Key tensor of shape (batch_size, ..., seq_len, d_k)
        V: Value tensor of shape (batch_size, ..., seq_len, d_v)
        mask: Mask tensor of shape (batch_size, ..., seq_len, seq_len)

    Returns:
        Output tensor of shape (batch_size, ..., seq_len, d_v)
    """
    d_k = K.shape[-1]
    QK = torch.einsum("... q d, ... k d -> ... q k", Q, K)
    QK = QK / (d_k ** 0.5)
    if mask is not None:
        QK = QK.masked_fill(~mask, float("-inf"))
    return torch.einsum("... q k, ... k d -> ... q d", softmax(QK, dimension=-1), V)

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    This module implements the scaled dot-product attention with multiple heads,
    optionally using rotary position embeddings.
    """
    def __init__(self, d_model: int, num_heads: int, theta: float | None = None, max_seq_len: int | None = None, device=None, dtype=None):
        """
        Construct a multi-head self-attention module.
        
        Args:
            d_model: int - Model dimension
            num_heads: int - Number of attention heads
            theta: float | None = None - Base frequency for rotary embeddings
            max_seq_len: int | None = None - Maximum sequence length for rotary embeddings
            device: torch.device | None = None - Device to store the parameters on
            dtype: torch.dtype | None = None - Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionEmbedding(theta, d_model // num_heads, max_seq_len, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply multi-head self-attention to input.
        
        Args:
            x: torch.Tensor - Input tensor of shape (batch_size, seq_len, d_model)
            mask: torch.Tensor | None = None - Attention mask of shape (seq_len, seq_len)
            token_positions: torch.Tensor | None = None - Position indices for rotary embeddings
            
        Returns:
            torch.Tensor - Attention output of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.shape[-2]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        x_q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        x_k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        x_v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        if self.rope is not None:
            x_q = self.rope(x_q, token_positions)
            x_k = self.rope(x_k, token_positions)

        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        x_q = scaled_dot_product_attention(x_q, x_k, x_v, mask)
        x_q = rearrange(x_q, "b h s d -> b s (h d)")
        return self.output_proj(x_q)

class TransformerBlock(nn.Module):
    """
    A single transformer block containing self-attention and feed-forward layers.
    
    This block implements the standard transformer architecture with pre-norm
    layer normalization and residual connections.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float | None = None, max_seq_len: int | None = None, device=None, dtype=None):
        """
        Construct a transformer block.
        
        Args:
            d_model: int - Model dimension
            num_heads: int - Number of attention heads
            d_ff: int - Feed-forward network hidden dimension
            theta: float | None = None - Base frequency for rotary embeddings
            max_seq_len: int | None = None - Maximum sequence length for rotary embeddings
            device: torch.device | None = None - Device to store the parameters on
            dtype: torch.dtype | None = None - Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer block to input.
        
        Args:
            x: torch.Tensor - Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor - Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.attn(self.ln1(x)) + x
        return self.ffn(self.ln2(x)) + x   

class TransformerLM(nn.Module):
    """
    Transformer-based Language Model.
    
    This is a complete transformer language model that can be used for
    autoregressive language modeling tasks.
    """
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        """
        Construct a transformer language model.
        
        Args:
            vocab_size: int - Size of the vocabulary
            context_length: int - Maximum sequence length
            d_model: int - Model dimension
            num_layers: int - Number of transformer layers
            num_heads: int - Number of attention heads
            d_ff: int - Feed-forward network hidden dimension
            rope_theta: float - Base frequency for rotary embeddings
            device: torch.device | None = None - Device to store the parameters on
            dtype: torch.dtype | None = None - Data type of the parameters
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer language model.
        
        Args:
            token_positions: torch.Tensor - Token IDs of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor - Logits of shape (batch_size, seq_len, vocab_size)
        """
        x = self.token_embeddings(token_positions)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))