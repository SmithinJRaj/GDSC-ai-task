import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        return: (batch, heads, seq_len, d_head)
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_head)
        return x.transpose(1, 2)

    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Q, K, V: (batch, heads, seq_len, d_head)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        mask: (batch, 1, 1, seq_len) or None
        """
        Q = self._split_heads(self.W_q(x))
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        out = self._scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous()
        batch_size, seq_len, num_heads, d_head = out.size()
        out = out.view(batch_size, seq_len, num_heads * d_head)


        return self.W_o(out)

mha = MultiHeadAttention(d_model=64, num_heads=8)

# Example input: batch of 2 sequences, each of length 5, embedding size 64
x = torch.randn(2, 5, 64)  # (batch, seq_len, d_model)

# Forward pass
out = mha(x)

print("Output shape:", out.shape)