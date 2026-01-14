import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)              # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class GPT2Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)  # token embeddings
        self.wpe = nn.Embedding(max_len, d_model)     # position embeddings

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0)  # (1, seq_len)

        return self.wte(input_ids) + self.wpe(positions)

vocab_size = 50257   # GPT-2 default
max_len = 1024       # GPT-2 context length

# pe = GPT2Embeddings(vocab_size, d_model=64, max_len=max_len)

pe = PositionalEncoding(d_model=64)

x = torch.zeros(2, 5, 64)

out = pe(x)

print(out.shape)

x = torch.zeros(2, 5, 64)
out = pe(x)

print(torch.allclose(out[0], out[1]))

print(torch.allclose(out[:, 0, :], out[:, 1, :]))

pos0 = pe.pe[0, :, 0]  # sin
pos1 = pe.pe[0, :, 1]  # cos

print((pos0**2 + pos1**2).mean())

x = torch.randn(2, 5, 64, requires_grad=True)
out = pe(x)
out.sum().backward()

print(x.grad is None)
print(hasattr(pe.pe, "grad"))

x1 = torch.zeros(1, 5, 64)
x2 = torch.zeros(1, 6, 64)

out1 = pe(x1)
out2 = pe(x2)

print(torch.allclose(out1[:, 1:5], out2[:, 1:5]))

