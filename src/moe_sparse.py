"""Sparse mixture-of-experts transformer block."""

import torch
import torch.nn as nn

from .experts import BlockSparseExperts


class SparseMoEBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, n_experts: int = 8,
                 top_k: int = 2, block_size: int = 64):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = BlockSparseExperts(d_model, d_hidden, n_experts, block_size, top_k)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x):
        h = self.norm1(x)
        router_logits = self.router(h.mean(dim=1))
        moe_out = self.experts(h, router_logits)
        x = x + moe_out
        x = x + self.ffn(self.norm2(x))
        return x
