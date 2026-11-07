"""Block-structured sparse expert layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockSparseExperts(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, n_experts: int = 8,
                 block_size: int = 64, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.block_size = block_size

        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, d_hidden))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_hidden, d_model))
        self.block_masks = nn.Parameter(torch.ones(n_experts, d_hidden // block_size))

        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def _expert_forward(self, x, idx):
        h = torch.einsum("btm,mh->bth", x, self.w1[idx])
        mask = self.block_masks[idx].repeat_interleave(self.block_size)
        h = F.gelu(h) * mask
        return torch.einsum("bth,hm->btm", h, self.w2[idx])

    def forward(self, x, router_logits):
        probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)

        out = torch.zeros_like(x)
        for slot in range(self.top_k):
            expert_idx = topk_idx[..., slot]
            weight = topk_probs[..., slot].unsqueeze(-1)
            for e in range(self.n_experts):
                mask = (expert_idx == e).unsqueeze(-1).float()
                expert_out = self._expert_forward(x, e)
                out = out + weight * mask * expert_out
        return out
