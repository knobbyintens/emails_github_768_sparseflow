"""Training loop for SparseFlow."""

import argparse

import torch
import yaml

from src.moe_sparse import SparseMoEBlock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    block = SparseMoEBlock(cfg["d_model"], cfg["d_hidden"],
                           cfg["n_experts"], cfg["top_k"]).cuda()
    optim = torch.optim.AdamW(block.parameters(), lr=cfg["lr"])

    block.train()
    for step in range(1, cfg["steps"] + 1):
        x = torch.randn(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device="cuda")
        loss = block(x).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 50 == 0:
            print({"step": step, "loss": float(loss)})
    torch.save(block.state_dict(), "checkpoints/sparseflow.pt")


if __name__ == "__main__":
    main()
