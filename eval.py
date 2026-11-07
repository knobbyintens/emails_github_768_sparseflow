"""Throughput and quality benchmark for SparseFlow."""

import argparse
import json
import time

import torch
import yaml

from src.moe_sparse import SparseMoEBlock


@torch.no_grad()
def benchmark(model, batch_size, seq_len, d_model, iters=20):
    x = torch.randn(batch_size, seq_len, d_model, device="cuda")
    for _ in range(3):
        model(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        model(x)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / iters
    tokens = batch_size * seq_len
    return {"tokens_per_second": tokens / elapsed,
            "latency_ms": elapsed * 1000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    model = SparseMoEBlock(cfg["d_model"], cfg["d_hidden"],
                           cfg["n_experts"], cfg["top_k"]).cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    report = {str(s): benchmark(model, cfg["batch_size"], s, cfg["d_model"])
              for s in (1024, 4096, 16384)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
