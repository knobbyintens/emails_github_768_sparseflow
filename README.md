# SparseFlow

Reference code for **SparseFlow: Structured Sparsity for Fast Inference of Mixture-of-Experts** (ICML 2026).

SparseFlow introduces block-structured expert masks and a latency-aware routing
objective. The router learns which expert blocks to keep resident for a batch,
so the sparse compute is predictable and kernels can be scheduled without
dynamic shapes.

## Layout

- `src/experts.py` — block-sparse expert layer
- `src/moe_sparse.py` — sparse mixture-of-experts transformer block
- `train.py`, `eval.py` — training and throughput benchmarks
- `configs/base.yaml`

## Usage

```bash
pip install -r requirements.txt
python train.py --config configs/base.yaml
python eval.py --config configs/base.yaml --checkpoint checkpoints/sparseflow.pt
```
