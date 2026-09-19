# vLLM serving benchmark: precision, latency, throughput, accuracy

<!-- TODO: one-line answer with your real numbers, then the chart from analysis/report.ipynb -->

**Question:** on a single NVIDIA L4, which way of serving a 7B model gives the cheapest tokens that still
meets a latency target, and what does each shortcut cost in accuracy?

**Model:** Qwen2.5-7B-Instruct · **GPU:** NVIDIA L4 (23 GB, Colab) · **Server:** vLLM 0.29.0, PyTorch 2.13 / CUDA 13.0

## Setup compared

| Label | Weights | Math runs at | How it is made |
|---|---|---|---|
| `bf16` | 16-bit float | 16-bit | as published |
| `fp8` | 8-bit float | 8-bit (L4 FP8 hardware) | converted at load: `--quantization fp8` |
| `awq` | 4-bit integer, group size 128 | 16-bit (converted on the fly) | pre-quantized: `Qwen2.5-7B-Instruct-AWQ` |

## Serving benchmark

512 input tokens, 256 output tokens per request, concurrency 1 / 4 / 16 / 32 / 64,
`vllm bench serve` with the random dataset. Raw runs: `results/raw/<label>_c<concurrency>.json`.

<!-- TODO: table generated from analysis/report.ipynb — never hand-typed -->

## Accuracy

GSM8K, all 1,319 test questions, temperature 0, answers parsed and matched numerically.
Same questions for every setup, so the comparison is paired (`eval/eval_acc.py --compare`).

| Setup | Correct | Accuracy | 95% CI | vs BF16 (McNemar, exact) |
|---|---|---|---|---|
| BF16 | 1199 / 1319 | 90.9% | 89.2–92.3% | — |
| FP8 | 1191 / 1319 | 90.3% | 88.6–91.8% | p = 0.382 |
| AWQ | 1194 / 1319 | 90.5% | 88.8–92.0% | p = 0.657 |

<!-- TODO: state the finding in one sentence, with the throughput multiple from the table above -->

## Honest limits

- One GPU (L4), one model, one prompt shape (512 in / 256 out). Results move with all three.
- 1,319 paired questions can rule out accuracy gaps of roughly 2 points, not smaller ones.
- GSM8K is arithmetic reasoning. A harder task may separate the setups where this one does not.
- Cost per 1M tokens is computed from a published L4 hourly rate, not from a bill.

## Reproduce

1. `colab/setup.ipynb` — GPU runtime, SSH tunnel, vLLM install. <!-- TODO: export the working cell -->
2. Start a server, e.g. `vllm serve Qwen/Qwen2.5-7B-Instruct --quantization fp8 --max-model-len 4096 --gpu-memory-utilization 0.90`
3. `python bench/sweep.py --config configs/fp8.yaml` <!-- TODO: write it -->
4. `python eval/eval_acc.py --label fp8_full --n 1319`
5. `python eval/eval_acc.py --compare bf16_full fp8_full`

## Layout

```
bench/sweep.py        the concurrency sweep (TODO)
eval/eval_acc.py      GSM8K accuracy + paired McNemar comparison
configs/              one file per serving setup (TODO)
results/raw/          every run, committed, never edited by hand
analysis/report.ipynb tables + charts, reads results/raw (TODO)
colab/setup.ipynb     runtime setup (TODO)
```
