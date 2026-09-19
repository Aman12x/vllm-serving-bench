"""Concurrency sweep against a running vLLM server.

The server is started separately (one precision per server), e.g.
    vllm serve Qwen/Qwen2.5-7B-Instruct --quantization fp8 --max-model-len 4096 --gpu-memory-utilization 0.90

Then:
    python bench/sweep.py --config configs/fp8.yaml
    python bench/sweep.py --config configs/fp8.yaml --dry-run      # print commands only

Each level writes <label>_c<concurrency>.json into the results dir and a one-line summary to stdout.
Prompt counts scale with load so low-concurrency levels do not take an hour.
"""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
from urllib.request import urlopen

DEFAULTS = {
    "input_len": 512,
    "output_len": 256,
    "concurrency": [1, 4, 16, 32, 64],
    "prompts_per_slot": 8,   # num_prompts = concurrency * this
    "min_prompts": 20,       # floor, so concurrency 1 still runs 20 requests
    "base_url": "http://localhost:8000",
    "results_dir": "results/raw",
}


def load_config(path):
    text = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        import yaml  # pyyaml ships with Colab; pip install pyyaml elsewhere
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    missing = {"label"} - cfg.keys()
    if missing:
        sys.exit(f"{path} is missing required key(s): {', '.join(sorted(missing))}")
    return {**DEFAULTS, **cfg}


def served_model(base_url):
    """Ask the server which model it is serving, so the label can never drift from reality."""
    try:
        with urlopen(f"{base_url}/v1/models", timeout=10) as r:
            return json.load(r)["data"][0]["id"]
    except Exception as e:
        sys.exit(f"No vLLM server answering at {base_url} ({e}). Start the server first.")


def run_level(cfg, model, concurrency, dry_run):
    n = max(cfg["min_prompts"], concurrency * cfg["prompts_per_slot"])
    out_name = f"{cfg['label']}_c{concurrency}.json"
    cmd = [
        "vllm", "bench", "serve",
        "--base-url", cfg["base_url"],
        "--model", model,
        "--dataset-name", "random",
        "--random-input-len", str(cfg["input_len"]),
        "--random-output-len", str(cfg["output_len"]),
        "--num-prompts", str(n),
        "--max-concurrency", str(concurrency),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,95,99",
        "--save-result",
        "--result-dir", cfg["results_dir"],
        "--result-filename", out_name,
    ]
    if dry_run:
        print(" ".join(cmd))
        return None
    print(f"--- {cfg['label']} concurrency={concurrency} prompts={n}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:] + proc.stderr[-2000:], file=sys.stderr)
        sys.exit(f"benchmark failed at concurrency {concurrency} (exit {proc.returncode})")
    path = Path(cfg["results_dir"]) / out_name
    d = json.loads(path.read_text())
    print(f"    {d['output_throughput']:8.2f} tok/s | tpot_med {d['median_tpot_ms']:6.2f} ms | "
          f"ttft_med {d['median_ttft_ms']:7.1f} ms | ttft_p95 {d['p95_ttft_ms']:7.1f} ms | "
          f"{time.time() - t0:.0f}s -> {path}", flush=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML or JSON config, see configs/")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--concurrency", type=int, nargs="*", help="override the config's concurrency list")
    args = ap.parse_args()

    if shutil.which("vllm") is None:
        sys.exit("vllm is not on PATH — run this on the machine serving the model.")
    cfg = load_config(args.config)
    if args.concurrency:
        cfg["concurrency"] = args.concurrency
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)

    model = served_model(cfg["base_url"])
    if cfg.get("expect_model") and cfg["expect_model"] != model:
        sys.exit(f"Server is serving {model}, but {args.config} expects {cfg['expect_model']}. "
                 f"Wrong server for this config.")
    print(f"label={cfg['label']}  model={model}  levels={cfg['concurrency']}  "
          f"{cfg['input_len']} in / {cfg['output_len']} out")
    if cfg.get("note"):
        print(f"note: {cfg['note']}")

    for c in cfg["concurrency"]:
        run_level(cfg, model, c, args.dry_run)


if __name__ == "__main__":
    main()
