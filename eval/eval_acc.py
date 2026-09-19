"""GSM8K accuracy eval against a running vLLM server.

Run once per server config:
    python eval_acc.py --label bf16
    python eval_acc.py --label fp8
    python eval_acc.py --label awq          (model is read from the server)
Compare two finished runs (paired McNemar test on the same questions):
    python eval_acc.py --compare bf16 awq
"""
import argparse, asyncio, json, math, random, re, time
from pathlib import Path

OUT = Path("/content/drive/MyDrive/vllm-bench")
PROMPT = "\n\nSolve step by step. End with a final line: Answer: <number>"
NUM = r"-?\d[\d,]*\.?\d*"


def to_float(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_pred(text):
    """Number after the last 'Answer:', else the last number anywhere in the reply."""
    text = text or ""
    tagged = re.findall(r"Answer:\s*\$?\s*(" + NUM + ")", text, flags=re.IGNORECASE)
    if tagged:
        return to_float(tagged[-1])
    nums = re.findall(NUM, text)
    return to_float(nums[-1]) if nums else None


def extract_gold(ans):
    return to_float(ans.split("####")[-1])


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


async def run(args):
    from datasets import load_dataset
    from openai import AsyncOpenAI

    ds = load_dataset("openai/gsm8k", "main", split="test")
    random.seed(args.seed)
    idx = sorted(random.sample(range(len(ds)), args.n))  # identical questions for every config
    client = AsyncOpenAI(base_url=args.base_url, api_key="x")
    served = [m.id for m in (await client.models.list()).data]
    if args.model is None:
        args.model = served[0]
    elif args.model not in served:
        raise SystemExit(f"Server is serving {served}, not {args.model}. Restart the server or drop --model.")
    print(f"label={args.label}  model={args.model}  n={args.n}")
    sem = asyncio.Semaphore(args.concurrency)

    async def ask(i):
        q = ds[i]
        async with sem:
            r = await client.chat.completions.create(
                model=args.model, temperature=0, max_tokens=512,
                messages=[{"role": "user", "content": q["question"] + PROMPT}])
        text = r.choices[0].message.content
        pred, gold = extract_pred(text), extract_gold(q["answer"])
        correct = pred is not None and gold is not None and abs(pred - gold) < 1e-6
        return {"id": i, "correct": correct, "pred": pred, "gold": gold,
                "output_tokens": r.usage.completion_tokens, "reply_tail": text[-200:]}

    t0 = time.time()
    rows = await asyncio.gather(*(ask(i) for i in idx))
    k = sum(r["correct"] for r in rows)
    res = {"label": args.label, "model": args.model, "n": args.n, "seed": args.seed,
           "correct": k, "accuracy": round(k / args.n, 4), "ci95": wilson(k, args.n),
           "unparsed": sum(r["pred"] is None for r in rows),
           "seconds": round(time.time() - t0, 1), "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"acc_{args.label}.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"{args.label}: {k}/{args.n} = {res['accuracy']:.1%}  95% CI {res['ci95']}  "
          f"unparsed={res['unparsed']}  ({res['seconds']}s) -> {path}")


def compare(a, b):
    from statsmodels.stats.contingency_tables import mcnemar

    A = {r["id"]: r["correct"] for r in json.loads((OUT / f"acc_{a}.json").read_text())["rows"]}
    B = {r["id"]: r["correct"] for r in json.loads((OUT / f"acc_{b}.json").read_text())["rows"]}
    ids = sorted(A.keys() & B.keys())
    only_a = sum(A[i] and not B[i] for i in ids)
    only_b = sum(B[i] and not A[i] for i in ids)
    both = sum(A[i] and B[i] for i in ids)
    neither = len(ids) - both - only_a - only_b
    p = mcnemar([[both, only_a], [only_b, neither]], exact=True).pvalue
    print(f"{a} vs {b} on {len(ids)} shared questions: only {a} right={only_a}, "
          f"only {b} right={only_b}, McNemar exact p={p:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--model", default=None, help="defaults to whatever the server is serving")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        if not a.label:
            ap.error("--label is required unless --compare is used")
        asyncio.run(run(a))
