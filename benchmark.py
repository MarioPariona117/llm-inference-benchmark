"""
LLM inference performance benchmark.

Measures generation latency, decode throughput (tokens/sec), and peak memory
for a causal language model across a sweep of batch sizes and generation
lengths — and across **precision** (fp32 vs fp16) and **hardware** (CUDA / Apple
MPS / CPU).

The point is data-driven analysis of how inference behaves across settings —
the question a performance-analysis role actually asks: *given a workload, how
does it behave across the system, and where does the time (and memory) go?*

Usage:
    # A real small LLM in half precision on the Apple GPU:
    python benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --device mps --dtype fp16 --batch-sizes 1 2 4 8 --new-tokens 32 128

    # Same in full precision, to compare fp32 vs fp16:
    python benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --device mps --dtype fp32 --batch-sizes 1 2 4 8 --new-tokens 32 128

Results are written to results/benchmark_<model>_<device>_<dtype>.csv.
No results are committed until you run it yourself.
"""
import argparse
import csv
import gc
import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def reset_and_read_memory(device):
    """Free cached memory and return current allocated MB (0 on CPU where we
    don't have a cheap per-op counter)."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device == "mps":
        torch.mps.empty_cache()
    return current_memory_mb(device)


def current_memory_mb(device):
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    if device == "mps":
        return torch.mps.current_allocated_memory() / 1024**2
    return 0.0


def time_generation(model, input_ids, attention_mask, new_tokens, device):
    """Wall-clock seconds for one greedy generation of exactly `new_tokens`."""
    sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,  # force a fixed amount of decode work
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=model.config.eos_token_id,
        )
    sync(device)
    return time.perf_counter() - start


def run(args):
    device = pick_device(args.device)
    dtype = DTYPES[args.dtype]
    print(f"Device: {device} | Model: {args.model} | dtype: {args.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    prompt = "The history of computing is"
    rows = []

    for batch_size in args.batch_sizes:
        for new_tokens in args.new_tokens:
            prompts = [prompt] * batch_size
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]

            reset_and_read_memory(device)
            for _ in range(args.warmup):
                time_generation(model, input_ids, attention_mask, new_tokens, device)

            samples = [
                time_generation(model, input_ids, attention_mask, new_tokens, device)
                for _ in range(args.repeats)
            ]
            peak_mem = current_memory_mb(device)  # footprint at end of generation
            mean_s = statistics.mean(samples)
            std_s = statistics.pstdev(samples) if len(samples) > 1 else 0.0
            total_tokens = batch_size * new_tokens
            throughput = total_tokens / mean_s

            rows.append({
                "model": args.model,
                "device": device,
                "dtype": args.dtype,
                "batch_size": batch_size,
                "new_tokens": new_tokens,
                "mean_latency_s": round(mean_s, 5),
                "std_latency_s": round(std_s, 5),
                "throughput_tokens_per_s": round(throughput, 2),
                "peak_mem_mb": round(peak_mem, 1),
            })
            print(f"  batch={batch_size:>3} tok={new_tokens:>4} "
                  f"latency={mean_s*1000:8.1f} ms  thr={throughput:8.1f} tok/s  "
                  f"mem={peak_mem:7.0f} MB")
            reset_and_read_memory(device)

    os.makedirs("results", exist_ok=True)
    safe = args.model.replace("/", "_")
    out_path = os.path.join("results", f"benchmark_{safe}_{device}_{args.dtype}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                   help="HuggingFace model id (default: a real 1.1B LLM)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--dtype", default="fp16", choices=list(DTYPES),
                   help="numerical precision (fp16 halves memory vs fp32)")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--new-tokens", type=int, nargs="+", default=[32, 128])
    p.add_argument("--repeats", type=int, default=3, help="timed runs per config")
    p.add_argument("--warmup", type=int, default=1, help="untimed warmup runs per config")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
