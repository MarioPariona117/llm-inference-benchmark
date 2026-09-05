"""
Operator-level profiling of LLM inference with torch.profiler.

Where the wall-clock benchmark (benchmark.py) answers *"how fast is the whole
generation?"*, this answers the next question a performance engineer asks:
*"where does that time actually go, operator by operator?"*

It runs one generation under `torch.profiler`, prints the top operators by
self-CPU time, and exports a **trace** you can open in a timeline viewer
(chrome://tracing or https://ui.perfetto.dev). torch.profiler captures real
compute times on CPU and CUDA; it does not support Apple MPS, so we profile on
CPU — which is exactly where you get a meaningful per-operator breakdown.

Usage:
    python profile_run.py --model distilgpt2 --new-tokens 16
"""
import argparse
import os

import torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(args):
    device = "cpu"  # torch.profiler gives real per-op times here (not on MPS)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.to(device).eval()

    enc = tokenizer("The history of computing is", return_tensors="pt").to(device)
    gen_kwargs = dict(max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens,
                      do_sample=False, num_beams=1, use_cache=True,
                      pad_token_id=model.config.eos_token_id)

    # Warm up so we profile steady-state, not one-off allocation.
    with torch.no_grad():
        model.generate(**enc, **{**gen_kwargs, "max_new_tokens": 4, "min_new_tokens": 4})

    print(f"Profiling {args.model} generating {args.new_tokens} tokens on CPU...")
    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        model.generate(**enc, **gen_kwargs)

    os.makedirs("results", exist_ok=True)
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15)
    print("\n" + table)

    with open("results/profile_summary.txt", "w") as f:
        f.write(f"torch.profiler operator breakdown — {args.model}, {args.new_tokens} tokens, CPU\n\n")
        f.write(table)
    prof.export_chrome_trace("results/profile_trace.json")
    print("\nWrote results/profile_summary.txt and results/profile_trace.json")
    print("Open the trace at chrome://tracing or https://ui.perfetto.dev")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="distilgpt2",
                   help="small model keeps profiling fast; the operator mix is representative of any transformer")
    p.add_argument("--new-tokens", type=int, default=16)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
