# A Plain-English Guide to What This Benchmark Measures

You don't need any ML background to read this. By the end you'll understand
exactly what the numbers mean, what's being compared, and why each comparison
matters. Real figures below come from `TinyLlama-1.1B` on an Apple GPU (MPS).

---

## 1. What "LLM inference" actually is

An LLM (large language model) generates text **one token at a time**. A *token*
is roughly a word-piece — "computing" might be one token, "unbelievable" might be
three ("un", "believ", "able"). To write a sentence, the model runs in a loop:

1. Read everything so far → predict the **next** token.
2. Append that token to the text.
3. Go back to step 1, now with one more token of context.
4. Stop after N tokens (or at an end-of-text token).

"**Inference**" just means *running* a trained model to get output (as opposed
to *training* it). This benchmark measures how fast and how memory-hungry that
generation loop is.

There are two phases in a single generation:

- **Prefill** — the model reads your whole prompt at once (fast, done in parallel).
- **Decode** — the model then produces new tokens one by one (slow, sequential —
  each token must wait for the previous one). Decode is where almost all the time
  goes, which is why this benchmark focuses on it.

> **Why "one at a time" is the whole story:** you can't generate token #50 until
> you've generated token #49, because #50 depends on it. This sequential
> dependency is *the* defining constraint of LLM inference, and it explains most
> of what we measure.

---

## 2. The two numbers: latency and throughput

Every performance conversation comes down to these two, and they're **not** the
same thing:

- **Latency** — how long *one* request takes, start to finish. Measured in
  milliseconds/seconds. *"How long until this user gets their answer?"* Lower is
  better.
- **Throughput** — how many tokens the system produces *per second in total*,
  across everything it's doing at once. *"How many users can we serve?"* Higher is
  better.

The catch: **they trade off against each other.** You can often get more
throughput (serve more people) at the cost of higher latency (each person waits
a bit longer), and vice-versa. A big part of inference engineering is picking the
right point on that trade-off for your use case (a chatbot cares about latency; a
batch document-processing job cares about throughput).

Example from the real data (one prompt, 32 new tokens): latency **917 ms**,
throughput **34.9 tokens/sec**.

---

## 3. Why we sweep **generation length** (new tokens)

Because decode is sequential (§1), generating twice as many tokens takes roughly
twice as long. The data confirms it — same batch of 1, on MPS:

| New tokens | Latency |
|---|---|
| 32  | 917 ms |
| 128 | 4,700 ms |

4× the tokens → ~5× the time. Latency grows (roughly) **linearly** with how much
text you ask for. This is why "make the model answer more concisely" is itself a
real latency optimization.

---

## 4. What a **batch** is, and why we sweep batch size

A **batch** means running several prompts *through the model at the same time*
instead of one after another. Batch size 8 = eight prompts processed together in
one pass of the loop.

**Why would that help?** Because the hardware (especially a GPU) is massively
parallel — it has thousands of tiny compute units. Running one prompt leaves most
of them idle; the model is small relative to the hardware, so you're paying for a
big machine to do one small thing. Batching fills those idle units with useful
work, so you get **much more total throughput for almost the same latency** — up
to a point.

The real data shows exactly this (32 new tokens, MPS, fp16):

| Batch size | Latency | Throughput |
|---|---|---|
| 1 | 917 ms   | 34.9 tok/s |
| 2 | 1,196 ms | 53.5 tok/s |
| 4 | 1,879 ms | 68.1 tok/s |
| 8 | 3,055 ms | 83.8 tok/s |

Notice: going from batch 1 → 8, throughput **2.4×'d** (34.9 → 83.8 tok/s) while
latency only ~3×'d. You served 8× the prompts for 3× the wait — a big efficiency
win. This is why real LLM services batch many users' requests together.

**The knee.** Batching only helps until the hardware is *full*. Once every compute
unit is busy, adding more to the batch just makes everyone queue — throughput
flattens and latency keeps climbing. Finding that "knee" is a core job of
inference performance analysis. (On a *tiny* model like `distilgpt2` we actually
saw throughput *drop* at the largest batch × longest length — the system tipped
past its sweet spot. The bigger `TinyLlama` scales more smoothly at these sizes.)

---

## 5. The comparisons, and why each matters

The benchmark varies four things. Here's what each one teaches:

| We vary… | To answer… |
|---|---|
| **Batch size** | How well does the workload use the hardware? Where's the knee? (§4) |
| **Generation length** | How much does the sequential decode loop cost? (§3) |
| **Hardware** (CPU vs GPU/MPS) | What does specialised hardware buy us? (§5a) |
| **Precision** (fp32 vs fp16) | Can we trade a little accuracy for big memory/speed wins? (§5b) |

### 5a. CPU vs GPU (MPS)

A **CPU** has a few powerful general-purpose cores. A **GPU** (Apple calls its
built-in one "MPS" — Metal Performance Shaders) has thousands of small cores built
for doing the same maths on lots of data at once — which is exactly what a neural
network is. So the GPU is usually much faster for this work, *especially* when
batching keeps it busy. Our earlier small-model run showed the GPU ~2.4–2.9×
faster than the CPU — but interestingly the gap **shrank to near-parity at the
heaviest workload**, a reminder that "the faster chip" isn't uniformly faster
across every workload.

### 5b. Precision: fp32 vs fp16 (this is the big edge-AI lever)

A model is millions of numbers ("weights"). **Precision** is how many bits we use
to store each one:

- **fp32** ("full precision") — 32 bits per number. The default.
- **fp16** ("half precision") — 16 bits per number. Half the storage.

Using fewer bits means each number is slightly less exact — but neural networks
are remarkably tolerant of this, so the output is usually nearly identical. The
textbook promise is *"half the memory **and** faster."*

Our measurements tell a more interesting (and more honest) story:

| | Peak memory | Peak throughput |
|---|---:|---:|
| **fp32** | 4,196 MB | **125.9 tok/s** |
| **fp16** | **2,098 MB** | 83.8 tok/s |

- **Memory: fp16 halved it, exactly as promised** (2.1 GB vs 4.2 GB) — the
  guaranteed, hardware-independent win.
- **Speed: fp16 was *not* faster here — fp32 actually won** on this Apple GPU (see
  `results/compare_throughput.png`). Apple's MPS backend clearly has better fp32
  code paths for this model than fp16.

That surprise is the whole point of *measuring* instead of *assuming*. The
"fp16 is faster" rule of thumb holds on many NVIDIA GPUs (which have dedicated
16-bit tensor cores) but **is not a law of nature** — it depends on the hardware
and the software stack. A performance engineer's job is exactly this: validate
the assumption on the actual target, and report what's really true.

**Why precision still matters enormously for edge AI:** on phones, cameras, and
ARM chips, memory and power are the binding constraints, so halving memory (and
the even bigger savings from 8-bit/4-bit "**quantization**") is often what makes a
model *fit at all* — regardless of whether it also speeds things up on a given
chip. Analysing that trade-off — memory and speed gained vs accuracy lost, per
target device — is a core part of an edge-AI performance role.

---

## 6. The KV cache (why memory grows, and why big batches can stall)

When the model generates token #50, it needs to "look back" at tokens #1–49. Re-
computing all of that every step would be enormously wasteful, so the model
**caches** the intermediate results for every past token. This is the **KV cache**
("key–value" cache).

The important consequence: the KV cache **grows with both batch size and sequence
length**. Its size is, per token per sequence:

```
2 (keys + values) × layers × kv_heads × head_dimension × bytes_per_number
```

For `TinyLlama-1.1B` (22 layers, 4 KV heads, head-dim 64, fp16 = 2 bytes) that's
about **22 KB per token, per sequence in the batch**. At batch 8 × 128 tokens that's
~22 MB — small, because TinyLlama uses a trick called **grouped-query attention**
(only 4 "KV heads" instead of 32) specifically to keep this cache small. Older
models without that trick have much larger KV caches, which is why *they* start to
choke (memory pressure, slower memory access) at large batch × length while
TinyLlama keeps scaling. **The KV cache is the usual reason "big batch + long
output" gets disproportionately expensive.**

---

## 7. Profiling: where the time actually *goes*

The benchmark answers *"how fast is it?"* **Profiling** answers the next
question: *"which parts are slow?"* — so you know what to fix.

A model's forward pass is built from many small mathematical **operations**
("operators") — mostly matrix multiplications, plus attention, activation
functions, and memory shuffles. A **profiler** (here, `torch.profiler`) times
every single one and shows how the total breaks down. Running it on one
generation gave:

| Operation | Share of time | What it is |
|---|---:|---|
| `addmm` | 50% | the linear layers (a matrix multiply plus a bias) |
| `mm` | 24% | more matrix multiplies (the final word-prediction layer) |
| attention | 3% | the "look back at previous tokens" step |
| everything else | ~23% | activations, copies, reshapes |

The headline: **~74% of the time is matrix multiplication.** Not attention, not
anything exotic — just multiplying big grids of numbers together.

**Why that single number is the whole edge-AI story:**

- The specialised chips that make AI fast — GPU "tensor cores," and the **NPUs**
  in phones and ARM devices — are, at their heart, **matrix-multiply
  accelerators**. Now you can see *why* they help: they attack that 74%.
- **Quantization** (§5b) speeds things up for the same reason — fewer bits makes
  each of those matrix multiplies cheaper.

So profiling converts a vague *"the model is slow"* into a precise, actionable
*"the model is matmul-bound — so the levers are matmul hardware and
cheaper-matmul precision."* That chain — measure, find the bottleneck, name the
fix — is the core of what a performance engineer does.

The profiler also saves a **trace**: a timeline of every operation, openable in a
viewer (`chrome://tracing` or [Perfetto](https://ui.perfetto.dev)) so you can
literally *see* the sequence of work over time.

---

## 8. How to read the plots

- **`*_throughput_vs_batch.png`** — going up-and-to-the-right means batching is
  helping. A flattening or downturn is the "knee" where the hardware is full.
- **`*_latency_vs_batch.png`** — how much each individual request slows down as you
  pack more into a batch. The throughput/latency trade-off (§2) lives in the gap
  between this plot and the one above.
- **`*_latency_vs_length.png`** — should rise roughly linearly; the slope is your
  "cost per generated token" (§3).
- **`compare_memory.png` / `compare_throughput.png`** — fp16 vs fp32 side by side.
  fp16 uses ~half the memory (always); whether it's also *faster* depends on the
  hardware — here it wasn't (§5b).

---

## 9. Why this matters for an edge-AI / ARM role

ARM chips power phones, cameras, cars, and IoT devices — all with tight memory,
power, and thermal budgets. Getting an AI model to run *well* on them is entirely
about the questions this benchmark asks:

- How does inference behave across batch sizes and sequence lengths? (utilisation)
- CPU vs GPU vs NPU — which hardware, for which workload? (heterogeneous compute)
- How much can precision/quantization shrink and speed up the model? (the core lever)
- Where does memory (the KV cache) become the bottleneck?
- Where does the time actually go, operator by operator? (profiling → matmul-bound)

A benchmark like this — *measure across settings, find the bottleneck, explain
why* — is the day-to-day of an AI-inference performance engineer.

---

## Glossary

- **Token** — a word-piece; the unit LLMs read and generate.
- **Inference** — running a trained model to get output (vs. training it).
- **Prefill / decode** — reading the prompt (parallel, fast) vs. generating new tokens (sequential, slow).
- **Latency** — time for one request. **Throughput** — tokens/sec across everything at once.
- **Batch** — several prompts processed together in one pass.
- **Precision (fp32/fp16)** — bits per number; fewer bits = less memory, often faster, slightly less exact.
- **Quantization** — deliberately using fewer bits (fp16, int8, int4) to shrink/speed up a model.
- **KV cache** — stored intermediate results for past tokens; grows with batch × length.
- **Operator** — one low-level math operation the model is built from (matrix multiply, attention, etc.).
- **Profiling** — timing each operator to see *where* the time goes (vs. benchmarking, which times the whole thing).
- **Trace** — a timeline recording of every operation, viewable in `chrome://tracing` or Perfetto.
- **Matmul-bound** — dominated by matrix multiplication (LLM inference is ~74% matmul); why matmul accelerators (tensor cores, NPUs) matter.
- **MPS** — Apple's built-in GPU backend (Metal Performance Shaders); **NPU** — a dedicated neural-network accelerator chip.
