# Rejoin Phase 0 — escrow survival and alignment study

**Goal:** measure whether the escrowed suffix after a speculative rejection is a
*better draft* than a fresh one, and — if it is not — whether it reattaches at
all once you allow a short bridge. These two numbers decide whether Rejoin is
worth building.

| File | What it is |
|---|---|
| `trace_core.py` | Model-agnostic core loop (cache bookkeeping, event pairing, alignment). Pure Python. |
| `test_core.py` | CPU-only tests with mock models. **Run this first, always.** |
| `spec_trace.py` | Real-model wiring: HF target/draft sessions, datasets, CLI, JSONL output. |
| `analyze_traces.py` | Turns traces into survival curves, bridge curves, and the cost-corrected go/no-go table. |

Traces are written outside the package (`traces/`, gitignored). Schema version
is `2`; every record carries `"schema"`.

---

## 0. Results

**The study was run and the mechanism does not pay.** Escrowed suffixes survive
far above chance, but only under alignment slack that is unavailable online, and
every implementable policy falls below break-even.

Qwen3-4B target + Qwen3-0.6B draft, γ=16, greedy, on a T4.

| run | prompts | events | tokens/target-pass |
|---|---|---|---|
| code (HumanEval, bf16) | 164 | 4004 | 6.08 |
| math + reasoning chains (GSM8K, `--thinking`, fp16) | 25 | 4574 | 6.03 |

### The escrow does survive

Escrow content reappears in the target's continuation far above the chance
control (escrows paired with *other* events' continuations):

| | code | reasoning | chance (code / reasoning) |
|---|---|---|---|
| `lcs ≥ 4` | 44.5% | 51.8% | 0.8% / 1.7% |
| `lcs ≥ 8` | 15.7% | 15.5% | 0.0% / 0.1% |
| mean `lcs_len` | 4.05 | 4.39 | 0.84 / 1.35 |

A 30–56× ratio at `lcs ≥ 4`. This is real and it is not boilerplate collision.
It is also invisible to offset-0 measurement, which is all schema 1 recorded.

### But it survives misaligned, and the reachable rungs fail

Mean accepted tokens per rejection, and the recycle-if-better gain `g` against
break-even `g_min = cost_factor × tokens-per-pass`:

| rung | code `g` | reasoning `g` | `g_min` (batched) | implementable |
|---|---|---|---|---|
| offset-0 (the original Rejoin design) | 0.06 | 0.07 | ~0.72 | yes |
| front-trim (batched candidate tree) | 0.13 | 0.22 | ~0.72 | yes |
| best-bridge | 0.36 | 0.34 | ~0.72 | no — needs the target's continuation |
| two-sided (`lcs`) | 1.55 | 1.69 | ~0.72 | no — and never pays for its bridge |

At **zero** verification cost — the ceiling regardless of engineering —
front-trim is worth **+1.8%** (code) and **+3.3%** (reasoning). The frequently
quoted "+10–13%" belongs to the two-sided rung, which needs the answer in
advance to build its candidate.

The structural reason: the escrow is `P(continuation | prefix + wrong_token)`
and the fresh draft is `P(continuation | prefix + correct_token)` — the same
model, with the fresh sample strictly better conditioned. `delta` is negative in
every domain, at every rung, across 8578 events.

### Selective application does not rescue it

Gating scales benefit and cost identically, so `(N + f·E·g_sel)/(P + c·f·E) >
N/P` reduces to `g_sel > c·T`: **coverage cancels**. A gate helps only if its
selected events clear `g_min` alone, and coverage instead caps the payoff.

Twenty online-observable gates were tested (event class, escrow length, mismatch
index, target entropy at both tails, `p_rejected`). **None clears `g_min` on
either workload.** The best is `p_rejected ≥ 0.4` at `g_sel = 0.42` against 0.72
— near-misses genuinely salvage about twice as well as blunders, and it is still
1.7× short at 7.7% coverage.

An ORACLE gate selecting exactly the winning events — the ceiling for any
predicate that could ever exist — caps at **+1.3%** (code) and **+2.5%**
(reasoning), because only 5–7% of rejections have any headroom at all.

### The roadmap's structural-repetition prediction did not hold

Reasoning chains were the strongest remaining case: the roadmap predicts benefit
rises with structural repetition, and FailFast reported high suffix utility
there. Thousand-token reasoning chains gave `lcs_len` 4.39 against short Python
functions' 4.05. Two maximally different workloads, the same answer.

### Secondary findings

- **Entropy does not predict survival.** A pilot signal vanished at scale (code
  0.58 low-entropy vs 0.54 high). Do not gate on it.
- **Most rejections are blunders, not near-misses**: 43.5% of code rejections
  have `p_rejected < 0.01` (24.6% in reasoning).
- **Tokenization mismatches are the best-salvaging class** (`L_survive` 2.52 vs
  0.50 for semantic) but only 2.1% of events — contribution ≈ 0.05 tokens/event.
- **Prompt format contaminated the pilot**: an unpinned output format put 27% of
  events in a maximum-escrow, zero-survival bucket. The prefill fixed it (2.5%).

### Caveats

- One model pair, one γ, greedy only. `g_min` scales with tokens-per-pass, so a
  much weaker draft or larger target lowers the bar — untested, and the escrow
  plausibly degrades just as fast.
- 164 and 25 prompts. Events are correlated within prompts, so intervals are
  wider than ~4000 events suggests. Do not quote three significant figures.
- The code run is bf16 and the reasoning run fp16 (T4 has no bf16 tensor cores).
  The agreement across a far larger domain gap makes dtype implausible as a
  driver, but it is a confound.
- The final runs carry no `--paranoid` checks; cache correctness was validated
  on the smoke run (26/26) and by the survival cross-check (0 disagreements).

---

## 1. What you're measuring

Standard greedy speculative decoding: the draft proposes γ tokens, the target
scores the block in **one** forward pass, tokens are accepted up to the first
mismatch, the target's own choice there is committed for free (the **bonus
token**), and everything after the rejection is discarded.

The scarce resource is **target forward passes**. Repairing and re-verifying the
escrow costs a pass; re-drafting and verifying fresh costs a pass. Same count.
So salvage only wins if the escrow is a *higher-quality draft* than a fresh one
from the same corrected prefix. Per rejection event:

- `m` — escrow length (tokens after the rejected one)
- `a` — mismatch index inside the block
- `L_survive` — escrow tokens accepted when attached at **offset 0**
- `L_bridge`, `bridge_k` — escrow tokens accepted when attached after a `k`-token
  bridge of the target's own continuation (`k ≤ 8`)
- `lcs_len` — longest contiguous run shared by escrow and realized continuation
- `L_fresh` — acceptance of the **fresh** draft from the identical corrected
  prefix, capped at `m` so both metrics share a ceiling

`delta = L_survive − L_fresh`, paired per event.

### Why `L_bridge` is the metric that actually decides the project

`suffix[0]` is the token immediately after the one that was just replaced — the
single position most contaminated by the correction. Measuring only offset-0
attachment asks the strictest possible question, and a null result there cannot
distinguish "rejections are semantic pivots" from "the plan survives but resumes
three tokens later." `L_bridge` separates them. Read it first.

A bridge policy is **not** implementable online (you would need the target's
continuation to build the bridge). Treat `L_bridge` as an upper bound on what a
repair model could buy — that is exactly its purpose.

### Why the instrumentation is free

Every committed token is the target's greedy choice, so **the committed tail
after an event is the target's greedy continuation from the corrected prefix.**
Survival and alignment are read off it directly: zero extra target passes.
`L_fresh` is likewise free (it is the next cycle's draft, since greedy drafting
is deterministic — "deferred pairing").

`--branch-verify` re-enables a dedicated branch pass per event purely as a
cross-check; `test_core.py::test_free_survival_matches_branch_pass` asserts the
two agree. Run it once on a smoke test, then never again.

The one thing the free path cannot resolve is **right-censoring**: if a prompt
ends fewer than `m` tokens after an event, the true survival is only known to be
`≥` what was observed. Those events are flagged `L_survive_censored` and the
analyzer reports the fraction.

**Why greedy everywhere:** any policy that only commits target-verified greedy
tokens produces bit-identical output to plain target decoding, so the
instrumentation cannot bias the trajectory.

---

## 2. The two implementation ideas worth understanding

**The cache invariant.** The KV cache always holds states for `committed[:-1]`,
and every verification forward feeds `[committed[-1]] + block`. Logits row `j`
predicts input position `j+1`, so row `j` scores `block[j]` for `j = 0..k−1`, and
row `k` is the bonus token. The bonus token never enters the cache when
committed — it just becomes `committed[-1]` and leads the next forward. After a
rejection at block index `a`, `cache.crop(len(committed) − 1)` handles both the
valid-prefix and garbage-suffix cases with one line. `cache_position` is passed
explicitly on every forward so nothing depends on HF inferring position from a
cache we crop behind its back.

**Alignment is where spec-decoding implementations die.** `logits[i]` predicts
token `i+1`. Compare **token IDs**, never decoded strings. `test_core.py` locks
this down with two mock targets: a position-only target (corrections can't change
the future ⇒ survival runs exactly to the next draft error, computed
independently) and a content-dependent target (a correction changes everything ⇒
survival must be 0). Both also assert the committed output equals a pure greedy
rollout.

---

## 3. Hardware and model pairs

Hard requirement: **draft and target must share an identical tokenizer** (the
script asserts this). Qwen3-0.6B and Qwen3-8B do.

| Hardware | Target | Notes |
|---|---|---|
| 24 GB GPU (3090/4090) | Qwen3-8B bf16 | Primary setup. Keep context ≤ 4K. |
| 16 GB GPU | Qwen3-8B `--load-in-8bit`, or Qwen3-4B bf16 | Quantizing redefines "the target"; the study stays internally consistent. |
| Colab free T4 (16 GB) | Qwen3-4B fp16 + 0.6B | T4 has no bf16; the script detects this. |
| Mac, 32 GB+ unified | Qwen3-8B on MPS | Works, several× slower. |

The draft now runs with a persistent KV cache instead of re-prefilling through
`model.generate` every cycle, which is most of the wall-clock cost on long runs.

---

## 4. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install "torch>=2.4" "transformers>=4.51,<5" datasets accelerate
```

---

## 5. Run order — do not reorder this

**Step 1 — logic tests (CPU, seconds):**

```bash
python test_core.py
```

**Step 2 — plumbing self-test (GPU, ~1 min).** Draft = target = Qwen3-0.6B. A
model drafting for itself must have ~zero rejections; more than a handful means
the torch wiring is broken, not the models. Read the printed text too — garbage
text with zero rejections is a chat-template bug.

```bash
python spec_trace.py --self-test --max-new 64 --out traces/self-test.jsonl
```

**Step 3 — smoke run with every cross-check enabled:**

```bash
python spec_trace.py --domain code --n 5 --gamma 16 --paranoid 5 --branch-verify --check-draft-cache 3 --out traces/smoke.jsonl
```

This validates three independent things and prints a verdict for each: the KV
cache against a cache-free recompute, the free survival measurement against a
branch pass, and the cached draft against `model.generate`. All three must
report `OK` before you trust a large run. Then open the JSONL and **read ten
events** — check that `rejected`/`correction` are plausible substitutions and
that `suffix_text`/`realized_text` look sensible.

**Step 4 — the real runs** (drop `--branch-verify`; it costs a pass per event and
step 3 already proved it redundant):

```bash
python spec_trace.py --domain code --n 200 --gamma 16 --paranoid 50 --out traces/code-g16.jsonl
```

Sweep γ, because γ=32 with a 0.6B draft puts most events in the pathological
corner — the draft diverges at block index 0 and the resulting 31-token escrow
was generated with zero verified grounding:

```bash
for g in 8 16 32; do
  python spec_trace.py --domain code --n 200 --gamma $g --out traces/code-g$g.jsonl
done
```

Then the other domains, and the reasoning ablation (needs a much larger
`--max-new`; long chains are where the roadmap predicts escrow value is highest):

```bash
python spec_trace.py --domain math --n 200 --gamma 16 --out traces/math.jsonl
python spec_trace.py --domain chat --n 200 --gamma 16 --out traces/chat.jsonl
python spec_trace.py --domain math --n 100 --gamma 16 --thinking --max-new 2048 --out traces/math-thinking.jsonl
```

Analyze one γ at a time — the analyzer warns if you mix them:

```bash
python analyze_traces.py traces/code-g16.jsonl
```

---

## 6. Reading the results

`analyze_traces.py` prints escrow geometry, offset-0 attachment, best-bridge
attachment, longest shared run, an event taxonomy
(whitespace / format / numeric / semantic), confidence buckets, and a
cost-corrected economics block.

**Economics.** Break-even gain per event is `g_min = cost_factor × baseline
tokens-per-pass`. With a sequential branch pass (`cost_factor = 1`) you must gain
a full baseline block per event, which is hopeless. Verifying the escrow and the
fresh draft as a **batch of 2 sharing one KV prefix** makes the second row nearly
free on memory-bound decode (`cost_factor ≈ 0.12`), dropping `g_min` by ~8×.
That is the only cost model under which recycle-if-better is worth implementing,
and it is why the original "same pass count, so it can't win" framing is too
pessimistic: you never have to choose between the two candidates, you verify both.

**Decision.**

- `L_bridge` high, `L_survive` low → the plan survives but resumes later. The
  bridge-length histogram is a repair model's addressable market. Proceed with
  bridge repair, not direct reattachment.
- `L_bridge ≈ L_survive`, both high, `delta ≈ 0` → escrows survive but the draft
  redrafts them anyway. Only draft-cost savings remain (FailFast reproduced).
- `L_bridge ≈ L_survive ≈ 0` → rejections are semantic pivots and no bridge model
  can help. Publish the measurement and stop.

---

## 7. Pitfalls checklist

- **Prompt-format divergence.** The instruct-tuned draft opens a ```` ```python ````
  fence and re-emits the docstring while the target goes straight to the body.
  In the first pilot this was 27% of all events, every one of them landing in the
  maximum-escrow bucket with zero survival. The code domain now commits the fence
  and the HumanEval signature+docstring as an assistant **prefill**, so neither
  can diverge. `--no-prefill` reproduces the artifact if you want to quantify it.
- **String vs token-ID comparison** — IDs only, always. (The analyzer's taxonomy
  uses decoded text, but only to *label* events, never to measure them.)
- **Off-by-one in logits** — covered by the invariant and tests; re-run
  `test_core.py` after any change to the loop.
- **EOS inside a block** — the loop truncates at EOS and skips event measurement
  on the final cycle. Don't "fix" this into measuring past EOS.
- **Censoring** — check the reported censored fraction before believing a low
  survival mean. Short generations (HumanEval solutions are 18–150 tokens)
  censor a lot of events.
- **bf16 nondeterminism** — changing batch/cache shapes can flip near-tie
  argmaxes. Isolated paranoid warnings are noise; a *rate* above ~2% is a bug.
- **Sampling leaking in** — everything must be `do_sample=False`.
- **Overshoot** — spec decoding commits up to γ+1 tokens past `--max-new`.

---

## 8. If the mechanism is real: the bridge study

`L_bridge` already gives you the oracle bound for free, using the target's own
continuation as the bridge. The next step is only worth taking if that bound is
large: enumerate candidate bridges from the top-k of the logits row at the
rejection position (already computed in `verify`), beam-search 1–4 tokens with
the target, append the escrow after each, and record the best survival. The gap
between that and `L_bridge` tells you how much of the oracle a *realizable*
bridge search captures; the gap between `L_bridge` and `L_survive` is the
addressable market itself.
