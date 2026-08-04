# Rejoin Speculative Decoding

Rejoin is a research implementation of suffix-preserving speculative decoding.
When a draft block first diverges from a target model, the decoder retains the
remaining draft tokens, inserts the target correction, and reverifies the
retained suffix under the repaired prefix.

The repository contains a dependency-free reference implementation for greedy
decoding and the Phase 0 study that measured whether the idea works.

## Findings: it does not pay

The crux experiment was run on Qwen3-4B + Qwen3-0.6B, γ=16, over 8578 rejection
events across two workloads (HumanEval code, and GSM8K with reasoning chains).

Escrowed suffixes **do** survive — their content reappears in the target's
continuation at 30–56× the chance rate. But they survive *misaligned*, and every
policy that can be implemented online falls below break-even:

| rung | gain per event | break-even | implementable |
|---|---|---|---|
| offset-0 (direct reattachment) | 0.06–0.07 | ~0.72 | yes |
| front-trim (batched candidate tree) | 0.13–0.22 | ~0.72 | yes |
| best-bridge | 0.34–0.36 | ~0.72 | no |
| two-sided | 1.55–1.69 | ~0.72 | no |

At zero verification cost — the ceiling regardless of engineering — the best
implementable policy is worth **+1.8% to +3.3%**. Selective application does not
help: of twenty online-observable gates none clears break-even, and a perfect
oracle gate caps at +2.5%. Reasoning chains, which the roadmap predicted would
be the best case, gave the same answer as code.

The structural reason is that the escrow is `P(continuation | prefix +
wrong_token)` while a fresh draft is `P(continuation | prefix + correct_token)`
— the same model, better conditioned.

Full numbers, method, and caveats: [studies/phase0/README.md](studies/phase0/README.md).

The reference implementation below remains a correct, tested demonstration of
exact suffix reattachment under greedy decoding; the measurement says it is not
worth putting on a GPU.

## Current scope

- Exact target-only greedy generation
- Exact ordinary speculative decoding
- Direct suffix reattachment with a configurable short-suffix fallback
- Structured per-step traces and JSONL collection
- Offline direct-rejoinability measurement
- Deterministic test doubles and output-equivalence tests

Sampling, anchor search, bridge generation, tree verification, and KV-state
reuse are intentionally deferred. See the
[design roadmap](rejoin_speculative_decoding_design_roadmap.md) for the full
research plan. [IMPLEMENTATION.md](IMPLEMENTATION.md) tracks the staged
engineering work and its decision gates.

## Quick start

The reference core supports Python 3.9+ and has no runtime dependencies.

```bash
python3 -m unittest discover -s tests -v
python3 -m examples.reference_demo
```

The central API accepts adapters implementing the small protocols in
`rejoin.models.protocols`:

```python
from rejoin import DecoderConfig, RejoinDecoder

decoder = RejoinDecoder(
    target=target_adapter,
    draft=draft_adapter,
    config=DecoderConfig(draft_block_size=16, min_suffix_length=4),
)

result = decoder.generate(prefix_tokens, max_new_tokens=128)
print(result.tokens)
```

Committed tokens always follow target-greedy behavior. Retained suffix tokens
are candidates only and are committed only after exact reverification.

## Layout

```text
rejoin/
├── api.py                  Public decoder facade
├── config.py               Validated runtime configuration
├── types.py                Immutable result and trace types
├── analysis/               Offline rejoinability measurements
├── metrics/                Exactness and salvage metrics
├── models/                 Model protocols and test/reference adapters
└── runtime/                Target, speculative, and rejoin decoding loops

studies/
└── phase0/                 GPU escrow survival + alignment study (schema 2)
```

The runtime remains inference-engine agnostic. A production backend should
implement batched `verify_greedy` directly so a draft block is teacher-forced in
one target invocation; the supplied base class is a correctness reference, not
a performance implementation.

## Development

Install optional tooling with:

```bash
python3 -m pip install -e '.[dev]'
pytest
ruff check .
```

The next engineering milestone is a Hugging Face or vLLM-backed adapter plus a
benchmark harness that records target invocations, wall time, device time, and
peak KV memory on a fixed target/draft model pair.
