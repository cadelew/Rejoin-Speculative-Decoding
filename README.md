# Rejoin Speculative Decoding

Rejoin is a research implementation of suffix-preserving speculative decoding.
When a draft block first diverges from a target model, the decoder retains the
remaining draft tokens, inserts the target correction, and reverifies the
retained suffix under the repaired prefix.

The repository currently contains a dependency-free reference implementation
for greedy decoding. Its purpose is to establish correctness, trace semantics,
and stable model interfaces before adding framework-specific GPU adapters.

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
