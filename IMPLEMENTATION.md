# Implementation Plan

This plan converts the research roadmap into small, testable engineering
milestones. A phase is complete only when its correctness gate passes; latency
work should not proceed on an unverified decoder.

## Milestone 1 — Reference core (implemented)

- [x] Define narrow target and draft model protocols.
- [x] Implement target-only greedy generation.
- [x] Implement ordinary greedy speculative decoding.
- [x] Implement exact direct suffix reattachment.
- [x] Add short-suffix fallback and EOS-safe commit handling.
- [x] Add immutable results, metrics, and versioned JSONL traces.
- [x] Add deterministic equivalence and edge-case tests.

Gate: all three runtimes produce the same target-greedy token sequence.

## Milestone 2 — Real model baseline

- [ ] Select a target/draft pair with a shared tokenizer.
- [ ] Add one framework adapter with batched teacher-forced verification.
- [ ] Make cache ownership and device placement explicit in the adapter.
- [ ] Add a prompt dataset format and reproducible configuration files.
- [ ] Benchmark target-only and ordinary speculation after warm-up.
- [ ] Record wall time, device time, target calls, committed tokens, and memory.

Gate: exact output equality on the fixed evaluation set and stable repeated
latency measurements. Initial recommendation: use Hugging Face Transformers for
research transparency, then add vLLM only after trace semantics are stable.

## Milestone 3 — Rejoinability study

Implemented in `studies/phase0/` (trace schema 2). See its README for the run
order and the decision table.

- [x] Collect mismatch traces without changing online decoding behavior.
- [x] Measure direct suffix survival after the target correction.
- [x] Search oracle anchor offsets offline (`L_bridge`, `bridge_k`, `lcs_len`).
- [x] Emit survival curves by workload, draft length, and mismatch category.
- [x] Estimate a break-even gain per event under explicit cost models.
- [ ] Run the study at scale across domains and draft-block lengths.

Gate: continue only if a meaningful workload has a nontrivial 8+ token survival
rate or a useful long tail of 16–64+ token survival.

The gate is evaluated on `L_bridge`, not on offset-0 `L_survive`. Offset-0
attachment puts the escrow's first token immediately after the token that was
just replaced — the position most contaminated by the correction — so a null
result there cannot distinguish a semantic pivot from a plan that resumes a few
tokens later. Anchor-offset search is therefore part of this milestone rather
than deferred to Milestone 5.

A pilot on Qwen3-4B + 0.6B (HumanEval, γ=32, 60 events) failed the gate on
offset-0 survival: 1.7% of events reached 8 tokens, mean paired delta −8.05,
zero wins. That pilot predates the prompt-format fix and has no anchor
measurement, so it is not yet a fair test of the gate.

## Milestone 4 — Online Rejoin-Exact

- [ ] Connect direct reattachment to the batched model adapter.
- [ ] Add latency-based activation instead of a fixed suffix threshold.
- [ ] Add conservative avalanche rules for numbers, negation, code identifiers,
      structured parser-state changes, and EOS.
- [ ] Compare output, latency, throughput, and memory against both baselines.

Gate: 100% greedy equality and positive end-to-end speedup on at least one
workload, with bounded memory.

## Milestone 5 — Anchors and bridge repair

- [ ] Add tokenization-safe anchor candidates and guard spans.
- [ ] Compare heuristic anchors with the offline oracle.
- [ ] Add utility ranking and strict branch budgets.
- [ ] Introduce bridge generation behind a separate repair-model protocol.
- [ ] Add tree-batched verification only after sequential candidates show value.

Gate: additional salvage must exceed repair, verification, and memory overhead.

## Engineering rules

- A token is committed only after exact target verification.
- No stale target KV state is reused after a changed token.
- Model-specific code stays in adapters; scheduling stays in `rejoin.runtime`.
- Trace schemas are versioned and append-only within a schema version.
- Benchmarks always report warm-up policy, model revisions, hardware, seeds, and
  complete latency scope.
- Generated datasets, traces, and model artifacts live outside source packages.
