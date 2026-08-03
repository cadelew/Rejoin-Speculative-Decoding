# Rejoin Speculative Decoding
## Design Choices, Mathematical Formulation, Edge Cases, and Implementation Roadmap

**Working title:** Rejoin: Suffix-Preserving Speculative Decoding Through Local Repair  
**Project status:** Research hypothesis / pre-implementation  
**Primary objective:** Determine whether speculative decoding errors are often local enough that later draft tokens can be salvaged after repairing an earlier mismatch.  
**Recommended first deployment:** Standalone inference backend evaluated in offline and shadow mode on general benchmarks and LawOS workloads.  
**Non-goal for the first version:** Reusing stale target-model KV states after changing an earlier token.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Core Hypothesis](#3-core-hypothesis)
4. [Terminology](#4-terminology)
5. [Baseline: Ordinary Speculative Decoding](#5-baseline-ordinary-speculative-decoding)
6. [Proposed Method: Repair and Rejoin](#6-proposed-method-repair-and-rejoin)
7. [Exactness and the Previous-Token Problem](#7-exactness-and-the-previous-token-problem)
8. [Mathematical Model](#8-mathematical-model)
9. [Representations and Data Structures](#9-representations-and-data-structures)
10. [Design Choice Matrix](#10-design-choice-matrix)
11. [Recommended Initial Architecture](#11-recommended-initial-architecture)
12. [Algorithms](#12-algorithms)
13. [Multiple Error Islands](#13-multiple-error-islands)
14. [Anchor Selection](#14-anchor-selection)
15. [Repair Models](#15-repair-models)
16. [Causal Stability and Rejoin Certificates](#16-causal-stability-and-rejoin-certificates)
17. [Model Count and Training Requirements](#17-model-count-and-training-requirements)
18. [Transformer Architecture Changes](#18-transformer-architecture-changes)
19. [Edge Cases and Failure Modes](#19-edge-cases-and-failure-modes)
20. [Evaluation Plan](#20-evaluation-plan)
21. [LawOS Integration Strategy](#21-lawos-integration-strategy)
22. [Implementation Roadmap](#22-implementation-roadmap)
23. [Repository Structure](#23-repository-structure)
24. [Decision Gates](#24-decision-gates)
25. [Advanced Research Directions](#25-advanced-research-directions)
26. [Recommended Direction](#26-recommended-direction)
27. [Open Questions](#27-open-questions)
28. [Appendix: Pseudocode and Schemas](#28-appendix-pseudocode-and-schemas)

---

# 1. Executive Summary

Ordinary speculative decoding uses a small draft model to propose a sequence of future tokens and a larger target model to verify those tokens in parallel. The target accepts the longest valid prefix. Once one draft token fails, every later draft token is discarded—even when many of those later tokens would still match the target after the local error is corrected.

**Rejoin speculative decoding** asks whether that discarded suffix can be retained as a candidate, connected to a corrected prefix through a short repair span, and reverified in one target pass.

Example:

```text
Prompt:
Why did the chicken

Draft:
cross → the → farm → ?

Target:
cross → the → road → ?
```

Ordinary speculative decoding accepts `cross the`, rejects `farm`, emits or samples `road`, and begins a new draft round. Rejoin instead constructs:

```text
cross → the → road → ?
```

and immediately reverifies the retained `?` under the corrected prefix.

The same concept becomes more valuable with a long draft:

```text
Draft:
[A B C] [wrong] [50-token suffix]

Target:
[A B C] [repair] [mostly the same 50-token suffix]
```

The core opportunity is not that the old suffix was already correct. It was scored under the wrong prefix and therefore must be reverified. The opportunity is that the old suffix may still be a high-quality proposal, avoiding serial redrafting and potentially committing many more tokens per target invocation.

The recommended first project is deliberately conservative:

- Use an ordinary autoregressive Transformer.
- Use greedy decoding.
- Preserve exact target output.
- Repair only the earliest unresolved mismatch.
- Reuse suffix token IDs as candidates.
- Recompute target states after the divergence point.
- Do not reuse stale KV cache.
- Do not begin with FPGA hardware.
- Do not begin with company-specific specialization.
- Use LawOS as one benchmark distribution, not as the definition of the method.

The first question to answer is empirical:

> After replacing the first wrong draft token with the target token, how often and how quickly does the original draft realign with the target continuation?

If rejoinability is low, the research direction should be stopped or narrowed to structured generation. If rejoinability is high, the project can progress from candidate reuse to learned anchor selection, local infilling, tree-batched verification, causal-state fingerprints, approximate KV repair, and eventually rejoin-native model architectures.

---

# 2. Problem Statement

Let an autoregressive target model define:

\[
p_T(x_{1:n}) = \prod_{i=1}^{n} p_T(x_i \mid x_{<i})
\]

and let a cheaper draft model define:

\[
p_D(x_{1:n}) = \prod_{i=1}^{n} p_D(x_i \mid x_{<i})
\]

At a decoding step, the draft model proposes:

\[
d_{1:k}
\]

conditioned on an already accepted prefix \(c\).

The target scores the proposed sequence. Under greedy decoding, the target accepts the longest prefix such that:

\[
d_i = \arg\max_x p_T(x \mid c, d_{<i})
\]

for every accepted position \(i\).

If the first mismatch occurs at index \(j\), conventional speculative decoding discards:

\[
d_{j+1:k}
\]

even though some or all of those suffix tokens may match the target continuation after \(d_j\) is replaced with the target correction.

The research problem is:

> Can the decoder preserve later draft tokens as a provisional suffix, repair the earliest divergence, and efficiently verify bridge-and-suffix candidates without violating target correctness?

---

# 3. Core Hypothesis

## 3.1 Primary hypothesis

For a meaningful subset of speculative decoding errors, the effect of an incorrect token is **local** rather than trajectory-wide.

Formally, let:

- \(d_j\) be the first rejected draft token.
- \(y_j\) be the target correction.
- \(d_{j+1:k}\) be the original suffix.
- \(y_{j+1:\infty}\) be the target continuation after the corrected prefix.

Define the direct suffix-survival length:

\[
L =
\max \left\{
\ell \ge 0 :
d_{j+r} = y_{j+r}
\text{ for all } 1 \le r \le \ell
\right\}
\]

The method is useful when the distribution of \(L\) has a meaningful tail, especially when:

- \(L\) is often larger than the expected next draft length.
- The repair span is short.
- The target can verify the repaired path efficiently.
- The overhead of anchor search and tree construction is small.

## 3.2 Stronger hypothesis

Even when direct reattachment fails, the original draft may realign after a short changed region.

Let \(a > j\) be an anchor in the old draft and let \(b_{1:m}\) be a repair bridge. A valid rejoin candidate is:

\[
c, d_{1:j-1}, b_{1:m}, d_{a:k}
\]

The stronger hypothesis is that many divergences have:

- A short repair bridge \(m\).
- A relatively early anchor \(a\).
- A long reusable suffix \(k-a+1\).

## 3.3 Long-term hypothesis

A model may be post-trained or architected so that local lexical errors have bounded future influence, allowing actual computational state rejoining rather than only candidate token reuse.

This is not required for the first validation.

---

# 4. Terminology

| Term | Meaning |
|---|---|
| **Accepted prefix** | Tokens already verified and committed by the target model. |
| **Draft block** | A sequence proposed by the draft model. |
| **First rejection** | Earliest draft token not accepted by the target. |
| **Target correction** | The token selected by the target at the first rejection point. |
| **Tentative suffix** | Draft tokens after the first rejection, preserved as candidates but not considered correct. |
| **Repair island** | A contiguous region that differs from the retained draft. |
| **Bridge** | Tokens generated to connect the corrected prefix to a retained suffix anchor. |
| **Anchor** | A point in the old suffix where a repaired path attempts to rejoin. |
| **Rejoin** | Reusing a retained suffix after a repair and reverifying it under the corrected prefix. |
| **Causal horizon** | Estimated distance over which a correction materially changes future tokens. |
| **Suffix survival length** | Number of retained suffix tokens accepted after repair. |
| **Rejoin certificate** | Scheduling evidence that a candidate suffix is worth further verification. It does not replace exact target verification. |
| **Textual rejoin** | Two paths contain the same future token string. |
| **Computational rejoin** | Two paths share equivalent target-model hidden or KV state. Much harder. |
| **Rejoin-Exact** | Every committed token is verified against the exact target behavior. |
| **Rejoin-Approx** | Allows semantic equivalence, approximate hidden-state reuse, or task-level rather than distribution-level equality. |

---

# 5. Baseline: Ordinary Speculative Decoding

## 5.1 Greedy baseline

Given accepted prefix \(c\), the draft model proposes:

\[
d_{1:k}
\]

The target evaluates all proposed positions in one forward pass. Let the target’s greedy token at draft position \(i\) be:

\[
t_i = \arg\max_x p_T(x \mid c, d_{<i})
\]

The accepted length is:

\[
A = \max \left\{
a :
d_i=t_i \quad \forall i \le a
\right\}
\]

If \(A < k\), the target correction is \(t_{A+1}\), and the suffix \(d_{A+2:k}\) is discarded.

## 5.2 Cost model

Let:

- \(T_D(k)\): time for the draft model to produce \(k\) tokens.
- \(T_T(k)\): time for the target to verify \(k\) proposed tokens.
- \(A\): accepted draft length.

A rough latency per committed token is:

\[
C_{\text{spec}}
=
\frac{T_D(k)+T_T(k)}
{\mathbb{E}[A]+1}
\]

where the extra \(1\) represents the target correction when rejection occurs.

This formulation hides batching, KV caching, and pipeline overlap, but it is useful for comparing methods.

## 5.3 Waste after rejection

After a rejection, speculative decoding wastes:

1. **Candidate-generation work** spent creating the suffix.
2. **Information** about likely future tokens.
3. **Potential extra decoding rounds** needed to reproduce the same suffix.
4. **Draft hidden states** that may contain useful continuation structure.
5. **Target logits** under the old path, which cannot prove correctness under the new path but may still provide predictive signals.

Rejoin attempts to recover value from items 1–4 while respecting the invalidation of item 5.

---

# 6. Proposed Method: Repair and Rejoin

## 6.1 Direct reattachment

The simplest method requires no repair model.

Suppose the draft is:

\[
d_{1:k}
\]

and the first mismatch is at \(j\). Let the target correction be \(y_j\).

Construct:

\[
r =
y_j, d_{j+1:k}
\]

and ask the target to verify \(r\) under:

\[
c, d_{1:j-1}
\]

If a long prefix of \(r\) matches, commit it.

## 6.2 Anchor skipping

Direct reattachment may fail because the correction affects several immediate tokens. Instead, choose a later anchor \(a\):

\[
r_a =
y_j, d_{a:k}
\]

This sacrifices tokens \(d_{j+1:a-1}\) but may recover a more stable suffix.

## 6.3 Bridge-and-rejoin

Generate a repair bridge \(b_{1:m}\) conditioned on:

- Left context: \(c,d_{1:j-1}\)
- Target correction: \(y_j\)
- Right context: \(d_{a:k}\)

Construct:

\[
r_{a,b} =
b_{1:m}, d_{a:k}
\]

and target-verify it.

## 6.4 Tree-batched candidate verification

Generate multiple bridge-anchor candidates:

\[
r_1, r_2, \dots, r_B
\]

Share the common accepted prefix and evaluate branches using a tree-compatible attention mask. Commit the longest target-valid branch.

## 6.5 Why this is a lattice conceptually

A tree represents branching but not reconnection. A lattice or DAG can represent:

```text
accepted prefix
    ├── repair A ─┐
    ├── repair B ─┼── retained suffix token buffer
    └── repair C ─┘
```

However, target-model states remain branch-specific:

```text
repair A → suffix state A
repair B → suffix state B
repair C → suffix state C
```

The suffix token IDs may be shared in storage, but hidden states and KV pages cannot be merged merely because the visible text becomes identical.

---

# 7. Exactness and the Previous-Token Problem

## 7.1 The central objection

Autoregressive probability depends on the complete prefix:

\[
p_T(x_i \mid x_{<i})
\]

Changing one earlier token changes the conditioning context for every later token.

Therefore:

\[
p_T(? \mid \text{cross the farm})
\neq
p_T(? \mid \text{cross the road})
\]

in general.

The old suffix is not proven correct after the repair.

## 7.2 Safe invariant for Rejoin-Exact

For greedy decoding:

> Every committed token must equal the token ordinary target-model greedy decoding would select under the full corrected prefix.

A candidate path:

\[
z_{1:m}
\]

is committed only up to the largest \(q\) such that:

\[
z_i =
\arg\max_x p_T(x \mid c,z_{<i})
\quad \forall i\le q
\]

This preserves exact greedy output.

## 7.3 What can be reused safely

| Reuse type | Safe with ordinary Transformer? | Requires revalidation? |
|---|---:|---:|
| Suffix text | Yes | Yes |
| Suffix token IDs | Yes, after splice retokenization | Yes |
| Candidate topology | Yes | Yes |
| Prefix KV before divergence | Yes | No |
| Branch KV after divergence | No cross-branch reuse | Must recompute |
| Old suffix target logits | As features only | Cannot certify correctness |
| Old suffix target KV | No | Must recompute or approximate |
| Parser state | Only if repaired path reaches same state | Must check |
| Semantic interpretation | Approximate | Requires separate policy |

## 7.4 Sampling

Exact sampling is harder than greedy decoding. After the prefix changes, the old draft probabilities are invalid:

\[
q_{\text{old}}(d_i)
\neq
q_{\text{repaired}}(d_i)
\]

An exact sampling extension would require teacher-forced rescoring of the retained suffix under the repaired prefix, followed by valid speculative acceptance corrections.

Recommendation:

> Start with temperature-zero greedy decoding. Add exact sampling only after the candidate-reuse mechanism produces a meaningful end-to-end win.

---

# 8. Mathematical Model

## 8.1 Suffix salvage ratio

For a candidate with \(R\) tokens after the first rejection and \(S\) reused tokens ultimately accepted:

\[
\mathrm{SSR}
=
\frac{S}{R}
\]

## 8.2 Correct-token discard rate

Let \(G\) be the ordinary target greedy continuation after correction. Let \(d_{j+1:k}\) be the discarded draft suffix. Count draft tokens that match the corrected target continuation at aligned positions:

\[
M =
\sum_{r=1}^{k-j}
\mathbf{1}
\left[
d_{j+r}=G_r
\right]
\]

Then:

\[
\mathrm{CTDR}
=
\frac{M}{k-j}
\]

This measures how much potentially useful candidate work standard speculative decoding throws away.

## 8.3 Rejoin utility

For anchor \(a\) and bridge \(b\), define:

- \(P_{\text{success}}(a,b)\): probability that the candidate survives long enough to be useful.
- \(L(a,b)\): expected accepted tokens.
- \(C_D(a,b)\): draft or repair cost.
- \(C_T(a,b)\): target verification cost.
- \(C_M(a,b)\): memory and tree-construction cost.

A simple utility is:

\[
U(a,b)
=
P_{\text{success}}(a,b)L(a,b)
-
\lambda_D C_D(a,b)
-
\lambda_T C_T(a,b)
-
\lambda_M C_M(a,b)
\]

Attempt rejoin only when:

\[
U(a,b) > U_{\text{fallback}}
\]

## 8.4 Break-even condition

Let:

- \(T_R\): repair and anchor-selection overhead.
- \(T_V\): additional target verification overhead.
- \(T_{\text{saved}}\): serial drafting and future verification time avoided.

Rejoin is profitable when:

\[
T_R + T_V < T_{\text{saved}}
\]

A practical estimator:

\[
T_{\text{saved}}
\approx
L_{\text{survived}}\cdot t_D
+
N_{\text{rounds avoided}}\cdot T_{\text{round}}
\]

where \(t_D\) is average draft time per token.

## 8.5 Causal influence

For correction at position \(j\) and later position \(i\), define:

\[
M_{j,i}
=
P\left(
x_i^{\text{old}} \ne x_i^{\text{repaired}}
\mid
\text{correction at } j
\right)
\]

The predicted causal horizon can be defined as the smallest \(h\) such that:

\[
M_{j,i}<\epsilon
\quad
\forall i \ge j+h
\]

This is a statistical rather than absolute causal claim.

## 8.6 Distribution shift

Compare old-path and repaired-path target distributions at a suffix position:

\[
\Delta_i =
D_{\mathrm{JS}}
\left(
p_T^{\text{old}}(\cdot \mid i),
p_T^{\text{repaired}}(\cdot \mid i)
\right)
\]

Signals of stability include:

- Low \(\Delta_i\)
- Same top-1 token
- High rank of retained token
- Low entropy
- Consecutive retained-token matches

These signals schedule verification. They do not bypass exact verification.

## 8.7 Future-state fingerprint

A learned projection:

\[
z_i = F(h_i)
\]

should encode future behavior, not just the next token.

One possible training distance:

\[
D_{\text{future}}(u,v)
=
\sum_{h \in \{1,2,4,8,16\}}
w_h
D_{\mathrm{JS}}
\left(
p_T^{(h)}(\cdot \mid u),
p_T^{(h)}(\cdot \mid v)
\right)
\]

The runtime approximates \(D_{\text{future}}\) using \(z\)-space distance.

## 8.8 Branch explosion

If there are \(n\) islands and each has \(b\) candidate repairs:

\[
N_{\text{paths}}=b^n
\]

This makes simultaneous independent island repair impractical without:

- Earliest-island scheduling
- Best-first search
- Strict branch budgets
- Path deduplication
- Dynamic fallback

---

# 9. Representations and Data Structures

A graph drawing is only a visualization. The actual system should combine multiple representations.

## 9.1 Sequence lattice

Represents candidate token paths:

```text
accepted prefix
    ├── correction + old suffix
    ├── bridge A + anchor 1 suffix
    ├── bridge B + anchor 1 suffix
    └── bridge C + anchor 2 suffix
```

Answers:

- Which strings are candidates?
- Which prefixes are shared?
- Where do branches diverge?
- Which suffix token buffers can be referenced?

## 9.2 Interval DAG

Token-level graphs become large and noisy. Use spans:

```text
[accepted prefix]
    ↓
[repair interval]
    ↓
[tentative suffix interval]
```

Each span edge contains a token sequence, source, score, and verification metadata.

## 9.3 Causal hypergraph

A single repair can affect multiple later regions. A hyperedge connects one source change to several dependent intervals.

```text
repair: "approved" → "rejected"
    ├── affects conclusion
    ├── affects justification
    ├── affects tool action
    └── affects final summary
```

Use the hypergraph to predict avalanches and provisional downstream islands.

## 9.4 Factor graph

A candidate path must satisfy multiple constraints:

- Tokenization
- Target probability
- Parser state
- Grammar
- Type consistency
- Source quotation fidelity
- Numeric consistency
- Safety and side-effect constraints

Represent these as factors, not one confidence scalar.

## 9.5 High-dimensional state manifold

Each checkpoint carries a future-behavior vector:

\[
z_i \in \mathbb{R}^d
\]

Nearby vectors indicate predicted future similarity.

Use this for:

- Anchor ranking
- Branch pruning
- Guard-window sizing
- Avalanche prediction
- Future research on approximate state merging

Do not initially use it to skip target verification.

## 9.6 Causal influence tensor

A richer representation:

\[
M[
\text{repair hypothesis},
\text{layer},
\text{source position},
\text{future position}
]
\]

A dense tensor is too expensive. Approximate with:

- Low-rank factors
- Sparse intervals
- Per-island embeddings
- Learned causal-cone summaries

## 9.7 Recommended representation stack

For the first implementation:

1. **Interval candidate tree** for actual verification.
2. **Suffix token buffer references** for memory efficiency.
3. **Checkpoint metadata** for parser state and scores.
4. **Optional causal-score vector** for anchor ranking.

Do not implement the full hyperlattice before validating suffix survival.

---

# 10. Design Choice Matrix

## 10.1 Verification mode

| Option | Correctness | Complexity | Potential speed | Recommendation |
|---|---:|---:|---:|---|
| Greedy exact | Exact target greedy output | Low | Medium | Start here |
| Exact sampling | Exact target distribution | High | Medium | Later |
| Semantic equivalence | Task-level correctness | Medium-high | High | Separate Rejoin-Approx track |
| No re-verification | Unsafe | Low | Very high on paper | Reject |

## 10.2 Repair strategy

| Strategy | Extra model? | Cost | Capability | Recommendation |
|---|---:|---:|---|---|
| Target correction + direct suffix | No | Lowest | Repairs one token | First baseline |
| Target correction + anchor skip | No | Low | Handles short unstable region | Early |
| AR bridge model | Usually yes | Medium | Flexible left-to-right repair | Compare |
| Fill-in-the-middle model | Yes or dual-mode draft | Medium | Uses left and right anchors | Preferred learned repair |
| Block diffusion infill | Yes | Higher | Repairs multiple positions in parallel | Later |
| Full target regeneration | No extra | High | Reliable fallback | Always available |

## 10.3 Candidate topology

| Structure | Strength | Weakness | Use |
|---|---|---|---|
| Chain | Simple | No alternatives | Baseline |
| Tree | Parallel repair branches | No conceptual merging | Verification runtime |
| DAG/lattice | Represents shared suffixes | Hidden states still branch-specific | Candidate planning/storage |
| Interval DAG | Compact | Less token-level detail | Recommended |
| Hypergraph | Models nonlocal dependencies | Complex | Controller research |
| Factor graph | Multiple constraints | Inference overhead | Structured tasks |
| Vector manifold | Similarity and ranking | Approximate | Learned controller |
| Influence tensor | Rich causal model | Expensive | Offline analysis / compressed runtime |

## 10.4 State reuse

| Level | What is reused | Architecture change? | Exact? | Recommendation |
|---|---|---:|---:|---|
| 0 | Nothing | No | Yes | Baseline |
| 1 | Suffix token IDs | No | Yes after reverify | Start |
| 2 | Candidate tree and prefix KV | No | Yes | Early optimization |
| 3 | Draft hidden states | No | Depends | Research |
| 4 | Approximate target KV via editor | Small module | No | Advanced |
| 5 | Canonical checkpoint state | Post-training/architecture | Possibly | Long-term |
| 6 | True hidden-state merging | Likely architecture change | Hard | Moonshot |

## 10.5 Number of models

| Configuration | Full models | Small modules | Use |
|---|---:|---:|---|
| Minimal | Target + draft | None | Rejoinability and direct reattach |
| Learned controller | Target + draft | Shared prediction heads | Anchor and survival prediction |
| Repair system | Target + dual-mode draft/repair | Shared heads | Main practical design |
| KV-edit system | Target + draft | KV editor | Advanced |
| Single-backbone native | One target | Multi-token, repair, checkpoint heads | Long-term architecture |

---

# 11. Recommended Initial Architecture

```text
┌───────────────────────────────────────────────────────┐
│ Inference API                                         │
│ generate(prompt, max_tokens, temperature=0)           │
└──────────────────────────┬────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────┐
│ Rejoin Controller                                     │
│                                                       │
│ Baseline speculative loop                             │
│ First-mismatch detection                              │
│ Suffix retention                                      │
│ Anchor policy                                         │
│ Branch budget                                         │
│ Fallback decision                                     │
└─────────────┬──────────────────────────────┬──────────┘
              │                              │
              ▼                              ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│ Small draft model       │       │ Large target model      │
│                         │       │                         │
│ AR draft generation     │       │ Exact verification      │
│ Optional FIM repair     │       │ Target correction       │
│ Optional controller     │       │ Tree verification       │
│ heads                   │       │                         │
└─────────────────────────┘       └─────────────────────────┘
```

## Initial constraints

- Greedy decoding only
- Same tokenizer for target and draft
- One unresolved island at a time
- Target correction is always the first repair token
- At most three anchors
- At most eight candidate branches
- No stale target KV reuse
- No side-effect execution before commit
- Strict target-only fallback
- Exact output equality with ordinary target greedy decoding

---

# 12. Algorithms

## 12.1 Offline rejoinability study

```python
def measure_rejoinability(target, draft, prompt, draft_length):
    prefix = tokenize(prompt)

    proposed = draft.generate_greedy(prefix, draft_length)
    target_result = target.verify_greedy(prefix, proposed)

    j = target_result.first_mismatch
    if j is None:
        return {"fully_accepted": True}

    correction = target_result.target_token_at(j)
    old_suffix = proposed[j + 1:]

    repaired_prefix = prefix + proposed[:j] + [correction]

    # Teacher-force old suffix under repaired prefix.
    rescored = target.verify_greedy(repaired_prefix, old_suffix)

    direct_survival = rescored.greedy_match_length

    return {
        "fully_accepted": False,
        "mismatch_index": j,
        "correction": correction,
        "suffix_length": len(old_suffix),
        "direct_survival": direct_survival,
    }
```

## 12.2 Rejoin-Exact direct reattachment

```python
def rejoin_exact_direct(target, prefix, draft_tokens):
    first = target.verify_greedy(prefix, draft_tokens)

    j = first.first_mismatch
    if j is None:
        return draft_tokens

    accepted = draft_tokens[:j]
    correction = first.target_token_at(j)
    suffix = draft_tokens[j + 1:]

    if len(suffix) < MIN_SUFFIX_LENGTH:
        return accepted + [correction]

    repaired_candidate = [correction] + suffix
    second = target.verify_greedy(prefix + accepted, repaired_candidate)

    commit_len = second.greedy_match_length

    if commit_len == 0:
        # The target correction itself should normally match,
        # but retain a defensive fallback.
        return accepted + [correction]

    return accepted + repaired_candidate[:commit_len]
```

## 12.3 Anchor skipping

```python
def build_anchor_candidates(correction, suffix, offsets):
    candidates = []

    for offset in offsets:
        if offset >= len(suffix):
            continue

        candidates.append({
            "anchor_offset": offset,
            "tokens": [correction] + suffix[offset:],
        })

    return candidates
```

This naive form omits the missing region. It is useful only if the correction token can connect grammatically to the farther suffix. The bridge version is more general.

## 12.4 Bridge-and-rejoin

```python
def propose_rejoin_paths(
    repair_model,
    accepted_prefix,
    correction,
    suffix,
    anchors,
    max_bridges_per_anchor,
):
    paths = []

    for anchor in anchors:
        right_context = suffix[anchor:]

        bridges = repair_model.fill_middle(
            left_context=accepted_prefix + [correction],
            right_context=right_context,
            max_candidates=max_bridges_per_anchor,
        )

        for bridge in bridges:
            paths.append({
                "anchor": anchor,
                "bridge": [correction] + bridge,
                "suffix": right_context,
                "tokens": [correction] + bridge + right_context,
            })

    return paths
```

## 12.5 Best-first utility controller

```python
def expected_utility(candidate, estimates, costs):
    return (
        estimates.success_probability * estimates.accepted_tokens
        - costs.repair_latency
        - costs.verification_latency
        - costs.memory_penalty
    )
```

Only verify candidates with estimated positive value.

---

# 13. Multiple Error Islands

## 13.1 Why initial island labels are unreliable

Suppose the original target pass appears to show:

```text
A B [X] D E [Y] G H
```

The second mismatch \(Y\) was scored under a prefix containing \(X\). Once \(X\) is corrected, \(Y\) may:

- Become correct
- Move to another position
- Turn into a different mismatch
- Cause a larger changed span
- Disappear

Therefore, later islands are provisional.

## 13.2 Earliest-unresolved-island scheduling

Recommended policy:

1. Find first rejection.
2. Mark all later tokens tentative.
3. Repair first rejection.
4. Reverify a bounded candidate.
5. Resegment based on the repaired path.
6. Process the new earliest rejection.

This preserves causality and avoids false downstream diagnoses.

## 13.3 Progressive multi-island mode

Later, the system may hold:

```text
[verified prefix]
[repair island 1]
[tentative stable span]
[provisional island 2]
[tentative suffix]
```

But it should not commit a later span before every earlier token on the selected path is verified.

## 13.4 Branch control

Use:

- Maximum active paths
- Maximum repair depth
- Maximum anchor count
- Maximum suffix length
- Best-first search
- Prefix-hash deduplication
- Immediate pruning after first candidate mismatch
- Fallback when expected utility is negative

---

# 14. Anchor Selection

## 14.1 Weak anchors

- A single punctuation token
- `the`
- Whitespace
- Common newline
- Repeated delimiter
- Common suffix token

## 14.2 Strong anchors

- Unique multi-token phrase
- Function/class boundary
- JSON key plus delimiter
- Markdown heading
- Exact quotation span
- Source-copy sequence
- Distinctive identifier
- Sentence boundary plus several following tokens
- Parser-state-compatible span
- Low-entropy suffix region

## 14.3 Anchor features

For candidate anchor \(a\), compute features such as:

- Distance from rejection
- Remaining suffix length
- N-gram uniqueness
- Original draft confidence
- Old target probability
- Token entropy
- Parser state
- Tokenization boundary validity
- Whether suffix is copied from prompt
- Whether suffix matches a known template
- Predicted causal influence
- Expected bridge length
- Estimated verification cost

## 14.4 Anchor scoring

\[
V(a)
=
\hat P_{\text{survive}}(a)
\cdot
\hat L_{\text{accepted}}(a)
-
\lambda_1 C_{\text{bridge}}(a)
-
\lambda_2 C_{\text{verify}}(a)
-
\lambda_3 C_{\text{memory}}(a)
\]

Rank anchors by \(V(a)\).

## 14.5 Guard spans

Do not define an anchor as one token. Prefer an anchor span:

\[
a_{1:g}
\]

where \(g\) may be 4–16 tokens depending on uniqueness and structure.

A longer anchor is more convincing but harder to reach.

---

# 15. Repair Models

## 15.1 No repair model

Use only the target correction and direct suffix reattachment.

Advantages:

- No training
- Lowest overhead
- Clean baseline
- Exact

Weakness:

- Cannot bridge multi-token divergence

## 15.2 Autoregressive bridge model

Generate left-to-right from the corrected prefix while rewarding reconnection to an anchor.

Weakness:

- Does not naturally use the right boundary
- May wander away from the retained suffix

## 15.3 Fill-in-the-middle model

Input:

```text
LEFT CONTEXT <HOLE> RIGHT ANCHOR
```

Output:

```text
BRIDGE
```

This directly solves the rejoin problem.

Recommended learned repair model.

## 15.4 Block-diffusion infill

Mask the repair island:

```text
accepted prefix [MASK × m] retained suffix
```

Predict several missing positions in parallel.

Potential advantage:

- Parallel local repair
- Uses both boundaries
- Natural for multiple short islands

Potential disadvantage:

- Additional model and refinement steps
- May cost more than redrafting
- Harder to preserve exact target sampling

## 15.5 Dual-mode small model

The preferred practical design is one small backbone with mode tokens:

```text
<AR_DRAFT>
<FIM_REPAIR>
<PREDICT_SURVIVAL>
<PREDICT_ANCHOR>
```

This avoids loading several full models.

---

# 16. Causal Stability and Rejoin Certificates

## 16.1 Rejoin certificate purpose

A certificate does not prove correctness. It predicts whether more verification is worth the cost.

Possible fields:

```python
@dataclass
class RejoinCertificate:
    anchor_offset: int
    anchor_length: int
    anchor_uniqueness: float

    direct_guard_matches: int
    guard_match_rate: float

    mean_target_probability: float
    mean_entropy: float
    distribution_shift: float

    parser_state_match: bool
    tokenizer_boundary_valid: bool
    copied_from_context: bool

    predicted_survival_length: float
    avalanche_probability: float
    expected_utility: float
```

## 16.2 Geometric guard verification

Verify progressively larger candidate windows:

```text
4 tokens
8 tokens
16 tokens
32 tokens
full suffix
```

Continue only when:

- Match rate is high
- Distribution shift is low
- Expected salvage remains positive
- No parser or tokenization failure occurs

## 16.3 Causal distance vector

Instead of one scalar distance:

\[
\mathbf d(u,v)=
\begin{bmatrix}
d_{\text{position}}\\
d_{\text{lexical}}\\
d_{\text{logit}}\\
d_{\text{latent}}\\
d_{\text{parser}}\\
d_{\text{semantic}}\\
d_{\text{causal}}\\
d_{\text{compute}}
\end{bmatrix}
\]

A controller maps this vector to an action:

```text
DIRECT_REJOIN
TRY_FARTHER_ANCHOR
GENERATE_BRIDGE
VERIFY_SHORT_WINDOW
VERIFY_FULL_SUFFIX
FALLBACK
```

## 16.4 Avalanche detector

High-risk correction categories:

- Negation changes
- Answer polarity changes
- Main entity changes
- Core verb changes
- Numeric result changes
- Code control-flow changes
- Tool-call changes
- EOS becomes likely
- Open/close delimiter changes
- The correction changes sentence intent

If avalanche risk is high, do not attempt suffix salvage.

---

# 17. Model Count and Training Requirements

## 17.1 Stage A: Validation

**Models:** 2

- Large target model
- Small draft model

**Training:** None

**Capabilities:**

- Baseline speculation
- Trace collection
- First-mismatch analysis
- Direct suffix reattachment
- Exact greedy evaluation

## 17.2 Stage B: Learned controller

**Full models:** 2  
**Small neural component:** 1 shared controller or several heads

Heads:

- Attempt-rejoin classifier
- Expected suffix survival
- Anchor scorer
- Causal horizon predictor
- Avalanche detector
- Branch utility predictor

Training data is generated automatically from target/draft traces.

## 17.3 Stage C: Dual-mode repair model

**Full models:** 2

- Target
- Small draft/repair model

The small model supports:

- AR drafting
- FIM bridge generation
- Optional multi-token prediction
- Shared controller heads

Training:

- Fine-tune the small model
- Target remains frozen
- Use target acceptance length as an objective

## 17.4 Stage D: KV repair

**Full models:** 2  
**Additional module:** KV editor

Possible forms:

- Low-rank layerwise adapter
- Small cross-attention network
- Recurrent state-correction network
- Hypernetwork emitting KV deltas

This is approximate and should be a separate research track.

## 17.5 Stage E: Rejoin-native model

Potentially one full backbone with:

- Standard LM head
- Multi-token draft heads
- Repair head
- Checkpoint head
- State-equivalence head
- Causal-impact head

This requires substantial post-training and should not precede candidate-level validation.

## 17.6 Training-data pipeline

For each prompt:

1. Generate draft continuation.
2. Target-verify it.
3. Find first mismatch.
4. Insert target correction.
5. Teacher-force old suffix under repaired prefix.
6. Record survival mask.
7. Test several anchors.
8. Generate candidate bridges.
9. Measure accepted length and latency.
10. Store structural and hidden-state features.

Example:

```json
{
  "accepted_prefix": "Why did the chicken cross the",
  "draft_token": "farm",
  "target_correction": "road",
  "old_suffix": ["?", "Because", "it", "wanted"],
  "survival_mask": [1, 1, 1, 1],
  "direct_survival_length": 4,
  "best_anchor_offset": 0,
  "causal_horizon": 1,
  "avalanche": false
}
```

## 17.7 Controller losses

### Survival mask

\[
\mathcal L_{\text{survival}}
=
-\sum_i
\left[
m_i \log \hat m_i
+
(1-m_i)\log(1-\hat m_i)
\right]
\]

### Horizon prediction

\[
\mathcal L_{\text{horizon}}
=
\mathrm{CE}(\hat H,H)
\]

### Anchor ranking

\[
U(a)
=
L_{\text{accepted}}(a)
-
\lambda C_{\text{verify}}(a)
-
\mu C_{\text{repair}}(a)
\]

Train with pairwise or listwise ranking.

### Bridge generation

Train on accepted bridge-and-suffix paths. Optimize:

- Target-accepted length
- Short bridge length
- Low latency
- Structural validity

Not just language-model likelihood.

---

# 18. Transformer Architecture Changes

## 18.1 No architecture change required for initial Rejoin

An ordinary causal Transformer can:

- Verify direct reattachments
- Verify bridge-and-suffix candidates
- Use tree attention
- Preserve exact greedy output

It must recompute target states after divergence.

## 18.2 When architecture change becomes necessary

Architecture modification becomes relevant if the goal is:

- Reuse target KV after an earlier token changes
- Merge branch states
- Bound error propagation
- Share future computation after textual reconnection
- Accept noncontiguous islands without full causal recomputation

## 18.3 KV editing

Approximate:

\[
KV_{\text{new}}
\approx
KV_{\text{old}}
+
\Delta KV
\]

where:

\[
\Delta KV =
R(
KV_{\text{old}},
\text{old token},
\text{new token},
\text{local context}
)
\]

Risks:

- Nonlinear error accumulation
- Layerwise mismatch
- Long-horizon drift
- False confidence
- Hard exactness guarantees

## 18.4 Checkpointed Transformer

Introduce semantic checkpoint states:

```text
Block 1 → <CHECKPOINT> → Block 2 → <CHECKPOINT> → Block 3
```

At checkpoint \(b\):

\[
c_b = C(h_{1:b})
\]

Future generation depends primarily on:

- Checkpoint \(c_b\)
- Recent local window
- Explicit memory/retrieval

Different lexical paths may converge to the same canonical checkpoint.

Trade-off:

- Less unrestricted token-level dependency
- Greater editability and state reuse
- Requires post-training or architectural modification

## 18.5 Confluence objective

For two semantically equivalent paths \(A\) and \(B\):

\[
\mathcal L_{\text{confluence}}
=
\|C(h_A)-C(h_B)\|^2
+
\lambda
D_{\mathrm{KL}}
\left(
p_T(\cdot\mid C(h_A))
\parallel
p_T(\cdot\mid C(h_B))
\right)
\]

This is a long-term research direction, not an initial requirement.

---

# 19. Edge Cases and Failure Modes

## 19.1 Negation and polarity

```text
may terminate
may not terminate
```

One token changes the meaning of the entire suffix.

Response:

- High avalanche score
- Abandon old suffix
- Full fallback

## 19.2 Entity replacement and coreference

```text
Alice approved...
Bob approved...
```

Later pronouns, facts, and actions may all change.

Response:

- Expand causal horizon
- Use entity-aware dependency checks
- Avoid aggressive rejoin

## 19.3 Numeric corrections

Changing one amount, date, or intermediate result can alter later calculations.

Response:

- Mark numbers as high-impact
- Execute deterministic dependency checks
- Recompute downstream calculations
- Do not trust lexical realignment alone

## 19.4 Code variable changes

A rename may affect many later references.

Response:

- Use parser/symbol graph
- Treat linked references as a distributed repair region
- Reject suffix if symbol state no longer matches

## 19.5 Control-flow changes

Changing a condition can invalidate the whole code suffix.

Response:

- High avalanche category
- Fallback or use AST-aware repair

## 19.6 JSON and grammar state

A quote, comma, brace, or array boundary can change parser state.

Response:

- Store parser state in checkpoints
- Require parser-state compatibility at anchor
- Verify grammar deterministically

## 19.7 Tokenization at splice

Token IDs can change around the repair boundary.

Response:

- Reconstruct text or bytes
- Retokenize overlap around splice
- Re-establish a valid token boundary
- Require same tokenizer in v1

## 19.8 Bridge length changes

Changing one token to several tokens shifts positions and RoPE indices.

Response:

- Assign new positions
- Recompute target states after divergence
- Represent anchors by text/span identity, not only absolute index

## 19.9 EOS

The corrected target may prefer end-of-sequence.

Response:

- Commit EOS
- Discard suffix
- Never force rejoin past target termination

## 19.10 Weak or repeated anchor

A token like `?` may occur frequently and provide little evidence of true realignment.

Response:

- Use multi-token anchor spans
- Include uniqueness and parser state
- Prefer distinctive phrases

## 19.11 Multiple provisional islands

Later mismatches may be artifacts of the first error.

Response:

- Repair earliest first
- Rescore
- Resegment
- Never treat later islands as independently verified

## 19.12 Branch explosion

Too many bridges and anchors can erase all gains.

Response:

- Hard branch budgets
- Best-first expansion
- Utility gating
- Deduplicate exact prefixes
- Fast fallback

## 19.13 Memory explosion

Each branch requires branch-specific post-divergence KV.

Response:

- Copy-on-write KV pages
- Shared accepted-prefix KV
- Prune aggressively
- Cap active paths

## 19.14 Verification dominates

If target verification is nearly all latency, saving draft work may not matter.

Response:

- Measure end-to-end, not theoretical speed
- Prefer cases with expensive draft or multiple avoided rounds
- Explore hierarchical or partial verification only later

## 19.15 Creative generation

Open-ended text may not naturally realign.

Response:

- Dynamic deactivation
- Benchmark separately
- Accept that the method may specialize in structured or low-temperature tasks

## 19.16 Reasoning avalanche

One wrong intermediate reasoning step may change everything.

Response:

- Treat reasoning-state changes as high impact
- Rejoin only after a stable semantic boundary
- Consider checkpoint-based methods later

## 19.17 Tool calls and side effects

Speculative branches may contain file writes, messages, payments, or API calls.

Response:

- Never execute speculative side effects
- Store as inert tokens or structured actions
- Execute only after target commitment and policy checks

## 19.18 Sampling

Old draft probabilities are invalid under repaired prefix.

Response:

- Recompute draft proposal probabilities under repaired prefix
- Apply valid speculative sampling corrections
- Delay until greedy system is proven

## 19.19 Different tokenizers

Target and draft tokenization mismatch complicates suffix alignment.

Response:

- Require shared tokenizer in v1
- Later represent anchors at byte/character level

## 19.20 Adversarial inputs

Inputs could intentionally cause branch explosion or low acceptance.

Response:

- Strict budgets
- OOD/avalanche detection
- Disable Rejoin when confidence is low
- Always maintain ordinary decoding fallback

---

# 20. Evaluation Plan

## 20.1 Stage 0: Rejoinability characterization

Before implementing an online decoder, measure:

- First mismatch position
- Direct suffix survival length
- Best possible rejoin anchor
- Realignment distance
- Causal horizon
- Correction category
- Task category
- Draft confidence
- Target entropy

## 20.2 Core distribution

Plot:

\[
P(L \ge \ell)
\]

for suffix-survival length \(L\).

This survival curve is more informative than the mean.

Questions:

- What fraction survives at least 4 tokens?
- At least 8?
- At least 16?
- At least 32?
- How does this vary by task and model pair?

## 20.3 Baselines

Compare against:

1. Target-only autoregressive decoding
2. Ordinary speculative decoding
3. Larger draft blocks
4. Tree-based speculation
5. Direct suffix reattachment
6. Anchor skipping
7. Bridge-and-rejoin
8. Optional block-diffusion repair
9. Oracle best-anchor upper bound

## 20.4 Metrics

### Exact output equality

For greedy Rejoin-Exact:

\[
\text{output}_{\text{Rejoin}}
=
\text{output}_{\text{target greedy}}
\]

### End-to-end latency

Include:

- Draft generation
- Target verification
- Anchor selection
- Bridge generation
- Tree construction
- KV allocation
- Kernel launch overhead
- Synchronization

### Throughput

Measure:

- Requests per second
- Tokens per second
- Useful committed tokens per target invocation

### Suffix salvage ratio

\[
\mathrm{SSR}=
\frac{\text{accepted reused suffix tokens}}
{\text{tokens after first rejection}}
\]

### Repair success rate

Fraction of failed draft rounds where a rejoin candidate commits more tokens than ordinary fallback.

### Break-even suffix length

Smallest suffix length for which Rejoin has positive latency benefit.

### Branch efficiency

\[
\frac{\text{winning branch compute}}
{\text{total branch compute}}
\]

### Memory overhead

- Peak KV memory
- Candidate token storage
- Number of active branches
- Copy-on-write page count

### Energy

Later:

- Joules per committed token
- Joules per completed request

## 20.5 Task categories

Use a broad evaluation:

- General conversation
- Summarization
- Structured JSON
- Code completion
- SQL
- Mathematical reasoning
- Citation-grounded generation
- Legal extraction
- Legal report generation

Hypothesis:

> Rejoin benefit increases as output entropy decreases and structural repetition increases.

## 20.6 Ablations

- Anchor count
- Anchor distance
- Guard-window length
- Draft block length
- Draft-model size
- Temperature
- Same versus different tokenizer
- Controller versus heuristics
- Direct reattach versus FIM bridge
- Tree budget
- Suffix-length cap
- Avalanche detector

---

# 21. LawOS Integration Strategy

LawOS should be a benchmark and shadow workload, not a tightly coupled dependency.

## 21.1 Interface

```python
class InferenceBackend(Protocol):
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        schema: dict | None = None,
    ) -> GenerationResult:
        ...
```

Backends:

```text
NormalAutoregressiveBackend
StandardSpeculativeBackend
RejoinExperimentalBackend
```

## 21.2 Best initial LawOS tasks

- Clause extraction
- Contract metadata extraction
- Structured issue spotting
- Citation-grounded summaries
- Contract playbook comparisons
- Template-based reports
- Defined-term and section-number extraction

## 21.3 Avoid initially

- Open-ended legal memoranda
- Multi-hop legal reasoning
- Agentic tool use
- High-stakes production decisions
- Any approximate KV reuse
- Any unverified semantic join

## 21.4 Shadow mode

For each production-style request:

1. Normal backend produces the authoritative output.
2. Rejoin backend runs offline or in parallel.
3. Compare exact output, latency, memory, and salvage.
4. Log failures and causal categories.
5. Do not expose Rejoin output to users until exactness and speed gates are met.

---

# 22. Implementation Roadmap

Durations are rough estimates for a focused student/researcher implementation and depend heavily on hardware access and inference-engine experience.

## Phase 0: Scope and infrastructure

**Goal:** Establish reproducible baselines.

Tasks:

- Choose target and draft models with shared tokenizer.
- Implement target greedy generation.
- Implement ordinary speculative decoding.
- Add trace logging.
- Build benchmark harness.
- Record exact latency and memory.
- Create unit tests for output equality.

Deliverables:

- Baseline repo
- Reproducible benchmark script
- Trace schema
- Initial LawOS prompt set

Go/no-go:

- Baseline outputs are correct.
- Latency measurements are stable.
- Trace collection does not distort runtime excessively.

## Phase 1: Rejoinability study

**Goal:** Determine whether suffix realignment is common enough.

Tasks:

- Collect 10,000–100,000 speculative mismatch traces.
- Insert target correction.
- Teacher-force retained suffix.
- Measure direct survival.
- Search oracle anchors.
- Categorize corrections.
- Plot survival curves.

Deliverables:

- Rejoinability report
- Task-specific statistics
- Oracle upper bound
- Failure taxonomy

Go/no-go:

Proceed if at least one meaningful workload has:

- A nontrivial fraction of mismatches with 8+ surviving tokens, or
- A strong long-tail of 16–64+ salvageable tokens, and
- Estimated speedup survives realistic target-verification cost.

## Phase 2: Rejoin-Exact v1

**Goal:** Implement direct suffix reattachment.

Tasks:

- Preserve suffix buffer.
- Insert target correction.
- Reverify repaired candidate.
- Commit exact matching prefix.
- Add short-suffix fallback.
- Add avalanche rules.
- Measure end-to-end speed.

Deliverables:

- Exact greedy Rejoin backend
- Output-equivalence tests
- Latency benchmark

Go/no-go:

- 100% greedy output equality
- Positive speedup on at least one workload
- No pathological memory growth

## Phase 3: Anchor search

**Goal:** Salvage suffixes after short unstable regions.

Tasks:

- Implement candidate anchor generation.
- Add multi-token anchor spans.
- Add tokenization-safe splice handling.
- Add parser-state checks.
- Add utility ranking.
- Compare heuristic anchors with oracle anchors.

Deliverables:

- Anchor-search module
- Anchor-quality analysis
- Break-even controller

## Phase 4: Learned controller

**Goal:** Attempt Rejoin only when profitable.

Tasks:

- Generate labeled traces.
- Train survival predictor.
- Train causal-horizon predictor.
- Train anchor ranker.
- Train avalanche detector.
- Calibrate probabilities.
- Integrate controller.

Deliverables:

- Shared controller model
- Calibration plots
- Runtime utility improvement

## Phase 5: Local bridge generation

**Goal:** Repair multi-token divergence.

Tasks:

- Fine-tune or adapt small model for FIM repair.
- Condition on target correction and right anchor.
- Generate a small candidate set.
- Optimize bridge length and target acceptance.
- Add branch budgets.

Deliverables:

- Dual-mode draft/repair model
- Bridge benchmark
- Repair success analysis

## Phase 6: Tree-batched verification

**Goal:** Verify several repair paths in one target call.

Tasks:

- Implement candidate-tree packing.
- Build tree attention masks.
- Share prefix KV.
- Use copy-on-write branch KV.
- Add branch pruning.
- Profile kernels and memory.

Deliverables:

- Tree verification runtime
- Comparison against sequential candidate verification
- Memory and utilization report

## Phase 7: LawOS shadow integration

**Goal:** Evaluate on realistic legal workflows.

Tasks:

- Add backend adapter.
- Collect structured legal traces.
- Run normal and Rejoin outputs side by side.
- Analyze legal-specific error categories.
- Measure citation and structured-output cases.

Deliverables:

- Shadow deployment report
- LawOS benchmark
- Product integration recommendation

## Phase 8: Exact sampling

**Goal:** Preserve target sampling distribution.

Tasks:

- Rescore retained suffix under repaired prefix.
- Derive proposal distribution for bridge paths.
- Implement correct acceptance correction.
- Validate statistically.

Deliverables:

- Sampling-capable Rejoin variant
- Distributional tests
- Performance comparison

## Phase 9: Advanced state reuse

Only begin if candidate-level Rejoin proves valuable.

Options:

- Future-state fingerprints
- Delta-KV editor
- Sparse checkpoint verification
- Canonical checkpoint tokens
- Confluence post-training
- Rejoin-native architecture

---

# 23. Repository Structure

```text
rejoin/
├── README.md
├── pyproject.toml
├── configs/
│   ├── models/
│   ├── benchmarks/
│   └── experiments/
├── rejoin/
│   ├── api.py
│   ├── controller.py
│   ├── types.py
│   ├── runtime/
│   │   ├── autoregressive.py
│   │   ├── speculative.py
│   │   ├── direct_reattach.py
│   │   ├── anchor_search.py
│   │   ├── bridge_repair.py
│   │   ├── tree_verify.py
│   │   ├── kv_pages.py
│   │   └── fallback.py
│   ├── analysis/
│   │   ├── trace_collector.py
│   │   ├── realignment.py
│   │   ├── causal_horizon.py
│   │   ├── survival_curves.py
│   │   └── error_taxonomy.py
│   ├── models/
│   │   ├── target_adapter.py
│   │   ├── draft_adapter.py
│   │   ├── repair_adapter.py
│   │   ├── controller_heads.py
│   │   └── future_fingerprint.py
│   ├── structures/
│   │   ├── interval_tree.py
│   │   ├── candidate_lattice.py
│   │   ├── checkpoint.py
│   │   └── factors.py
│   └── metrics/
│       ├── latency.py
│       ├── salvage.py
│       ├── memory.py
│       └── exactness.py
├── benchmarks/
│   ├── general/
│   ├── structured/
│   ├── code/
│   ├── reasoning/
│   └── lawos/
├── experiments/
│   ├── phase1_rejoinability/
│   ├── phase2_direct/
│   ├── phase3_anchors/
│   ├── phase4_controller/
│   └── phase5_repair/
└── tests/
    ├── test_exact_greedy.py
    ├── test_tokenization_splice.py
    ├── test_parser_state.py
    ├── test_branch_pruning.py
    └── test_fallback.py
```

---

# 24. Decision Gates

## Gate 1: Is the phenomenon real?

Stop or narrow the project if:

- Most mismatches have zero or one reusable suffix token.
- Realignment almost never occurs within the draft block.
- Only trivial punctuation survives.
- Oracle anchor search still gives weak salvage.

## Gate 2: Does candidate reuse help latency?

Stop or redesign if:

- Reverification cost exceeds saved drafting time.
- Direct reattachment slows end-to-end decoding.
- Additional target passes dominate.
- Tree verification has poor utilization.

## Gate 3: Can a controller identify profitable cases?

Stop automatic activation if:

- Predictor calibration is poor.
- False-positive rejoin attempts erase gains.
- Strong cases cannot be separated from avalanche cases.

## Gate 4: Is the method general or specialized?

Possible outcomes:

1. **General win:** Continue as general inference research.
2. **Structured-only win:** Position for code, JSON, extraction, and professional workflows.
3. **LawOS-only win:** Keep as a product optimization, not a general paper.
4. **No win:** Publish negative findings or pivot to rejoin-native training.

## Gate 5: Is architecture modification justified?

Only attempt KV editing or checkpoint architecture if:

- Candidate-level Rejoin works.
- Target suffix recomputation is the dominant remaining bottleneck.
- Paths frequently become behaviorally similar again.
- There is enough compute to train and evaluate state-repair modules.

---

# 25. Advanced Research Directions

## 25.1 Future-equivalence fingerprints

Train a projection that predicts whether two prefix states will generate similar futures.

Use for:

- Anchor ranking
- Branch pruning
- Guard-window sizing
- Approximate state reuse research

## 25.2 Delta-KV repair

Predict how stale suffix KV should change after a local prefix edit.

\[
KV_{\text{new}}
\approx
KV_{\text{old}}+\Delta KV
\]

Require periodic clean target checkpoints.

## 25.3 Canonical checkpoints

Post-train the target to compress a block into a state designed for future reuse.

This can bound error propagation.

## 25.4 Confluent decoding

Train semantically equivalent local paths to converge to equivalent future states.

Potential loss:

\[
\mathcal L_{\text{confluence}}
=
\|C(h_A)-C(h_B)\|^2
+
\lambda D_{\mathrm{KL}}(p_A\|p_B)
\]

## 25.5 Local diffusion repair

Use masked block diffusion only over the repair island, not the whole output block.

Possible advantage:

- Local parallel infilling
- Multiple bridge proposals
- Better use of both boundaries

## 25.6 Semantic rejoin

Allow different wording to be considered equivalent at a semantic checkpoint.

This leaves exact token-distribution preservation and belongs in Rejoin-Approx.

## 25.7 Proof-carrying output spans

Attach deterministic evidence:

- Parser state
- Source hashes
- Numeric checks
- Schema proofs
- Tool preconditions

Use these to reduce full-model verification for structured spans.

## 25.8 Hardware acceleration

Only after software validation:

- FPGA candidate-lattice manager
- Anchor search and suffix automata
- Parser/factor checking
- Sparse edit application
- Tiny repair model
- KV filtering
- Tree-attention scheduling

Hardware should accelerate a proven bottleneck, not define the initial research.

---

# 26. Recommended Direction

## Immediate recommendation

Build **Rejoin-Exact v1** as a standalone inference backend.

Use:

- One target model
- One smaller draft model
- Greedy decoding
- Shared tokenizer
- Direct suffix reattachment
- One unresolved island
- Exact target reverification
- Strict fallback
- No new training
- No Transformer changes
- No FPGA
- LawOS only as one benchmark and shadow workload

## Why this direction is best

It answers the most important question with the least complexity:

> Are speculative decoding errors local enough that suffix preservation has real value?

Every advanced idea depends on the answer.

## Recommended sequence

```text
Rejoinability study
    ↓
Direct exact suffix reattachment
    ↓
Anchor search
    ↓
Learned survival/controller heads
    ↓
FIM bridge repair
    ↓
Tree-batched verification
    ↓
LawOS shadow deployment
    ↓
Exact sampling
    ↓
Future-state fingerprints
    ↓
KV repair / checkpoint architecture
    ↓
Hardware specialization
```

## What not to do first

Do not begin with:

- Multiple error islands
- High-dimensional hypergraphs in the runtime
- Learned KV repair
- New Transformer pretraining
- Block diffusion target conversion
- Custom FPGA deployment
- Company-specific model compilation
- Semantic approximate acceptance

These are valid later directions but would hide whether the core phenomenon is real.

---

# 27. Open Questions

1. How often do speculative continuations naturally realign after one local correction?
2. Which task distributions have the highest suffix-survival tail?
3. How much of the gain comes from avoiding draft regeneration versus avoiding future rounds?
4. Does direct suffix reattachment already capture most of the available value?
5. How far ahead should anchors be placed?
6. Are punctuation and sentence boundaries useful or misleading anchors?
7. Can a tiny controller predict causal horizon accurately?
8. Is FIM repair materially better than AR repair?
9. How many branches can be verified before tree overhead dominates?
10. How does draft block length affect rejoinability?
11. Does a stronger draft model produce fewer but more local errors?
12. Does quantization change error locality?
13. Can exact sampling be supported without losing the speed benefit?
14. Can future-state fingerprints reliably predict multi-step equivalence?
15. Can stale KV be corrected locally without long-horizon drift?
16. Can post-training create bounded error propagation?
17. Is the method broadly useful or mainly useful for structured generation?
18. Does LawOS provide an unusually favorable benchmark that does not generalize?
19. Can Rejoin combine with block-diffusion drafting without excessive verification?
20. What is the strongest novelty claim supported by a formal literature and patent search?

---

# 28. Appendix: Pseudocode and Schemas

## 28.1 Core types

```python
from dataclasses import dataclass
from typing import Sequence


@dataclass
class CandidateSpan:
    tokens: Sequence[int]
    anchor_offset: int
    bridge_length: int
    expected_salvage: float
    expected_utility: float


@dataclass
class CheckpointState:
    token_position: int
    prefix_hash: bytes
    parser_state: str | None
    target_entropy: float
    predicted_survival: float
    exact_verified: bool


@dataclass
class VerificationResult:
    greedy_match_length: int
    target_tokens: Sequence[int]
    target_probabilities: Sequence[float]
```

## 28.2 Hardened exact loop

```python
def rejoin_exact_step(
    target,
    draft_model,
    prefix_tokens,
    draft_length: int,
    max_anchors: int = 3,
    max_paths: int = 8,
):
    draft_tokens = draft_model.generate_greedy(
        prefix_tokens,
        max_new_tokens=draft_length,
    )

    initial = target.verify_greedy(prefix_tokens, draft_tokens)

    if initial.greedy_match_length == len(draft_tokens):
        return draft_tokens

    j = initial.greedy_match_length
    accepted = draft_tokens[:j]
    correction = initial.target_tokens[j]
    suffix = draft_tokens[j + 1:]

    if not suffix:
        return accepted + [correction]

    if len(suffix) < MIN_SUFFIX_LENGTH:
        return accepted + [correction]

    base = prefix_tokens + accepted

    if avalanche_detector(base, correction, suffix):
        return accepted + [correction]

    candidates = [
        CandidateSpan(
            tokens=[correction] + suffix,
            anchor_offset=0,
            bridge_length=1,
            expected_salvage=len(suffix),
            expected_utility=0.0,
        )
    ]

    anchors = rank_anchors(
        base=base,
        correction=correction,
        suffix=suffix,
    )[:max_anchors]

    for anchor in anchors:
        for bridge in propose_bridges(
            left_context=base,
            correction=correction,
            right_context=suffix[anchor:],
        ):
            tokens = [correction] + bridge + suffix[anchor:]
            candidates.append(
                CandidateSpan(
                    tokens=tokens,
                    anchor_offset=anchor,
                    bridge_length=1 + len(bridge),
                    expected_salvage=len(suffix) - anchor,
                    expected_utility=estimate_utility(tokens),
                )
            )

    candidates = deduplicate_by_token_prefix(candidates)
    candidates = sorted(
        candidates,
        key=lambda c: c.expected_utility,
        reverse=True,
    )[:max_paths]

    results = target.verify_tree(
        prefix_tokens=base,
        candidate_paths=[c.tokens for c in candidates],
    )

    best_idx = max(
        range(len(results)),
        key=lambda i: results[i].greedy_match_length,
    )

    best = candidates[best_idx]
    matched = results[best_idx].greedy_match_length

    if matched <= 0:
        return accepted + [correction]

    return accepted + list(best.tokens[:matched])
```

## 28.3 Trace schema

```json
{
  "prompt_id": "example-001",
  "task_type": "structured_json",
  "target_model": "target-model-name",
  "draft_model": "draft-model-name",
  "draft_length": 32,

  "accepted_prefix_length": 11,
  "first_mismatch_position": 12,
  "draft_token": 4281,
  "target_correction": 913,

  "suffix_length": 19,
  "direct_survival_length": 8,
  "best_anchor_offset": 3,
  "best_anchor_survival": 14,

  "correction_category": "noun_substitution",
  "avalanche": false,
  "parser_state_match": true,

  "baseline_round_latency_ms": 18.4,
  "rejoin_latency_ms": 15.1,
  "tokens_committed_baseline": 12,
  "tokens_committed_rejoin": 24
}
```

## 28.4 Minimal experiment checklist

- [ ] Choose target and draft models with shared tokenizer.
- [ ] Reproduce ordinary target greedy output.
- [ ] Reproduce ordinary speculative decoding.
- [ ] Add exact trace logging.
- [ ] Collect at least 10,000 mismatch cases.
- [ ] Plot direct suffix-survival curve.
- [ ] Compute oracle anchor upper bound.
- [ ] Categorize avalanche corrections.
- [ ] Implement direct reattachment.
- [ ] Verify 100% exact output equality.
- [ ] Measure end-to-end latency.
- [ ] Decide whether to continue to learned repair.

---

## Final Position

Rejoin is worth implementing first as a **general, exact, candidate-reuse decoding method**. The most important near-term contribution is not a new Transformer architecture or FPGA design. It is establishing whether local speculative errors leave reusable future structure often enough to produce a measurable systems advantage.

The core research discipline should be:

> Preserve later draft tokens as provisional candidates, repair only the earliest causal break, and never confuse textual reconnection with computational-state equivalence.

If that conservative method works, the project has a credible path toward learned causal representations, local diffusion repair, approximate KV editing, checkpointed model architectures, and specialized hardware. If it does not work, those advanced layers would only optimize a phenomenon that is too rare to matter.
