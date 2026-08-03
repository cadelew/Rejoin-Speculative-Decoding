"""trace_core.py — model-agnostic core of the Phase 0 suffix-survival study.

Contains ZERO torch/transformers code so the tricky bookkeeping (off-by-ones,
cache cropping, deferred event pairing, escrow alignment) can be unit-tested on
CPU with mock models (see test_core.py). spec_trace.py wires real HF models in.

Invariant 1 — the cache invariant.
    The KV cache always holds states for committed[:-1]. Every verification
    forward therefore feeds [committed[-1]] + candidate_block, so logits row j
    is the target's prediction for input position j+1:
        row j  -> prediction for block[j]   (j = 0 .. k-1)
        row k  -> prediction for the token AFTER the block (the bonus token)
    After a rejection at block index `a`, states for accepted tokens are valid
    and everything from the rejected position on is garbage;
    `crop_to(len(committed) - 1)` handles both cases with one line.

Invariant 2 — why survival measurement is free.
    Every committed token is the target's greedy choice, so the committed tail
    after an event IS the target's greedy continuation from the corrected
    prefix. Teacher-forcing the escrowed suffix in a separate branch pass
    recomputes what the main loop already produces. `L_survive` is therefore
    derived from the realized continuation at zero extra target passes, and
    `branch_verify` is retained only as an opt-in cross-check.

    Consequence: escrow ALIGNMENT (does the escrow reattach after a k-token
    bridge?) is also free, because the realized continuation is recorded. This
    is what schema 1 could not answer — it discarded the suffix token ids and
    never stored the continuation.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 2

# How far past the escrow to record the target's realized continuation. Must
# exceed MAX_BRIDGE so bridge search never runs off the end of the record.
DEFAULT_REALIZED_HORIZON = 24
MAX_BRIDGE = 8


# ------------------------------- alignment ---------------------------------


def first_mismatch(cand: Sequence[int], ref: Sequence[int]) -> int:
    """Index of the first position where cand[i] != ref[i]; min length if all match.

    Under teacher-forced greedy verification this IS the acceptance length.
    """
    n = min(len(cand), len(ref))
    for i in range(n):
        if cand[i] != ref[i]:
            return i
    return n


def prefix_match(cand: Sequence[int], ref: Sequence[int]) -> Tuple[int, bool]:
    """Like `first_mismatch`, but reports censoring.

    Returns (matched, censored). `censored` is True when the reference ran out
    before the candidate mismatched, i.e. the true match length is >= matched
    but unknown. This happens when a prompt ends (EOS or max_new) fewer than
    `len(cand)` tokens after an event, and it is the one thing the free
    measurement cannot resolve that a branch pass can.
    """
    n = min(len(cand), len(ref))
    for i in range(n):
        if cand[i] != ref[i]:
            return i, False
    return n, n < len(cand)


def best_bridge(
    suffix: Sequence[int], realized: Sequence[int], max_bridge: int = MAX_BRIDGE
) -> Tuple[int, int]:
    """Best whole-escrow reattachment after a short bridge.

    For each bridge length k, ask how many escrow tokens match the realized
    continuation starting at offset k. k=0 is `L_survive`, so the result is
    always >= L_survive.

    The bridge here is the target's OWN realized continuation, which makes this
    an oracle measurement -- but a free one. It bounds what the beam-searched
    bridge study of the original guide (its section 8) could find, without
    spending a single extra target pass, and without gating it behind Gate 1.

    Returns (length, k).
    """
    best_len, best_k = 0, 0
    limit = min(max_bridge, len(realized))
    for k in range(limit + 1):
        matched, _ = prefix_match(suffix, realized[k:])
        if matched > best_len:
            best_len, best_k = matched, k
    return best_len, best_k


def best_trim(
    suffix: Sequence[int], realized: Sequence[int], max_trim: int = MAX_BRIDGE
) -> Tuple[int, int]:
    """Best attachment when the escrow is trimmed at the FRONT and attaches
    immediately -- no bridge.

    Unlike `best_bridge`, this one is implementable online: propose escrow[j:]
    for several small j as a batched candidate tree sharing the corrected
    prefix, and take whichever verifies longest. Nothing about the target's
    continuation needs to be known in advance. j=0 reduces to `L_survive`.

    Returns (length, j).
    """
    best_len, best_j = 0, 0
    limit = min(max_trim, len(suffix))
    for j in range(limit + 1):
        matched, _ = prefix_match(suffix[j:], realized)
        if matched > best_len:
            best_len, best_j = matched, j
    return best_len, best_j


def longest_common_run(a: Sequence, b: Sequence) -> Tuple[int, int, int]:
    """Longest contiguous common subsequence. Returns (length, start_in_a, start_in_b).

    Catches the case where the escrow's plan survives but the draft also has to
    skip some of its own tokens -- `best_bridge` only allows slack on the
    target side, this allows slack on both. Works on token-id lists and on
    strings (for the tokenization-insensitive variant).
    """
    if not a or not b:
        return 0, 0, 0
    best = (0, 0, 0)
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                run = prev[j - 1] + 1
                cur[j] = run
                if run > best[0]:
                    best = (run, i - run, j - run)
        prev = cur
    return best


# --------------------------------- session ----------------------------------


class TargetSession:
    """Interface the target model must implement (see HFTargetSession)."""

    def prefill(self, ids: List[int]) -> None:
        """Reset and fill the cache with `ids` (= committed[:-1])."""
        raise NotImplementedError

    def verify(self, last: int, block: List[int]):
        """Forward [last] + block on top of the cache (cache GROWS by 1+len(block)).

        Returns (tg, bonus, extras):
          tg     list[int], len(block): target greedy token for each block position
          bonus  int: target greedy token after the full block
          extras dict: optional diagnostics. Recognised keys:
                 "entropies" list[float]  per-position target entropy
                 "p_draft"   list[float]  target probability of the DRAFT token
                 "p_top1"    list[float]  target probability of its own choice
        """
        raise NotImplementedError

    def branch_verify(self, last: int, block: List[int]) -> List[int]:
        """Like verify() but restores the cache to its pre-call length after.

        Only used by the opt-in survival cross-check; the study no longer needs
        it for its primary measurement.
        """
        raise NotImplementedError

    def crop_to(self, n: int) -> None:
        """Crop the cache to the first n token positions."""
        raise NotImplementedError

    def paranoid_check(self, committed: List[int]) -> bool:
        """Recompute the next-token prediction WITHOUT the cache and compare.

        Returns True when the cached and cache-free argmax agree. The default
        implementation opts out by reporting agreement.
        """
        return True


# ---------------------------------- events ----------------------------------


@dataclass
class SuffixEvent:
    """One rejection with a non-empty escrowed suffix.

    L_survive  acceptance length of the escrow attached at offset 0, derived
               from the realized continuation (free). This is the metric the
               original study reported, and it is the strictest possible
               question: suffix[0] is the token immediately after the one that
               was just replaced, i.e. the position most contaminated by the
               correction.
    L_bridge   acceptance length of the escrow attached after `bridge_k` tokens
               of the target's own continuation. L_bridge >= L_survive by
               construction. This is the measurement schema 1 could not make.
    L_fresh    acceptance length of the NEXT cycle's fresh draft from the very
               same corrected prefix, capped at m so the paired comparison has
               equal ceilings. Filled one cycle later ("deferred pairing"),
               which costs zero extra target passes.
    """

    step: int
    committed_len: int  # length of the corrected prefix; realized starts here
    a: int  # mismatch index within the draft block
    block_len: int
    m: int  # escrow length
    rejected: int
    correction: int
    suffix: List[int] = field(default_factory=list)
    realized: List[int] = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    L_survive: Optional[int] = None
    L_survive_censored: bool = False
    L_survive_branch: Optional[int] = None  # opt-in cross-check
    L_bridge: Optional[int] = None
    bridge_k: Optional[int] = None
    lcs_len: Optional[int] = None
    lcs_i: Optional[int] = None  # start in suffix
    lcs_j: Optional[int] = None  # start in realized
    L_fresh: Optional[int] = None
    L_fresh_censored: bool = False


@dataclass
class PromptResult:
    committed: List[int]
    prompt_len: int
    events: List[SuffixEvent]
    n_cycles: int = 0
    n_main_passes: int = 0
    n_branch_passes: int = 0
    n_dropped_events: int = 0  # events whose pairing cycle never happened
    n_paranoid_checks: int = 0
    n_paranoid_mismatches: int = 0
    total_accepted: int = 0  # draft tokens accepted across cycles (excl. bonus)


# ----------------------------------- loop -----------------------------------


def run_prompt(
    prompt: List[int],
    draft_fn: Callable[[List[int], int], List[int]],
    session: TargetSession,
    gamma: int,
    max_new: int,
    eos_ids: set,
    branch_verify: bool = False,
    paranoid_every: int = 0,
    realized_horizon: int = DEFAULT_REALIZED_HORIZON,
    max_bridge: int = MAX_BRIDGE,
) -> PromptResult:
    """Greedy speculative decoding with suffix-survival instrumentation.

    Output text is IDENTICAL to pure greedy decoding of the target (every
    committed token is target-verified greedy), so the instrumentation cannot
    bias the trajectory. With branch_verify=False the instrumentation is also
    free: no extra target passes at all.
    """
    committed = list(prompt)
    res = PromptResult(committed=committed, prompt_len=len(prompt), events=[])
    session.prefill(committed[:-1])
    pending: Optional[SuffixEvent] = None
    done = False

    while not done and len(committed) - res.prompt_len < max_new:
        block = draft_fn(committed, gamma)
        if not block:
            break
        tg, bonus, extras = session.verify(committed[-1], block)
        res.n_main_passes += 1
        res.n_cycles += 1

        a = first_mismatch(block, tg)  # acceptance length within block
        correction = tg[a] if a < len(block) else bonus
        res.total_accepted += a

        # Complete the pending event: this cycle's block IS the fresh draft
        # from the same corrected prefix (greedy drafting is deterministic).
        if pending is not None:
            pending.L_fresh = min(a, pending.m)
            # The fresh draft can only be censored if it EOS'd short of m.
            pending.L_fresh_censored = a == len(block) and len(block) < pending.m
            res.events.append(pending)
            pending = None

        # Commit accepted tokens + correction/bonus; truncate at EOS.
        added = block[:a] + [correction]
        cut = next((i for i, t in enumerate(added) if t in eos_ids), None)
        if cut is not None:
            added = added[: cut + 1]
            done = True
        committed.extend(added)
        # Cache states are valid exactly for committed[:-1] (accepted tokens
        # match what was fed; the rejected position and everything after were
        # computed under wrong tokens and must be dropped).
        session.crop_to(len(committed) - 1)

        if paranoid_every and res.n_cycles % paranoid_every == 0:
            res.n_paranoid_checks += 1
            if not session.paranoid_check(committed):
                res.n_paranoid_mismatches += 1

        # Escrow bookkeeping. L_survive/L_bridge are filled in after the whole
        # prompt is decoded, from the realized continuation.
        if not done and a < len(block):
            suffix = block[a + 1 :]
            if suffix:
                evt_extras = {}
                for key, idx in (("entropies", a), ("p_draft", a), ("p_top1", a)):
                    seq = extras.get(key)
                    if seq is not None and idx < len(seq):
                        name = {
                            "entropies": "entropy_at_rejection",
                            "p_draft": "p_rejected",
                            "p_top1": "p_correction",
                        }[key]
                        evt_extras[name] = seq[idx]
                evt = SuffixEvent(
                    step=res.n_cycles,
                    committed_len=len(committed),
                    a=a,
                    block_len=len(block),
                    m=len(suffix),
                    rejected=block[a],
                    correction=correction,
                    suffix=list(suffix),
                    extras=evt_extras,
                )
                if branch_verify:
                    tgs = session.branch_verify(committed[-1], suffix)
                    res.n_branch_passes += 1
                    evt.L_survive_branch = first_mismatch(suffix, tgs)
                pending = evt

    if pending is not None:
        res.n_dropped_events += 1
    res.committed = committed
    fill_alignment(res, realized_horizon=realized_horizon, max_bridge=max_bridge)
    return res


def fill_alignment(
    res: PromptResult,
    realized_horizon: int = DEFAULT_REALIZED_HORIZON,
    max_bridge: int = MAX_BRIDGE,
) -> None:
    """Derive survival and alignment for every event from the realized tail.

    Zero target passes: the committed tail after `committed_len` is exactly the
    target's greedy continuation from the corrected prefix (invariant 2).
    """
    for e in res.events:
        e.realized = res.committed[e.committed_len : e.committed_len + e.m + realized_horizon]
        matched, censored = prefix_match(e.suffix, e.realized)
        e.L_survive = matched
        e.L_survive_censored = censored
        e.L_bridge, e.bridge_k = best_bridge(e.suffix, e.realized, max_bridge)
        e.lcs_len, e.lcs_i, e.lcs_j = longest_common_run(e.suffix, e.realized)


def greedy_rollout(prompt: List[int], next_fn, max_new: int, eos_ids: set) -> List[int]:
    """Reference pure greedy decode (for tests / paranoid checks)."""
    out = list(prompt)
    for _ in range(max_new):
        t = next_fn(out)
        out.append(t)
        if t in eos_ids:
            break
    return out
