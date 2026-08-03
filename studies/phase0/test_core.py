"""test_core.py — CPU-only tests of the trace loop using mock models.

Run `python test_core.py` BEFORE touching real models. If these pass, the
off-by-one alignment, cache-crop bookkeeping, deferred event pairing, and the
free survival/alignment derivation are correct; any later bug is in the torch
wiring, not the algorithm.
"""

import trace_core as tc

VOCAB = 1000
EOS = 999


class MockSession(tc.TargetSession):
    """Implements the session contract over a deterministic next-token fn.

    Mirrors the cache invariant literally: self.toks plays the role of the KV
    cache and must always equal committed[:-1] between cycles.
    """

    def __init__(self, next_fn):
        self.f = next_fn
        self.toks = []

    def prefill(self, ids):
        self.toks = list(ids)

    def _teacher_forced(self, last, block):
        prefix = self.toks + [last]
        tg = []
        for b in block:
            tg.append(self.f(prefix))
            prefix = prefix + [b]  # teacher forcing: condition on DRAFT token
        bonus = self.f(prefix)
        return tg, bonus

    def verify(self, last, block):
        tg, bonus = self._teacher_forced(last, block)
        self.toks = self.toks + [last] + list(block)  # cache grows
        return tg, bonus, {}

    def branch_verify(self, last, block):
        tg, _ = self._teacher_forced(last, block)
        return tg  # cache restored (never grew)

    def crop_to(self, n):
        assert n <= len(self.toks), "crop_to beyond cache length"
        self.toks = self.toks[:n]

    def paranoid_check(self, committed):
        assert self.toks == committed[:-1], "cache invariant violated"
        return True


def make_positional_target(prompt_len):
    """Next token depends ONLY on position -> corrections never change the
    future -> every escrowed suffix should fully survive (up to later draft
    errors inside it)."""
    def f(prefix):
        return 100 + (len(prefix) - prompt_len) % 700
    return f


def make_content_target():
    """Next token depends on the last 3 tokens -> a correction changes the
    entire continuation -> L_survive must be 0 at every event."""
    def f(prefix):
        return (hash(tuple(prefix[-3:])) % 700) + 100
    return f


def make_flipping_draft(next_fn, flip_every, flip_offset):
    """Greedy rollout of the target, but corrupt the token whenever the
    generated index g satisfies g % flip_every == flip_offset."""
    def draft_fn(committed, gamma, prompt_len):
        out = []
        prefix = list(committed)
        for _ in range(gamma):
            t = next_fn(prefix)
            g = len(prefix) - prompt_len
            if g % flip_every == flip_offset:
                t = (t + 1) % VOCAB
            out.append(t)
            prefix.append(t)
        return out
    return draft_fn


def run(target_fn, draft_raw, prompt, gamma=8, max_new=120, **kw):
    session = MockSession(target_fn)
    draft_fn = lambda committed, g: draft_raw(committed, g, len(prompt))
    return tc.run_prompt(prompt, draft_fn, session, gamma, max_new, {EOS}, **kw)


# ----------------------------- alignment units ------------------------------


def test_prefix_match_censoring():
    assert tc.prefix_match([1, 2, 3], [1, 2, 3, 4]) == (3, False)
    assert tc.prefix_match([1, 2, 3], [1, 9, 3]) == (1, False)
    assert tc.prefix_match([1, 2, 3], [1, 2]) == (2, True), "short ref must censor"
    assert tc.prefix_match([1, 2, 3], []) == (0, True)
    assert tc.prefix_match([], [1]) == (0, False)
    print("  prefix_match censoring: OK")


def test_best_bridge():
    suffix = [7, 8, 9]
    # target emits one bridge token, then exactly the escrow
    assert tc.best_bridge(suffix, [5, 7, 8, 9, 1]) == (3, 1)
    # offset 0 already perfect -> k=0 wins
    assert tc.best_bridge(suffix, [7, 8, 9, 1]) == (3, 0)
    # escrow never reappears
    assert tc.best_bridge(suffix, [1, 2, 3, 4]) == (0, 0)
    # bridge beyond the search window is not found
    assert tc.best_bridge(suffix, [0] * 10 + [7, 8, 9], max_bridge=2) == (0, 0)
    assert tc.best_bridge(suffix, [0] * 10 + [7, 8, 9], max_bridge=10)[0] == 3
    print("  best_bridge: OK")


def test_longest_common_run():
    assert tc.longest_common_run([1, 2, 3, 4], [9, 2, 3, 9]) == (2, 1, 1)
    assert tc.longest_common_run([1, 2, 3], [1, 2, 3]) == (3, 0, 0)
    assert tc.longest_common_run([1, 2], [3, 4]) == (0, 0, 0)
    assert tc.longest_common_run([], [1]) == (0, 0, 0)
    # works on strings too (the tokenization-insensitive variant)
    assert tc.longest_common_run("xxabcyy", "zzabcz")[0] == 3
    print("  longest_common_run: OK")


def test_fill_alignment_detects_bridge():
    """End-to-end proof that the bridge measurement sees what offset-0 misses."""
    suffix = [41, 42, 43, 44]
    prefix_len = 10
    # target's realized continuation: one bridge token, then the whole escrow
    committed = list(range(prefix_len)) + [77] + suffix + [55, 56]
    e = tc.SuffixEvent(step=1, committed_len=prefix_len, a=0, block_len=8,
                       m=len(suffix), rejected=1, correction=2, suffix=list(suffix))
    res = tc.PromptResult(committed=committed, prompt_len=0, events=[e])
    tc.fill_alignment(res)
    assert e.L_survive == 0, f"offset-0 must miss it, got {e.L_survive}"
    assert e.L_bridge == 4 and e.bridge_k == 1, f"got L_bridge={e.L_bridge} k={e.bridge_k}"
    assert e.lcs_len == 4
    assert not e.L_survive_censored
    print("  fill_alignment bridge detection: OK")


def test_fill_alignment_censoring():
    suffix = [41, 42, 43, 44]
    committed = list(range(10)) + [41, 42]  # prompt ends mid-escrow
    e = tc.SuffixEvent(step=1, committed_len=10, a=0, block_len=8,
                       m=len(suffix), rejected=1, correction=2, suffix=list(suffix))
    res = tc.PromptResult(committed=committed, prompt_len=0, events=[e])
    tc.fill_alignment(res)
    assert e.L_survive == 2 and e.L_survive_censored
    print("  fill_alignment censoring: OK")


# ------------------------------- loop tests ---------------------------------


def test_self_speculation():
    """Draft == target greedy -> zero rejections, zero events, and the output
    equals the pure greedy rollout."""
    prompt = [1, 2, 3]
    f = make_content_target()
    perfect = lambda committed, g, pl: tc.greedy_rollout(committed, f, g, {EOS})[len(committed):]
    res = run(f, perfect, prompt)
    ref = tc.greedy_rollout(prompt, f, 200, {EOS})
    n = min(len(res.committed), len(ref))
    assert res.committed[:n] == ref[:n], "self-spec output != greedy rollout"
    assert not res.events and res.n_branch_passes == 0
    assert res.total_accepted + res.n_cycles == len(res.committed) - len(prompt)
    print(f"  self-speculation: OK ({res.n_cycles} cycles, 0 events)")


def test_output_invariance_and_events(name, target_fn, **kw):
    """CORE INVARIANT: whatever the draft does, committed output must equal the
    pure greedy rollout of the target. Also sanity-checks event fields."""
    prompt = [1, 2, 3]
    draft = make_flipping_draft(target_fn, flip_every=11, flip_offset=2)
    res = run(target_fn, draft, prompt, **kw)
    ref = tc.greedy_rollout(prompt, target_fn, 200, {EOS})
    n = min(len(res.committed), len(ref))
    assert res.committed[:n] == ref[:n], f"[{name}] output diverged from greedy rollout"
    assert res.events, f"[{name}] expected events (draft flips every 11 tokens)"
    for e in res.events:
        assert 0 <= e.L_survive <= e.m
        assert e.L_fresh is not None and 0 <= e.L_fresh <= e.m
        assert e.rejected != e.correction
        assert e.L_bridge >= e.L_survive, "bridge must dominate offset-0 by construction"
        assert e.lcs_len >= e.L_survive
        assert e.block_len > e.a >= 0
        assert e.m == e.block_len - e.a - 1
    print(f"  {name}: OK ({len(res.events)} events, output == greedy rollout)")
    return res


def test_free_survival_matches_branch_pass():
    """THE claim behind the rewrite: L_survive derived from the realized
    continuation equals L_survive measured with a dedicated branch pass, so the
    branch pass -- one extra target forward per event -- is pure waste."""
    prompt = [1, 2, 3]
    checked = 0
    for name, f in (("positional", make_positional_target(3)), ("content", make_content_target())):
        draft = make_flipping_draft(f, flip_every=11, flip_offset=2)
        res = run(f, draft, prompt, branch_verify=True)
        assert res.n_branch_passes == len(res.events) + res.n_dropped_events
        for e in res.events:
            if e.L_survive_censored:
                assert e.L_survive <= e.L_survive_branch
                continue
            assert e.L_survive == e.L_survive_branch, (
                f"[{name}] free={e.L_survive} branch={e.L_survive_branch} at step {e.step}")
            checked += 1
    assert checked >= 5, f"only {checked} uncensored comparisons; test is too weak"
    print(f"  free survival == branch pass: OK ({checked} uncensored events)")


def test_positional_survival():
    """Position-only target: suffix survives exactly until the next flipped
    token inside it (the correction cannot break anything)."""
    prompt = [1, 2, 3]
    f = make_positional_target(len(prompt))
    res = test_output_invariance_and_events("positional-target", f)
    for e in res.events:
        start_g = e.committed_len - len(prompt)  # gen-index of suffix[0]
        expected = e.m
        for i in range(e.m):
            if (start_g + i) % 11 == 2:  # that token was flipped
                expected = i
                break
        if e.L_survive_censored:
            assert e.L_survive <= expected
        else:
            assert e.L_survive == expected, (
                f"positional survival wrong: got {e.L_survive}, expected {expected}")
    print("  positional survival formula: OK")


def test_content_zero_survival():
    """Content-dependent target: the correction changes the continuation, so
    every recycled suffix must die immediately (L_survive == 0) unless the
    corrected token coincidentally regenerates the same next token (hash
    collision -- allowed, so we assert 'almost all zero')."""
    prompt = [1, 2, 3]
    f = make_content_target()
    res = test_output_invariance_and_events("content-target", f)
    zeros = sum(1 for e in res.events if e.L_survive == 0)
    assert zeros >= len(res.events) - 1, "content target should kill suffixes"
    print(f"  content zero-survival: OK ({zeros}/{len(res.events)} events at 0)")


def test_paranoid_hook_fires_per_cycle():
    """Schema 1 incremented its counter once per PROMPT, so the guide's own
    smoke command (--n 5 --paranoid 10) never ran a single check."""
    prompt = [1, 2, 3]
    f = make_content_target()
    draft = make_flipping_draft(f, flip_every=11, flip_offset=2)
    res = run(f, draft, prompt, paranoid_every=2)
    assert res.n_paranoid_checks == res.n_cycles // 2, (
        f"expected {res.n_cycles // 2} checks, got {res.n_paranoid_checks}")
    assert res.n_paranoid_checks > 0
    assert res.n_paranoid_mismatches == 0
    print(f"  paranoid hook fires per cycle: OK ({res.n_paranoid_checks} checks)")


if __name__ == "__main__":
    print("running core tests (no GPU needed)...")
    test_prefix_match_censoring()
    test_best_bridge()
    test_longest_common_run()
    test_fill_alignment_detects_bridge()
    test_fill_alignment_censoring()
    test_self_speculation()
    test_positional_survival()
    test_content_zero_survival()
    test_free_survival_matches_branch_pass()
    test_paranoid_hook_fires_per_cycle()
    print("ALL CORE TESTS PASSED")
