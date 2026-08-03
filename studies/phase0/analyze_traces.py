"""analyze_traces.py — turn Phase 0 JSONL traces into the go/no-go numbers.

Usage: python analyze_traces.py traces/code.jsonl [traces/math.jsonl ...]

Reads schema 2 traces (with escrow alignment). Schema 1 traces still load, but
every alignment statistic is skipped because that schema discarded the suffix
token ids and never recorded the target's realized continuation.
"""

import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict

import trace_core as tc

KS = [1, 2, 4, 8, 16, 32]

# Cost of one extra target pass per event, as a fraction of a main pass.
#   sequential: a second full forward                              -> 1.00
#   batched:    escrow and fresh draft verified as a batch of 2
#               sharing one KV prefix; decode is memory-bound, so
#               the second row is close to free                    -> ~0.12
COST_MODELS = [("sequential branch pass", 1.00), ("batched (batch-2, shared prefix)", 0.12)]


def load(paths):
    events, summaries, metas = defaultdict(list), defaultdict(list), []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            kind = r.get("type")
            if kind == "meta":
                metas.append(r)
            elif kind == "event":
                events[r["domain"]].append(r)
            elif kind == "summary":
                summaries[r["domain"]].append(r)
    return events, summaries, metas


def classify(e):
    rej, cor = e.get("rejected", ""), e.get("correction", "")
    if rej.strip() == "" and cor.strip() == "":
        return "whitespace"
    if any(m in rej or m in cor for m in ("```", '"""', "'''")):
        return "format"
    if rej.strip().isdigit() or cor.strip().isdigit():
        return "numeric"
    return "semantic"


def null_alignment(ev, seed=0):
    """Chance-level alignment: pair each escrow with ANOTHER event's continuation.

    Python boilerplate repeats itself -- `    return result`, `):\\n        `,
    `for i in range(len(` -- so a shared 4-token run is not by itself evidence
    that the escrow's plan survived. This is the null: same escrows, same
    continuations, wrong pairings. Any observed statistic must clear it before
    it means anything.
    """
    n = len(ev)
    if n < 2:
        return None
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    for i in range(n):  # derange: nobody paired with themselves
        if order[i] == i:
            j = rng.choice([k for k in range(n) if k != i])
            order[i], order[j] = order[j], order[i]

    bridges, runs = [], []
    for i, j in enumerate(order):
        suf = ev[i].get("suffix_ids") or []
        real = ev[j].get("realized_ids") or []
        if not suf or not real:
            continue
        bridges.append(tc.best_bridge(suf, real)[0])
        runs.append(tc.longest_common_run(suf, real)[0])
    return (bridges, runs) if bridges else None


def curve(vals, ks=KS):
    n = max(len(vals), 1)
    return "  ".join(f">={k}:{sum(v >= k for v in vals)/n:5.1%}" for k in ks)


def band(label, vals, width=7):
    if not vals:
        return f"  {label:<24} (none)"
    return (f"  {label:<24} mean {st.mean(vals):>{width}.2f}   "
            f"median {st.median(vals):>{width}.1f}   max {max(vals):>4}")


def report_domain(dom, ev, summ, has_align):
    n = len(ev)
    Ls = [e["L_survive"] for e in ev]
    Lf = [e["L_fresh"] for e in ev]
    d = [a - b for a, b in zip(Ls, Lf)]
    wins = sum(x > 0 for x in d)
    ties = sum(x == 0 for x in d)

    print(f"\n{'='*74}\n== domain: {dom}  ({n} events, {len(summ)} prompts) ==\n{'='*74}")

    cens = sum(1 for e in ev if e.get("L_survive_censored"))
    if cens:
        print(f"  NOTE {cens}/{n} ({cens/n:.0%}) survival measurements are right-censored "
              f"(prompt ended before the escrow could be fully compared).")

    print("\n-- escrow geometry --")
    print(band("escrow length m", [e["m"] for e in ev]))
    if "a" in ev[0]:
        A = [e["a"] for e in ev]
        print(band("mismatch index a", A))
        print(f"  {'a == 0 (immediate)':<24} {sum(x == 0 for x in A)/n:5.1%}   "
              f"a <= 2: {sum(x <= 2 for x in A)/n:5.1%}")

    print("\n-- offset-0 attachment (what schema 1 measured) --")
    print(band("L_survive", Ls))
    print(band("L_fresh (redraft)", Lf))
    print(band("delta = survive - fresh", d))
    print(f"  {'recycled wins':<24} {wins/n:5.1%}   ties {ties/n:5.1%}   "
          f"loses {(n-wins-ties)/n:5.1%}")
    print("  survival curve         " + curve(Ls))

    gains = {"offset-0": st.mean([max(0, x) for x in d])}

    if has_align:
        Lb = [e["L_bridge"] for e in ev]
        db = [a - b for a, b in zip(Lb, Lf)]
        bwins = sum(x > 0 for x in db)
        print("\n-- best-bridge attachment (free oracle; schema 1 could not see this) --")
        print(band("L_bridge", Lb))
        print(band("delta_bridge", db))
        print(f"  {'bridge wins':<24} {bwins/n:5.1%}")
        print("  bridge curve           " + curve(Lb))
        ks = Counter(e["bridge_k"] for e in ev if e["L_bridge"] > 0)
        if ks:
            tot = sum(ks.values())
            print("  winning bridge length  " + "  ".join(
                f"k={k}:{c/tot:4.0%}" for k, c in sorted(ks.items())))
            print(f"  headroom over offset-0 {st.mean(Lb) - st.mean(Ls):+.2f} tokens/event")
        gains["best-bridge"] = st.mean([max(0, x) for x in db])

        # Front-trim: escrow[j:] attached immediately. Recomputed here rather
        # than in the trace because raw ids are logged -- no GPU re-run needed.
        raw = [e for e in ev if e.get("suffix_ids") and e.get("realized_ids") is not None]
        if len(raw) == len(ev):
            trims = [tc.best_trim(e["suffix_ids"], e["realized_ids"]) for e in ev]
            Lt = [t[0] for t in trims]
            dt = [a - b for a, b in zip(Lt, Lf)]
            print("\n-- front-trim attachment (IMPLEMENTABLE: batched candidate tree) --")
            print(band("L_trim", Lt))
            print(band("delta_trim", dt))
            print(f"  {'trim wins':<24} {sum(x > 0 for x in dt)/n:5.1%}")
            print("  trim curve             " + curve(Lt))
            tj = Counter(t[1] for t in trims if t[0] > 0)
            if tj:
                tot = sum(tj.values())
                print("  winning trim length    " + "  ".join(
                    f"j={j}:{c/tot:4.0%}" for j, c in sorted(tj.items())))
            gains["front-trim"] = st.mean([max(0, x) for x in dt])

        Lc = [e["lcs_len"] for e in ev]
        print("\n-- longest shared run (slack on BOTH sides; upper bound) --")
        print(band("lcs_len", Lc))
        print("  lcs curve              " + curve(Lc))
        li = Counter(e["lcs_i"] for e in ev if e["lcs_len"] >= 4)
        lj = Counter(e["lcs_j"] for e in ev if e["lcs_len"] >= 4)
        if li:
            tot = sum(li.values())
            print(f"  where the run starts (events with lcs >= 4, n={tot}):")
            print("    in the escrow  (trim) " + "  ".join(
                f"i={i}:{c/tot:4.0%}" for i, c in sorted(li.items())[:6]))
            print("    in the target (bridge) " + "  ".join(
                f"j={j}:{c/tot:4.0%}" for j, c in sorted(lj.items())[:6]))
            only_trim = sum(c for j, c in lj.items() if j == 0) / tot
            print(f"    j==0 (front trim alone suffices): {only_trim:5.1%}")
        # Ceiling: both-sided oracle, and it does not pay for the bridge tokens
        # it assumes. Real policies cannot reach this; it bounds the addressable
        # market for a repair model that predicts (trim, bridge).
        gains["two-sided CEIL"] = st.mean([max(0, a - b) for a, b in zip(Lc, Lf)])

        null = null_alignment(ev)
        if null:
            nb, nr = null
            print("\n-- chance control (escrows paired with the WRONG continuation) --")
            print(band("null L_bridge", nb))
            print(band("null lcs_len", nr))
            print(f"  {'signal over chance':<24} L_bridge {st.mean(Lb) - st.mean(nb):+.2f}   "
                  f"lcs {st.mean(Lc) - st.mean(nr):+.2f} tokens/event")
            print(f"  {'lcs curve (null)':<24} " + curve(nr))
            if st.mean(Lc) - st.mean(nr) < 1.0:
                print("  WARNING: lcs barely clears chance. Shared runs here are "
                      "boilerplate n-gram collisions, not surviving plan.")

    print("\n-- event taxonomy --")
    tax = Counter(classify(e) for e in ev)
    for k, c in tax.most_common():
        sub = [e for e in ev if classify(e) == k]
        print(f"  {k:<12} {c:>4} ({c/n:5.1%})   mean L_survive {st.mean([x['L_survive'] for x in sub]):5.2f}"
              + (f"   mean L_bridge {st.mean([x['L_bridge'] for x in sub]):5.2f}" if has_align else ""))

    ents = [e["entropy_at_rejection"] for e in ev if e.get("entropy_at_rejection") is not None]
    if ents:
        print("\n-- target confidence at the rejection --")
        lo = [e for e in ev if (e.get("entropy_at_rejection") or 0) < 0.05]
        hi = [e for e in ev if (e.get("entropy_at_rejection") or 0) >= 0.05]
        print(f"  median entropy {st.median(ents):.3g}   frac < 0.05: {len(lo)/n:.1%}")
        for name, grp in (("low-entropy (<0.05)", lo), ("high-entropy (>=0.05)", hi)):
            if grp:
                print(f"  {name:<24} n={len(grp):>4}  mean L_survive "
                      f"{st.mean([e['L_survive'] for e in grp]):5.2f}"
                      + (f"  mean L_bridge {st.mean([e['L_bridge'] for e in grp]):5.2f}"
                         if has_align else ""))
        pr = [e["p_rejected"] for e in ev if e.get("p_rejected") is not None]
        if pr:
            print(f"  target P(draft's rejected token): median {st.median(pr):.3g}   "
                  f"frac < 0.01 (blunder, not near-miss): {sum(x < 0.01 for x in pr)/len(pr):.1%}")

    # ------------------------------ economics ------------------------------
    if summ:
        N = sum(r["new_tokens"] for r in summ)
        P = sum(r["main_passes"] for r in summ)
        E = n
        T = N / max(P, 1)
        print("\n-- economics --")
        print(f"  baseline throughput    {T:.2f} tokens / target pass "
              f"({N} tokens, {P} passes)")
        print(f"  events per target pass {E/max(P,1):.2f}")
        print("  break-even gain per event, g_min = cost_factor x baseline throughput:")
        for label, c in COST_MODELS:
            g_min = c * T
            print(f"    {label:<34} g_min {g_min:6.2f} tok/event")
            for gname, g in gains.items():
                new = (N + g * E) / (P + c * E)
                verdict = "PAYS" if g > g_min else "does not pay"
                print(f"      recycle-if-better [{gname:<11}] g={g:5.2f}  "
                      f"-> {new:5.2f} tok/pass ({new/T - 1:+6.1%})  {verdict}")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python analyze_traces.py <trace.jsonl> [...]")
    events, summaries, metas = load(sys.argv[1:])

    if metas:
        print("-- runs --")
        for m in metas:
            print(f"  {m.get('started_utc','?')}  schema {m.get('schema')}  {m.get('domain')}  "
                  f"{m.get('target')} <- {m.get('draft')}  gamma={m.get('gamma')}  "
                  f"n={m.get('n_prompts')}  prefill={m.get('prefill')}  "
                  f"thinking={m.get('thinking')}  git={m.get('git_sha')}")
        gammas = {m.get("gamma") for m in metas}
        if len(gammas) > 1:
            print(f"  WARNING: mixed gamma values {sorted(gammas)} in one analysis; "
                  f"split the files before drawing conclusions.")
    bad = sum(r.get("paranoid_mismatches", 0) for s in summaries.values() for r in s)
    checks = sum(r.get("paranoid_checks", 0) for s in summaries.values() for r in s)
    if checks:
        print(f"  paranoid checks: {bad}/{checks} mismatches"
              + ("  <- INVESTIGATE, likely a cache bug" if bad > checks * 0.02 else ""))
    elif summaries:
        print("  WARNING: no paranoid checks were run; cache/alignment correctness is "
              "unvalidated for this trace. Re-run a smoke test with --paranoid 5.")

    for dom in sorted(events):
        ev = [e for e in events[dom] if e.get("L_fresh") is not None]
        if not ev:
            print(f"\n== {dom}: no completed events ==")
            continue
        has_align = all(e.get("L_bridge") is not None for e in ev)
        if not has_align:
            print(f"\n  NOTE {dom}: schema 1 trace — alignment statistics unavailable "
                  f"(suffix ids and realized continuation were not recorded).")
        report_domain(dom, ev, summaries.get(dom, []), has_align)

    print("""
======================================================================
-- interpretation --

  Read L_bridge FIRST. It is the mechanism test; L_survive is only the
  mechanism test restricted to the single worst attachment offset.

  L_bridge high, L_survive low
      The escrow's plan survives but resumes a few tokens later. Offset-0
      reattachment can never capture it. The bridge-length histogram is the
      addressable market for a repair model, and short winning k is the
      strongest possible case for a small infiller. Proceed to Phase 1 with
      bridge repair, NOT with direct reattachment.

  L_bridge ~ L_survive, both high, delta ~ 0
      Escrows survive but the draft redrafts them anyway. Only draft-cost
      savings remain (the FailFast result). Rejoin as designed is not worth
      building; write up the negative and release the traces.

  L_bridge ~ L_survive ~ 0
      Rejections are semantic pivots. No bridge model can help, because the
      escrow does not reappear in the target's continuation at any offset.
      The escrow premise fails. Publish the measurement and stop.

  Then read the economics block. `recycle-if-better [offset-0]` is the only
  implementable policy of the three -- a bridge policy needs the target's
  continuation to build the bridge, which is what you are trying to avoid
  computing. Treat every bridge number as an upper bound on what repair could
  buy, never as an achieved speedup.
======================================================================
""")


if __name__ == "__main__":
    main()
