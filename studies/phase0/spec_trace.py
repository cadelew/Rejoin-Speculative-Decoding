"""spec_trace.py — Phase 0 suffix-survival study with real HF models (schema 2).

Runs greedy speculative decoding (target verifies draft blocks in parallel)
and, at every rejection with a non-empty escrowed suffix, records:

  L_survive : acceptance length of the escrow attached at offset 0
  L_bridge  : acceptance length of the escrow attached after a k-token bridge
              of the target's own continuation (k <= 8), with the winning k
  lcs       : longest contiguous run shared by the escrow and the realized
              continuation, with its offset in each
  L_fresh   : acceptance length of the next FRESH draft from the same corrected
              prefix, capped at the escrow length

All four are free -- committed tokens are target-greedy, so the committed tail
after an event is the target's greedy continuation from the corrected prefix.
Schema 1 spent one branch pass per event recomputing L_survive and then threw
away the suffix token ids, which is why it could not answer the bridge question
offline. `--branch-verify` re-enables the branch pass purely as a cross-check.

Usage (see README.md):
  python test_core.py                                     # CPU logic tests first
  python spec_trace.py --self-test                        # plumbing test on GPU
  python spec_trace.py --domain code --n 5 --paranoid 5 \
      --branch-verify --check-draft-cache 5 --out traces/smoke.jsonl
  python spec_trace.py --domain code --n 200 --out traces/code.jsonl
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import trace_core as tc


# ----------------------------- target session ------------------------------


class HFTargetSession(tc.TargetSession):
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.cache = DynamicCache()
        self.cache_len = 0

    def _forward(self, ids):
        input_ids = torch.tensor([ids], device=self.device)
        attn = torch.ones(
            (1, self.cache_len + len(ids)), device=self.device, dtype=torch.long
        )
        # Passing cache_position explicitly removes any reliance on HF inferring
        # position from a cache that we crop behind its back.
        cache_position = torch.arange(
            self.cache_len, self.cache_len + len(ids), device=self.device
        )
        out = self.model(
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=cache_position,
        )
        self.cache = out.past_key_values
        self.cache_len += len(ids)
        return out.logits[0].float()  # [len(ids), vocab]

    def prefill(self, ids):
        self.cache = DynamicCache()
        self.cache_len = 0
        if ids:
            self._forward(ids)

    def verify(self, last, block):
        logits = self._forward([last] + list(block))  # rows 0..k
        greedy = logits.argmax(-1).tolist()
        tg, bonus = greedy[: len(block)], greedy[len(block)]
        # Per-position diagnostics (row j scores block[j]).
        rows = logits[: len(block)]
        logp = torch.log_softmax(rows, dim=-1)
        p = logp.exp()
        ent = (-(p * logp).sum(-1)).tolist()
        blk = torch.tensor(list(block), device=rows.device).unsqueeze(-1)
        p_draft = p.gather(-1, blk).squeeze(-1).tolist()
        p_top1 = p.max(-1).values.tolist()
        return tg, bonus, {"entropies": ent, "p_draft": p_draft, "p_top1": p_top1}

    def branch_verify(self, last, block):
        pre = self.cache_len
        logits = self._forward([last] + list(block))
        tg = logits.argmax(-1).tolist()[: len(block)]
        self.crop_to(pre)
        return tg

    def crop_to(self, n):
        self.cache.crop(n)
        self.cache_len = n

    @torch.inference_mode()
    def paranoid_check(self, committed):
        """Recompute the next-token prediction WITHOUT the cache and compare.

        Called every N cycles from inside the decode loop (schema 1 called this
        once per prompt, so `--n 5 --paranoid 10` never fired at all).
        """
        ids = torch.tensor([committed], device=self.device)
        full = self.model(input_ids=ids, use_cache=False).logits[0, -1].float()
        g_full = int(full.argmax())
        g_cached = self.branch_verify(committed[-1], [g_full])[0]
        return g_full == g_cached


# --------------------------------- draft -----------------------------------


class HFDraftSession:
    """Greedy drafting with a persistent KV cache.

    `committed` is append-only within a prompt, so the cache never needs to be
    rewound past the committed prefix -- only trimmed of the speculative tail
    it just produced. Schema 1 called `model.generate` from scratch every
    cycle, re-prefilling the whole prompt each time; on a 200-prompt run that
    dominates wall clock.
    """

    def __init__(self, model, device, eos_ids):
        self.model = model
        self.device = device
        self.eos = set(eos_ids)
        self.cache = DynamicCache()
        self.cache_len = 0

    def reset(self):
        self.cache = DynamicCache()
        self.cache_len = 0

    def _step(self, ids):
        input_ids = torch.tensor([ids], device=self.device)
        attn = torch.ones(
            (1, self.cache_len + len(ids)), device=self.device, dtype=torch.long
        )
        cache_position = torch.arange(
            self.cache_len, self.cache_len + len(ids), device=self.device
        )
        out = self.model(
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=cache_position,
        )
        self.cache = out.past_key_values
        self.cache_len += len(ids)
        return out.logits[0, -1]

    def _crop(self, n):
        if n < self.cache_len:
            self.cache.crop(n)
            self.cache_len = n

    @torch.inference_mode()
    def logits_at(self, committed, agreed):
        """Cached-path logits for the position after committed + agreed.

        Replays exactly what `draft` does, so the result is the vector that
        actually chose the token -- which is what a cache bug would corrupt.
        """
        base = len(committed)
        self._crop(base)
        new = committed[self.cache_len :]
        if not new:
            self._crop(base - 1)
            new = committed[base - 1 :]
        logits = self._step(new)
        for t in agreed:
            logits = self._step([t])
        self._crop(base)
        return logits.float()

    @torch.inference_mode()
    def draft(self, committed, gamma):
        base = len(committed)
        self._crop(base)
        new = committed[self.cache_len :]
        if not new:  # cache already covers everything; back off one position
            self._crop(base - 1)
            new = committed[base - 1 :]
        logits = self._step(new)
        out = []
        for _ in range(gamma):
            t = int(logits.argmax())
            out.append(t)
            if t in self.eos or len(out) == gamma:
                break
            logits = self._step([t])
        self._crop(base)
        return out


@torch.inference_mode()
def draft_divergence_report(model, draft_session, device, committed, agreed):
    """Why the cached and uncached draft paths chose different tokens.

    An argmax flip is only a symptom. The cause is measurable: compare the two
    paths' full logit vectors at the divergence point.

      max_abs_delta  largest disagreement between the two logit vectors
      ulps           that delta in bf16 units-at-this-magnitude (mantissa 2^-8)
      logit_gap      the top-2 margin the flip had to cross

    Chunked prefill (cache path) and full prefill (`generate`) reduce in
    different orders, so bf16 logits differ in their last bits -- a handful of
    ULPs. If max_abs_delta is at that scale AND is large enough to cross
    logit_gap, the flip is fully explained by numerics. A cache bug instead
    corrupts attention outright, putting the delta thousands of ULPs out.
    """
    seq = list(committed) + list(agreed)
    ids = torch.tensor([seq], device=device)
    full = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    cached = draft_session.logits_at(committed, agreed)

    delta = float((cached - full).abs().max())
    scale = float(full.abs().max())
    ulp = scale * 2**-8  # bf16 spacing at this magnitude
    top = full.topk(2)
    probs = torch.softmax(full, dim=-1)
    gap = float(top.values[0] - top.values[1])
    return {
        "logit_gap": gap,
        "p1": float(probs[top.indices[0]]),
        "p2": float(probs[top.indices[1]]),
        "ids": [int(x) for x in top.indices],
        "max_abs_delta": delta,
        "ulps": delta / max(ulp, 1e-12),
        "explains_flip": delta >= gap,
        # A few dozen ULPs is reduction-order noise. Thousands is a broken cache.
        "numerical": delta / max(ulp, 1e-12) < 100.0,
    }


def make_uncached_draft_fn(model, device, eos_ids):
    """Reference drafting path (no draft cache). Used by --check-draft-cache."""
    eos_list = sorted(eos_ids)

    @torch.inference_mode()
    def draft_fn(committed, gamma):
        ids = torch.tensor([committed], device=device)
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=gamma,
            do_sample=False,
            eos_token_id=eos_list,
            pad_token_id=eos_list[0],
        )
        return out[0, len(committed) :].tolist()

    return draft_fn


# -------------------------------- datasets ---------------------------------


def load_prompts(domain, n):
    """Return a list of (user_text, assistant_prefill) pairs.

    The prefill is committed before decoding starts. For code this pins the
    output format for BOTH models: schema 1 let the 0.6B open a ```python fence
    and re-emit the docstring while the target went straight to the body, which
    produced 27% of all events in the pilot -- every one of them a stylistic
    prior mismatch landing in the maximum-escrow bucket.
    """
    from datasets import load_dataset

    if domain == "code":
        ds = load_dataset("openai_humaneval", split="test")
        # The signature + docstring go into the prefill, so generation starts at
        # the function body and neither the fence nor the docstring can diverge.
        return [
            (
                "Complete the following Python function.\n\n"
                "```python\n" + r["prompt"] + "```",
                "```python\n" + r["prompt"],
            )
            for r in ds
        ][:n]
    if domain == "math":
        ds = load_dataset("gsm8k", "main", split="test")
        return [
            (r["question"] + "\nThink step by step and give the final answer.", "")
            for r in ds
        ][:n]
    if domain == "chat":
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        return [(r["instruction"], "") for r in ds if not r["input"]][:n]
    raise ValueError(f"unknown domain {domain!r} (use code|math|chat, or add a loader)")


def build_prompt_ids(tok, user_text, prefill="", thinking=False):
    msgs = [{"role": "user", "content": user_text}]
    try:  # Qwen3 exposes thinking mode through the template
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, enable_thinking=thinking
        )
    except TypeError:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    ids = list(ids)
    if prefill:
        ids += tok.encode(prefill, add_special_tokens=False)
    return ids


# ---------------------------------- meta -----------------------------------


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def model_revision(model):
    return getattr(getattr(model, "config", None), "_commit_hash", None)


def dtype_kwarg(dtype):
    """Transformers renamed `torch_dtype` to `dtype` in 4.56."""
    try:
        parts = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    except ValueError:
        parts = (0, 0)
    return {"dtype": dtype} if parts >= (4, 56) else {"torch_dtype": dtype}


# ---------------------------------- main -----------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--domain", default="code", help="code | math | chat")
    ap.add_argument("--n", type=int, default=100, help="number of prompts")
    ap.add_argument("--start", type=int, default=0,
                    help="skip the first N prompts (resume after a disconnect; "
                         "prompt_id stays absolute so traces concatenate cleanly)")
    ap.add_argument("--gamma", type=int, default=32, help="draft block length")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--out", default=None, help="JSONL output path")
    ap.add_argument("--load-in-8bit", action="store_true",
                    help="quantize the TARGET (redefines the target; internally consistent)")
    ap.add_argument("--paranoid", type=int, default=0,
                    help="every N CYCLES, cross-check cached vs cache-free logits")
    ap.add_argument("--branch-verify", action="store_true",
                    help="also measure L_survive with a branch pass and assert it "
                         "matches the free measurement (costs 1 pass per event)")
    ap.add_argument("--check-draft-cache", type=int, default=0,
                    help="for the first N cycles of each prompt, recompute the draft "
                         "block without the draft cache and assert equality")
    ap.add_argument("--no-draft-cache", action="store_true",
                    help="use the slow uncached draft path")
    ap.add_argument("--thinking", action="store_true",
                    help="enable Qwen3 thinking mode (the reasoning-chain ablation)")
    ap.add_argument("--no-prefill", action="store_true",
                    help="disable the assistant prefill (reproduces the schema-1 "
                         "format-divergence artifact)")
    ap.add_argument("--self-test", action="store_true",
                    help="use the draft model as BOTH draft and target; expect ~zero rejections")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float16
    print(f"device={device} dtype={dtype}")

    target_name = args.draft if args.self_test else args.target
    tok = AutoTokenizer.from_pretrained(target_name)
    dtok = AutoTokenizer.from_pretrained(args.draft)
    assert tok.get_vocab() == dtok.get_vocab(), (
        "draft and target MUST share an identical tokenizer (token-ID comparison "
        "is meaningless otherwise). Qwen3-0.6B/8B share one; mixed families don't."
    )

    kw = dict(device_map=device, **dtype_kwarg(dtype))
    if args.load_in_8bit and not args.self_test:
        kw = dict(device_map=device, load_in_8bit=True)
    print(f"loading target {target_name} ...")
    target = AutoModelForCausalLM.from_pretrained(target_name, **kw).eval()
    print(f"loading draft {args.draft} ...")
    draft = (target if args.self_test else AutoModelForCausalLM.from_pretrained(
        args.draft, device_map=device, **dtype_kwarg(dtype)).eval())

    eos_ids = {tok.eos_token_id}
    ge = target.generation_config.eos_token_id
    eos_ids |= set(ge) if isinstance(ge, (list, tuple)) else ({ge} if ge is not None else set())
    eos_ids.discard(None)

    session = HFTargetSession(target, device)
    draft_session = HFDraftSession(draft, device, eos_ids)
    uncached_draft_fn = make_uncached_draft_fn(draft, device, eos_ids)

    if args.self_test:
        pairs = [("Write a haiku about the ocean.", ""), ("Explain what a hash map is.", "")]
    else:
        pairs = load_prompts(args.domain, args.n)
    if args.start:
        pairs = pairs[args.start:]
        print(f"resuming at prompt {args.start} ({len(pairs)} remaining)")

    out_path = args.out or f"traces/{args.domain}.jsonl"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    f = open(out_path, "a")

    f.write(json.dumps({
        "type": "meta", "schema": tc.SCHEMA_VERSION, "domain": args.domain,
        "target": target_name, "draft": args.draft,
        "target_revision": model_revision(target), "draft_revision": model_revision(draft),
        "device": device, "dtype": str(dtype),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "gamma": args.gamma, "max_new": args.max_new, "n_prompts": len(pairs),
        "start": args.start,
        "thinking": args.thinking, "prefill": not args.no_prefill,
        "branch_verify": args.branch_verify, "draft_cache": not args.no_draft_cache,
        "load_in_8bit": args.load_in_8bit,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "git_sha": git_sha(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }) + "\n")
    f.flush()

    tot_events = tot_cycles = tot_tokens = 0
    tot_paranoid = tot_paranoid_bad = 0
    survive_disagreements = draft_cache_disagreements = 0
    draft_cache_checks = draft_cache_near_ties = 0
    t0 = time.time()

    with torch.inference_mode():
        for pi, (text, prefill) in enumerate(pairs, start=args.start):
            if args.no_prefill:
                prefill = ""
            ids = build_prompt_ids(tok, text, prefill, thinking=args.thinking)
            draft_session.reset()

            state = {"cycle": 0, "draft_bad": 0, "draft_checks": 0, "draft_near_tie": 0}

            def draft_fn(committed, gamma, _s=state, _pi=pi):
                _s["cycle"] += 1
                if args.no_draft_cache:
                    return uncached_draft_fn(committed, gamma)
                block = draft_session.draft(committed, gamma)
                if args.check_draft_cache and _s["cycle"] <= args.check_draft_cache:
                    _s["draft_checks"] += 1
                    ref = uncached_draft_fn(committed, gamma)
                    if block != ref:
                        _s["draft_bad"] += 1
                        i = tc.first_mismatch(block, ref)
                        d = draft_divergence_report(
                            draft, draft_session, device, committed, block[:i])
                        benign = d["numerical"] and d["explains_flip"]
                        _s["draft_near_tie"] += int(benign)
                        print(f"  [draft-cache] MISMATCH prompt {_pi} cycle {_s['cycle']} "
                              f"at draft index {i}")
                        print(f"      cached   {tok.decode(block[i:i+4])!r}")
                        print(f"      uncached {tok.decode(ref[i:i+4])!r}")
                        print(f"      top-2 {[tok.decode([t]) for t in d['ids']]}  "
                              f"gap {d['logit_gap']:.4f}  p {d['p1']:.4f} vs {d['p2']:.4f}")
                        print(f"      cached-vs-uncached logits: max delta "
                              f"{d['max_abs_delta']:.4f} = {d['ulps']:.1f} bf16 ULP"
                              f"{'s' if d['ulps'] >= 2 else ''}, "
                              f"{'crosses' if d['explains_flip'] else 'does NOT cross'} the gap")
                        print("      -> " + (
                            "NUMERICAL (reduction-order noise; cannot bias the study)"
                            if benign else
                            "UNEXPLAINED — investigate the draft cache"))
                return block

            res = tc.run_prompt(
                ids, draft_fn, session, args.gamma, args.max_new, eos_ids,
                branch_verify=args.branch_verify, paranoid_every=args.paranoid,
            )
            draft_cache_disagreements += state["draft_bad"]
            draft_cache_checks += state["draft_checks"]
            draft_cache_near_ties += state["draft_near_tie"]
            new_tokens = len(res.committed) - res.prompt_len

            for e in res.events:
                if e.L_survive_branch is not None and not e.L_survive_censored:
                    if e.L_survive_branch != e.L_survive:
                        survive_disagreements += 1
                        print(f"  [survive] MISMATCH prompt {pi} step {e.step}: "
                              f"free={e.L_survive} branch={e.L_survive_branch}")
                f.write(json.dumps({
                    "type": "event", "schema": tc.SCHEMA_VERSION,
                    "domain": args.domain, "prompt_id": pi,
                    "step": e.step, "committed_len": e.committed_len,
                    "a": e.a, "block_len": e.block_len, "m": e.m,
                    "L_survive": e.L_survive, "L_survive_censored": e.L_survive_censored,
                    "L_survive_branch": e.L_survive_branch,
                    "L_bridge": e.L_bridge, "bridge_k": e.bridge_k,
                    "lcs_len": e.lcs_len, "lcs_i": e.lcs_i, "lcs_j": e.lcs_j,
                    "L_fresh": e.L_fresh, "L_fresh_censored": e.L_fresh_censored,
                    "delta": e.L_survive - e.L_fresh,
                    "delta_bridge": e.L_bridge - e.L_fresh,
                    # Raw ids so every alignment question stays answerable offline.
                    "suffix_ids": e.suffix, "realized_ids": e.realized,
                    "rejected": tok.decode([e.rejected]),
                    "correction": tok.decode([e.correction]),
                    "suffix_text": tok.decode(e.suffix),
                    "realized_text": tok.decode(e.realized),
                    "entropy_at_rejection": e.extras.get("entropy_at_rejection"),
                    "p_rejected": e.extras.get("p_rejected"),
                    "p_correction": e.extras.get("p_correction"),
                }) + "\n")

            f.write(json.dumps({
                "type": "summary", "schema": tc.SCHEMA_VERSION,
                "domain": args.domain, "prompt_id": pi,
                "new_tokens": new_tokens, "cycles": res.n_cycles,
                "main_passes": res.n_main_passes, "branch_passes": res.n_branch_passes,
                "tokens_per_main_pass": round(new_tokens / max(res.n_main_passes, 1), 3),
                "events": len(res.events), "dropped_events": res.n_dropped_events,
                "paranoid_checks": res.n_paranoid_checks,
                "paranoid_mismatches": res.n_paranoid_mismatches,
            }) + "\n")
            f.flush()

            tot_events += len(res.events)
            tot_cycles += res.n_cycles
            tot_tokens += new_tokens
            tot_paranoid += res.n_paranoid_checks
            tot_paranoid_bad += res.n_paranoid_mismatches

            if args.self_test:
                n_rej = len(res.events) + res.n_dropped_events
                print(f"  [self-test] prompt {pi}: {new_tokens} tokens, {res.n_cycles} cycles, "
                      f"{n_rej} rejection events (expect ~0; a few near-ties are OK)")
                print("  output:", tok.decode(res.committed[res.prompt_len:])[:200].replace("\n", " "))
            elif (pi + 1) % 5 == 0:
                dt = time.time() - t0
                print(f"[{pi+1}/{len(pairs)}] {tot_tokens} tokens, {tot_events} events, "
                      f"{tot_tokens/max(dt,1e-9):.1f} tok/s wall")
    f.close()

    print(f"\ndone. traces -> {out_path}  |  events: {tot_events}  "
          f"tokens/main-pass: {tot_tokens/max(tot_cycles,1):.2f}")
    if tot_paranoid:
        print(f"paranoid: {tot_paranoid_bad}/{tot_paranoid} cached-vs-uncached mismatches "
              f"({'OK' if tot_paranoid_bad <= tot_paranoid * 0.02 else 'INVESTIGATE — likely a cache bug'})")
    if args.branch_verify:
        print(f"survival cross-check: {survive_disagreements} disagreements "
              f"({'OK' if survive_disagreements == 0 else 'INVESTIGATE'})")
    if args.check_draft_cache:
        hard = draft_cache_disagreements - draft_cache_near_ties
        # Numerically-explained flips cannot bias the study: the draft only
        # proposes, every committed token is target-verified greedy, and both
        # survival and alignment are measured against the target. They change
        # which escrows exist, not what is measured.
        print(f"draft-cache cross-check: {draft_cache_disagreements}/{draft_cache_checks} "
              f"cycles disagree ({draft_cache_near_ties} numerical, {hard} unexplained) "
              f"({'OK' if hard == 0 else 'INVESTIGATE'})")
    print("next: python analyze_traces.py " + out_path)


if __name__ == "__main__":
    main()
