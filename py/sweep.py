#!/usr/bin/env python3
"""
Self-report provenance sweep. Runs battery_v1.json across checkpoints.

Works in two modes:
  1. Local (preferred): reads checkpoints prefetched by download.py from./checkpoints/{model}/{revision}/ and never touches the network.
  2. Streaming: if a revision isn't found locally, downloads it to a temp dir, runs it, deletes it.

Streams one CSV row per measurement; resumable; --delete-after frees disk as it walks the local checkpoints (use once the run is final).

Usage:
  python sweep.py --model EleutherAI/pythia-1.4b --revisions step1000,step143000
  python sweep.py --model EleutherAI/pythia-410m --revisions local
  python sweep.py --model allenai/OLMo-2-0425-1B-Instruct --revisions main --chat

Requires: torch, transformers, huggingface_hub
"""

import argparse
import csv
import gc
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FIELDS = ["model", "revision", "block", "item_id", "frame", "variant", "measure", "value", "text"]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def step_num(name):
    return int(re.sub(r"\D", "", name) or 0)


# ---------------------------------------------------------------- scoring

def continuation_logprob(model, tok, context, continuation, device):
    """Sum of log P(continuation tokens | context), scored as the suffix of
    the jointly tokenized string, so multi-token candidates and merges are
    handled; returns NaN when the continuation merges into the context."""
    ctx_ids = tok(context, return_tensors="pt").input_ids.to(device)
    full_ids = tok(context + continuation, return_tensors="pt").input_ids.to(device)
    n_ctx = ctx_ids.shape[1]
    if full_ids.shape[1] <= n_ctx:
        return float("nan")
    with torch.no_grad():
        logits = model(full_ids).logits
    logprobs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = full_ids[:, 1:]
    lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return lp[0, n_ctx - 1:].sum().item()


def sequence_avg_logprob(model, tok, sentence, device):
    ids = tok(sentence, return_tensors="pt").input_ids.to(device)
    if ids.shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        logits = model(ids).logits
    logprobs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = ids[:, 1:]
    lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return lp.mean().item()


def normalized_candidate_probs(model, tok, context, candidates, device):
    lps = [continuation_logprob(model, tok, context, c, device) for c in candidates]
    if all(math.isnan(x) for x in lps):
        return lps
    z = torch.logsumexp(
        torch.tensor([x if not math.isnan(x) else -1e30 for x in lps]), dim=0
    ).item()
    return [math.exp(x - z) if not math.isnan(x) else float("nan") for x in lps]


def classify(text, rubric):
    t = text.lower()
    for label in ("DENY", "ASSERT", "HEDGE", "DEFLECT"):
        for pat in rubric[label]:
            if re.search(pat, t):
                return label
    return "OTHER"


def generate_k(model, tok, prompt, device, k, temp, max_new, chat):
    if chat and tok.chat_template:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt",
        ).to(device)
    else:
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    outs = []
    with torch.no_grad():
        for _ in range(k):
            out = model.generate(
                ids, do_sample=True, temperature=temp, max_new_tokens=max_new,
                pad_token_id=tok.eos_token_id,
            )
            outs.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
    return outs


# ---------------------------------------------------------------- blocks

def run_block1(model, tok, battery, device, writer, meta):
    for item in battery["block1_logit_items"]:
        probs = normalized_candidate_probs(model, tok, item["context"], item["candidates"], device)
        for cand, p in zip(item["candidates"], probs):
            writer.writerow(meta | {"block": 1, "item_id": item["id"], "frame": "",
                                    "variant": cand.strip(), "measure": "norm_prob",
                                    "value": p, "text": ""})


def run_block2(model, tok, battery, device, writer, meta, k, temp, max_new, chat):
    rubric = battery["rubric"]
    for prop in battery["block2_propositions"]:
        for fname, template in battery["block2_frames"].items():
            prompt = template.format(question=prop["question"],
                                     assertion=prop["assertion"],
                                     indirect=prop["indirect"])
            for i, completion in enumerate(
                    generate_k(model, tok, prompt, device, k, temp, max_new, chat)):
                writer.writerow(meta | {"block": 2, "item_id": prop["id"], "frame": fname,
                                        "variant": f"sample{i}",
                                        "measure": "class:" + classify(completion, rubric),
                                        "value": "",
                                        "text": completion.replace("\n", " ")[:300]})


def run_block3(model, tok, battery, device, writer, meta):
    for tr in battery["block3_triples"]:
        for role in ("mental_self", "control_self", "mental_other"):
            writer.writerow(meta | {"block": 3, "item_id": tr["id"], "frame": "",
                                    "variant": role, "measure": "avg_logprob",
                                    "value": sequence_avg_logprob(model, tok, tr[role], device),
                                    "text": ""})


def run_block4(model, tok, battery, device, writer, meta, k, temp, max_new):
    rubric = battery["rubric"]
    for item in battery["block4_chat_items"]:
        for i, completion in enumerate(
                generate_k(model, tok, item["prompt"], device, k, temp, max_new, chat=True)):
            writer.writerow(meta | {"block": 4, "item_id": item["id"], "frame": "chat",
                                    "variant": f"sample{i}",
                                    "measure": "class:" + classify(completion, rubric),
                                    "value": "",
                                    "text": completion.replace("\n", " ")[:300]})


# ---------------------------------------------------------------- main

def resolve(args):
    """Yield (revision, load_path_or_repo, is_local, local_dir)."""
    root = Path(args.ckpt_dir) / args.model.replace("/", "__")
    if args.revisions == "local":
        if not root.exists():
            sys.exit(f"no local checkpoints under {root}; run download.py first")
        revs = sorted((d.name for d in root.iterdir()
                       if (d / ".complete").exists()), key=step_num)
        if not revs:
            sys.exit(f"no completed checkpoints under {root}")
        for r in revs:
            yield r, str(root / r), True, root / r
    else:
        for r in [x.strip() for x in args.revisions.split(",")]:
            local = root / r
            if (local / ".complete").exists():
                yield r, str(local), True, local
            else:
                yield r, args.model, False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revisions", default="local",
                    help="'local' (all prefetched), 'main', or comma-separated list")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--battery", default="battery_v1.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--delete-after", action="store_true",
                    help="delete each local checkpoint once its rows are written")
    args = ap.parse_args()

    battery = json.loads(Path(args.battery).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype) if device == "cuda" else torch.float32

    out_path = Path(args.out or (args.model.split("/")[-1] + "_sweep.csv"))
    new_file = not out_path.exists()
    done = set()
    if not new_file:
        with out_path.open() as f:
            done = {row["revision"] for row in csv.DictReader(f)}

    log(f"{args.model}: device={device}, out={out_path}")

    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        for rev, source, is_local, local_dir in resolve(args):
            if rev in done:
                log(f"  {rev}: already in CSV, skipping")
                continue
            tmp = None
            try:
                if is_local:
                    log(f"  {rev}: loading from disk")
                    tok = AutoTokenizer.from_pretrained(source)
                    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype)
                else:
                    log(f"  {rev}: not prefetched, streaming from hub")
                    tmp = tempfile.mkdtemp(prefix="sweep_ckpt_")
                    tok = AutoTokenizer.from_pretrained(source, revision=rev, cache_dir=tmp)
                    model = AutoModelForCausalLM.from_pretrained(
                        source, revision=rev, cache_dir=tmp, torch_dtype=dtype)
                model = model.to(device).eval()

                meta = {"model": args.model, "revision": rev}
                run_block1(model, tok, battery, device, writer, meta)
                run_block3(model, tok, battery, device, writer, meta)
                run_block2(model, tok, battery, device, writer, meta,
                           args.k, args.temp, args.max_new, chat=args.chat)
                if args.chat:
                    run_block4(model, tok, battery, device, writer, meta,
                               args.k, args.temp, args.max_new)
                f.flush()
                log(f"  {rev}: done")
                if is_local and args.delete_after:
                    shutil.rmtree(local_dir, ignore_errors=True)
                    log(f"  {rev}: deleted from disk")
            finally:
                try:
                    del model, tok
                except NameError:
                    pass
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                if tmp:
                    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
