#!/usr/bin/env python3
"""
Checkpoint prefetcher.
Fetches model checkpoints (HF revisions) into ./checkpoints/{model}/{revision}/
so the sweep can run fully offline afterwards. Skips revisions already present.
Pulls only weights + tokenizer + config (no optimizer states, no training logs).

Usage:
  python download.py --model EleutherAI/pythia-1.4b --revisions step1000,step143000
  python download.py --model EleutherAI/pythia-410m --revisions all
  python download.py --model EleutherAI/pythia-410m --revisions subsample30
  python download.py --model allenai/OLMo-2-0425-1B-Instruct --revisions main

Requires: pip install huggingface_hub
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

from huggingface_hub import list_repo_refs, snapshot_download

PATTERNS = [
    "*.safetensors", "*.bin", "*.json", "*.txt", "*.model",
    "tokenizer*", "*.tiktoken",
]

def log(*a):
    print(*a, file=sys.stderr, flush=True)


def step_num(name: str) -> int:
    return int(re.sub(r"\D", "", name) or 0)


def resolve_revisions(model: str, spec: str) -> list:
    if spec not in ("all", "subsample30"):
        return [r.strip() for r in spec.split(",") if r.strip()]
    refs = list_repo_refs(model)
    steps = sorted(
        (b.name for b in refs.branches if b.name.startswith(("step", "stage"))),
        key=step_num,
    )
    if not steps:
        return ["main"]
    if spec == "all":
        return steps
    # subsample30: keep all log-spaced early checkpoints (step number <= 1000),
    # then a uniform stride through the rest, always keeping the final checkpoint.
    early = [s for s in steps if step_num(s) <= 1000]
    late = [s for s in steps if step_num(s) > 1000]
    budget = max(1, 30 - len(early))
    stride = max(1, len(late) // budget)
    picked = late[::stride]
    if late and late[-1] not in picked:
        picked.append(late[-1])
    return early + picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revisions", default="all",
                    help="'all', 'subsample30', 'main', or comma-separated list")
    ap.add_argument("--dir", default="checkpoints", help="root download directory")
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched and exit")
    args = ap.parse_args()

    revisions = resolve_revisions(args.model, args.revisions)
    root = Path(args.dir) / args.model.replace("/", "__")

    log(f"{args.model}: {len(revisions)} revisions -> {root}")
    if args.dry_run:
        for r in revisions:
            print(r)
        return

    done, failed = 0, []
    for i, rev in enumerate(revisions, 1):
        target = root / rev
        marker = target / ".complete"
        if marker.exists():
            log(f"[{i}/{len(revisions)}] {rev}: already present, skipping")
            done += 1
            continue
        log(f"[{i}/{len(revisions)}] {rev}: downloading")
        try:
            snapshot_download(
                repo_id=args.model,
                revision=rev,
                local_dir=str(target),
                allow_patterns=PATTERNS,
            )
            marker.touch()
            done += 1
            free_gb = shutil.disk_usage(root).free / 1e9
            log(f"[{i}/{len(revisions)}] {rev}: done ({free_gb:.0f} GB free)")
        except KeyboardInterrupt:
            log("interrupted; partial revision left without .complete marker, rerun to resume")
            raise
        except Exception as e:
            log(f"[{i}/{len(revisions)}] {rev}: FAILED ({e})")
            failed.append(rev)

    log(f"finished: {done}/{len(revisions)} present" + (f", failed: {','.join(failed)}" if failed else ""))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
