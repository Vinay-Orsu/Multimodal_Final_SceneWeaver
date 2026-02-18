import argparse
import json
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

# Import your dataset factory (adjust import if your function name differs)
from data_sources.load_dataset import get_dataset


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    """Try multiple possible keys (datasets often differ in column names)."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def iter_examples(ds) -> Iterable[Dict[str, Any]]:
    """
    Supports both standard datasets (indexable) and streaming datasets (iterable).
    """
    if hasattr(ds, "__len__") and hasattr(ds, "__getitem__"):
        for i in range(len(ds)):
            yield ds[i]
    else:
        # streaming datasets
        for ex in ds:
            yield ex


def export_jsonl(
    examples: List[Dict[str, Any]],
    out_path: str,
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Export dataset captions/prompts to JSONL.")
    parser.add_argument("--dataset", type=str, default="msrvtt", help="Dataset name (e.g., msrvtt, activitynet).")
    parser.add_argument("--split", type=str, default="train", help="Split name (train/val/test).")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode (avoids full download).")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Output directory.")
    parser.add_argument("--limit", type=int, default=0, help="Max number of prompts to export (0 = all).")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before taking limit/subset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle/subset.")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    ds = get_dataset(args.dataset, split=args.split, streaming=args.streaming)

    # Collect prompts
    collected: List[Dict[str, Any]] = []
    rng = random.Random(args.seed)

    # Some common caption keys across video datasets:
    caption_keys = [
        "caption", "captions", "sentence", "sentences", "description", "descriptions", "text"
    ]
    id_keys = ["video_id", "video", "id", "clip_id", "name"]

    # If streaming, we need to decide shuffle strategy:
    # - True shuffle isn't feasible without buffering everything.
    # - We'll do a simple reservoir-like approach if shuffle+limit is used.
    buffer_for_shuffle: List[Dict[str, Any]] = []

    for idx, ex in enumerate(iter_examples(ds)):
        vid = safe_get(ex, id_keys, default=str(idx))
        cap = safe_get(ex, caption_keys, default=None)

        # If captions is a list, pick the first by default (simple baseline)
        if isinstance(cap, list) and len(cap) > 0:
            cap = cap[0]

        if cap is None:
            # Skip if no caption text
            continue

        row = {
            "uid": f"{args.dataset}_{args.split}_{idx}",
            "source_dataset": args.dataset,
            "split": args.split,
            "video_id": str(vid),
            "prompt": str(cap).strip(),
        }

        if args.streaming and args.shuffle and args.limit > 0:
            buffer_for_shuffle.append(row)
            # Keep buffer bounded (2x limit is enough for decent shuffle)
            if len(buffer_for_shuffle) >= max(100, args.limit * 2):
                rng.shuffle(buffer_for_shuffle)
                while buffer_for_shuffle and (args.limit == 0 or len(collected) < args.limit):
                    collected.append(buffer_for_shuffle.pop())
        else:
            collected.append(row)

        if args.limit > 0 and len(collected) >= args.limit and not (args.streaming and args.shuffle):
            break

    # If we used streaming shuffle buffer, flush remaining
    if buffer_for_shuffle and (args.limit == 0 or len(collected) < args.limit):
        rng.shuffle(buffer_for_shuffle)
        for row in buffer_for_shuffle:
            if args.limit > 0 and len(collected) >= args.limit:
                break
            collected.append(row)

    # Non-streaming shuffle is easy
    if (not args.streaming) and args.shuffle:
        rng.shuffle(collected)

    # If shuffle and limit were used non-streaming, trim after shuffle
    if args.limit > 0:
        collected = collected[: args.limit]

    out_jsonl = os.path.join(args.out_dir, f"{args.dataset}_{args.split}.jsonl")
    export_jsonl(collected, out_jsonl)

    # Save metadata for reproducibility
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "streaming": args.streaming,
        "limit": args.limit,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "num_exported": len(collected),
        "output_jsonl": out_jsonl,
    }
    out_meta = os.path.join(args.out_dir, f"{args.dataset}_{args.split}.meta.json")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f" Exported {len(collected)} prompts {out_jsonl}")
    print(f" Metadata saved {out_meta}")


if __name__ == "__main__":
    main()
