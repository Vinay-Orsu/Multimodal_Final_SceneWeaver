from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel


SHOT_ID_RE = re.compile(r"^(s_\d+_e_\d+)_shot_(\d+)_(\d+)$")


def parse_shot_id(global_id: str) -> Optional[Tuple[str, int, int]]:
    match = SHOT_ID_RE.match(global_id)
    if match is None:
        return None
    episode_id, start, end = match.groups()
    return episode_id, int(start), int(end)


def build_continuity_pairs(
    split_ids: Sequence[str],
    available_ids: set[str],
    max_pairs: Optional[int],
) -> List[Tuple[str, str]]:
    by_episode: Dict[str, List[Tuple[int, int, str]]] = {}
    for shot_id in split_ids:
        if shot_id not in available_ids:
            continue
        parsed = parse_shot_id(shot_id)
        if parsed is None:
            continue
        episode_id, start, end = parsed
        by_episode.setdefault(episode_id, []).append((start, end, shot_id))

    pairs: List[Tuple[str, str]] = []
    for episode_id in sorted(by_episode.keys()):
        shots = sorted(by_episode[episode_id], key=lambda x: (x[0], x[1]))
        if len(shots) < 2:
            continue
        for idx in range(len(shots) - 1):
            prev_shot = shots[idx][2]
            next_shot = shots[idx + 1][2]
            pairs.append((prev_shot, next_shot))

    if max_pairs is not None and max_pairs > 0:
        pairs = pairs[:max_pairs]
    return pairs


class FlintstonesContinuityDataset(Dataset):
    def __init__(
        self,
        zip_path: Path,
        pairs: Sequence[Tuple[str, str]],
    ) -> None:
        self.zip_path = Path(zip_path)
        self.pairs = list(pairs)
        self._zip_file: Optional[zipfile.ZipFile] = None

    def __len__(self) -> int:
        return len(self.pairs)

    def _zip(self) -> zipfile.ZipFile:
        if self._zip_file is None:
            self._zip_file = zipfile.ZipFile(self.zip_path.as_posix(), mode="r")
        return self._zip_file

    def _load_clip_frames(self, shot_id: str) -> np.ndarray:
        name = f"video_frames_sampled/{shot_id}.npy"
        raw = self._zip().read(name)
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"Unexpected frame shape for {shot_id}: {arr.shape}")
        return arr

    def __getitem__(self, index: int) -> Dict[str, object]:
        prev_shot, next_shot = self.pairs[index]
        prev_clip = self._load_clip_frames(prev_shot)
        next_clip = self._load_clip_frames(next_shot)
        anchor = prev_clip[-1]
        positive = next_clip[0]
        return {
            "anchor": anchor,
            "positive": positive,
            "prev_shot": prev_shot,
            "next_shot": next_shot,
        }


class ContinuityProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def collate_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    anchors = [item["anchor"] for item in batch]
    positives = [item["positive"] for item in batch]
    prev_shots = [item["prev_shot"] for item in batch]
    next_shots = [item["next_shot"] for item in batch]
    return {
        "anchors": anchors,
        "positives": positives,
        "prev_shots": prev_shots,
        "next_shots": next_shots,
    }


def device_from_arg(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_frames(
    images: Sequence[np.ndarray],
    processor: AutoImageProcessor,
    backbone: AutoModel,
    projector: ContinuityProjector,
    device: str,
    train_backbone: bool,
) -> torch.Tensor:
    inputs = processor(images=list(images), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if train_backbone:
        outputs = backbone(**inputs)
    else:
        with torch.no_grad():
            outputs = backbone(**inputs)
    features = outputs.last_hidden_state.mean(dim=1)
    projected = projector(features)
    return F.normalize(projected, dim=-1)


def compute_bidirectional_infonce(
    anchor_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, float]:
    logits = (anchor_emb @ positive_emb.T) / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_a = F.cross_entropy(logits, targets)
    loss_b = F.cross_entropy(logits.T, targets)
    loss = 0.5 * (loss_a + loss_b)
    acc = (logits.argmax(dim=1) == targets).float().mean().item()
    return loss, acc


def run_epoch(
    loader: DataLoader,
    processor: AutoImageProcessor,
    backbone: AutoModel,
    projector: ContinuityProjector,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
    temperature: float,
    train_backbone: bool,
) -> Tuple[float, float]:
    training = optimizer is not None
    if training:
        projector.train()
        backbone.train(mode=train_backbone)
    else:
        projector.eval()
        backbone.eval()

    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    for batch in loader:
        anchor_images = batch["anchors"]
        positive_images = batch["positives"]

        anchor_emb = encode_frames(
            images=anchor_images,
            processor=processor,
            backbone=backbone,
            projector=projector,
            device=device,
            train_backbone=train_backbone and training,
        )
        positive_emb = encode_frames(
            images=positive_images,
            processor=processor,
            backbone=backbone,
            projector=projector,
            device=device,
            train_backbone=train_backbone and training,
        )

        loss, acc = compute_bidirectional_infonce(
            anchor_emb=anchor_emb,
            positive_emb=positive_emb,
            temperature=temperature,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_acc += float(acc)
        steps += 1

    if steps == 0:
        return math.nan, math.nan
    return total_loss / steps, total_acc / steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a DINO-based continuity adapter on FlintstonesSV."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/vault/v123be/v123be36/FlintstonesSV",
        help="Path containing FlintstonesSV files.",
    )
    parser.add_argument(
        "--dino_model_id",
        type=str,
        default="/home/vault/v123be/v123be36/facebook/dinov2-base",
        help="HF model id or local path for DINOv2.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_projector", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--unfreeze_backbone",
        action="store_true",
        help="Enable full DINOv2 backbone finetuning. Default trains projector only.",
    )
    parser.add_argument("--max_train_pairs", type=int, default=0, help="0 means use all train pairs.")
    parser.add_argument("--max_val_pairs", type=int, default=0, help="0 means use all val pairs.")
    parser.add_argument(
        "--save_path",
        type=str,
        default="outputs/flintstones_continuity_adapter.pt",
        help="Checkpoint output path.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    annotations_path = dataset_root / "flintstones_annotations_v1-0.json"
    split_path = dataset_root / "train-val-test_split.json"
    zip_path = dataset_root / "video_frames_sampled.zip"

    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing file: {annotations_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing file: {split_path}")
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing file: {zip_path}")

    with split_path.open("r", encoding="utf-8") as f:
        split_data = json.load(f)

    with zipfile.ZipFile(zip_path.as_posix(), mode="r") as zf:
        zip_ids = {
            Path(name).stem
            for name in zf.namelist()
            if name.startswith("video_frames_sampled/") and name.endswith(".npy")
        }

    max_train_pairs = args.max_train_pairs if args.max_train_pairs > 0 else None
    max_val_pairs = args.max_val_pairs if args.max_val_pairs > 0 else None

    train_pairs = build_continuity_pairs(
        split_ids=split_data.get("train", []),
        available_ids=zip_ids,
        max_pairs=max_train_pairs,
    )
    val_pairs = build_continuity_pairs(
        split_ids=split_data.get("val", []),
        available_ids=zip_ids,
        max_pairs=max_val_pairs,
    )

    if len(train_pairs) == 0:
        raise RuntimeError("No training continuity pairs were built.")
    if len(val_pairs) == 0:
        raise RuntimeError("No validation continuity pairs were built.")

    print(f"train_pairs={len(train_pairs)} val_pairs={len(val_pairs)}")

    train_ds = FlintstonesContinuityDataset(zip_path=zip_path, pairs=train_pairs)
    val_ds = FlintstonesContinuityDataset(zip_path=zip_path, pairs=val_pairs)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )

    device = device_from_arg(args.device)
    print(f"device={device}")

    processor = AutoImageProcessor.from_pretrained(args.dino_model_id)
    backbone = AutoModel.from_pretrained(args.dino_model_id).to(device)

    if not args.unfreeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    with torch.no_grad():
        probe = processor(images=[np.zeros((128, 128, 3), dtype=np.uint8)], return_tensors="pt")
        probe = {k: v.to(device) for k, v in probe.items()}
        out = backbone(**probe)
        feature_dim = int(out.last_hidden_state.shape[-1])

    projector = ContinuityProjector(
        input_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.proj_dim,
    ).to(device)

    optim_groups = [{"params": projector.parameters(), "lr": args.lr_projector}]
    if args.unfreeze_backbone:
        optim_groups.append({"params": backbone.parameters(), "lr": args.lr_backbone})
    optimizer = torch.optim.AdamW(
        optim_groups,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            loader=train_loader,
            processor=processor,
            backbone=backbone,
            projector=projector,
            optimizer=optimizer,
            device=device,
            temperature=args.temperature,
            train_backbone=args.unfreeze_backbone,
        )
        val_loss, val_acc = run_epoch(
            loader=val_loader,
            processor=processor,
            backbone=backbone,
            projector=projector,
            optimizer=None,
            device=device,
            temperature=args.temperature,
            train_backbone=False,
        )

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "projector_state_dict": projector.state_dict(),
                "args": vars(args),
                "dino_model_id": args.dino_model_id,
                "feature_dim": feature_dim,
                "history": history,
            }
            torch.save(ckpt, save_path.as_posix())
            print(f"saved_best={save_path.as_posix()} val_loss={val_loss:.4f}")

    history_path = save_path.with_suffix(".history.json")
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"history={history_path.as_posix()}")


if __name__ == "__main__":
    main()
