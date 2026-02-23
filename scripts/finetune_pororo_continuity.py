from __future__ import annotations

import argparse
import io
import json
import math
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel


class ContinuityProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PororoContinuityDataset(Dataset):
    def __init__(
        self,
        zip_path: Path,
        frame_windows: np.ndarray,
        indices: Sequence[int],
    ) -> None:
        self.zip_path = Path(zip_path)
        self.frame_windows = frame_windows
        self.indices = list(indices)
        self._zip_file: Optional[zipfile.ZipFile] = None

    def __len__(self) -> int:
        return len(self.indices)

    def _zip(self) -> zipfile.ZipFile:
        if self._zip_file is None:
            self._zip_file = zipfile.ZipFile(self.zip_path.as_posix(), mode="r")
        return self._zip_file

    @staticmethod
    def _decode_path(value: object) -> str:
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode("utf-8")
        return str(value)

    def _read_png(self, rel_path: str) -> np.ndarray:
        # Paths in metadata are relative to this root in the zip.
        in_zip = rel_path
        if not in_zip.startswith("pororo_png_filtered_blip2/"):
            in_zip = f"pororo_png_filtered_blip2/{in_zip}"

        raw = self._zip().read(in_zip)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row_idx = self.indices[index]
        frames = self.frame_windows[row_idx]
        if len(frames) < 2:
            raise ValueError(f"Unexpected window size at index {row_idx}: {len(frames)}")

        # Use transition pair in each cached sequence.
        anchor_path = self._decode_path(frames[-2])
        positive_path = self._decode_path(frames[-1])

        anchor = self._read_png(anchor_path)
        positive = self._read_png(positive_path)
        return {
            "anchor": anchor,
            "positive": positive,
            "anchor_path": anchor_path,
            "positive_path": positive_path,
        }


def collate_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    anchors = [item["anchor"] for item in batch]
    positives = [item["positive"] for item in batch]
    anchor_paths = [item["anchor_path"] for item in batch]
    positive_paths = [item["positive_path"] for item in batch]
    return {
        "anchors": anchors,
        "positives": positives,
        "anchor_paths": anchor_paths,
        "positive_paths": positive_paths,
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
        description="Fine-tune a DINO-based continuity adapter on PororoSV."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/vault/v123be/v123be36/PororoSV",
        help="Path containing PororoSV files.",
    )
    parser.add_argument(
        "--dino_model_id",
        type=str,
        default="/home/vault/v123be/v123be36/facebook/dinov2-base",
        help="HF model id or local path for DINOv2.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_projector", type=float, default=5e-5)
    parser.add_argument("--lr_backbone", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--unfreeze_backbone",
        action="store_true",
        help="Enable full DINOv2 backbone finetuning. Default trains projector only.",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="seen",
        choices=["seen", "unseen", "both"],
        help="Which Pororo validation split to use.",
    )
    parser.add_argument("--max_train_pairs", type=int, default=0, help="0 means use all train pairs.")
    parser.add_argument("--max_val_pairs", type=int, default=0, help="0 means use all val pairs.")
    parser.add_argument(
        "--save_path",
        type=str,
        default="outputs/pororo_continuity_adapter.pt",
        help="Checkpoint output path.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    following_path = dataset_root / "following_cache4.npy"
    split_path = dataset_root / "train_seen_unseen_ids.npy"
    zip_path = dataset_root / "pororo_png_filtered_blip2.zip"

    if not following_path.exists():
        raise FileNotFoundError(f"Missing file: {following_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing file: {split_path}")
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing file: {zip_path}")

    frame_windows = np.load(following_path.as_posix(), allow_pickle=True)
    split_data = np.load(split_path.as_posix(), allow_pickle=True)

    train_ids = np.asarray(split_data[0], dtype=np.int64)
    seen_ids = np.asarray(split_data[1], dtype=np.int64)
    unseen_ids = np.asarray(split_data[2], dtype=np.int64)

    if args.val_split == "seen":
        val_ids = seen_ids
    elif args.val_split == "unseen":
        val_ids = unseen_ids
    else:
        val_ids = np.concatenate([seen_ids, unseen_ids], axis=0)

    if args.max_train_pairs > 0:
        train_ids = train_ids[: args.max_train_pairs]
    if args.max_val_pairs > 0:
        val_ids = val_ids[: args.max_val_pairs]

    if len(train_ids) == 0:
        raise RuntimeError("No training continuity pairs were selected.")
    if len(val_ids) == 0:
        raise RuntimeError("No validation continuity pairs were selected.")

    print(f"train_pairs={len(train_ids)} val_pairs={len(val_ids)} val_split={args.val_split}")

    train_ds = PororoContinuityDataset(zip_path=zip_path, frame_windows=frame_windows, indices=train_ids)
    val_ds = PororoContinuityDataset(zip_path=zip_path, frame_windows=frame_windows, indices=val_ids)

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
