#!/usr/bin/env python3
"""Compute mIoU between rendered part masks and original masks.

This script is tailored for the current split-dataset layout, e.g.:

- Original example data (per-object):
    examples_ori/<object_id>/masks/00000.png

- Split dataset (per-object):
    datasets/<object_id>/
        eval/split_info.json
        mask_0/00000.png
        mask_1/00000.png

`split_info.json` contains `train_original_indices`, mapping each train-frame
index in the split dataset to the corresponding original frame index.

For each train frame index i:
  - original index = train_original_indices[i]
  - GT mask path   = examples_ori/<id>/masks/{original_index:05d}.png
  - pred0 path     = datasets/<id>/mask_0/{i:05d}.png
  - pred1 path     = datasets/<id>/mask_1/{i:05d}.png

The script computes IoU on the foreground (object) class for:
  1) union(mask_0, mask_1) vs GT (overall object mask)
  2) mask_0 vs GT
  3) mask_1 vs GT
and reports the mean IoU (mIoU) over all train frames.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _load_binary_mask(path: Path) -> np.ndarray:
    """Load a PNG mask and convert to a boolean foreground mask.

    Any pixel > 0 is treated as foreground.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    return arr > 0


def compute_miou_metrics(
    object_id: str,
    examples_root: Path,
    datasets_root: Path,
) -> dict[str, float | int | str]:
    """Compute part-wise mIoU using GT masks in rendered_masks, robust to 0/1 swap.

    Assumptions for a given object ID:
      - GT part masks:
          examples_root/<id>/rendered_masks/mask_0/{orig_idx:04d or 05d}.png
          examples_root/<id>/rendered_masks/mask_1/{orig_idx:04d or 05d}.png
      - Predicted part masks:
          datasets_root/<id>/mask_0/{split_idx:05d}.png
          datasets_root/<id>/mask_1/{split_idx:05d}.png
      - Mapping between split_idx and orig_idx via train_original_indices.

    We evaluate two assignments between predicted and GT parts:
      A: pred0->gt0, pred1->gt1
      B: pred0->gt1, pred1->gt0
    and report the average mIoU over the two parts for each assignment,
    as well as the best of the two assignments.
    """
    obj_examples_dir = examples_root / object_id
    obj_datasets_dir = datasets_root / object_id

    gt_part_root = obj_examples_dir / "rendered_masks"
    gt_part0_dir = gt_part_root / "mask_0"
    gt_part1_dir = gt_part_root / "mask_1"
    split_info_path = obj_datasets_dir / "eval" / "split_info.json"
    mask0_dir = obj_datasets_dir / "mask_0"
    mask1_dir = obj_datasets_dir / "mask_1"

    if not gt_part0_dir.is_dir() or not gt_part1_dir.is_dir():
        raise FileNotFoundError(
            f"GT part-wise masks not found under {gt_part_root} (expect mask_0/ and mask_1/)."
        )
    if not split_info_path.is_file():
        raise FileNotFoundError(f"split_info.json not found: {split_info_path}")
    if not mask0_dir.is_dir() or not mask1_dir.is_dir():
        raise FileNotFoundError(
            f"mask_0 or mask_1 directory missing under {obj_datasets_dir}"
        )

    with open(split_info_path, "r") as f:
        split_info = json.load(f)

    train_original_indices = split_info.get("train_original_indices")
    if train_original_indices is None:
        raise KeyError("train_original_indices not found in split_info.json")

    train_original_indices = list(train_original_indices)
    num_train = len(train_original_indices)

    pred_indices = sorted(
        int(path.stem)
        for path in mask0_dir.glob("*.png")
        if path.is_file() and path.stem.isdigit() and (mask1_dir / path.name).is_file()
    )
    num_pred = len(pred_indices)
    num_gt = len(list(gt_part0_dir.glob("*.png")))

    use_direct_idx = num_pred > 0 and num_pred == num_gt
    if num_pred == 0:
        raise FileNotFoundError(f"No matching predicted mask pairs found under {mask0_dir} and {mask1_dir}")
    if use_direct_idx:
        eval_split_indices = pred_indices
        num_train = num_pred
    else:
        eval_split_indices = [idx for idx in pred_indices if idx < len(train_original_indices)]
        num_train = len(eval_split_indices)

    def _pick_gt_part(path_root: Path, idx: int) -> Path:
        name4 = f"{idx:04d}.png"
        p4 = path_root / name4
        if p4.is_file():
            return p4
        name5 = f"{idx:05d}.png"
        p5 = path_root / name5
        if p5.is_file():
            return p5
        raise FileNotFoundError(
            f"GT part mask not found for index {idx} under {path_root}"
        )

    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        uni = np.logical_or(a, b).sum()
        if uni == 0:
            # No foreground anywhere; treat as perfect match
            return 1.0
        return float(inter) / float(uni)

    part_iou00: list[float] = []  # pred0 vs gt0
    part_iou11: list[float] = []  # pred1 vs gt1
    part_iou01: list[float] = []  # pred0 vs gt1
    part_iou10: list[float] = []  # pred1 vs gt0

    for split_idx in eval_split_indices:
        split_name = f"{split_idx:05d}.png"

        if use_direct_idx:
            orig_idx = split_idx
        else:
            orig_idx = train_original_indices[split_idx]

        pred0_path = mask0_dir / split_name
        pred1_path = mask1_dir / split_name
        gt0_path = _pick_gt_part(gt_part0_dir, orig_idx)
        gt1_path = _pick_gt_part(gt_part1_dir, orig_idx)

        gt0 = _load_binary_mask(gt0_path)
        gt1 = _load_binary_mask(gt1_path)
        pred0 = _load_binary_mask(pred0_path)
        pred1 = _load_binary_mask(pred1_path)

        if gt0.shape != pred0.shape or gt1.shape != pred1.shape:
            raise ValueError(
                "Shape mismatch in part-wise GT/pred: "
                f"gt0 {gt0.shape}, gt1 {gt1.shape}, "
                f"pred0 {pred0.shape}, pred1 {pred1.shape}"
            )

        part_iou00.append(_iou(pred0, gt0))
        part_iou11.append(_iou(pred1, gt1))
        part_iou01.append(_iou(pred0, gt1))
        part_iou10.append(_iou(pred1, gt0))

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    # Assignment A: pred0->gt0, pred1->gt1
    m00 = _mean(part_iou00)
    m11 = _mean(part_iou11)
    m_avg_A = 0.5 * (m00 + m11)

    # Assignment B: pred0->gt1, pred1->gt0
    m01 = _mean(part_iou01)
    m10 = _mean(part_iou10)
    m_avg_B = 0.5 * (m01 + m10)

    if m_avg_A >= m_avg_B:
        best_mapping = "pred0->gt0, pred1->gt1"
        best_avg = m_avg_A
    else:
        best_mapping = "pred0->gt1, pred1->gt0"
        best_avg = m_avg_B

    return {
        "object_id": object_id,
        "frames": int(num_train),
        "miou_assignment_a": float(m_avg_A),
        "miou_assignment_b": float(m_avg_B),
        "miou_best": float(best_avg),
        "best_mapping": best_mapping,
    }


def compute_miou_for_object(
    object_id: str,
    examples_root: Path,
    datasets_root: Path,
) -> dict[str, float | int | str]:
    """Compute and print part-wise mIoU for one object."""
    metrics = compute_miou_metrics(object_id, examples_root, datasets_root)

    print(f"Object: {object_id}")
    print(f"  Train frames (split): {metrics['frames']}")
    print(f"  mIoU A (pred0->gt0, pred1->gt1, avg over parts): {metrics['miou_assignment_a']:.4f}")
    print(f"  mIoU B (pred0->gt1, pred1->gt0, avg over parts): {metrics['miou_assignment_b']:.4f}")
    print(
        f"  mIoU best over assignments: {metrics['miou_best']:.4f} "
        f"(assignment: {metrics['best_mapping']})"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute mIoU between original masks in examples_ori and "
            "split masks (mask_0/mask_1) under datasets/<object_id>."
        )
    )
    parser.add_argument(
        "--object_id",
        type=str,
        required=True,
        help="Object/category ID, e.g., 4552",
    )
    parser.add_argument(
        "--examples_root",
        type=str,
        default="examples_ori",
        help="Root directory containing original examples (default: examples_ori)",
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        default="datasets",
        help="Root directory containing split datasets (default: datasets)",
    )

    args = parser.parse_args()

    examples_root = Path(args.examples_root)
    datasets_root = Path(args.datasets_root)

    compute_miou_for_object(args.object_id, examples_root, datasets_root)


if __name__ == "__main__":
    main()
