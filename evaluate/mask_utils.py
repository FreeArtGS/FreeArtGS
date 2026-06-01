import json
import os
from pathlib import Path

from compute_mask_miou import compute_miou_metrics


def load_motion_part_from_manifest(results_dir: str) -> int | None:
    manifest_path = os.path.join(results_dir, "export_manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    motion_part = manifest.get("motion_part")
    if motion_part in (0, 1):
        return int(motion_part)
    return None


def _miou_required_paths(dataset_dir: str) -> list[Path]:
    dataset_path = Path(dataset_dir)
    return [
        dataset_path / "mask_0",
        dataset_path / "mask_1",
        dataset_path / "eval" / "split_info.json",
    ]


def _format_motion_part_scores(metrics: dict[str, float | int | str] | None) -> dict[str, float | int | str] | None:
    if metrics is None:
        return None
    return {
        "miou_assignment_a": float(metrics["miou_assignment_a"]),
        "miou_assignment_b": float(metrics["miou_assignment_b"]),
        "miou_best": float(metrics["miou_best"]),
        "best_mapping": str(metrics["best_mapping"]),
    }


def resolve_rendered_masks_root(dataset_dir: str) -> Path | None:
    dataset_path = Path(dataset_dir)
    if (dataset_path / "rendered_masks" / "mask_0").is_dir() and (dataset_path / "rendered_masks" / "mask_1").is_dir():
        return dataset_path
    return None


def infer_psnr_motion_part_from_masks(
    object_id: str,
    dataset_dir: str,
    repo_root: Path,
) -> tuple[int | None, dict[str, float | int | str] | None]:
    rendered_root = resolve_rendered_masks_root(dataset_dir)
    if rendered_root is None:
        return None, None

    dataset_path = Path(dataset_dir)
    required_paths = _miou_required_paths(dataset_dir)
    if not all(path.exists() for path in required_paths):
        return None, None

    metrics = compute_miou_metrics(object_id, rendered_root.parent, dataset_path.parent)

    best_mapping = str(metrics["best_mapping"])
    if best_mapping == "pred0->gt0, pred1->gt1":
        return 1, metrics
    if best_mapping == "pred0->gt1, pred1->gt0":
        return 0, metrics
    return None, metrics


def compute_part_mask_miou(object_id: str, dataset_dir: str, repo_root: Path) -> dict[str, float | int | str] | None:
    rendered_root = resolve_rendered_masks_root(dataset_dir)
    if rendered_root is None:
        return None
    dataset_path = Path(dataset_dir)
    required_dirs = _miou_required_paths(dataset_dir)
    if not all(path.exists() for path in required_dirs):
        return None
    metrics = compute_miou_metrics(object_id, rendered_root.parent, dataset_path.parent)
    return metrics


def resolve_psnr_motion_part(
    results_dir: str,
    dataset_dir: str,
    object_id: str,
    repo_root: Path,
    cli_motion_part: int,
) -> tuple[int, bool, dict[str, float | int | str] | None, dict[str, float | int | str] | None]:
    manifest_motion_part = load_motion_part_from_manifest(results_dir)
    base_motion_part = cli_motion_part if manifest_motion_part is None else manifest_motion_part
    mask_inferred_motion_part, mask_assignment_metrics = infer_psnr_motion_part_from_masks(
        object_id=object_id,
        dataset_dir=dataset_dir,
        repo_root=repo_root,
    )
    if mask_inferred_motion_part is None:
        return int(base_motion_part), False, None, mask_assignment_metrics

    psnr_motion_part = int(mask_inferred_motion_part)
    flipped = psnr_motion_part != int(base_motion_part)
    scores = _format_motion_part_scores(mask_assignment_metrics)
    return psnr_motion_part, flipped, scores, mask_assignment_metrics
