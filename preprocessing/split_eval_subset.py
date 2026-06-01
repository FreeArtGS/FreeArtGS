import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.pipeline_config import get_section, load_pipeline_config, resolve_pipeline_paths


FIVE_DIGIT_WIDTH = 5
DEFAULT_SUBDIRS = [
    "rgb",
    "depth",
    "masks",
    "vis",
    "rendered_masks/mask_0",
    "rendered_masks/mask_1",
]


def zero_pad(index: int, width: int = FIVE_DIGIT_WIDTH) -> str:
    return str(index).zfill(width)


def discover_indices_and_exts(directory: Path) -> Dict[int, str]:
    """Return mapping: frame_index -> file extension for all numeric-named files in a directory."""
    index_to_ext: Dict[int, str] = {}
    if not directory.exists():
        return index_to_ext
    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue
        stem = entry.stem
        if not stem.isdigit():
            continue
        try:
            index = int(stem)
        except ValueError:
            continue
        index_to_ext[index] = entry.suffix
    return index_to_ext


def read_json_list(json_path: Path) -> List:
    with json_path.open("r") as f:
        return json.load(f)


def write_json_list(json_path: Path, data: List) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(data, f, indent=2)


def sample_eval_indices(num_frames: int, percentage: float, seed: int) -> List[int]:
    assert 0.0 < percentage < 1.0, "percentage must be in (0,1)"
    sample_count = max(1, int(round(num_frames * percentage)))
    rng = random.Random(seed)
    indices = list(range(num_frames))
    if sample_count >= num_frames:
        return indices
    sampled = sorted(rng.sample(indices, sample_count))
    return sampled


def move_eval_files(dataset_dir: Path, eval_dir: Path, subdirs: List[str], eval_indices: List[int], subdir_index_to_ext: Dict[str, Dict[int, str]], old_to_eval_new: Dict[int, int]) -> None:
    for subdir_name in subdirs:
        src_dir = dataset_dir / subdir_name
        if not src_dir.exists():
            continue
        dst_dir = eval_dir / subdir_name
        index_to_ext = subdir_index_to_ext.get(subdir_name, {})
        if not index_to_ext:
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for old_idx in eval_indices:
            ext = index_to_ext.get(old_idx)
            if ext is None:
                # Skip if this subdir does not have this frame
                continue
            src_path = src_dir / f"{zero_pad(old_idx)}{ext}"
            if not src_path.exists():
                continue
            new_idx = old_to_eval_new[old_idx]
            dst_path = dst_dir / f"{zero_pad(new_idx)}{ext}"
            shutil.move(str(src_path), str(dst_path))


def rename_remaining_to_sequential(dataset_dir: Path, subdirs: List[str], train_indices: List[int], subdir_index_to_ext: Dict[str, Dict[int, str]], old_to_train_new: Dict[int, int]) -> None:
    # First rename to temporary names to avoid collisions, then rename to final names
    for subdir_name in subdirs:
        dir_path = dataset_dir / subdir_name
        if not dir_path.exists():
            continue
        index_to_ext = subdir_index_to_ext.get(subdir_name, {})
        if not index_to_ext:
            continue
        # Temp rename
        for old_idx in train_indices:
            ext = index_to_ext.get(old_idx)
            if ext is None:
                continue
            src_path = dir_path / f"{zero_pad(old_idx)}{ext}"
            if not src_path.exists():
                continue
            tmp_path = dir_path / f"__tmp__{zero_pad(old_idx)}{ext}"
            os.rename(src_path, tmp_path)
        # Final rename
        for old_idx in train_indices:
            ext = index_to_ext.get(old_idx)
            if ext is None:
                continue
            tmp_path = dir_path / f"__tmp__{zero_pad(old_idx)}{ext}"
            if not tmp_path.exists():
                continue
            new_idx = old_to_train_new[old_idx]
            final_path = dir_path / f"{zero_pad(new_idx)}{ext}"
            os.rename(tmp_path, final_path)


def split_and_write_jsons(dataset_dir: Path, eval_dir: Path, eval_indices: List[int], train_indices: List[int]) -> None:
    # JSON files: object_poses.json, camera_params.json, joint_states.json
    # We assume all are list-like aligned with frame indices
    json_files = [
        "object_poses.json",
        "camera_params.json",
        "joint_states.json",
    ]
    for json_name in json_files:
        src_path = dataset_dir / json_name
        if not src_path.exists():
            continue
        data_list = read_json_list(src_path)
        # Defensive checks
        total_len = len(data_list)
        if max(max(eval_indices, default=-1), max(train_indices, default=-1)) >= total_len:
            raise RuntimeError(f"JSON {json_name} length ({total_len}) smaller than required index")

        eval_subset = [data_list[i] for i in eval_indices]
        train_subset = [data_list[i] for i in train_indices]

        # write eval subset
        write_json_list(eval_dir / json_name, eval_subset)
        # overwrite original with train subset
        write_json_list(src_path, train_subset)

    # cam_K.txt is global; copy to eval
    cam_k_src = dataset_dir / "cam_K.txt"
    if cam_k_src.exists():
        shutil.copy2(str(cam_k_src), str(eval_dir / "cam_K.txt"))


def write_split_info(eval_dir: Path, eval_indices: List[int], train_indices: List[int]) -> None:
    info = {
        "eval_original_indices": eval_indices,
        "train_original_indices": train_indices,
    }
    write_json_list(eval_dir / "split_info.json", info)  # type: ignore[arg-type]


def main():
    parser = argparse.ArgumentParser(description="Split dataset frames into eval subset and reindex remaining frames.")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline config YAML")
    parser.add_argument("--object_name", type=str, default=None, help="Object name override used with --config")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to the dataset directory, e.g., .../examples/102033")
    parser.add_argument("--percentage", type=float, default=0.05, help="Fraction of frames to put into eval. Default: 0.05 (5%%)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--subdirs", type=str, nargs="*", default=list(DEFAULT_SUBDIRS), help="Subdirectories containing per-frame files")
    args = parser.parse_args()

    if args.config:
        config = load_pipeline_config(args.config, object_name_override=args.object_name)
        runtime_paths = resolve_pipeline_paths(config, object_name_override=args.object_name)
        split_eval_cfg = get_section(config, "split_eval")
        if not bool(split_eval_cfg.get("enabled", True)):
            print("split_eval is disabled in config, skipping.")
            return
        if args.dataset_dir is None:
            args.dataset_dir = runtime_paths["original_object_dir"]
        args.percentage = split_eval_cfg.get("percentage", args.percentage)
        args.seed = split_eval_cfg.get("seed", args.seed)
        args.subdirs = split_eval_cfg.get("subdirs", args.subdirs)

    if args.dataset_dir is None:
        parser.error("--dataset_dir is required when --config is not provided")

    dataset_dir = Path(args.dataset_dir).resolve()
    assert dataset_dir.exists(), f"Dataset dir not found: {dataset_dir}"
    eval_dir = dataset_dir / "eval"
    if not eval_dir.exists():
        eval_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"eval dir already exists, skipping splitting: {eval_dir}")
        return

    # Discover indices from the canonical subdir: prefer rgb, else pick first available
    canonical_subdir = None
    for name in ["rgb", "depth", "masks", "vis"]:
        if (dataset_dir / name).exists():
            canonical_subdir = name
            break
    if canonical_subdir is None:
        raise RuntimeError("No known subdirectories (rgb/depth/masks/vis) found under dataset_dir")

    # Build mapping of indices->ext for each subdir to operate generically
    subdir_index_to_ext: Dict[str, Dict[int, str]] = {}
    for subdir_name in args.subdirs:
        subdir_path = dataset_dir / subdir_name
        subdir_index_to_ext[subdir_name] = discover_indices_and_exts(subdir_path)

    # Frame indices are those present in the canonical dir
    canonical_map = subdir_index_to_ext.get(canonical_subdir, {})
    frame_indices_sorted = sorted(canonical_map.keys())
    if not frame_indices_sorted:
        raise RuntimeError(f"No frames found in {dataset_dir / canonical_subdir}")

    # Ensure continuous 0..N-1 to detect anomalies (we can still proceed)
    num_frames = len(frame_indices_sorted)
    print(f"Discovered {num_frames} frames (from {canonical_subdir}).")

    # Compute eval/train split
    eval_indices = sample_eval_indices(num_frames=num_frames, percentage=args.percentage, seed=args.seed)
    train_indices = [i for i in frame_indices_sorted if i not in set(eval_indices)]
    print(f"Sampling {len(eval_indices)} eval frames ({args.percentage*100:.2f}%). Remaining {len(train_indices)} frames for training.")

    # Build reindex mappings
    old_to_eval_new: Dict[int, int] = {old_idx: new_idx for new_idx, old_idx in enumerate(eval_indices)}
    old_to_train_new: Dict[int, int] = {old_idx: new_idx for new_idx, old_idx in enumerate(train_indices)}

    # Move eval files into eval/ subdir with new indices starting from 0
    move_eval_files(dataset_dir, eval_dir, args.subdirs, eval_indices, subdir_index_to_ext, old_to_eval_new)

    # Rename remaining files in-place to start from 0 and be contiguous
    rename_remaining_to_sequential(dataset_dir, args.subdirs, train_indices, subdir_index_to_ext, old_to_train_new)

    # Split JSONs: write eval JSONs and overwrite originals to match remaining frames
    split_and_write_jsons(dataset_dir, eval_dir, eval_indices, train_indices)

    # Copy intrinsics and write split info
    write_split_info(eval_dir, eval_indices, train_indices)

    print("Done. Eval subset written to:", eval_dir)
    print("Remaining frames and JSONs reindexed in:", dataset_dir)


if __name__ == "__main__":
    main()
