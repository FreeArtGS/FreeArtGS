import os
import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from termcolor import colored

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.pipeline_config import get_section, load_pipeline_config, resolve_pipeline_paths

def cprint(text, color):
    print(colored(text, color))

def render_segmap(seg_map, object_mask=None):
    finite_mask = np.isfinite(seg_map)
    rendered = np.full(seg_map.shape + (3,), 255, dtype=np.uint8)
    if not np.any(finite_mask):
        return rendered
    clipped = np.clip(seg_map, 0.0, 1.0)
    colorized = cv2.applyColorMap((clipped * 255).astype(np.uint8), cv2.COLORMAP_JET)
    rendered[finite_mask] = colorized[finite_mask]
    if object_mask is not None:
        if object_mask.shape != seg_map.shape:
            object_mask = cv2.resize(
                object_mask.astype(np.uint8),
                (seg_map.shape[1], seg_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        else:
            object_mask = object_mask.astype(bool)
        rendered[~object_mask] = 255
    return rendered


def load_object_masks(dataset_dir):
    base = Path(dataset_dir)
    candidate_dirs = [
        base / "masks",
        base.parent / "masks",
        base.parent.parent / "masks",
    ]

    for mask_dir in candidate_dirs:
        if not mask_dir.is_dir():
            continue
        mask_files = sorted(
            path for path in mask_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if mask_files:
            cprint(f"Loaded {len(mask_files)} masks from {mask_dir}", "cyan")
            return mask_files

    cprint(
        f"No mask directory with images found near {dataset_dir}, keeping default background rendering.",
        "yellow",
    )
    return []


def load_mask_for_frame(mask_files, frame_idx):
    num_masks = len(mask_files)
    if num_masks == 0 or frame_idx < 0:
        return None

    if frame_idx >= num_masks:
        if num_masks == 1:
            frame_idx = 0
        else:
            # Map ping-pong frame index back to the original sequence index.
            period = 2 * num_masks - 2
            idx = frame_idx % period
            frame_idx = idx if idx < num_masks else period - idx

    mask = cv2.imread(str(mask_files[frame_idx]), cv2.IMREAD_UNCHANGED)
    if mask is None:
        cprint(f"Warning: Could not read mask {mask_files[frame_idx]}", "yellow")
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask > 0


def find_all_images_in_tracking(tracking_dir):
    """Find all seg_map_*.npy files in the tracking directory, sorted by window and frame."""
    all_images = {}  # frame_idx -> npy_path
    
    # Iterate over all window directories.
    for window_dir in sorted(Path(tracking_dir).glob("window_*")):
        if not window_dir.is_dir():
            continue
            
        window_idx = int(window_dir.name.split('_')[1])
        
        # Read window metadata to get frame indices.
        window_info_path = window_dir / "window_info.json"
        frame_indices = []
        
        if window_info_path.exists():
            try:
                with open(window_info_path, 'r') as f:
                    window_info = json.load(f)
                frame_indices = window_info.get('frame_indices', [])
            except:
                cprint(f"Warning: Could not read {window_info_path}", "yellow")
        
        segmap_files = sorted(window_dir.glob("seg_map_*.npy"))

        
        if not segmap_files:
            cprint(f"No seg_map_*.npy files found in window {window_idx}", "yellow")
            continue
            
        cprint(f"Window {window_idx}: found {len(segmap_files)} seg_map arrays, frames {frame_indices}", "cyan")
        
        def extract_segmap_number(path):
            import re
            match = re.search(r'seg_map_(\d+)\.npy', path.name)
            return int(match.group(1)) if match else 0

        segmap_files.sort(key=extract_segmap_number)
        
        # Map images to frame indices.
        for i, img_path in enumerate(segmap_files):
            if i < len(frame_indices):
                frame_idx = frame_indices[i]
                all_images[frame_idx] = {
                    'path': str(img_path),
                    'window_idx': window_idx
                }
            elif not frame_indices:  # If frame_indices are missing, use sequential numbering.
                frame_idx = window_idx * 1000 + i  # Simple numbering scheme.
                all_images[frame_idx] = {
                    'path': str(img_path),
                    'window_idx': window_idx
                }
    
    return all_images

def create_video_from_images(args):
    """Create a video from seg_map_*.npy files."""
    
    # Find all images.
    all_images = find_all_images_in_tracking(args.tracking_dir)
    
    if not all_images:
        cprint("No images found in tracking directory!", "red")
        return
    
    # Sort frames.
    sorted_frame_indices = sorted(all_images.keys())
    cprint(f"Found {len(sorted_frame_indices)} images from frame {sorted_frame_indices[0]} to {sorted_frame_indices[-1]}", "green")

    total_frames = len(sorted_frame_indices)
    dataset_dir = os.path.dirname(os.path.abspath(args.tracking_dir))
    mask_files = load_object_masks(dataset_dir)

    def _count_images(dir_path):
        if not os.path.isdir(dir_path):
            return 0
        files = [f for f in os.listdir(dir_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        return len(sorted(files))

    orig_candidates = [
        os.path.join(dataset_dir, "images_ori"),
        os.path.join(dataset_dir, "images"),
        os.path.join(dataset_dir, "rgb"),
        os.path.join(dataset_dir, "images_ori_jpg"),
    ]
    orig_len = 0
    for d in orig_candidates:
        orig_len = _count_images(d)
        if orig_len > 0:
            break

    if orig_len > 0 and total_frames >= max(2 * orig_len - 2, orig_len + 1):
        start_idx = max(0, total_frames - orig_len)
        kept_indices = sorted_frame_indices[start_idx:]
        sorted_frame_indices = kept_indices[::-1]
        cprint(f"Detected ping-pong sequence: total={total_frames}, orig={orig_len}. Keeping latter half only.", "cyan")
    else:
        cprint(f"No ping-pong detected or unknown original length (total={total_frames}, orig={orig_len}). Using full sequence.", "cyan")
    
    first_image_path = all_images[sorted_frame_indices[0]]['path']
    try:
        first_seg_map = np.load(first_image_path)
    except Exception as exc:
        cprint(f"Could not read first seg map: {first_image_path} ({exc})", "red")
        return

    first_mask = load_mask_for_frame(mask_files, sorted_frame_indices[0]) if mask_files else None
    first_frame = render_segmap(first_seg_map, first_mask)
    height, width = first_frame.shape[:2]
    cprint(f"Video dimensions: {width}x{height}", "blue")
    
    # Create output directory.
    output_dir = os.path.join(args.tracking_dir , "seg_map_video")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up video writer.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(output_dir, f"segmentation_video.mp4")
    out = cv2.VideoWriter(video_path, fourcc, args.fps, (width, height))
    
    # Process each frame.
    cprint("Creating video...", "cyan")
    
    for frame_idx in tqdm(sorted_frame_indices, desc="Processing frames"):
        image_info = all_images[frame_idx]
        image_path = image_info['path']
        window_idx = image_info['window_idx']
        
        try:
            try:
                seg_map = np.load(image_path)
                object_mask = load_mask_for_frame(mask_files, frame_idx) if mask_files else None
                frame = render_segmap(seg_map, object_mask)
            except Exception:
                cprint(f"Warning: Could not read {image_path}", "yellow")
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            # Write frame.
            out.write(frame)
            
        except Exception as e:
            cprint(f"Error processing frame {frame_idx}: {e}", "red")
            # Write a black frame.
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)
            out.write(black_frame)
    
    # Release resources.
    out.release()
    
    cprint(f"Video created successfully!", "green")
    cprint(f"Output path: {video_path}", "blue")
    cprint(f"Video stats: {len(sorted_frame_indices)} frames, {width}x{height}px, {args.fps}fps", "blue")
    
    # Generate preview information.
    total_duration = len(sorted_frame_indices) / args.fps
    cprint(f" Duration: {total_duration:.1f} seconds", "blue")

def main():
    parser = argparse.ArgumentParser(description="Create video from segmentation images in tracking windows")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline config YAML")
    parser.add_argument("--object_name", '-on', type=str, 
                       default=None,
                       help="Path to tracking directory containing window_* folders")
    parser.add_argument("--tracking_dir", type=str, 
                       default=None,
                       help="Path to tracking directory containing window_* folders")
    
    parser.add_argument("--fps", type=int, default=30,
                       help="Video frame rate (frames per second)")
    
    args = parser.parse_args()
    if args.config:
        config = load_pipeline_config(args.config, object_name_override=args.object_name)
        runtime_paths = resolve_pipeline_paths(config, object_name_override=args.object_name)
        tracking_postprocess_cfg = get_section(config, "tracking_postprocess")
        if args.object_name is None:
            args.object_name = runtime_paths["object_name"]
        if args.tracking_dir is None:
            args.tracking_dir = runtime_paths["tracking_dir"]
        args.fps = tracking_postprocess_cfg.get("segmap_video_fps", args.fps)
    elif args.tracking_dir is None and args.object_name is not None:
        args.tracking_dir = f"./datasets/{args.object_name}/tracking"

    if args.tracking_dir is None:
        parser.error("--tracking_dir or --object_name is required when --config is not provided")
    # Check input directory.
    if not os.path.exists(args.tracking_dir):
        cprint(f" Tracking directory not found: {args.tracking_dir}", "red")
        return
    
    # Show the path to process.
    cprint(f"Scanning tracking directory: {args.tracking_dir}", "cyan")
    
    create_video_from_images(args)

if __name__ == "__main__":
    main()
