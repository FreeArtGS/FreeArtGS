#!/usr/bin/env python3
"""
Export moving_map.npz from rendered segmentation masks.

This script generates the moving_map.npz file needed for evaluation.
It requires rendering the trained model to get per-frame part segmentation.
"""

import os
import argparse
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import json


def load_part_weight_rendering(render_dir, frame_indices=None):
    """
    Load part weight renderings from a directory.
    
    Expected format: render_dir should contain part_weight_*.png or part_weight_*.npy files
    where the part weight is a grayscale image with values in [0, 1] or [0, 255]
    """
    render_dir = Path(render_dir)
    
    # Try different file patterns
    patterns = [
        "part_weight_*.png",
        "part_weight_*.npy",
        "part_*.png",
        "moving_mask_*.png",
        "weight_*.png"
    ]
    
    files = []
    for pattern in patterns:
        files = sorted(render_dir.glob(pattern))
        if files:
            print(f"Found {len(files)} files matching pattern: {pattern}")
            break
    
    if not files:
        raise FileNotFoundError(
            f"No part weight renderings found in {render_dir}\n"
            f"Tried patterns: {patterns}"
        )
    
    moving_maps = []
    
    for file_path in files:
        if file_path.suffix == '.npy':
            # Load numpy array
            weight = np.load(file_path)
        else:
            # Load image
            img = Image.open(file_path).convert('L')
            weight = np.array(img).astype(np.float32)
            # Normalize to [0, 1] if needed
            if weight.max() > 1.0:
                weight = weight / 255.0
        
        moving_maps.append(weight)
    
    moving_maps = np.stack(moving_maps)
    print(f"Loaded {len(moving_maps)} moving maps with shape {moving_maps[0].shape}")
    print(f"Value range: [{moving_maps.min():.4f}, {moving_maps.max():.4f}]")
    
    return moving_maps


def create_moving_map_from_part_weights(part_weights, threshold=0.5):
    """
    Convert part weights to binary moving masks.
    
    Args:
        part_weights: (N, H, W) array of part weights in [0, 1]
        threshold: threshold for binarization (default: 0.5)
    
    Returns:
        moving_masks: (N, H, W) boolean array
    """
    moving_masks = part_weights > threshold
    print(f"Created binary moving masks with threshold={threshold}")
    print(f"Moving pixels: {moving_masks.sum()} / {moving_masks.size} "
          f"({100*moving_masks.sum()/moving_masks.size:.2f}%)")
    
    return moving_masks


def export_moving_map(
    render_dir,
    output_path,
    threshold=0.5,
    resize=None
):
    """
    Export moving_map.npz from rendered part weights.
    
    Args:
        render_dir: Directory containing rendered part weight images
        output_path: Output path for moving_map.npz
        threshold: Threshold for binarizing part weights (default: 0.5)
        resize: Optional (H, W) to resize the masks
    """
    # Load part weight renderings
    part_weights = load_part_weight_rendering(render_dir)
    
    # Resize if needed
    if resize is not None:
        H, W = resize
        print(f"Resizing from {part_weights.shape[1:]} to ({H}, {W})...")
        resized_weights = []
        for weight in part_weights:
            resized = cv2.resize(weight, (W, H), interpolation=cv2.INTER_LINEAR)
            resized_weights.append(resized)
        part_weights = np.stack(resized_weights)
    
    # Create binary moving masks
    moving_masks = create_moving_map_from_part_weights(part_weights, threshold)
    
    # Convert to float for compatibility
    moving_masks = moving_masks.astype(np.float32)
    
    # Save to npz with key 'a' (as expected by evaluation script)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(output_path, a=moving_masks)
    print(f"\nSaved moving_map.npz to {output_path}")
    print(f"Shape: {moving_masks.shape}")
    print(f"Dtype: {moving_masks.dtype}")


def export_moving_map_from_gaussian_splatting(
    splatfacto_art_dir,
    output_dir,
    image_size=(480, 640),
    threshold=0.5,
    joint_type=None
):
    """
    Alternative method: Generate moving map by rendering the trained model.
    
    This is a placeholder - you need to implement the actual rendering logic
    using your splatfacto-art model.
    
    Args:
        splatfacto_art_dir: Path to splatfacto-art output directory
        output_dir: Output directory for evaluation data
        image_size: (H, W) for rendered images
        threshold: Threshold for part weights
        joint_type: 'revolute' or 'prismatic' (or None to auto-detect)
    """
    splatfacto_art_dir = Path(splatfacto_art_dir)
    output_dir = Path(output_dir)
    
    # Auto-detect joint type if not provided
    if joint_type is None:
        articulation_path = splatfacto_art_dir / "articulation.json"
        if articulation_path.exists():
            with open(articulation_path, 'r') as f:
                data = json.load(f)
                joint_type = data['joint_type']
                print(f"Auto-detected joint type: {joint_type}")
        else:
            raise ValueError("Cannot auto-detect joint type. Please specify --joint_type")
    
    # Check if renders already exist
    render_dir = splatfacto_art_dir / "renders"
    if render_dir.exists():
        print(f"Using existing renders from {render_dir}")
        output_path = output_dir / joint_type / "moving_map.npz"
        export_moving_map(render_dir, output_path, threshold, resize=image_size)
    else:
        print(f"\n{'='*60}")
        print("ERROR: No renders found!")
        print(f"Expected directory: {render_dir}")
        print("\nYou need to render the trained model first.")
        print("Please use the nerfstudio rendering pipeline:")
        print(f"\n  ns-render dataset --load-config {splatfacto_art_dir}/config.yml \\")
        print(f"             --output-path {render_dir} \\")
        print(f"             --rendered-output-names part_weight")
        print(f"\nOr implement custom rendering in this script.")
        print(f"{'='*60}\n")
        raise FileNotFoundError(f"Render directory not found: {render_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Export moving_map.npz from rendered part weight images"
    )
    
    # Two modes: from existing renders or from model
    subparsers = parser.add_subparsers(dest='mode', help='Export mode')
    
    # Mode 1: From existing render directory
    parser_renders = subparsers.add_parser(
        'from_renders',
        help='Export from existing rendered images'
    )
    parser_renders.add_argument(
        "--render_dir",
        type=str,
        required=True,
        help="Directory containing rendered part weight images"
    )
    parser_renders.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for moving_map.npz"
    )
    parser_renders.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binarizing part weights (default: 0.5)"
    )
    parser_renders.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=('H', 'W'),
        help="Resize masks to (H, W), e.g., --resize 480 640"
    )
    
    # Mode 2: From splatfacto-art model
    parser_model = subparsers.add_parser(
        'from_model',
        help='Export from splatfacto-art model (requires rendering first)'
    )
    parser_model.add_argument(
        "--splatfacto_art_dir",
        type=str,
        required=True,
        help="Path to splatfacto-art output directory"
    )
    parser_model.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for evaluation data"
    )
    parser_model.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[480, 640],
        metavar=('H', 'W'),
        help="Image size (H, W) for rendered images (default: 480 640)"
    )
    parser_model.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for part weights (default: 0.5)"
    )
    parser_model.add_argument(
        "--joint_type",
        type=str,
        choices=['revolute', 'prismatic'],
        help="Joint type (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    if args.mode == 'from_renders':
        resize = tuple(args.resize) if args.resize else None
        export_moving_map(
            render_dir=args.render_dir,
            output_path=args.output_path,
            threshold=args.threshold,
            resize=resize
        )
    elif args.mode == 'from_model':
        export_moving_map_from_gaussian_splatting(
            splatfacto_art_dir=args.splatfacto_art_dir,
            output_dir=args.output_dir,
            image_size=tuple(args.image_size),
            threshold=args.threshold,
            joint_type=args.joint_type
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

