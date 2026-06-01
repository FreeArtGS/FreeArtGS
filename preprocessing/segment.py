import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image
import shutil
import cv2
from termcolor import cprint
import numpy as np 
import argparse
from grounded_sam_toolkit import load_model_once, do_grounded_sam
from utils.pipeline_config import load_pipeline_config, resolve_pipeline_paths, resolve_preprocess, set_arg_defaults
def extract_frames(args):
    video_path = os.path.join(args.input_dir, "rgb_video.mp4")
    frame_dir = os.path.join(args.output_dir, "images_ori")
    num_frames = 100

    os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = range(total_frames)
    if num_frames is not None and num_frames < total_frames:
        # Uniform sampling.
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    saved_count = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_path = os.path.join(frame_dir, f"{idx:05d}.png")
        cv2.imwrite(str(frame_path), frame)
        saved_count += 1
    cap.release()
    cprint(f"extract {saved_count}/{total_frames} frames to {frame_dir}", "green")

def convert_png_to_jpg(args):
    frame_dir = os.path.join(args.output_dir, "images_ori")
    
    jpg_dir = os.path.join(args.output_dir, "images_ori_jpg")
    if os.path.exists(jpg_dir):
        shutil.rmtree(jpg_dir)
    os.makedirs(jpg_dir, exist_ok=True)
    for fname in os.listdir(frame_dir):
        fpath = os.path.join(frame_dir, fname)
        if fname.lower().endswith(".png"):
            jpg_fname = fname.replace(".png", ".jpg")
            jpg_path = os.path.join(jpg_dir, jpg_fname)
            Image.open(fpath).convert('RGB').save(jpg_path, 'JPEG')

    cprint(f"convert {len(os.listdir(frame_dir))} png files to {len(os.listdir(jpg_dir))} jpg files", "green")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment a video using SAM2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline config YAML")
    parser.add_argument("--object_name", type=str, default=None, help="Object name override used with --config")
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("--output_dir", type=str, default=None, 
                          help="Directory to save the output dataset")
    io_group.add_argument("--input_dir", type=str, default=None, 
                          help="Input directory containing the video or RGB images")

    # Segmentation options.
    segment_group = parser.add_argument_group("Segmentation")
    segment_group.add_argument("--segment_prompt", type=str, default=None, 
                               help="Prompt for segmentation (e.g., object name or description)")
    segment_group.add_argument("--track_sam", action="store_true", default=None,
                               help="Enable SAM-based tracking")
    segment_group.add_argument("--interactive_mask", action="store_true", default=None,
                               help="Enable interactive mask editing")
    segment_group.add_argument("--mask_only", action="store_true", 
                               help="Generate masks only without further processing")
    segment_group.add_argument("--bg_color", type=str, default=None, choices=['white', 'black'], 
                               help="Background color for visualization")
    segment_group.add_argument("--skip_segment", action="store_true", default=None,
                               help="Skip segmentation")
    args = parser.parse_args()

    if args.config:
        config = load_pipeline_config(args.config, object_name_override=args.object_name)
        runtime_paths = resolve_pipeline_paths(config, object_name_override=args.object_name)
        preprocess_cfg = resolve_preprocess(config)
        if args.input_dir is None:
            args.input_dir = runtime_paths["original_object_dir"]
        if args.output_dir is None:
            args.output_dir = runtime_paths["dataset_root"]
    else:
        preprocess_cfg = resolve_preprocess({})

    set_arg_defaults(
        args,
        {
            "segment_prompt": preprocess_cfg["segment_prompt"],
            "track_sam": preprocess_cfg["track_sam"],
            "interactive_mask": preprocess_cfg["interactive_mask"],
            "skip_segment": preprocess_cfg["skip_segment"],
            "bg_color": preprocess_cfg["bg_color"],
        },
    )

    if args.input_dir is None:
        parser.error("--input_dir is required when --config is not provided")
    if args.output_dir is None:
        args.output_dir = "datasets"

    cprint("="*60, "cyan")
    cprint("Preprocessing raw frames", "green", attrs=['bold'])
    cprint("-"*60, "cyan")
    cprint("Extracting/copying frames", "green")

    args.object_name = args.input_dir.split("/")[-1]
    args.output_dir = f"{args.output_dir}/{args.object_name}"
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isdir(os.path.join(args.input_dir, "rgb")):
        rgb_dir = os.path.join(args.input_dir, "rgb")
        for fname in os.listdir(args.input_dir):
            src_path = os.path.join(args.input_dir, fname)
            dst_path = os.path.join(args.output_dir, fname)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
            elif os.path.isdir(src_path):
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                if fname == "rgb":
                    dst_path = os.path.join(args.output_dir, "images_ori")
                    if os.path.exists(dst_path):
                        shutil.rmtree(dst_path)
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

        cprint(f"copy {len(os.listdir(rgb_dir))} images to {args.output_dir}", "green")

    elif os.path.isfile(os.path.join(args.input_dir, "rgb_video.mp4")):
        extract_frames(args)
    else:
        cprint(f"input_dir {args.input_dir} is not a valid directory", "red")
        raise ValueError(f"input_dir {args.input_dir} is not a valid directory")
        

    convert_png_to_jpg(args)
    cprint("-"*60, "cyan")

    if not args.skip_segment:
        cprint(f"Do segment\nsegment_prompt: {args.segment_prompt}", "green")
        sam2_model_cfg, predictor, model, sam2_checkpoint = load_model_once()
        do_grounded_sam(args, sam2_model_cfg, predictor, model, sam2_checkpoint)
    else:
        cprint("Generating images using mask and rgb ...", "green")
        rgb_dir = os.path.join(args.output_dir, "images_ori")
        mask_dir = os.path.join(args.output_dir, "masks")
        output_img_dir = os.path.join(args.output_dir, "images")
        os.makedirs(output_img_dir, exist_ok=True)

        rgb_files = sorted([
            f for f in os.listdir(rgb_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        mask_files = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        from PIL import Image
        import numpy as np

        for fname in rgb_files:
            rgb_path = os.path.join(rgb_dir, fname)
            mask_path = os.path.join(mask_dir, fname)
            if not os.path.exists(mask_path):
                basename = os.path.splitext(fname)[0]
                mask_candidates = [m for m in mask_files if os.path.splitext(m)[0]==basename]
                if mask_candidates:
                    mask_path = os.path.join(mask_dir, mask_candidates[0])
                else:
                    cprint(f"[warn] No mask found for {fname}, skipping.", "yellow")
                    continue
            rgb_img = Image.open(rgb_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("L")
            # Binarize mask.
            mask = np.array(mask_img)
            mask_bin = (mask > 127).astype(np.uint8)[..., None]
            rgb_arr = np.array(rgb_img)
            output_arr = rgb_arr * mask_bin + 255 * (1-mask_bin)
            output_arr = output_arr.astype(np.uint8)
            out_img = Image.fromarray(output_arr)
            out_path = os.path.join(output_img_dir, fname)
            out_img.save(out_path)
