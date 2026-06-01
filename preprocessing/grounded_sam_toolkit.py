import argparse
import os
import sys
import numpy as np
import json
import torch
from PIL import Image

PREPROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
GROUNDING_DINO_DIR = os.path.join(PREPROCESS_DIR, "GroundingDINO")
SAM2_DIR = os.path.join(PREPROCESS_DIR, "segment-anything-2")

if GROUNDING_DINO_DIR not in sys.path:
    sys.path.append(GROUNDING_DINO_DIR)
if SAM2_DIR not in sys.path:
    sys.path.insert(0, SAM2_DIR)

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    masked_ax = ax.imshow(mask_image)
    return masked_ax
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    pos_scatter = ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    neg_scatter = ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    return pos_scatter, neg_scatter

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def load_image(image_path):

    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    pred_logits = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            pred_logits.append(logit.max().item())
        else:
            pred_phrases.append(pred_phrase)
            pred_logits.append(1.0)

    return boxes_filt, pred_phrases, pred_logits

def do_grounding(image_path, model, text_prompt, box_threshold, text_threshold, device):
    # load image
    image_pil, image = load_image(image_path)
    size = image_pil.size
    H, W = size[1], size[0]
    all_boxes_filt = []
    for subtext in text_prompt.split("_"):
        # run grounding dino model
        boxes_filt, pred_phrases, pred_logits = get_grounding_output(
            model, image, subtext, box_threshold, text_threshold, device=device
        )
        max_box_idx = np.argmax(pred_logits)
        boxes_filt = boxes_filt[max_box_idx:max_box_idx+1]
        boxes_filt = boxes_filt * torch.tensor([W, H, W, H], dtype=boxes_filt.dtype, device=boxes_filt.device).unsqueeze(0)
        boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
        boxes_filt[:, 2:] += boxes_filt[:, :2]
        boxes_filt = boxes_filt.cpu()
        all_boxes_filt.append(boxes_filt)
    all_boxes_filt = torch.cat(all_boxes_filt, dim=0)
    ori_image = cv2.imread(image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    return all_boxes_filt, image, ori_image

def convert_png_to_jpg(png_image_path, jpg_image_path):
    try:
        with Image.open(png_image_path) as img:
            img = img.convert('RGB')
            img.save(jpg_image_path, 'JPEG')
    except Exception as e:
        print(f"Error during conversion: {e}")

def load_model_once():
    sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
    sam2_model_cfg = "sam2_hiera_l.yaml"
    device = "cuda"

    config_file = 'preprocessing/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py'
    grounded_checkpoint = 'checkpoints/groundingdino_swinb_cogcoor.pth'
    model = load_model(config_file, grounded_checkpoint, device=device)
    sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("GroundingDINO model loaded")
    print("SAM2 model loaded")
    return sam2_model_cfg, predictor, model, sam2_checkpoint,

mask0 = []
def do_grounded_sam(args, sam2_model_cfg = None, predictor = None, model = None, sam2_checkpoint = None):
    global mask0
    # cfg
    image_ori_dir = os.path.join(args.output_dir, "images_ori")
    image_dir_jpg = os.path.join(args.output_dir, "images_ori_jpg")
    output_dir = os.path.join(args.output_dir, "masks")
    masked_rgb_dir = os.path.join(args.output_dir, "images")
    grounding_vis_dir = os.path.join(args.output_dir, "grounding_visualizations")
    device = "cuda"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(masked_rgb_dir, exist_ok=True)
    os.makedirs(image_dir_jpg, exist_ok=True)
    text_prompt = args.segment_prompt
    box_threshold = 0.3
    text_threshold = 0.25
    if sam2_model_cfg is None :
        sam2_model_cfg, predictor, model, sam2_checkpoint = load_model_once()
    
    if args.track_sam:
        predictor_video = build_sam2_video_predictor(sam2_model_cfg, sam2_checkpoint)
        image_filelist = sorted(os.listdir(image_dir_jpg))
        image_path = os.path.join(image_dir_jpg, image_filelist[0])
        boxes_filt, image, _ = do_grounding(image_path, model, text_prompt, box_threshold, text_threshold, device)
        predictor.set_image(image)
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            if args.interactive_mask:
                if len(mask0) == 0:
                    for i, prompt, box_filt in zip(range(len(text_prompt.split("_"))), text_prompt.split("_"), boxes_filt):
                        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
                        point_coords = np.empty((0, 2))
                        point_labels = np.empty((0, ))
                        pos_scatter = None
                        neg_scatter = None
                        def onclick(event):
                            nonlocal point_coords
                            nonlocal point_labels
                            nonlocal pos_scatter
                            nonlocal neg_scatter
                            nonlocal masked_ax
                            nonlocal mask
                            #nonlocal mask0
                            if event.button == 1:
                                # left button
                                point_coords = np.append(point_coords, np.array([[int(event.xdata), int(event.ydata)]]), axis=0)
                                point_labels = np.append(point_labels, np.array([1, ]), axis=0)
                            elif event.button == 3:
                                # right button
                                point_coords = np.append(point_coords, np.array([[int(event.xdata), int(event.ydata)]]), axis=0)
                                point_labels = np.append(point_labels, np.array([0, ]), axis=0)
                            if pos_scatter is not None:
                                pos_scatter.remove()
                            if neg_scatter is not None:
                                neg_scatter.remove()
                            # import ipdb
                            # ipdb.set_trace()
                            masks, _, _ = predictor.predict(
                                point_coords = point_coords,
                                point_labels = point_labels,
                                # box = box_filt,
                                box = None,
                                multimask_output = False,
                            )
                            #if len(boxes_filt) == 1:
                            mask = masks[0]
                            # else:
                            #     mask = masks.sum(0).astype(bool).astype(np.float32)[0]
                            masked_ax.remove()
                            masked_ax = show_mask(mask, plt.gca(), borders=True)
                            pos_scatter, neg_scatter = show_points(point_coords, point_labels, plt.gca())
                            plt.draw()
                        def on_key_press(event):
                            nonlocal point_coords
                            nonlocal point_labels
                            nonlocal pos_scatter
                            nonlocal neg_scatter
                            nonlocal masked_ax
                            nonlocal mask
                            #nonlocal mask0
                            if event.key == 'q':
                                plt.close()
                            elif event.key == 'z':
                                if point_coords.shape[0] >= 1:
                                    point_coords = point_coords[:-1]
                                    point_labels = point_labels[:-1]
                                masks, _, _ = predictor.predict(
                                    point_coords = point_coords,
                                    point_labels = point_labels,
                                    box = box_filt,
                                    #box= None,
                                    multimask_output = False,
                                )
                                # if len(boxes_filt) == 1:
                                mask = masks[0]
                                # else:
                                #     mask = masks.sum(0).astype(bool).astype(np.float32)[0]
                                if pos_scatter is not None:
                                    pos_scatter.remove()
                                if neg_scatter is not None:
                                    neg_scatter.remove()
                                masked_ax.remove()
                                masked_ax = show_mask(mask, plt.gca(), borders=True)
                                pos_scatter, neg_scatter = show_points(point_coords, point_labels, plt.gca())
                                plt.draw()
                        fig = plt.figure(figsize=(10, 10))
                        plt.imshow(image)
                        cid = fig.canvas.mpl_connect('button_press_event', onclick)
                        cid = fig.canvas.mpl_connect('key_press_event', on_key_press)
                        masked_ax = show_mask(mask, plt.gca(), borders=True)
                        plt.axis('off')
                        plt.show()
                        state = predictor_video.init_state(image_dir_jpg)
                        # add new prompts and instantly get the output on the same frame
                        mask0.append(mask)
                        frame_idx, object_ids, masks = predictor_video.add_new_mask(state, 0, 0, mask)

                        # propagate the prompts to get masklets throughout the video
                        for frame_idx, object_ids, masks in predictor_video.propagate_in_video(state):
                            if len(text_prompt.split("_")) == 1:
                                image_filename = image_filelist[frame_idx].replace(".jpg", ".png")
                            else:
                                image_filename = image_filelist[frame_idx].replace(".jpg", f"_{i}.png")
                            ori_image = cv2.imread(os.path.join(image_ori_dir, image_filelist[frame_idx].replace(".jpg", ".png")))
                            mask_bool = (masks[0, 0] > 0).cpu().numpy()
                            mask_uint8 = mask_bool.astype(np.uint8) * 255
                            cv2.imwrite(os.path.join(output_dir, image_filename), mask_uint8)
                            bg_color = 255 if args.bg_color == "white" else 0
                            bg_image = np.ones_like(ori_image) * bg_color
                            masked_image = np.where(mask_bool[..., None], ori_image, bg_image)
                            cv2.imwrite(os.path.join(masked_rgb_dir, image_filename), np.array(masked_image, dtype=np.uint8))
                else:
                    for i in range(len(text_prompt.split("_"))):
                        state = predictor_video.init_state(image_dir_jpg)
                            # add new prompts and instantly get the output on the same frame
                        frame_idx, object_ids, masks = predictor_video.add_new_mask(state, 0, 0, mask0[i])

                            # propagate the prompts to get masklets throughout the video
                        for frame_idx, object_ids, masks in predictor_video.propagate_in_video(state):
                            if len(text_prompt.split("_")) == 1:
                                image_filename = image_filelist[frame_idx].replace(".jpg", ".png")
                            else:
                                image_filename = image_filelist[frame_idx].replace(".jpg", f"_{i}.png")
                            ori_image = cv2.imread(os.path.join(image_ori_dir, image_filelist[frame_idx].replace(".jpg", ".png")))
                            mask_bool = (masks[0, 0] > 0).cpu().numpy()
                            mask_uint8 = mask_bool.astype(np.uint8) * 255
                            cv2.imwrite(os.path.join(output_dir, image_filename), mask_uint8)
                            bg_color = 255 if args.bg_color == "white" else 0
                            bg_image = np.ones_like(ori_image) * bg_color
                            masked_image = np.where(mask_bool[..., None], ori_image, bg_image)
                            cv2.imwrite(os.path.join(masked_rgb_dir, image_filename), np.array(masked_image, dtype=np.uint8))
            else:
                masks, _, _ = predictor.predict(
                    point_coords = None,
                    point_labels = None,
                    box = boxes_filt,
                    multimask_output = False,
                )
                if len(boxes_filt) == 1:
                    mask0 = masks[0]
                else:
                    mask0 = masks.sum(0).astype(bool).astype(np.float32)[0]
                state = predictor_video.init_state(image_dir_jpg)
                # add new prompts and instantly get the output on the same frame
                frame_idx, object_ids, masks = predictor_video.add_new_mask(state, 0, 0, mask0)

                # propagate the prompts to get masklets throughout the video
                for frame_idx, object_ids, masks in predictor_video.propagate_in_video(state):
                    image_filename = image_filelist[frame_idx].replace(".jpg", ".png")
                    ori_image = cv2.imread(os.path.join(image_ori_dir, image_filename))
                    mask_bool = (masks[0, 0] > 0).cpu().numpy()
                    mask_uint8 = mask_bool.astype(np.uint8) * 255
                    cv2.imwrite(os.path.join(output_dir, image_filename), mask_uint8)
                    bg_color = 255 if args.bg_color == "white" else 0
                    bg_image = np.ones_like(ori_image) * bg_color
                    masked_image = np.where(mask_bool[..., None], ori_image, bg_image)
                    cv2.imwrite(os.path.join(masked_rgb_dir, image_filename), np.array(masked_image, dtype=np.uint8))
            if args.mask_only:
                import shutil
                shutil.rmtree(masked_rgb_dir)
                shutil.rmtree(image_dir_jpg)
    else:
        image_filelist = sorted(os.listdir(image_dir_jpg))
        print("Processing each frame with text prompts...")
        
        for idx, image_file in enumerate(tqdm(image_filelist)):
            image_path = os.path.join(image_dir_jpg, image_file)
            
            boxes_filt, image, _ = do_grounding(image_path, model, text_prompt, box_threshold, text_threshold, device)
            
            predictor.set_image(image)
            
            with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
                if args.interactive_mask and idx == 0:
                    print("Interactive mode only supported with --track_sam flag")
                    print("Using automatic mode instead...")
            
                for i, (prompt, box_filt) in enumerate(zip(text_prompt.split("_"), boxes_filt)):
                    masks, _, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box_filt.unsqueeze(0),
                        multimask_output=False,
                    )
                    
                    mask = masks[0]
                    
                    if len(text_prompt.split("_")) == 1:
                        image_filename = image_file.replace(".jpg", ".png")
                    else:
                        image_filename = image_file.replace(".jpg", f"_{i}.png")
                    
                    ori_image = cv2.imread(os.path.join(image_ori_dir, image_file.replace(".jpg", ".png")))
                    
                    # Save mask.
                    mask_uint8 = mask.astype(np.uint8) * 255
                    cv2.imwrite(os.path.join(output_dir, image_filename), mask_uint8)
                    
                    if not args.mask_only:
                        bg_color = 255 if args.bg_color == "white" else 0
                        bg_image = np.ones_like(ori_image) * bg_color
                        mask_bool = mask.astype(bool)
                        masked_image = np.where(mask_bool[..., None], ori_image, bg_image)
                        cv2.imwrite(os.path.join(masked_rgb_dir, image_filename), np.array(masked_image, dtype=np.uint8))

        print(f"Processed {len(image_filelist)} frames")
        
        if args.mask_only:
            import shutil
            shutil.rmtree(masked_rgb_dir)
            shutil.rmtree(image_dir_jpg)

def propagate_mask_in_video(args, first_frame_mask, image_dir_jpg, image_ori_dir, sam2_model_cfg = None, predictor = None, model = None, sam2_checkpoint = None):

    device = "cuda"
    
    predictor_video = build_sam2_video_predictor(sam2_model_cfg, sam2_checkpoint)
    
    image_filelist = sorted(os.listdir(image_dir_jpg))
    
    state = predictor_video.init_state(image_dir_jpg)
    
    frame_idx, object_ids, masks = predictor_video.add_new_mask(state, 0, 0, first_frame_mask)
    
    for frame_idx, object_ids, masks in predictor_video.propagate_in_video(state):

        image_filename = image_filelist[frame_idx].replace(".jpg", ".png")
        
        # Read original image.
        ori_image = cv2.imread(os.path.join(image_ori_dir, image_filelist[frame_idx].replace(".jpg", ".png")))
        
        # Process mask.
        mask_bool = (masks[0, 0] > 0).cpu().numpy()
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        # import ipdb
        # ipdb.set_trace()
        save_dir = args.output_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 
        # Save mask.
        cv2.imwrite(os.path.join(args.output_dir, image_filename), mask_uint8)
        
        # Create masked image.
        bg_color_value = 255 if args.bg_color == "white" else 0
        bg_image = np.ones_like(ori_image) * bg_color_value
        masked_image = np.where(mask_bool[..., None], ori_image, bg_image)
        cv2.imwrite(os.path.join(args.masked_rgb_dir, image_filename), np.array(masked_image, dtype=np.uint8))
    
    print(f"Video mask propagation completed. Output written to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--segment_prompt', type=str)
    parser.add_argument('--track_sam', action="store_true")
    parser.add_argument("--bg_color", type=str, default="white", choices=['white', 'black'])
    parser.add_argument("--interactive_mask", action="store_true")
    parser.add_argument("--mask_only", action="store_true")
    args = parser.parse_args()
    do_grounded_sam(args)
    #python preprocessing/grounded_sam_for_opt.py --output_dir logs --segment_prompt "franka"  --bg_color black --input_dir logs/images_ori --track_sam
