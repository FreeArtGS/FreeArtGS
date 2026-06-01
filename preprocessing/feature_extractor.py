import os, sys
import argparse
from pathlib import Path
from typing import Optional
from termcolor import cprint
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.pipeline_config import load_pipeline_config, resolve_pipeline_paths, resolve_preprocess, set_arg_defaults
from utils.utils import read_image_folder

class FeatureExtractor:
    """
    A class to extract deep features from images using the DINOv3 model.
    """
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    MASK_FG_THRESHOLD = 0.5
    MODEL_TO_NUM_LAYERS = {
        "dinov3_vits16": 12,
        "dinov3_vits16plus": 12,
        "dinov3_vitb16": 12,
        "dinov3_vitl16": 24,
        "dinov3_vith16plus": 32,
        "dinov3_vit7b16": 40,
    }

    def __init__(self, model_name="dinov3_vitl16", patch_size=16, image_size=768, device="cuda", checkpoint_path: Optional[str] = None):
        """
        Initializes the FeatureExtractor.

        Args:
            model_name (str): The name of the DINOv3 model to use.
            patch_size (int): The patch size for the model.
            image_size (int): The target image size for processing.
            device (str): The device to run the model on ('cuda' or 'cpu').
            checkpoint_path (str | None): Path to a local checkpoint (.safetensors or .pth). If provided, it will be loaded.
        """
        self.patch_size = patch_size
        self.image_size = image_size
        self.device = device
        self.model_name = model_name
        
        # Default to repository-local checkpoint if not provided
        if checkpoint_path is None:
            checkpoint_path = Path(__file__).resolve().parents[1] / "checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        self.checkpoint_path = str(checkpoint_path)
        
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        self.pca = None
        
        # quantization filter for the given patch size
        self.patch_quant_filter = torch.nn.Conv2d(1, 1, self.patch_size, stride=self.patch_size, bias=False)
        self.patch_quant_filter.weight.data.fill_(1.0 / (self.patch_size * self.patch_size))
        self.patch_quant_filter.to(self.device)

    def _load_model(self):
        """Loads the DINOv3 model and (optionally) local weights from checkpoints/dinov3/..."""
        # Prefer bundled local dinov3 repo if available; otherwise fall back to env var or GitHub
        local_repo = Path(__file__).resolve().parent / "dinov3"
        if local_repo.exists():
            dinov3_location = str(local_repo)
            source = "local"
        else:
            dinov3_location = os.getenv("DINOV3_LOCATION", "facebookresearch/dinov3")
            source = "local" if os.getenv("DINOV3_LOCATION") else "github"
        #print(f"DINOv3 location set to {dinov3_location} (source={source})")
        
        # Load architecture only to avoid internal weight downloading/hashing constraints
        model = torch.hub.load(
            repo_or_dir=dinov3_location,
            model=self.model_name,
            source=source,
            pretrained=False,
        )

        # Try loading local checkpoint if present
        ckpt_path = Path(self.checkpoint_path)
        if ckpt_path.exists():
            print(f"Loading weights from checkpoint: {ckpt_path}")
            state_dict = None
            try:
                # Prefer safetensors when file extension matches
                if ckpt_path.suffix == ".safetensors":
                    from safetensors.torch import load_file as safe_load_file
                    state_dict = safe_load_file(str(ckpt_path), device="cpu")
                else:
                    # Use weights_only=True to avoid pickle execution and silence FutureWarning (PyTorch >= 2.4)
                    try:
                        state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
                    except TypeError:
                        # Fallback for older PyTorch without weights_only
                        state_dict = torch.load(str(ckpt_path), map_location="cpu")
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint from {ckpt_path}: {e}")

            # Unwrap common containers
            if isinstance(state_dict, dict):
                # Some checkpoints store under 'model' or 'state_dict'
                for key in ["state_dict", "model", "module"]:
                    if key in state_dict and isinstance(state_dict[key], dict):
                        state_dict = state_dict[key]
                        break

            # Remove a possible 'module.' prefix from DDP
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Warning: {len(missing)} missing keys when loading checkpoint (showing first 5): {missing[:5]}")
            if unexpected:
                print(f"Warning: {len(unexpected)} unexpected keys in checkpoint (showing first 5): {unexpected[:5]}")
        else:
            print(f"Checkpoint not found at {ckpt_path}. Using randomly initialized weights.")
        return model

    def _resize_transform(self, image: Image) -> torch.Tensor:
        """Resizes and converts an image to a tensor."""
        w, h = image.size
        h_patches = int(self.image_size / self.patch_size)
        w_patches = int((w * self.image_size) / (h * self.patch_size))
        return TF.to_tensor(TF.resize(image, (h_patches * self.patch_size, w_patches * self.patch_size)))


    def process_image(self, image_array) -> torch.Tensor:
        """Like process_image, but takes an in-memory RGB numpy array instead of a path."""
        image = Image.fromarray(image_array).convert('RGB')
        image_resized = self._resize_transform(image)
        image_resized = TF.normalize(image_resized, mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        image_resized = image_resized.unsqueeze(0).to(self.device)
        
        n_layers = self.MODEL_TO_NUM_LAYERS[self.model_name]
        with torch.inference_mode():
            with torch.autocast(device_type=self.device.split(':')[0], dtype=torch.float32):
                feats = self.model.get_intermediate_layers(image_resized, n=range(n_layers), reshape=True, norm=True)
                patch_features = feats[-1].squeeze().detach().cpu()
        return patch_features

    def fit_pca(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Fits PCA (n_components=3) on features. If a mask is provided, fits only on foreground features."""
        self.pca = PCA(n_components=3, whiten=True)
        dim, h, w = features.shape
        features_reshaped = features.view(dim, -1).permute(1, 0)
        
        if mask is not None:
            # Flatten mask and select foreground features for fitting
            mask_flat = mask.view(-1)
            fg_features = features_reshaped[mask_flat > self.MASK_FG_THRESHOLD]
            print(f"Fitting PCA on {len(fg_features)} foreground features...")
            self.pca.fit(fg_features.numpy())
        else:
            print("Fitting PCA on all features...")
            self.pca.fit(features_reshaped.numpy())

    def visualize_and_save(self, features: torch.Tensor, output_path: str, mask: Optional[torch.Tensor] = None, target_hw: Optional[tuple] = None, per_frame_pca: bool = False):
        """
        Projects features to 3 dimensions using PCA and saves as a visualization image.
        If target_hw is provided, features are first upsampled to the original image size;
        the mask is used directly at its original pixel-level resolution without interpolation.
        When per_frame_pca=True: PCA is fitted individually for each frame within its mask,
        not sharing standards with other frames.

        Args:
            features: (C,H,W) patch-level features.
            output_path: Output path for the visualization.
            mask: Pixel-level foreground mask (H_img,W_img) or patch-level (H_patch,W_patch).
                  Ignored if dimensions do not match the final visualization size.
            target_hw: (target_h, target_w) Original image dimensions for upsampling.
            per_frame_pca: Whether to fit PCA individually for each frame.
        """
        dim, h, w = features.shape

        # Upsample features to original image size (features only)
        if target_hw is not None:
            features_for_vis = F.interpolate(
                features.unsqueeze(0),
                size=target_hw,
                mode="bilinear",
                align_corners=False
            ).squeeze(0)  # (C, target_h, target_w)
            vis_h, vis_w = target_hw
        else:
            features_for_vis = features
            vis_h, vis_w = h, w

        # Process mask (without changing its size)
        mask_tensor = None
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask_tensor = mask.float()
            else:
                mask_tensor = torch.from_numpy(np.array(mask)).float()
            if mask_tensor.max() > 1.0:
                mask_tensor = mask_tensor / 255.0
            if mask_tensor.shape[-2:] != (vis_h, vis_w):
                mask_tensor = None  # Discard if size doesn't match

        # Flatten features (N,C)
        features_reshaped = features_for_vis.view(dim, -1).permute(1, 0)  # (N,C)

        # Fit / select PCA
        if per_frame_pca:
            pca_local = PCA(n_components=3, whiten=True)
            if mask_tensor is not None:
                fg_idx = (mask_tensor.view(-1) > self.MASK_FG_THRESHOLD).nonzero(as_tuple=True)[0]
                if len(fg_idx) >= 3:  # Can fit 3 components
                    pca_local.fit(features_reshaped[fg_idx].numpy())
                else:  # Fallback to global if foreground is too small
                    pca_local.fit(features_reshaped.numpy())
            else:
                pca_local.fit(features_reshaped.numpy())
            projected_features = torch.from_numpy(pca_local.transform(features_reshaped.numpy())).view(vis_h, vis_w, 3)
        else:
            # Reuse / lazy-load global PCA
            if self.pca is None:
                if mask_tensor is not None:
                    fg_idx = (mask_tensor.view(-1) > self.MASK_FG_THRESHOLD).nonzero(as_tuple=True)[0]
                    self.pca = PCA(n_components=3, whiten=True)
                    if len(fg_idx) >= 3:
                        self.pca.fit(features_reshaped[fg_idx].numpy())
                    else:
                        self.pca.fit(features_reshaped.numpy())
                else:
                    self.fit_pca(features)  # Fit using original patch-level features
            projected_features = torch.from_numpy(self.pca.transform(features_reshaped.numpy())).view(vis_h, vis_w, 3)

        projected_image = torch.sigmoid(projected_features * 2.0).permute(2, 0, 1)  # (3,H,W)

        if mask_tensor is not None:
            projected_image *= (mask_tensor > self.MASK_FG_THRESHOLD)[None, :, :]

        plt.imsave(output_path, projected_image.permute(1, 2, 0).numpy())

    def process_and_save_features(self, image_dir: str, mask_dir:str, output_dir: str, visualize: bool = False):
        """Reads images from a directory into memory, then extracts and saves features for each."""
        images, image_files = read_image_folder(image_dir)
        masks, mask_files = read_image_folder(mask_dir, gray=True)
        
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        if visualize and not masks:
            raise ValueError("Visualization requires masks, but mask directory is empty or not found.")
        
        for idx, img_np in tqdm(list(enumerate(images)), desc="Extracting features"):
            features = self.process_image(img_np)
            dim = features.shape[0]
            features_reshaped = features.view(dim, -1)
            features_normalized = F.normalize(features_reshaped, p=2, dim=0)

            stem = Path(image_files[idx]).stem
            out_path = output_dir / f"{stem}.npz"
            np.savez_compressed(out_path, feat=features_normalized.view(dim, features.shape[1], features.shape[2]).cpu().numpy())

            if visualize:
                mask = Image.fromarray(masks[idx])  # Original pixel mask
                orig_w, orig_h = mask.size
                mask_tensor = TF.to_tensor(mask).squeeze(0)  # (H,W) in [0,1]
                vis_output_dir = output_dir / "visualizations"
                vis_output_dir.mkdir(exist_ok=True)
                vis_output_path = vis_output_dir / f"{stem}.png"
                self.visualize_and_save(features, str(vis_output_path), mask=mask_tensor, target_hw=(orig_h, orig_w), per_frame_pca=True)


def main():
    parser = argparse.ArgumentParser(description="Extract features from a directory of images using DINOv3.")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline config YAML")
    parser.add_argument("--object_name", type=str, default=None, help="Object name override used with --config")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to the dataset directory.")
    # parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the images.")
    # parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the extracted features.")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the DINOv3 model to use.")
    parser.add_argument("--image_size", type=int, default=None, help="Target image size.")
    parser.add_argument("--patch_size", type=int, default=None, help="Patch size for the model.")
    parser.add_argument("--visualize", action='store_true', default=None, help="Generate and save feature visualizations.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to local checkpoint (.safetensors or .pth).")
    
    args = parser.parse_args()
    if args.config:
        config = load_pipeline_config(args.config, object_name_override=args.object_name)
        runtime_paths = resolve_pipeline_paths(config, object_name_override=args.object_name)
        preprocess_cfg = resolve_preprocess(config)
        if args.dataset_dir is None:
            args.dataset_dir = runtime_paths["dataset_dir"]
    else:
        preprocess_cfg = resolve_preprocess({})
    set_arg_defaults(
        args,
        {
            "model_name": preprocess_cfg["feature_model_name"],
            "image_size": preprocess_cfg["feature_image_size"],
            "patch_size": preprocess_cfg["feature_patch_size"],
            "visualize": preprocess_cfg["visualize_features"],
            "checkpoint_path": preprocess_cfg["feature_checkpoint_path"],
        },
    )
    if args.dataset_dir is None:
        parser.error("--dataset_dir is required when --config is not provided")
    cprint("="*60, "green")
    cprint("Extracting features using DINOv3", "green")
    #cprint("-"*60, "green")
    args.image_dir = os.path.join(args.dataset_dir, "images_ori")
    args.output_dir = os.path.join(args.dataset_dir, "features")
    args.mask_dir = os.path.join(args.dataset_dir, "masks")
    extractor = FeatureExtractor(
        model_name=args.model_name,
        patch_size=args.patch_size,
        image_size=args.image_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_path=args.checkpoint_path,
    )
    
    extractor.process_and_save_features(args.image_dir, args.mask_dir, args.output_dir, args.visualize)


if __name__ == "__main__":
    main()
