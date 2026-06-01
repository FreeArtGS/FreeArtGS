import os
import cv2
import numpy as np
from prettytable import PrettyTable
from cprint import cprint
import argparse
import yaml
import torch
from tqdm import tqdm

def load_images(input_dir, append_reversed=True, exclude_boundary=True):
    """
    Load RGB images, depth maps, masks, and camera intrinsics from a directory.
    
    Args:
        input_dir (str): Path to the input directory containing image data
        append_reversed (bool): If True, append a reversed copy of the sequence to the end.
            This creates a forward-then-backward "replay" effect.
        exclude_boundary (bool): When appending reversed frames, avoid duplicating the last
            frame at the turning point (ping-pong: [0..N-1] + [N-2..0]).
        
    Returns:
        dict: Dictionary containing loaded data with keys:
            - rgbs: List of RGB images in BGR format
            - rgbs_tensor: Tensor of RGB images in format (1, T, C, H, W)
            - depths: List of depth maps
            - masks: List of binary masks
            - cam_K: Camera intrinsic matrix (3x3)
    """
    intrinsics_file = os.path.join(input_dir, "cam_K.txt")

    with open(intrinsics_file, 'r') as f:
        lines = f.readlines()
        fx, _, cx = map(float, lines[0].strip().split())
        _, fy, cy = map(float, lines[1].strip().split())
    cam_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    print(f"Loading images from {input_dir}...")
    
    # Load RGB images with progress bar
    image_dir = os.path.join(input_dir, "images_ori")
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])
    rgbs = []
    for image_file in tqdm(image_files, desc="Loading RGB images", unit="image"):
        rgb = cv2.imread(image_file)
        rgbs.append(rgb)

    # Load depth images with progress bar
    depth_dir = os.path.join(input_dir, "depth")
    depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(('.npz'))])
    depths = []
    for depth_file in tqdm(depth_files, desc="Loading depth images", unit="depth"):
        depth = np.load(depth_file)['depth']
        depths.append(depth)

    # Load mask images with progress bar
    mask_dir = os.path.join(input_dir, "masks")
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(('.png'))])
    masks = []
    for mask_file in tqdm(mask_files, desc="Loading mask images", unit="mask"):
        mask = cv2.imread(mask_file)
        masks.append(mask)

    feature_dir = os.path.join(input_dir, "features")
    if os.path.exists(feature_dir):
        feature_files = sorted([os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith(('.npz'))])
        features = []
        for feature_file in tqdm(feature_files, desc="Loading feature maps", unit="feature"):
            feature = torch.from_numpy(np.load(feature_file)['feat']).float()
            features.append(feature)
    else:
        features = None

    # Optionally append reversed copies to create a forward+backward sequence
    if append_reversed:
        def _pingpong(seq):
            if seq is None:
                return None
            if not isinstance(seq, list):
                return seq
            if len(seq) <= 1:
                return seq.copy()
            if exclude_boundary:
                return seq + list(reversed(seq[:-1]))
            else:
                return seq + list(reversed(seq))

        rgbs = _pingpong(rgbs)
        depths = _pingpong(depths)
        masks = _pingpong(masks)
        features = _pingpong(features)

    # Convert RGB images to tensor format with progress bar
    print("Converting RGB images to tensors...")
    rgb_tensor_list = []
    for rgb in tqdm(rgbs, desc="Converting to tensors", unit="tensor"):
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        rgb_tensor_list.append(rgb_tensor)
    
    # Stack tensors to create batch tensor with shape (1, T, C, H, W)
    print("Stacking tensors...")
    rgbs_tensor = torch.stack(rgb_tensor_list, dim=0).unsqueeze(0).float() if rgb_tensor_list else None

    images_dict = {
        "rgbs": rgbs,
        "rgbs_tensor": rgbs_tensor,
        "depths": depths,
        "masks": masks,
        "cam_K": cam_K,
        "features": features
    }
    return images_dict

def count_parameters(model):
    """
    Count and display the number of trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Total number of trainable parameters
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        # Only show modules with more than 100K parameters to avoid clutter
        if param > 100000:
            table.add_row([name, param])
        total_params+=param
    print(table)
    print('total params: %.2f M' % (total_params/1000000.0))

    return total_params

def read_image_folder(folder_path, gray=False):
    """
    Read all images from a folder and convert them to RGB format.
    
    Args:
        folder_path (str): Path to the folder containing images
        gray (bool): If True, read images as grayscale.
        
    Returns:
        list: List of RGB images as numpy arrays
    """
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))])
    frames = []
    if not image_files:
        cprint.err(f"No images found in {folder_path}")
        return frames, image_files
    for image_file in image_files:
        if gray:
            frame = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        else:
            frame = cv2.imread(image_file)

        if frame is not None:
            if not gray:
                # Convert from BGR to RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    return frames, image_files

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    args = argparse.Namespace(**config)
    return args

def resize_images(images_dict, max_resolution):
    """
    Resize images to a maximum resolution while maintaining aspect ratio and ensuring dimensions are divisible by 8.
    This is important for deep learning models that require specific input dimensions.
    
    Args:
        images_dict (dict): Dictionary containing images, depths, masks, and camera matrix
        max_resolution (int): Maximum resolution for the longer dimension
        
    Returns:
        tuple: (resized_images_dict, H, W) where:
            - resized_images_dict: Dictionary with resized data
            - H: New height
            - W: New width
    """

    rgbs = images_dict["rgbs"]
    depths = images_dict["depths"]
    masks = images_dict["masks"]
    cam_K = images_dict["cam_K"]
    rgbs_tensor = images_dict.get("rgbs_tensor", [])
    
    # Calculate new dimensions while maintaining aspect ratio
    H_orig, W_orig = rgbs[0].shape[:2]
    HH = max_resolution
    scale = min(HH/H_orig, HH/W_orig) if H_orig > 0 and W_orig > 0 else 1.0
    H, W = int(H_orig*scale), int(W_orig*scale)
    
    # Ensure dimensions are divisible by 8 (required by many deep learning models)
    if H % 8 != 0 or W % 8 != 0:
        cprint.warn(f"H or W is not divisible by 8, adjusting. H: {H}, W: {W}")
        H, W = H//8 * 8, W//8 * 8
    
    print(f"Resizing images from {H_orig}x{W_orig} to {H}x{W}...")
    
    # Resize RGB images using bilinear interpolation
    resized_rgbs = []
    for rgb in rgbs:
        resized_rgb = cv2.resize(rgb, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
        resized_rgbs.append(resized_rgb)
    
    # Resize masks using nearest neighbor interpolation to preserve binary values
    resized_masks = []
    for mask in masks:
        resized_mask = cv2.resize(mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST) > 0
        resized_masks.append(resized_mask)
    
    # Handle 3-channel masks by taking only the first channel
    if resized_masks and len(resized_masks[0].shape) == 3:
        resized_masks = [mask[:,:,0] for mask in resized_masks]
    
    # Compute eroded version for tracking-only usage (default: 3x3 kernel, 1 iteration)
    kernel = np.ones((3, 3), np.uint8)
    eroded_masks = []
    for mask in resized_masks:
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
        eroded_masks.append(eroded)
    
    # Resize depth maps using nearest neighbor to preserve depth values
    resized_depths = []
    for depth in depths:
        resized_depth = cv2.resize(depth, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        resized_depths.append(resized_depth)
    
    # Resize RGB tensors if they exist using PyTorch interpolation
    if rgbs_tensor is not None:
        print("Resizing RGB tensors...")
        # Input tensor shape: (1, T, C, H, W)
        batch_size, time_steps, channels, orig_h, orig_w = rgbs_tensor.shape
        # Reshape to (T, C, H, W) for interpolation
        rgbs_tensor_resized = torch.nn.functional.interpolate(
            rgbs_tensor.view(-1, channels, orig_h, orig_w),  # (T, C, H, W)
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        # Reshape back to (1, T, C, H, W)
        rgbs_tensor_resized = rgbs_tensor_resized.view(batch_size, time_steps, channels, H, W)
    else:
        rgbs_tensor_resized = None

    # Scale camera intrinsics to match the new image dimensions
    cam_K = images_dict["cam_K"]    
    cam_K_scaled = cam_K.copy()
    cam_K_scaled[:2, :] *= scale  # Scale fx, fy, cx, cy proportionally

    resized_images_dict = {
        "rgbs": resized_rgbs,
        "rgbs_tensor": rgbs_tensor_resized,
        "depths": resized_depths,
        "masks": resized_masks,
        "masks_eroded": eroded_masks,
        "cam_K": cam_K_scaled,
        "features": images_dict.get("features", None)
    }
    return resized_images_dict, H, W


class ImageLoader:
    """
    A class that supports slicing operations for image data loading.
    Provides convenient indexing and slicing operations for image sequences.
    
    This class allows easy manipulation of image sequences including:
    - Individual frame access via indexing
    - Sequence slicing for extracting subsequences
    - Frame duplication and extension operations
    - Convenient property access for different data types
    """
    
    def __init__(self, input_dir=None, images_dict=None, append_reversed=True, exclude_boundary=True):
        """
        Initialize the image loader.
        
        Args:
            input_dir (str, optional): Path to the data directory
            images_dict (dict, optional): Pre-loaded image dictionary
            append_reversed (bool): If True and loading from disk, append reversed frames on load.
            exclude_boundary (bool): If True, avoid duplicating the last frame at the turning point.
            
        Raises:
            ValueError: If neither input_dir nor images_dict is provided
        """
        if images_dict is not None:
            self.data = images_dict
        elif input_dir is not None:
            self.data = load_images(input_dir, append_reversed=append_reversed, exclude_boundary=exclude_boundary)
        else:
            raise ValueError("Must provide either input_dir or images_dict")
        
        # Get sequence length from RGB images
        self.length = len(self.data["rgbs"])
        # Store keys that can be sliced (list data with same length as sequence)
        self.sliceable_keys = []
        for key, value in self.data.items():
            if isinstance(value, list) and len(value) == self.length:
                self.sliceable_keys.append(key)
        
        # Camera intrinsics and rgbs_tensor don't need slicing, remove from sliceable keys
        if "cam_K" in self.sliceable_keys:
            self.sliceable_keys.remove("cam_K")
        if "rgbs_tensor" in self.sliceable_keys:
            self.sliceable_keys.remove("rgbs_tensor")
    
    def __len__(self):
        """Return the sequence length."""
        return self.length
    
    def __getitem__(self, index):
        """
        Support indexing and slicing operations.
        
        Args:
            index: Integer index or slice object
            
        Returns:
            If integer index: Dictionary containing single frame data
            If slice: New ImageLoader object with sliced data
            
        Raises:
            IndexError: If index is out of range
            TypeError: If index is neither int nor slice
        """
        if isinstance(index, int):
            # Handle negative indices
            if index < 0:
                index = self.length + index
            
            if index < 0 or index >= self.length:
                raise IndexError(f"Index {index} out of range [0, {self.length})")
            
            # Return single frame data
            frame_data = {}
            for key, value in self.data.items():
                if key in self.sliceable_keys:
                    frame_data[key] = value[index]
                elif key == "rgbs_tensor" and value is not None:
                    # For rgbs_tensor, return frame at index: (1, T, C, H, W) -> (C, H, W)
                    frame_data[key] = value[0, index, :, :, :]
                else:
                    frame_data[key] = value  # Non-list data (like cam_K) is copied directly

            return frame_data
        
        elif isinstance(index, slice):
            # Handle slice operations
            start, stop, step = index.indices(self.length)
            
            sliced_data = {}
            for key, value in self.data.items():
                if key in self.sliceable_keys:
                    sliced_data[key] = value[start:stop:step]
                elif key == "rgbs_tensor" and value is not None:
                    # For rgbs_tensor, slice time dimension: (1, T, C, H, W) -> (1, T', C, H, W)
                    sliced_data[key] = value[:, start:stop:step, :, :, :]
                else:
                    sliced_data[key] = value  # Non-list data is copied directly
            
            # Return new ImageLoader object
            return ImageLoader(images_dict=sliced_data)
        
        else:
            raise TypeError(f"Index type must be int or slice, got {type(index)}")
    
    def __iter__(self):
        """Support iteration over frames."""
        for i in range(self.length):
            yield self[i]
    
    def resize(self, max_resolution):
        """
        Resize images to a maximum resolution.
        
        Args:
            max_resolution (int): Maximum resolution for the longer dimension
            
        Returns:
            tuple: (new_ImageLoader, H, W) where H and W are new dimensions
        """
        resized_dict, H, W = resize_images(self.data, max_resolution)
        return ImageLoader(images_dict=resized_dict), H, W
    
    def get_frame_range(self, start, end):
        """
        Get data for a specific frame range.
        
        Args:
            start (int): Starting frame index
            end (int): Ending frame index (exclusive)
            
        Returns:
            ImageLoader: New ImageLoader object with specified frame range
        """
        return self[start:end]
    
    def get_nth_frames(self, n):
        """
        Take every nth frame from the sequence.
        
        Args:
            n (int): Frame interval
            
        Returns:
            ImageLoader: New ImageLoader object with every nth frame
        """
        return self[::n]
    
    def get_data_dict(self):
        """Return the raw data dictionary."""
        return self.data
    
    def duplicate_last_frame(self):
        """
        Duplicate the last frame's data and append it to the end of the sequence.
        Similar to extending the window by one frame.
        
        Returns:
            ImageLoader: New ImageLoader object with duplicated last frame
            
        Raises:
            ValueError: If the sequence is empty
        """
        if self.length == 0:
            raise ValueError("Cannot duplicate last frame of empty sequence")
        
        duplicated_data = {}
        
        for key, value in self.data.items():
            if key in self.sliceable_keys:
                # For list data, duplicate the last element
                last_item = value[-1]
                duplicated_data[key] = value + [last_item]
            elif key == "rgbs_tensor" and value is not None:
                # For rgbs_tensor, duplicate last frame: (1, T, C, H, W) -> (1, T+1, C, H, W)
                last_frame = value[:, -1:, :, :, :]  # (1, 1, C, H, W)
                duplicated_data[key] = torch.cat([value, last_frame], dim=1)
            else:
                # For non-list data (like cam_K), copy directly
                duplicated_data[key] = value
        
        return ImageLoader(images_dict=duplicated_data)

    def append_reversed(self, exclude_boundary=True):
        """
        Return a new ImageLoader with all frame-aligned lists appended with their reversed order.
        If exclude_boundary is True, avoid duplicating the last frame at the turning point.
        """
        if self.length <= 1:
            return ImageLoader(images_dict=self.data)

        def _pingpong(seq):
            if not isinstance(seq, list):
                return seq
            if exclude_boundary:
                return seq + list(reversed(seq[:-1]))
            else:
                return seq + list(reversed(seq))

        new_data = {}
        for key, value in self.data.items():
            if key in self.sliceable_keys:
                new_data[key] = _pingpong(value)
            elif key == "rgbs_tensor" and value is not None:
                # Rebuild rgbs_tensor from appended rgbs to ensure consistency
                # Convert images back to tensor for the new sequence
                rgb_list = _pingpong(self.data["rgbs"]) if "rgbs" in self.data else None
                if rgb_list:
                    rgb_tensor_list = [torch.from_numpy(rgb).permute(2, 0, 1) for rgb in rgb_list]
                    new_data[key] = torch.stack(rgb_tensor_list, dim=0).unsqueeze(0).float()
                else:
                    new_data[key] = value
            else:
                new_data[key] = value

        return ImageLoader(images_dict=new_data)
    
    def get_windowed_tensors(self):
        """
        Get tensor format data suitable for windowed operations.
        Similar to the window_rgb_tensor format in your code examples.
        
        Returns:
            dict: Dictionary containing various tensor formats:
                - rgb_tensor: (1, T, C, H, W) format RGB tensor
                - rgb_list: List of RGB images
                - depth_list: List of depth images  
                - mask_list: List of mask images
                
        Raises:
            ValueError: If the sequence is empty
        """
        if self.length == 0:
            raise ValueError("Empty sequence cannot create windowed tensors")
        
        # rgbs_tensor is already in (1, T, C, H, W) format
        window_rgb_tensor = self.data.get("rgbs_tensor")
        
        return {
            'rgb_tensor': window_rgb_tensor,
            'rgb_list': self.rgbs.copy(),
            'depth_list': self.depths.copy(),
            'mask_list': self.masks.copy(),
        }
    
    def duplicate_and_extend_window(self):
        """
        Duplicate the last frame and extend the window, fully mimicking your provided code logic.
        
        Returns:
            dict: Extended window data containing:
                - rgb_tensor: (1, T+1, C, H, W) format
                - rgb_list: Extended RGB list
                - depth_list: Extended depth list
                - mask_list: Extended mask list
        """
        # Get current window data
        window_data = self.get_windowed_tensors()
        
        window_rgb_tensor = window_data['rgb_tensor']
        window_rgb_list = window_data['rgb_list']
        window_depth_list = window_data['depth_list']
        window_mask_list = window_data['mask_list']
        
        if window_rgb_tensor is not None:
            # Duplicate last frame tensor: extract from (1, T, C, H, W) -> (1, 1, C, H, W)
            duplicated_rgb_tensor = window_rgb_tensor[:, -1:, :, :, :]
            
            # Duplicate last frame's other data
            duplicated_rgb = window_rgb_list[-1]
            duplicated_depth = window_depth_list[-1]
            duplicated_mask = window_mask_list[-1]
            
            # Extend tensors and lists
            window_rgb_tensor = torch.cat([window_rgb_tensor, duplicated_rgb_tensor], dim=1)
            window_rgb_list = window_rgb_list + [duplicated_rgb]
            window_depth_list = window_depth_list + [duplicated_depth]
            window_mask_list = window_mask_list + [duplicated_mask]
        
        return {
            'rgb_tensor': window_rgb_tensor,
            'rgb_list': window_rgb_list,
            'depth_list': window_depth_list,
            'mask_list': window_mask_list,
        }
    
    @property
    def rgbs(self):
        """Get RGB image list."""
        return self.data["rgbs"]
    
    @property
    def rgbs_tensor(self):
        """Get RGB image tensor in (1, T, C, H, W) format."""
        return self.data.get("rgbs_tensor")
    
    @property
    def depths(self):
        """Get depth image list."""
        return self.data["depths"]
    
    @property
    def masks(self):
        """Get mask image list."""
        return self.data["masks"]
    
    @property
    def cam_K(self):
        """Get camera intrinsic matrix."""
        return self.data["cam_K"]
    
    def __repr__(self):
        """String representation of the ImageLoader."""
        return f"ImageLoader(length={self.length}, keys={list(self.data.keys())})"
