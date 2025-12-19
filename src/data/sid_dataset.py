"""
SID (See in the Dark) dataset implementation.

Provides dataset classes for loading RAW image pairs for
low-light image denoising training.
"""

import os
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
import rawpy


def read_paired_fns(list_path: str) -> List[Tuple[str, str]]:
    """
    Read paired filenames from list file.
    
    Args:
        list_path: Path to list file with input/target pairs
        
    Returns:
        List of (input_fn, target_fn) tuples
    """
    paired_fns = []
    with open(list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    paired_fns.append((parts[0], parts[1]))
    return paired_fns


def pack_raw_bayer(raw) -> np.ndarray:
    """
    Pack Bayer RAW image to 4 channels (RGGB).
    
    Args:
        raw: rawpy raw object
        
    Returns:
        Packed image [4, H/2, W/2] in [0, 1] range
    """
    im = raw.raw_image_visible.astype(np.float32)
    raw_pattern = raw.raw_pattern
    
    R = np.where(raw_pattern == 0)
    G1 = np.where(raw_pattern == 1)
    B = np.where(raw_pattern == 2)
    G2 = np.where(raw_pattern == 3)
    
    white_point = 16383
    H, W = im.shape
    
    out = np.stack([
        im[R[0][0]:H:2, R[1][0]:W:2],   # R
        im[G1[0][0]:H:2, G1[1][0]:W:2], # G1
        im[B[0][0]:H:2, B[1][0]:W:2],   # B
        im[G2[0][0]:H:2, G2[1][0]:W:2], # G2
    ], axis=0).astype(np.float32)
    
    black_level = np.array(raw.black_level_per_channel)[:, None, None].astype(np.float32)
    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0, 1)
    
    return out


def compute_expo_ratio(input_fn: str, target_fn: str) -> float:
    """Compute exposure ratio from filenames."""
    in_exposure = float(input_fn.split('_')[-1][:-5])
    gt_exposure = float(target_fn.split('_')[-1][:-5])
    return min(gt_exposure / in_exposure, 300)


def metainfo(rawpath: str) -> Tuple[int, float]:
    """Extract ISO and exposure time from RAW file."""
    import exifread
    with open(rawpath, 'rb') as f:
        tags = exifread.process_file(f)
        _, suffix = os.path.splitext(os.path.basename(rawpath))
        
        if suffix.lower() == '.dng':
            expo = eval(str(tags['Image ExposureTime']))
            iso = eval(str(tags['Image ISOSpeedRatings']))
        else:
            expo = eval(str(tags['EXIF ExposureTime']))
            iso = eval(str(tags['EXIF ISOSpeedRatings']))
    
    return iso, expo


def resolve_raw_path(base_dir: str, candidate: str, fallback_subdir: str) -> str:
    """Resolve RAW file path."""
    candidate = candidate.strip()
    if os.path.isabs(candidate):
        return os.path.normpath(candidate)
    
    cleaned = candidate.replace('\\', '/').lstrip('./')
    lower = cleaned.lower()
    
    for marker in ('short/', 'long/'):
        idx = lower.find(marker)
        if idx != -1:
            cleaned = cleaned[idx:]
            break
    
    if lower.startswith('short/') or lower.startswith('long/'):
        return os.path.normpath(os.path.join(base_dir, cleaned))
    
    return os.path.normpath(os.path.join(base_dir, fallback_subdir, cleaned))


class SIDRawDenoiseDataset(Dataset):
    """
    SID RAW denoising dataset.
    
    Loads paired short/long exposure RAW images for training.
    
    Args:
        dataset_root: Root directory of SID dataset
        list_path: Path to list file with input/target pairs
        patchsize: Size of random patches (None for full images)
        augment: Whether to apply augmentation
    """
    
    def __init__(
        self,
        dataset_root: str,
        list_path: str,
        patchsize: int = 512,
        augment: bool = True,
    ):
        super().__init__()
        
        self.dataset_root = os.path.abspath(dataset_root)
        self.list_path = os.path.abspath(list_path)
        self.patchsize = patchsize
        self.augment = augment
        
        if not os.path.isdir(self.dataset_root):
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")
        if not os.path.isfile(self.list_path):
            raise FileNotFoundError(f"List file not found: {self.list_path}")
        
        self.paired_fns = read_paired_fns(self.list_path)
        
        # Cache for loaded images (optional)
        self.input_cache = {}
        self.target_cache = {}
    
    def __len__(self) -> int:
        return len(self.paired_fns)
    
    def __getitem__(self, idx: int) -> dict:
        input_fn, target_fn = self.paired_fns[idx]
        
        input_path = resolve_raw_path(self.dataset_root, input_fn, 'short')
        target_path = resolve_raw_path(self.dataset_root, target_fn, 'long')
        
        ratio = compute_expo_ratio(input_fn, target_fn)
        
        try:
            iso, _ = metainfo(input_path)
        except Exception:
            iso = 6400  # fallback
        
        # Load RAW images
        with rawpy.imread(input_path) as raw:
            input_image = pack_raw_bayer(raw) * ratio
        
        with rawpy.imread(target_path) as raw:
            target_image = pack_raw_bayer(raw)
        
        # Apply augmentation
        if self.augment and self.patchsize is not None:
            input_image, target_image = self._augment(input_image, target_image)
        
        # Clip to valid range
        input_image = np.clip(input_image, 0, 1).astype(np.float32)
        target_image = np.clip(target_image, 0, 1).astype(np.float32)
        
        return {
            'clean': torch.from_numpy(target_image),
            'noisy': torch.from_numpy(input_image),
            'ratio': torch.tensor([ratio], dtype=torch.float32),
            'ISO': torch.tensor([iso], dtype=torch.float32),
        }
    
    def _augment(
        self, 
        input_img: np.ndarray, 
        target_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentation."""
        C, H, W = input_img.shape
        ps = self.patchsize
        
        # Random crop
        if H > ps and W > ps:
            xx = np.random.randint(0, W - ps)
            yy = np.random.randint(0, H - ps)
            input_img = input_img[:, yy:yy+ps, xx:xx+ps]
            target_img = target_img[:, yy:yy+ps, xx:xx+ps]
        
        # Random horizontal flip
        if np.random.rand() > 0.5:
            input_img = np.flip(input_img, axis=2).copy()
            target_img = np.flip(target_img, axis=2).copy()
        
        # Random vertical flip
        if np.random.rand() > 0.5:
            input_img = np.flip(input_img, axis=1).copy()
            target_img = np.flip(target_img, axis=1).copy()
        
        # Random transpose
        if np.random.rand() > 0.5:
            input_img = np.transpose(input_img, (0, 2, 1)).copy()
            target_img = np.transpose(target_img, (0, 2, 1)).copy()
        
        return input_img, target_img


class ELDEvalDataset(Dataset):
    """
    ELD (Extreme Low-light Denoising) evaluation dataset.
    
    Args:
        basedir: Base directory containing camera folders
        camera_suffix: Tuple of (camera_name, file_suffix)
        scenes: List of scene indices
        img_ids: List of image IDs to load
    """
    
    def __init__(
        self,
        basedir: str,
        camera_suffix: Tuple[str, str],
        scenes: List[int],
        img_ids: List[int],
    ):
        super().__init__()
        self.basedir = basedir
        self.camera_suffix = camera_suffix
        self.scenes = scenes
        self.img_ids = img_ids
    
    def __len__(self) -> int:
        return len(self.scenes) * len(self.img_ids)
    
    def __getitem__(self, idx: int) -> dict:
        camera, suffix = self.camera_suffix
        
        scene_id = idx // len(self.img_ids)
        img_id = idx % len(self.img_ids)
        
        scene = f'scene-{self.scenes[scene_id]}'
        datadir = os.path.join(self.basedir, camera, scene)
        
        input_path = os.path.join(datadir, f'IMG_{self.img_ids[img_id]:04d}{suffix}')
        
        gt_ids = np.array([1, 6, 11, 16])
        ind = np.argmin(np.abs(self.img_ids[img_id] - gt_ids))
        target_path = os.path.join(datadir, f'IMG_{gt_ids[ind]:04d}{suffix}')
        
        # Get exposure info
        iso, expo = metainfo(target_path)
        target_expo = iso * expo
        iso_input, expo_input = metainfo(input_path)
        ratio = target_expo / (iso_input * expo_input)
        
        # Load images
        with rawpy.imread(input_path) as raw:
            input_img = pack_raw_bayer(raw) * ratio
        
        with rawpy.imread(target_path) as raw:
            target_img = pack_raw_bayer(raw)
        
        input_img = np.clip(input_img, 0, 1).astype(np.float32)
        target_img = np.clip(target_img, 0, 1).astype(np.float32)
        
        return {
            'input': torch.from_numpy(input_img),
            'target': torch.from_numpy(target_img),
            'fn': input_path,
            'input_path': input_path,
            'rawpath': target_path,
            'ISO': iso_input,
            'ratio': ratio,
        }


__all__ = [
    "SIDRawDenoiseDataset",
    "ELDEvalDataset",
    "read_paired_fns",
    "pack_raw_bayer",
    "compute_expo_ratio",
]

