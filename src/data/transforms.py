"""
Data augmentation transforms for RAW image denoising.

Provides augmentation functions that preserve the physical
properties of RAW images while increasing data diversity.
"""

from typing import Callable, Optional, Tuple

import numpy as np
import torch


class RandomCrop:
    """
    Random crop for RAW images.
    
    Args:
        size: Crop size (height, width) or single int for square crop
    """
    
    def __init__(self, size: int):
        self.size = size if isinstance(size, tuple) else (size, size)
    
    def __call__(
        self, 
        input_img: np.ndarray, 
        target_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, H, W = input_img.shape
        crop_h, crop_w = self.size
        
        if H > crop_h and W > crop_w:
            y = np.random.randint(0, H - crop_h)
            x = np.random.randint(0, W - crop_w)
            input_img = input_img[:, y:y+crop_h, x:x+crop_w]
            target_img = target_img[:, y:y+crop_h, x:x+crop_w]
        
        return input_img, target_img


class RandomFlip:
    """
    Random horizontal and vertical flips.
    
    Args:
        horizontal_prob: Probability of horizontal flip
        vertical_prob: Probability of vertical flip
    """
    
    def __init__(
        self, 
        horizontal_prob: float = 0.5, 
        vertical_prob: float = 0.5
    ):
        self.horizontal_prob = horizontal_prob
        self.vertical_prob = vertical_prob
    
    def __call__(
        self, 
        input_img: np.ndarray, 
        target_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.horizontal_prob:
            input_img = np.flip(input_img, axis=2).copy()
            target_img = np.flip(target_img, axis=2).copy()
        
        if np.random.rand() < self.vertical_prob:
            input_img = np.flip(input_img, axis=1).copy()
            target_img = np.flip(target_img, axis=1).copy()
        
        return input_img, target_img


class RandomTranspose:
    """
    Random transpose (90-degree rotation).
    
    Args:
        prob: Probability of transpose
    """
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(
        self, 
        input_img: np.ndarray, 
        target_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.prob:
            input_img = np.transpose(input_img, (0, 2, 1)).copy()
            target_img = np.transpose(target_img, (0, 2, 1)).copy()
        
        return input_img, target_img


class Compose:
    """
    Compose multiple transforms.
    
    Args:
        transforms: List of transform functions
    """
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(
        self, 
        input_img: np.ndarray, 
        target_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        for transform in self.transforms:
            input_img, target_img = transform(input_img, target_img)
        return input_img, target_img


def get_train_transforms(patch_size: int = 512) -> Compose:
    """
    Get training transforms.
    
    Args:
        patch_size: Size of random crop
        
    Returns:
        Composed transform function
    """
    return Compose([
        RandomCrop(patch_size),
        RandomFlip(),
        RandomTranspose(),
    ])


def get_val_transforms() -> Optional[Callable]:
    """
    Get validation transforms (typically no augmentation).
    
    Returns:
        None (no transforms for validation)
    """
    return None


__all__ = [
    "RandomCrop",
    "RandomFlip",
    "RandomTranspose",
    "Compose",
    "get_train_transforms",
    "get_val_transforms",
]

