"""
PyTorch Lightning DataModule for EMVA 1288 diffusion training.

Provides unified data loading for training, validation, and testing
with support for multiple dataset types (SID, ELD, Fuji).
"""

import os
from typing import Optional, Callable

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import torch

from data.sid_dataset import SIDRawDenoiseDataset
from data.transforms import get_train_transforms, get_val_transforms


class EMVA1288DataModule(pl.LightningDataModule):
    """
    Lightning DataModule for EMVA 1288 diffusion model training.
    
    Handles data loading for:
    - SID (See in the Dark) dataset
    - ELD (Extreme Low-light Denoising) dataset
    - Fuji 2025 dataset
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory (optional)
        train_list: Path to training pairs list file
        val_list: Path to validation pairs list file (optional)
        batch_size: Training batch size
        num_workers: Number of data loading workers
        patch_size: Size of random patches for training
        pin_memory: Whether to pin memory for faster GPU transfer
        use_sid_raw: Use SID RAW data format
        use_fuji_raw: Use Fuji RAW data format
        fuji_train_list: Path to Fuji training list (optional)
        fuji_val_list: Path to Fuji validation list (optional)
    """
    
    def __init__(
        self,
        train_dir: str = '../dataset/SID/Sony',
        val_dir: Optional[str] = None,
        train_list: str = './dataset/Sony_train.txt',
        val_list: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 4,
        patch_size: int = 512,
        pin_memory: bool = True,
        use_sid_raw: bool = True,
        use_fuji_raw: bool = False,
        fuji_train_list: Optional[str] = None,
        fuji_val_list: Optional[str] = None,
        val_split: float = 0.1,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.train_dir = train_dir
        self.val_dir = val_dir or train_dir
        self.train_list = train_list
        self.val_list = val_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.pin_memory = pin_memory
        self.use_sid_raw = use_sid_raw
        self.use_fuji_raw = use_fuji_raw
        self.fuji_train_list = fuji_train_list
        self.fuji_val_list = fuji_val_list
        self.val_split = val_split
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training, validation, and testing.
        
        Args:
            stage: 'fit', 'validate', 'test', or None
        """
        if stage == 'fit' or stage is None:
            # Build training dataset
            train_datasets = []
            
            if self.use_sid_raw and self.train_list:
                # Resolve paths relative to workspace root if they're relative
                workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                
                if not os.path.isabs(self.train_dir):
                    # Remove leading ../ or ./ and resolve from workspace root
                    clean_path = self.train_dir
                    while clean_path.startswith('../') or clean_path.startswith('./'):
                        if clean_path.startswith('../'):
                            clean_path = clean_path[3:]
                        elif clean_path.startswith('./'):
                            clean_path = clean_path[2:]
                    train_dir_candidate = os.path.join(workspace_root, clean_path)
                    if os.path.exists(train_dir_candidate):
                        train_dir = os.path.abspath(train_dir_candidate)
                    else:
                        # Fallback to original resolution
                        train_dir = os.path.abspath(self.train_dir)
                else:
                    train_dir = self.train_dir
                
                if not os.path.isabs(self.train_list):
                    # Remove leading ../ or ./ and resolve from workspace root
                    clean_path = self.train_list
                    while clean_path.startswith('../') or clean_path.startswith('./'):
                        if clean_path.startswith('../'):
                            clean_path = clean_path[3:]
                        elif clean_path.startswith('./'):
                            clean_path = clean_path[2:]
                    train_list_candidate = os.path.join(workspace_root, clean_path)
                    if os.path.exists(train_list_candidate):
                        train_list = os.path.abspath(train_list_candidate)
                    else:
                        # Fallback to original resolution
                        train_list = os.path.abspath(self.train_list)
                else:
                    train_list = self.train_list
                
                if os.path.exists(train_list):
                    sid_train = SIDRawDenoiseDataset(
                        dataset_root=train_dir,
                        list_path=train_list,
                        patchsize=self.patch_size,
                    )
                    train_datasets.append(sid_train)
                    print(f"Loaded SID training dataset: {len(sid_train)} samples")
                else:
                    pass
            
            if self.use_fuji_raw and self.fuji_train_list:
                fuji_dir = os.path.abspath(self.train_dir)
                fuji_list = os.path.abspath(self.fuji_train_list)
                
                if os.path.exists(fuji_list):
                    fuji_train = SIDRawDenoiseDataset(
                        dataset_root=fuji_dir,
                        list_path=fuji_list,
                        patchsize=self.patch_size,
                    )
                    train_datasets.append(fuji_train)
                    print(f"Loaded Fuji training dataset: {len(fuji_train)} samples")
            
            if train_datasets:
                if len(train_datasets) > 1:
                    combined = ConcatDataset(train_datasets)
                    print(f"Combined training dataset: {len(combined)} samples")
                else:
                    combined = train_datasets[0]
                
                # Split for validation if no explicit val set
                if self.val_list is None and self.fuji_val_list is None:
                    total_size = len(combined)
                    val_size = max(1, min(int(total_size * self.val_split), int(total_size * 0.2)))
                    train_size = total_size - val_size
                    
                    self.train_dataset, self.val_dataset = random_split(
                        combined,
                        [train_size, val_size],
                        generator=torch.Generator().manual_seed(42)
                    )
                    print(f"Split: {train_size} train, {val_size} validation")
                else:
                    self.train_dataset = combined
            else:
                pass
            
            # Build explicit validation dataset
            if self.val_list is not None:
                # Resolve paths relative to workspace root if they're relative
                workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                
                if not os.path.isabs(self.val_dir):
                    # Remove leading ../ or ./ and resolve from workspace root
                    clean_path = self.val_dir
                    while clean_path.startswith('../') or clean_path.startswith('./'):
                        if clean_path.startswith('../'):
                            clean_path = clean_path[3:]
                        elif clean_path.startswith('./'):
                            clean_path = clean_path[2:]
                    val_dir_candidate = os.path.join(workspace_root, clean_path)
                    if os.path.exists(val_dir_candidate):
                        val_dir = os.path.abspath(val_dir_candidate)
                    else:
                        # Fallback to original resolution
                        val_dir = os.path.abspath(self.val_dir)
                else:
                    val_dir = self.val_dir
                
                if not os.path.isabs(self.val_list):
                    # Remove leading ../ or ./ and resolve from workspace root
                    clean_path = self.val_list
                    while clean_path.startswith('../') or clean_path.startswith('./'):
                        if clean_path.startswith('../'):
                            clean_path = clean_path[3:]
                        elif clean_path.startswith('./'):
                            clean_path = clean_path[2:]
                    val_list_candidate = os.path.join(workspace_root, clean_path)
                    if os.path.exists(val_list_candidate):
                        val_list = os.path.abspath(val_list_candidate)
                    else:
                        # Fallback to original resolution
                        val_list = os.path.abspath(self.val_list)
                else:
                    val_list = self.val_list
                
                if os.path.exists(val_list):
                    self.val_dataset = SIDRawDenoiseDataset(
                        dataset_root=val_dir,
                        list_path=val_list,
                        patchsize=self.patch_size,
                    )
                    print(f"Loaded validation dataset: {len(self.val_dataset)} samples")
        
        if stage == 'test' or stage is None:
            # Test dataset setup can be customized
            pass
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self.val_dataset is None:
            print("Warning: No validation dataset available")
            return []
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Create test dataloader."""
        if self.test_dataset is None:
            return None
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


__all__ = ["EMVA1288DataModule"]

