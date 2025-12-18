"""
Dataset preparation module for EMVA 1288 Physics-Guided Diffusion Model

This module provides unified dataset preparation functions and Lightning
DataModule classes for training, validation, and testing.
"""

import os
from torch.utils.data import ConcatDataset, random_split
from dataset_loader import SID_Dataset_Denoise_raw
from dataset_loader_sid import build_sid_raw_dataset, build_fuji_raw_dataset


def build_dataset(args, split='train'):
    """
    Unified function to build datasets for train, validation, or test splits.
    
    Args:
        args: Configuration object with dataset parameters
        split: One of 'train', 'val', or 'test'
    
    Returns:
        Dataset object or None if not available
    """
    # Determine which list file to use based on split
    if split == 'train':
        list_attr = 'train_list'
        fuji_list_attr = 'fuji_train_list'
        dataset_root_attr = 'trainset_path'
        fuji_root_attr = 'fuji_trainset_path'
    elif split == 'val':
        list_attr = 'val_list'
        fuji_list_attr = 'fuji_val_list'
        dataset_root_attr = 'trainset_path'  # Validation often uses same root
        fuji_root_attr = 'fuji_trainset_path'
    elif split == 'test':
        list_attr = 'test_list'
        fuji_list_attr = 'fuji_test_list'
        dataset_root_attr = 'testset_path' if hasattr(args, 'testset_path') else 'trainset_path'
        fuji_root_attr = 'fuji_testset_path' if hasattr(args, 'fuji_testset_path') else 'fuji_trainset_path'
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
    
    dataset_root = os.path.abspath(getattr(args, dataset_root_attr))
    # Update args with absolute path for consistency
    setattr(args, dataset_root_attr, dataset_root)
    datasets_list = []
    
    # Check if using SID dataset
    use_sid = getattr(args, 'use_sid_raw', False) or os.path.isdir(os.path.join(dataset_root, 'short'))
    if use_sid:
        list_path = getattr(args, list_attr, None)
        if list_path is None:
            if split == 'train':
                raise ValueError("train_list must be specified when using SID RAW data")
            else:
                # For val/test, it's okay if not specified
                print(f"No {split} list specified for SID dataset, skipping...")
        else:
            list_path_abs = os.path.abspath(list_path)
            sid_dataset = build_sid_raw_dataset(dataset_root, list_path_abs, patchsize=args.patch_size)
            datasets_list.append(sid_dataset)
            print(f"Added SID {split} dataset with {len(sid_dataset)} samples")
    
    # Check if using Fuji dataset
    use_fuji = getattr(args, 'use_fuji_raw', False) or (getattr(args, fuji_list_attr, None) is not None)
    if use_fuji:
        fuji_list = getattr(args, fuji_list_attr, None)
        if fuji_list is None:
            if split == 'train':
                raise ValueError("fuji_train_list must be specified when using Fuji RAW data")
            else:
                # For val/test, it's okay if not specified
                print(f"No {split} list specified for Fuji dataset, skipping...")
        else:
            fuji_list_path = os.path.abspath(fuji_list)
            fuji_dataset_root = os.path.abspath(getattr(args, fuji_root_attr, dataset_root))
            fuji_dataset = build_fuji_raw_dataset(fuji_dataset_root, fuji_list_path, patchsize=args.patch_size)
            datasets_list.append(fuji_dataset)
            print(f"Added Fuji {split} dataset with {len(fuji_dataset)} samples")
    
    # If no RAW datasets specified and this is training, fall back to MAT format
    if len(datasets_list) == 0:
        if split == 'train':
            return SID_Dataset_Denoise_raw(dataset_root, patchsize=args.patch_size)
        else:
            # For val/test, return None if no datasets found
            return None
    
    # Combine multiple datasets if both are specified
    if len(datasets_list) > 1:
        combined_dataset = ConcatDataset(datasets_list)
        print(f"Combined {split} dataset with {len(combined_dataset)} total samples")
        return combined_dataset
    
    return datasets_list[0]


def build_train_dataset(args):
    """Build training dataset - wrapper for backward compatibility"""
    return build_dataset(args, split='train')


def build_val_dataset(args):
    """Build validation dataset - wrapper for backward compatibility"""
    return build_dataset(args, split='val')


def build_test_dataset(args):
    """Build test dataset"""
    return build_dataset(args, split='test')


# PyTorch Lightning DataModule
try:
    # Try modern lightning package first (lightning >= 2.0)
    try:
        import lightning.pytorch as pl
    except ImportError:
        # Fall back to legacy pytorch_lightning package
        import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    
    class EMVA1288DataModule(pl.LightningDataModule):
        """
        Lightning DataModule for EMVA 1288 training.
        
        This DataModule handles train/val/test splits and provides DataLoaders
        for Lightning training.
        """
        
        def __init__(
            self,
            args,
            batch_size=None,
            num_workers=None,
            pin_memory=True,
            val_split_ratio=0.1,
            val_split_seed=42,
        ):
            """
            Args:
                args: Configuration object with dataset parameters
                batch_size: Batch size (defaults to args.batch_size)
                num_workers: Number of data loading workers (defaults to args.load_thread)
                pin_memory: Whether to pin memory in DataLoader
                val_split_ratio: Ratio of training data to use for validation if no val set provided
                val_split_seed: Random seed for train/val split
            """
            super().__init__()
            self.args = args
            self.batch_size = batch_size if batch_size is not None else args.batch_size
            self.num_workers = num_workers if num_workers is not None else args.load_thread
            self.pin_memory = pin_memory
            self.val_split_ratio = val_split_ratio
            self.val_split_seed = val_split_seed
            
            # These will be set in setup()
            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None
        
        def setup(self, stage=None):
            """
            Setup datasets for the current stage (fit, validate, test, or predict).
            
            Args:
                stage: 'fit', 'validate', 'test', or 'predict'
            """
            if stage == 'fit' or stage is None:
                # Build training dataset
                self.train_dataset = build_train_dataset(self.args)
                
                # Build validation dataset
                self.val_dataset = build_val_dataset(self.args)
                
                # If no validation dataset provided, split training data
                if self.val_dataset is None:
                    total_size = len(self.train_dataset)
                    val_size = max(1, min(int(total_size * self.val_split_ratio), int(total_size * 0.2)))
                    train_size = total_size - val_size
                    
                    print(f"No validation dataset provided. Splitting training data: "
                          f"{train_size} train, {val_size} validation")
                    self.train_dataset, self.val_dataset = random_split(
                        self.train_dataset,
                        [train_size, val_size],
                        generator=torch.Generator().manual_seed(self.val_split_seed)
                    )
                    print(f"Created validation split: {len(self.val_dataset)} samples from training data")
            
            if stage == 'test' or stage is None:
                # Build test dataset
                self.test_dataset = build_test_dataset(self.args)
                if self.test_dataset is None:
                    print("No test dataset available")
        
        def train_dataloader(self):
            """Create DataLoader for training"""
            if self.train_dataset is None:
                raise RuntimeError("train_dataset not set. Call setup('fit') first.")
            return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False
            )
        
        def val_dataloader(self):
            """Create DataLoader for validation"""
            if self.val_dataset is None:
                raise RuntimeError("val_dataset not set. Call setup('fit') first.")
            return DataLoader(
                dataset=self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False
            )
        
        def test_dataloader(self):
            """Create DataLoader for testing"""
            if self.test_dataset is None:
                return None
            return DataLoader(
                dataset=self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False
            )
    
    # Import torch here to avoid issues if lightning is not available
    import torch

except ImportError:
    # Lightning not available - provide a fallback
    print("Warning: lightning (or pytorch_lightning) not available. DataModule classes will not be available.")
    EMVA1288DataModule = None
