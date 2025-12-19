"""
Dataset preparation module for EMVA 1288 Physics-Guided Diffusion Model

This module provides unified dataset preparation functions for training,
validation, and testing.
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
