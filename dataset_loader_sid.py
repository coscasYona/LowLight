import os

import torch
from torch.utils.data import Dataset

from dataset import read_paired_fns
from dataset.sid_dataset import SIDDataset


class SIDRawDenoiseDataset(Dataset):
    """Wrapper around the generic SIDDataset that returns tensors ready for training."""

    def __init__(self, dataset_root, list_path, patchsize=512):
        super().__init__()
        dataset_root = os.path.abspath(dataset_root)
        list_path = os.path.abspath(list_path)

        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(f"SID root directory not found: {dataset_root}")
        if not os.path.isfile(list_path):
            raise FileNotFoundError(f"SID train list not found: {list_path}")

        paired_fns = read_paired_fns(list_path)
        self.sid_dataset = SIDDataset(
            dataset_root,
            paired_fns,
            size=None,
            augment=True,
            memorize=False,
            stage_in="raw",
            stage_out="raw",
            gt_wb=0,
            CRF=False,
        )
        if patchsize is not None:
            self.sid_dataset.patch_size = patchsize

    def __len__(self):
        return len(self.sid_dataset)

    def __getitem__(self, idx):
        sample = self.sid_dataset[idx]
        noisy = torch.from_numpy(sample["input"]).float()
        clean = torch.from_numpy(sample["target"]).float()
        ratio = torch.tensor([sample["ratio"]], dtype=torch.float32)
        iso = torch.tensor([sample["ISO"]], dtype=torch.float32)
        return {"clean": clean, "noisy": noisy, "ratio": ratio, "ISO": iso}


def build_sid_raw_dataset(dataset_root, list_path, patchsize):
    """Factory helper to keep the main dataloader selection logic simple."""
    return SIDRawDenoiseDataset(dataset_root, list_path, patchsize=patchsize)

