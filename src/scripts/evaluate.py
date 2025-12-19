#!/usr/bin/env python3
"""
Evaluation script for EMVA 1288 Physics-Guided Diffusion Model.

Evaluates trained models on SID and ELD datasets.

Usage:
    python evaluate.py --checkpoint path/to/checkpoint.ckpt
    python evaluate.py --checkpoint path/to/checkpoint.ckpt --dataset sid
    python evaluate.py --checkpoint path/to/checkpoint.ckpt --dataset eld
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm

from training import EMVA1288LightningModule
from training.metrics import DenoisingMetrics, IlluminanceCorrect
from data.sid_dataset import SIDRawDenoiseDataset, ELDEvalDataset


def evaluate_sid(
    model: EMVA1288LightningModule,
    eval_dir: str,
    val_list: str,
    test_list: str,
    device: torch.device,
) -> dict:
    """
    Evaluate model on SID dataset.
    
    Args:
        model: Trained model
        eval_dir: SID dataset directory
        val_list: Validation list file
        test_list: Test list file
        device: Torch device
        
    Returns:
        Dict of evaluation metrics
    """
    from data.sid_dataset import read_paired_fns
    
    print("\n" + "=" * 60)
    print("Evaluating on SID Dataset")
    print("=" * 60)
    
    # Read paired filenames
    val_fns = read_paired_fns(val_list) if os.path.exists(val_list) else []
    test_fns = read_paired_fns(test_list) if os.path.exists(test_list) else []
    all_fns = val_fns + test_fns
    
    if not all_fns:
        print("No evaluation files found!")
        return {}
    
    # Group by exposure ratio
    expo_ratios = [100, 250, 300]
    
    def get_ratio(fn_pair):
        in_expo = float(fn_pair[0].split('_')[-1][:-5])
        gt_expo = float(fn_pair[1].split('_')[-1][:-5])
        return min(gt_expo / in_expo, 300)
    
    model.eval()
    metrics_calc = DenoisingMetrics(data_range=1.0)
    illum_correct = IlluminanceCorrect()
    
    results = {}
    
    for ratio_target in expo_ratios:
        # Filter files by ratio
        ratio_fns = [fn for fn in all_fns if abs(get_ratio(fn) - ratio_target) < 50]
        
        if not ratio_fns:
            continue
        
        print(f"\nRatio {ratio_target}: {len(ratio_fns)} samples")
        
        psnr_sum = 0.0
        ssim_sum = 0.0
        count = 0
        
        dataset = SIDRawDenoiseDataset(
            dataset_root=eval_dir,
            list_path=val_list,  # We'll filter manually
            patchsize=None,  # Full images
            augment=False,
        )
        dataset.paired_fns = ratio_fns
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        for batch in tqdm(loader, desc=f"Ratio {ratio_target}"):
            clean = batch['clean'].to(device)
            noisy = batch['noisy'].to(device)
            ratio = batch['ratio'].to(device)
            iso = batch['ISO'].to(device)
            
            with torch.no_grad():
                denoised = model.model.sample(noisy, iso=iso, ratio=ratio)
                denoised_corrected = illum_correct(denoised, clean)
            
            # Compute metrics
            psnr = metrics_calc.compute_psnr(denoised_corrected, clean)
            ssim = metrics_calc.compute_ssim(denoised_corrected, clean)
            
            psnr_sum += psnr.item()
            ssim_sum += ssim.item()
            count += 1
        
        if count > 0:
            avg_psnr = psnr_sum / count
            avg_ssim = ssim_sum / count
            results[f'ratio_{ratio_target}_psnr'] = avg_psnr
            results[f'ratio_{ratio_target}_ssim'] = avg_ssim
            print(f"  PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
    
    return results


def evaluate_eld(
    model: EMVA1288LightningModule,
    eval_dir: str,
    device: torch.device,
) -> dict:
    """
    Evaluate model on ELD dataset.
    
    Args:
        model: Trained model
        eval_dir: ELD dataset directory
        device: Torch device
        
    Returns:
        Dict of evaluation metrics
    """
    print("\n" + "=" * 60)
    print("Evaluating on ELD Dataset")
    print("=" * 60)
    
    cameras = [('SonyA7S2', '.ARW')]
    scenes = list(range(1, 11))
    img_ids_sets = [[4, 9, 14], [5, 10, 15]]
    
    model.eval()
    metrics_calc = DenoisingMetrics(data_range=1.0)
    illum_correct = IlluminanceCorrect()
    
    results = {}
    
    for camera, suffix in cameras:
        for set_idx, img_ids in enumerate(img_ids_sets):
            dataset = ELDEvalDataset(
                basedir=eval_dir,
                camera_suffix=(camera, suffix),
                scenes=scenes,
                img_ids=img_ids,
            )
            
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            psnr_sum = 0.0
            ssim_sum = 0.0
            count = 0
            
            print(f"\n{camera} - Set {set_idx + 1}:")
            
            for batch in tqdm(loader, desc=f"{camera} Set {set_idx + 1}"):
                clean = batch['target'].to(device)
                noisy = batch['input'].to(device)
                ratio = torch.tensor([[batch['ratio']]], device=device, dtype=torch.float32)
                iso = torch.tensor([[batch['ISO']]], device=device, dtype=torch.float32)
                
                with torch.no_grad():
                    denoised = model.model.sample(noisy, iso=iso, ratio=ratio)
                    denoised_corrected = illum_correct(denoised, clean)
                
                psnr = metrics_calc.compute_psnr(denoised_corrected, clean)
                ssim = metrics_calc.compute_ssim(denoised_corrected, clean)
                
                psnr_sum += psnr.item()
                ssim_sum += ssim.item()
                count += 1
            
            if count > 0:
                avg_psnr = psnr_sum / count
                avg_ssim = ssim_sum / count
                results[f'{camera}_set{set_idx + 1}_psnr'] = avg_psnr
                results[f'{camera}_set{set_idx + 1}_ssim'] = avg_ssim
                print(f"  PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate EMVA 1288 Diffusion Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'sid', 'eld'],
                        help='Dataset to evaluate on')
    parser.add_argument('--sid_dir', type=str, default='../dataset/SID/Sony',
                        help='SID dataset directory')
    parser.add_argument('--eld_dir', type=str, default='../dataset/ELD_new',
                        help='ELD dataset directory')
    parser.add_argument('--val_list', type=str, default='./dataset/Sony_val.txt',
                        help='SID validation list')
    parser.add_argument('--test_list', type=str, default='./dataset/Sony_test.txt',
                        help='SID test list')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = EMVA1288LightningModule.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()
    
    all_results = {}
    
    # Evaluate on SID
    if args.dataset in ['all', 'sid']:
        sid_results = evaluate_sid(
            model, args.sid_dir, args.val_list, args.test_list, device
        )
        all_results.update(sid_results)
    
    # Evaluate on ELD
    if args.dataset in ['all', 'eld']:
        eld_results = evaluate_eld(model, args.eld_dir, device)
        all_results.update(eld_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for key, value in sorted(all_results.items()):
        print(f"{key}: {value:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

