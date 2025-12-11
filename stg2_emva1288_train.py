"""
Training script for EMVA 1288 Physics-Guided Diffusion Model

This script trains a diffusion model that uses the EMVA 1288 standard
to accurately model CMOS camera noise. The forward diffusion process
uses actual noise distributions (shot + read + row + quantization)
instead of simple Gaussian scaling.

Based on: https://kmdouglass.github.io/posts/modeling-noise-for-image-simulations/
"""

import os
import random
import glob
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import scipy.io as sio
from dataset_loader import SID_Dataset_Denoise_raw
from dataset_loader_sid import build_sid_raw_dataset, build_fuji_raw_dataset
from torch.utils.data import ConcatDataset
from stg2_denoise_options import opt
from net.EMVA1288Diffusion import EMVA1288Diffusion
from torch.utils.tensorboard import SummaryWriter
import util.util as util
from data_process.process import sample_params_max

random.seed()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def findLastCheckpoint(save_dir, save_pre):
    file_list = glob.glob(os.path.join(save_dir, save_pre + '*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*" + save_pre +"(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def build_train_dataset(args):
    dataset_root = os.path.abspath(args.trainset_path)
    args.trainset_path = dataset_root
    
    datasets_list = []
    
    # Check if using SID dataset
    use_sid = args.use_sid_raw or os.path.isdir(os.path.join(dataset_root, 'short'))
    if use_sid:
        train_list = args.train_list
        if train_list is None:
            raise ValueError("train_list must be specified when using SID RAW data")
        train_list_path = os.path.abspath(train_list)
        sid_dataset = build_sid_raw_dataset(dataset_root, train_list_path, patchsize=args.patch_size)
        datasets_list.append(sid_dataset)
        print(f"Added SID dataset with {len(sid_dataset)} samples")
    
    # Check if using Fuji dataset
    use_fuji = args.use_fuji_raw or (args.fuji_train_list is not None)
    if use_fuji:
        fuji_train_list = args.fuji_train_list
        if fuji_train_list is None:
            raise ValueError("fuji_train_list must be specified when using Fuji RAW data")
        fuji_train_list_path = os.path.abspath(fuji_train_list)
        fuji_dataset_root = os.path.abspath(args.fuji_trainset_path) if args.fuji_trainset_path else dataset_root
        fuji_dataset = build_fuji_raw_dataset(fuji_dataset_root, fuji_train_list_path, patchsize=args.patch_size)
        datasets_list.append(fuji_dataset)
        print(f"Added Fuji 2025 dataset with {len(fuji_dataset)} samples")
    
    # If no RAW datasets specified, fall back to MAT format
    if len(datasets_list) == 0:
        return SID_Dataset_Denoise_raw(dataset_root, patchsize=args.patch_size)
    
    # Combine multiple datasets if both are specified
    if len(datasets_list) > 1:
        combined_dataset = ConcatDataset(datasets_list)
        print(f"Combined dataset with {len(combined_dataset)} total samples")
        return combined_dataset
    
    return datasets_list[0]


def build_val_dataset(args):
    """Build validation dataset similar to training dataset"""
    # Check if validation paths are specified
    if not hasattr(args, 'val_list') or args.val_list is None:
        print("No validation dataset specified (val_list not set)")
        return None
    
    dataset_root = os.path.abspath(args.trainset_path)
    datasets_list = []
    
    # Check if using SID validation dataset
    use_sid = args.use_sid_raw or os.path.isdir(os.path.join(dataset_root, 'short'))
    if use_sid:
        val_list = args.val_list
        val_list_path = os.path.abspath(val_list)
        sid_val_dataset = build_sid_raw_dataset(dataset_root, val_list_path, patchsize=args.patch_size)
        datasets_list.append(sid_val_dataset)
        print(f"Added SID validation dataset with {len(sid_val_dataset)} samples")
    
    # Check if using Fuji validation dataset
    use_fuji = args.use_fuji_raw or (args.fuji_train_list is not None)
    if use_fuji and hasattr(args, 'fuji_val_list') and args.fuji_val_list is not None:
        fuji_val_list = args.fuji_val_list
        fuji_val_list_path = os.path.abspath(fuji_val_list)
        fuji_dataset_root = os.path.abspath(args.fuji_trainset_path) if args.fuji_trainset_path else dataset_root
        fuji_val_dataset = build_fuji_raw_dataset(fuji_dataset_root, fuji_val_list_path, patchsize=args.patch_size)
        datasets_list.append(fuji_val_dataset)
        print(f"Added Fuji validation dataset with {len(fuji_val_dataset)} samples")
    
    # If no datasets were added, return None
    if len(datasets_list) == 0:
        return None
    
    # Combine multiple datasets if both are specified
    if len(datasets_list) > 1:
        combined_dataset = ConcatDataset(datasets_list)
        print(f"Combined validation dataset with {len(combined_dataset)} total samples")
        return combined_dataset
    
    return datasets_list[0]


def validate_epoch(dn_model, val_loader, criterion_mse, criterion_l1, compute_gradient_loss, 
                   l1_weight, gradient_weight, loss_scale, args, camera_type, epoch, writer):
    """Run validation on validation set and log metrics"""
    dn_model.eval()
    val_losses = []
    val_batch_data = None
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            img_gt = data['clean'].cuda()
            ratio = data['ratio'].cuda()
            iso = data['ISO'].cuda()
            
            batch, _, _, _ = img_gt.size()
            # Sample random timesteps
            timesteps = torch.randint(
                0, args.sd_num_steps, (batch,), device=img_gt.device, dtype=torch.long
            )
            
            # Base Gaussian noise for blending
            base_noise = torch.randn_like(img_gt)
            
            # Get camera parameters for physics-based noise generation
            base_model = dn_model.module if hasattr(dn_model, 'module') else dn_model
            
            # Sample camera parameters based on ISO
            iso_np = iso.cpu().numpy().flatten()
            ratio_np = ratio.cpu().numpy().flatten()
            
            iso_val = int(iso_np[0]) if len(iso_np) > 0 else 6400
            ratio_val = float(ratio_np[0]) if len(ratio_np) > 0 else 200.0
            
            camera_params = sample_params_max(
                camera_type=camera_type,
                iso=iso_val,
                ratio=ratio_val
            )
            
            # Forward diffusion with EMVA 1288 physics noise
            noisy_state = base_model.q_sample(
                img_gt, 
                base_noise, 
                timesteps,
                iso=iso,
                ratio=ratio,
                camera_params=camera_params,
                use_physics_noise=True,
            )
            
            # Store first batch for image logging
            if i == 0:
                val_batch_data = {
                    'img_gt': img_gt[:min(4, batch)].detach(),
                    'noisy_state': noisy_state[:min(4, batch)].detach(),
                    'iso': iso[:min(4, batch)],
                    'ratio': ratio[:min(4, batch)],
                    'camera_params': camera_params
                }
            
            # Model predicts the noise
            pred_noise = dn_model(
                noisy_state,
                iso=iso,
                ratio=ratio,
                timesteps=timesteps,
                predict_noise=True,
                camera_params=camera_params,
            )
            
            # Compute actual noise that was added
            sqrt_alpha = base_model._extract(
                base_model.sqrt_alphas_cumprod, timesteps, img_gt.shape
            )
            sqrt_one_minus_alpha = base_model._extract(
                base_model.sqrt_one_minus_alphas_cumprod, timesteps, img_gt.shape
            )
            sqrt_one_minus_alpha = torch.clamp(sqrt_one_minus_alpha, min=1e-6)
            actual_noise = (noisy_state - sqrt_alpha * img_gt) / sqrt_one_minus_alpha
            actual_noise = torch.where(
                torch.isfinite(actual_noise),
                actual_noise,
                torch.zeros_like(actual_noise)
            )
            actual_noise = torch.clamp(actual_noise, min=-10.0, max=10.0)
            pred_noise = torch.where(
                torch.isfinite(pred_noise),
                pred_noise,
                torch.zeros_like(pred_noise)
            )
            
            # Compute loss
            loss_mse = criterion_mse(pred_noise, actual_noise)
            loss_l1 = criterion_l1(pred_noise, actual_noise)
            base_weight = 1.0 - gradient_weight
            loss = base_weight * ((1.0 - l1_weight) * loss_mse + l1_weight * loss_l1)
            
            if gradient_weight > 0:
                loss_grad = compute_gradient_loss(pred_noise, actual_noise)
                loss = loss + gradient_weight * loss_grad
            
            if torch.isfinite(loss):
                val_losses.append(loss.item())
    
    # Compute average validation loss
    if val_losses:
        avg_val_loss = np.mean(val_losses)
        writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
        print(f"Validation Loss: {avg_val_loss:.4f}")
    else:
        avg_val_loss = None
    
    # Log validation images
    if val_batch_data is not None:
        util.log_validation_images(
            writer=writer,
            epoch=epoch,
            model=dn_model,
            image_data=val_batch_data,
            save_path=args.save_path
        )
    
    return avg_val_loss


def main(args):
    base_save_path = os.path.abspath(args.save_path)
    patch_folder = f"patch_{args.patch_size}"
    save_root = os.path.join(base_save_path, patch_folder)
    if not save_root.endswith(os.sep):
        save_root = save_root + os.sep
    args.save_path = save_root
    print(f"Checkpoints will be stored under: {args.save_path}")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    
    # Initialize TensorBoard writer
    log_dir = os.path.join(args.save_path, 'tensorboard_logs')
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be stored under: {log_dir}")
    print(f"To view TensorBoard, run: tensorboard --logdir={log_dir}")
    
    if not os.path.exists('test_epoch_psnr_emva1288.mat'):
        s = {}
        s["tep"] = np.zeros((13, 1))
        sio.savemat('test_epoch_psnr_emva1288.mat', s)
    
    # new or continue
    initial_epoch = findLastCheckpoint(save_dir=args.save_path, save_pre=args.save_prefix)
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        args.resume = "continue"
        args.last_ckpt = args.save_path + args.save_prefix + str(initial_epoch) + '.pth'

    # Get attention type (default to 'linear' for efficient training)
    attn_type = getattr(args, 'sd_attn_type', 'linear')
    scheduler_type = getattr(args, 'sd_scheduler', 'ddpm')
    camera_type = getattr(args, 'emva_camera_type', 'SonyA7S2')
    noise_code = getattr(args, 'emva_noise_code', 'prq')  # p=Poisson shot, r=row, q=quantization
    
    if attn_type is None:
        attn_type = 'linear'
    
    print(f"Using attention type: {attn_type} (memory-efficient)")
    print(f"Camera type: {camera_type}, Noise code: {noise_code}")
    
    # Network architecture with EMVA 1288 physics
    # Ensure channel_mults is a tuple of integers (not a string)
    if isinstance(args.sd_channel_mults, str):
        channel_mults = tuple(int(val) for val in args.sd_channel_mults.split(',') if val)
    else:
        channel_mults = args.sd_channel_mults
    
    dn_net = EMVA1288Diffusion(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        base_channels=args.sd_base_channels,
        channel_mults=channel_mults,
        num_steps=args.sd_num_steps,
        time_embed_dim=args.sd_time_embed_dim,
        cond_embed_dim=args.sd_cond_dim,
        attn_type=attn_type,
        scheduler=scheduler_type,
        camera_type=camera_type,
        noise_code=noise_code,
    )
    
    print(f"Model architecture: attention_type={attn_type}, scheduler={scheduler_type}")
    print(f"EMVA 1288 physics: camera_type={camera_type}, noise_code={noise_code}")

    # Loss functions - hybrid MSE + L1 + gradient for sharper images
    criterion_mse = nn.MSELoss().to(DEVICE)
    criterion_l1 = nn.L1Loss().to(DEVICE)
    l1_weight = getattr(args, 'l1_weight', 0.8)
    gradient_weight = getattr(args, 'gradient_weight', 0.1)
    
    # Pre-create Sobel kernels once (memory efficient)
    # Sobel is better than simple gradients: includes smoothing and is more robust to noise
    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                  dtype=torch.float32, device=DEVICE).view(1, 1, 3, 3)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                  dtype=torch.float32, device=DEVICE).view(1, 1, 3, 3)
    
    def compute_gradient_loss(pred, target):
        """Memory-optimized illuminance-invariant gradient loss using Sobel"""
        n_channels = pred.size(1)
        
        # Reuse pre-created Sobel kernels
        kernel_x = sobel_kernel_x.repeat(n_channels, 1, 1, 1)
        kernel_y = sobel_kernel_y.repeat(n_channels, 1, 1, 1)
        
        # Compute local mean for illuminance normalization (smaller 3x3 kernel saves memory)
        pred_mean = F.avg_pool2d(torch.abs(pred), kernel_size=3, stride=1, padding=1)
        
        # Apply Sobel filters to prediction (needs gradients)
        pred_gx = F.conv2d(pred, kernel_x, groups=n_channels, padding=1)
        pred_gy = F.conv2d(pred, kernel_y, groups=n_channels, padding=1)
        
        # Normalize by local illuminance BEFORE computing magnitude (saves one tensor)
        epsilon = 1e-3
        pred_gx_norm = pred_gx / (pred_mean + epsilon)
        pred_gy_norm = pred_gy / (pred_mean + epsilon)
        
        # Compute target without gradients (saves memory)
        with torch.no_grad():
            target_mean = F.avg_pool2d(torch.abs(target), kernel_size=3, stride=1, padding=1)
            target_gx = F.conv2d(target, kernel_x, groups=n_channels, padding=1)
            target_gy = F.conv2d(target, kernel_y, groups=n_channels, padding=1)
            target_gx_norm = target_gx / (target_mean + epsilon)
            target_gy_norm = target_gy / (target_mean + epsilon)
        
        # L1 loss on normalized gradient components (no magnitude tensor needed)
        loss_x = F.l1_loss(pred_gx_norm, target_gx_norm)
        loss_y = F.l1_loss(pred_gy_norm, target_gy_norm)
        
        return (loss_x + loss_y) * 0.5
    
    # Loss scaling and gradient clipping settings
    loss_scale = getattr(args, 'loss_scale', 10.0)
    use_grad_clip = getattr(args, 'use_grad_clip', False)
    grad_clip_max = getattr(args, 'grad_clip_max', 1.0)
    loss_components = ["MSE", "L1"]
    if gradient_weight > 0:
        loss_components.append("Gradient")
    print(f"Loss: Hybrid {' + '.join(loss_components)} (L1: {l1_weight}, Gradient: {gradient_weight})")
    print(f"Loss scaling: {loss_scale}x")
    if use_grad_clip:
        print(f"Gradient clipping enabled: max_norm={grad_clip_max}")
    else:
        print("Gradient clipping disabled")
    
    # Move to device / DataParallel if available
    dn_net = dn_net.to(DEVICE)
    
    # Enable gradient checkpointing if requested (saves memory)
    if getattr(args, 'use_gradient_checkpointing', False):
        print("Gradient checkpointing enabled (trades compute for memory)")
    
    if DEVICE.type == 'cuda' and torch.cuda.device_count() > 1:
        dn_model = nn.DataParallel(dn_net)
    else:
        dn_model = dn_net
    
    # Optimizer
    optimizer_dn = None
    resume_loaded = False
    start_epoch = 1

    if args.resume == "continue":
        try:
            tmp_ckpt = torch.load(args.last_ckpt, map_location=DEVICE)
            pretrained_dict = tmp_ckpt['state_dict']
            model_dict = dn_model.state_dict()
            compatible = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            missing_keys = [k for k in model_dict.keys() if k not in compatible]
            unexpected_keys = [k for k in pretrained_dict.keys() if k not in compatible]
            
            if missing_keys or unexpected_keys:
                print("Checkpoint mismatch detected; restarting from scratch.")
                print(f"Missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")
            else:
                model_dict.update(compatible)
                dn_model.load_state_dict(model_dict)
                optimizer_dn = optim.Adam(dn_model.parameters(), lr=args.learning_rate_dtcn)
                optimizer_dn.load_state_dict(tmp_ckpt['optimizer_state'])
                start_epoch = initial_epoch + 1
                resume_loaded = True
                print(f"✓ Successfully loaded checkpoint from epoch {initial_epoch}")
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            print(f"Failed to load checkpoint '{args.last_ckpt}': {exc}. Starting new training.")

    if not resume_loaded:
        args.resume = "new"
        optimizer_dn = optim.Adam(dn_model.parameters(), lr=args.learning_rate_dtcn)
    
    if args.resume == "continue" and not args.skip_eval:
        import stg2_denoise_test_SID
        import stg2_denoise_test_ELD
        # test SID
        stg2_denoise_test_SID.valid(args)
        # test ELD
        stg2_denoise_test_ELD.valid(args)

    # Set training set DataLoader
    train_dataset = build_train_dataset(args)
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.load_thread, 
        pin_memory=True, 
        drop_last=False
    )
    
    # Set validation set DataLoader (if validation list is provided)
    val_dataset = build_val_dataset(args)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # No shuffle for validation
            num_workers=args.load_thread,
            pin_memory=True,
            drop_last=False
        )
        print(f"Validation dataset loaded with {len(val_dataset)} samples")

    # Training
    global_step = 0
    for epoch in range(start_epoch, args.epoch + 1):
        dn_model.train()
        i = 0
        total_step = len(train_loader)
        lr_s = 1e-4
        if epoch >= 100:
            lr_s = 5e-5
        if epoch >= 180:
            lr_s = 1e-5
        if epoch == start_epoch or epoch == 100 or epoch == 180:
            for group in optimizer_dn.param_groups:
                group['lr'] = lr_s
        
        # Track epoch-level metrics
        epoch_losses = []
        # Store last batch for image logging
        last_batch_data = None
        
        for i, data in enumerate(train_loader, 0):
            img_gt = data['clean'].cuda()
            ratio = data['ratio'].cuda()
            iso = data['ISO'].cuda()
            optimizer_dn.zero_grad()

            batch, _, _, _ = img_gt.size()
            # Sample random timesteps
            timesteps = torch.randint(
                0, args.sd_num_steps, (batch,), device=img_gt.device, dtype=torch.long
            )
            
            # Base Gaussian noise for blending
            base_noise = torch.randn_like(img_gt)
            
            # Get camera parameters for physics-based noise generation
            base_model = dn_model.module if hasattr(dn_model, 'module') else dn_model
            
            # Sample camera parameters based on ISO (for accurate physics modeling)
            iso_np = iso.cpu().numpy().flatten()
            ratio_np = ratio.cpu().numpy().flatten()
            
            # Use first ISO in batch to get params (batch should have same camera)
            iso_val = int(iso_np[0]) if len(iso_np) > 0 else 6400
            ratio_val = float(ratio_np[0]) if len(ratio_np) > 0 else 200.0
            
            camera_params = sample_params_max(
                camera_type=camera_type,
                iso=iso_val,
                ratio=ratio_val
            )
            
            # Forward diffusion with EMVA 1288 physics noise
            # q_sample will generate CMOS noise internally
            noisy_state = base_model.q_sample(
                img_gt, 
                base_noise, 
                timesteps,
                iso=iso,
                ratio=ratio,
                camera_params=camera_params,
                use_physics_noise=True,  # Use actual CMOS noise
            )
            
            # Store last batch for image logging
            if i == len(train_loader) - 1:
                last_batch_data = {
                    'img_gt': img_gt[:min(4, batch)].detach(),
                    'noisy_state': noisy_state[:min(4, batch)].detach(),
                    'iso': iso[:min(4, batch)],
                    'ratio': ratio[:min(4, batch)],
                    'camera_params': camera_params
                }
            
            # Model predicts the noise (which is CMOS noise, not just Gaussian)
            pred_noise = dn_model(
                noisy_state,
                iso=iso,
                ratio=ratio,
                timesteps=timesteps,
                predict_noise=True,
                camera_params=camera_params,
            )
            
            # Loss: predict the actual noise that was added
            # The noise is a blend of CMOS noise and Gaussian, so we need to
            # compute what noise was actually added
            sqrt_alpha = base_model._extract(
                base_model.sqrt_alphas_cumprod, timesteps, img_gt.shape
            )
            sqrt_one_minus_alpha = base_model._extract(
                base_model.sqrt_one_minus_alphas_cumprod, timesteps, img_gt.shape
            )
            
            # Clamp sqrt_one_minus_alpha to avoid division by very small numbers
            # This prevents numerical instability when timesteps are close to 0
            sqrt_one_minus_alpha = torch.clamp(sqrt_one_minus_alpha, min=1e-6)
            
            # Compute actual noise that was added
            actual_noise = (noisy_state - sqrt_alpha * img_gt) / sqrt_one_minus_alpha
            
            # Check for NaN/inf in actual_noise and replace with zeros
            actual_noise = torch.where(
                torch.isfinite(actual_noise),
                actual_noise,
                torch.zeros_like(actual_noise)
            )
            
            # Clamp actual_noise to reasonable range to prevent extreme values
            actual_noise = torch.clamp(actual_noise, min=-10.0, max=10.0)
            
            # Check for NaN/inf in pred_noise and replace with zeros
            pred_noise = torch.where(
                torch.isfinite(pred_noise),
                pred_noise,
                torch.zeros_like(pred_noise)
            )
            
            # Hybrid loss: MSE + L1 + Gradient for sharper images
            # L1 reduces blur, gradient loss preserves edges
            loss_mse = criterion_mse(pred_noise, actual_noise)
            loss_l1 = criterion_l1(pred_noise, actual_noise)
            
            # Base loss: weighted combination of MSE and L1
            base_weight = 1.0 - gradient_weight
            loss = base_weight * ((1.0 - l1_weight) * loss_mse + l1_weight * loss_l1)
            
            # Add gradient loss for edge preservation
            if gradient_weight > 0:
                loss_grad = compute_gradient_loss(pred_noise, actual_noise)
                loss = loss + gradient_weight * loss_grad
            
            # Check if loss is NaN/inf and skip this batch if so
            if not torch.isfinite(loss):
                print(f"Warning: NaN/inf loss detected at batch {i+1}, skipping...")
                global_step += 1  # Still increment global_step to maintain consistency
                continue
            
            # Scale loss to increase gradient strength for robust training
            loss = loss * loss_scale
            loss.backward()
            
            # Gradient clipping to stabilize training and prevent vanishing gradients
            if use_grad_clip:
                grad_norm = torch.nn.utils.clip_grad_norm_(dn_model.parameters(), max_norm=grad_clip_max)
                if global_step % 100 == 0:
                    writer.add_scalar('Train/GradientNormClipped', grad_norm, global_step)
            
            optimizer_dn.step()
            i = i + 1
            
            # Log training metrics to TensorBoard
            # Note: loss_value is the scaled loss, log both scaled and unscaled
            loss_value = loss.item()
            loss_unscaled = loss_value / loss_scale  # Unscaled loss for reference
            epoch_losses.append(loss_unscaled)  # Store unscaled loss for epoch average
            writer.add_scalar('Train/Loss', loss_unscaled, global_step)  # Log unscaled loss
            writer.add_scalar('Train/LossScaled', loss_value, global_step)  # Log scaled loss
            writer.add_scalar('Train/LearningRate', lr_s, global_step)
            
            # Log gradient norms periodically
            if global_step % 100 == 0:
                total_norm = 0
                param_count = 0
                for p in dn_model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                if param_count > 0:
                    total_norm = total_norm ** (1. / 2)
                    writer.add_scalar('Train/GradientNorm', total_norm, global_step)
            
            print("Epoch:[{}/{}] Batch: [{}/{}] loss = {:.4f}".format(
                epoch, args.epoch, i, total_step, loss_value
            ))
            global_step += 1
        
        # Log epoch-level average loss
        if epoch_losses:
            avg_epoch_loss = np.mean(epoch_losses)
            writer.add_scalar('Train/EpochLoss', avg_epoch_loss, epoch)
            args._final_epoch_loss = avg_epoch_loss

        # Log image artifacts at end of each epoch
        if last_batch_data is not None:
            util.log_training_images(
                writer=writer,
                epoch=epoch,
                model=dn_model,
                image_data=last_batch_data,
                save_path=args.save_path
            )
        
        # Run validation if validation loader is available
        if val_loader is not None:
            val_loss = validate_epoch(
                dn_model=dn_model,
                val_loader=val_loader,
                criterion_mse=criterion_mse,
                criterion_l1=criterion_l1,
                compute_gradient_loss=compute_gradient_loss,
                l1_weight=l1_weight,
                gradient_weight=gradient_weight,
                loss_scale=loss_scale,
                args=args,
                camera_type=camera_type,
                epoch=epoch,
                writer=writer
            )
            if val_loss is not None:
                args._final_val_loss = val_loss

        if epoch % args.save_every_epochs == 0:
            # Save model and checkpoint
            save_dict = {
                'state_dict': dn_model.state_dict(),
                'optimizer_state': optimizer_dn.state_dict()
            }
            torch.save(save_dict, os.path.join(args.save_path + args.save_prefix + '{}.pth'.format(epoch)))
            del save_dict
            if not args.skip_eval:
                import stg2_denoise_test_SID
                import stg2_denoise_test_ELD
                # test SID/Fuji - pass writer for logging
                stg2_denoise_test_SID.valid(args, writer=writer, epoch=epoch)
                # test ELD - pass writer for logging
                stg2_denoise_test_ELD.valid(args, writer=writer, epoch=epoch)
    
    # Close TensorBoard writer
    writer.close()
    print("Training completed. TensorBoard logs saved.")
    
    return getattr(args, '_final_epoch_loss', None)


if __name__ == "__main__":
    main(opt)
    exit(0)

