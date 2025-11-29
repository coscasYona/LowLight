#
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
from net.UNetSeeInDark import Net
from torch.utils.tensorboard import SummaryWriter
import util.util as util
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
    if not os.path.exists('test_epoch_psnr_dncnn.mat'):
        s = {}
        s["tep"] = np.zeros((13, 1))
        sio.savemat('test_epoch_psnr_dncnn.mat', s)
    # new or continue
    initial_epoch = findLastCheckpoint(save_dir=args.save_path, save_pre = args.save_prefix)
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        args.resume = "continue"
        args.last_ckpt = args.save_path + args.save_prefix + str(initial_epoch) + '.pth'

    # net architecture
    dn_net = Net(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        base_channels=args.sd_base_channels,
        channel_mults=args.sd_channel_mults,
        num_steps=args.sd_num_steps,
        time_embed_dim=args.sd_time_embed_dim,
        cond_embed_dim=args.sd_cond_dim,
        attn_type=getattr(args, 'sd_attn_type', 'linear'),
        scheduler=getattr(args, 'sd_scheduler', 'ddpm'),
    )

    # loss function
    criterion = nn.MSELoss().to(DEVICE)
    # Move to device / DataParallel if available
    dn_net = dn_net.to(DEVICE)
    
    # Enable gradient checkpointing if requested (saves memory)
    if getattr(args, 'use_gradient_checkpointing', False):
        if hasattr(dn_net, 'unet'):
            dn_net.unet.enable_gradient_checkpointing = True
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
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            print(f"Failed to load checkpoint '{args.last_ckpt}': {exc}. Starting new training.")

    if not resume_loaded:
        args.resume = "new"
        optimizer_dn = optim.Adam(dn_model.parameters(), lr=args.learning_rate_dtcn)
    if args.resume=="continue" and not args.skip_eval:
        import stg2_denoise_test_SID
        import stg2_denoise_test_ELD
        # test SID
        stg2_denoise_test_SID.valid(args)
        # test ELD
        stg2_denoise_test_ELD.valid(args)

    # set training set DataLoader
    train_dataset = build_train_dataset(args)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.load_thread, pin_memory=True, drop_last=False)

    # training
    global_step = 0
    for epoch in range(start_epoch, args.epoch+1):
        dn_model.train()
        # train
        i = 0
        total_step = len(train_loader)
        lr_s = 1e-4
        if epoch >= 100:
            lr_s = 5e-5
        if epoch >= 180:
            lr_s = 1e-5
        if epoch == start_epoch or epoch == 100 or epoch == 180:
            # for optimizer in optimizer_dn:
            for group in optimizer_dn.param_groups:
                group['lr'] = lr_s
        
        # Track epoch-level metrics
        epoch_losses = []
        
        for i, data in enumerate(train_loader, 0):
            img_gt = data['clean'].cuda()
            ratio = data['ratio'].cuda()
            iso = data['ISO'].cuda()
            optimizer_dn.zero_grad()

            batch, _, _, _ = img_gt.size()
            if batch == args.batch_size:
                timesteps = torch.randint(
                    0, args.sd_num_steps, (batch,), device=img_gt.device, dtype=torch.long
                )
                noise = torch.randn_like(img_gt)
                base_model = dn_model.module if hasattr(dn_model, 'module') else dn_model
                noise_scale = base_model.physics_noise_scale(iso, ratio)
                scaled_noise = noise * noise_scale
                noisy_state = base_model.q_sample(img_gt, scaled_noise, timesteps)
                pred_noise = dn_model(
                    noisy_state,
                    iso=iso,
                    ratio=ratio,
                    timesteps=timesteps,
                    predict_noise=True,
                )
                loss = criterion(pred_noise, scaled_noise)
                loss.backward()
                optimizer_dn.step()
                i = i + 1
                
                # Log training metrics to TensorBoard
                loss_value = loss.item()
                epoch_losses.append(loss_value)
                writer.add_scalar('Train/Loss', loss_value, global_step)
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
                
                print("Epoch:[{}/{}] Batch: [{}/{}] loss = {:.4f}".format(epoch, args.epoch, i, total_step, loss_value))
                global_step += 1
        
        # Log epoch-level average loss
        if epoch_losses:
            avg_epoch_loss = np.mean(epoch_losses)
            writer.add_scalar('Train/EpochLoss', avg_epoch_loss, epoch)

        if epoch % args.save_every_epochs == 0:
            # save model and checkpoint
            save_dict = {'state_dict': dn_model.state_dict(),
                        'optimizer_state': optimizer_dn.state_dict()}
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

if __name__ == "__main__":

    main(opt)

    exit(0)



