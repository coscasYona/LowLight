

import argparse
import datetime
import torch
from pathlib import Path

parser = argparse.ArgumentParser(description="DResTCN_Gray")

parser.add_argument("--trainset_path", type=str, default='../dataset/denoising/sid_raw/Train_Paired_RGBG_p512_N4480_FPN', help="path to train set")
parser.add_argument("--train_dir", type=str, default='../dataset/denoising/', help="path to train set")
parser.add_argument("--eval_dir", type=str, default='../dataset/SID/Sony', help="path to eval set")
parser.add_argument("--fuji_eval_dir", type=str, default=None, help="path to Fuji 2025 eval set")
# parser.add_argument("--trainset_path", type=str, default="/share/data/cy/trainset_p50_s10_rgb.h5", help="path to train set")
# parser.add_argument("--trainset_path", type=str, default="/share/data/cy/trainset_p120_s200_gray.h5", help="path to train set")
# local
# parser.add_argument("--trainset_path", type=str, default="./h5_files/trainset_gray.h5", help="path to train set")







# Validation Set
# server
# parser.add_argument("--valset_path", type=str, default="/home/xhwu/lowlevel/DilatedTCN/data_val/", help="path to val set")
#
parser.add_argument("--patch_size", type=int, default=512, help="the patch size of input")
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--load_thread", type=int, default=0, help="thread for data loader")
parser.add_argument("--train_list", type=str, default="./dataset/Sony_train.txt",
                    help="Paired filename list for SID RAW training")
parser.add_argument("--use_sid_raw", action="store_true",
                    help="Use SID RAW directory structure (short/long)")
parser.add_argument("--fuji_trainset_path", type=str, default=None,
                    help="Path to Fuji 2025 training dataset root directory")
parser.add_argument("--fuji_train_list", type=str, default=None,
                    help="Paired filename list for Fuji 2025 RAW training")
parser.add_argument("--fuji_val_list", type=str, default=None,
                    help="Paired filename list for Fuji 2025 RAW validation")
parser.add_argument("--fuji_test_list", type=str, default=None,
                    help="Paired filename list for Fuji 2025 RAW testing")
parser.add_argument("--use_fuji_raw", action="store_true",
                    help="Use Fuji 2025 RAW directory structure (short/long)")
parser.add_argument("--skip_eval", action="store_true",
                    help="Skip validation loops after each checkpoint save")
# net
parser.add_argument("--in_channels", type=int, default=4, help="Input RAW channels")
parser.add_argument("--out_channels", type=int, default=4, help="Output RAW channels")
parser.add_argument("--sd_base_channels", type=int, default=32, help="Base width for SD denoiser")
parser.add_argument("--sd_channel_mults", type=str, default="1,2,4", help="Comma separated channel multipliers")
parser.add_argument("--sd_num_steps", type=int, default=4, help="Diffusion refinement iterations")
parser.add_argument("--sd_time_embed_dim", type=int, default=64, help="Time embedding width")
parser.add_argument("--sd_cond_dim", type=int, default=64, help="Physics conditioning width")
parser.add_argument("--sd_attn_type", type=str, default="linear", choices=["linear", "channel", "standard", "flash"],
                    help="Attention type: linear (O(n) memory, recommended), channel (O(C) memory, most efficient), standard (O(n²), original), flash (requires flash-attn)")
parser.add_argument("--sd_scheduler", type=str, default="ddpm", choices=["ddpm", "ddim"],
                    help="Diffusion scheduler: ddpm (stochastic, original), ddim (deterministic, faster)")
parser.add_argument("--use_gradient_checkpointing", action="store_true",
                    help="Enable gradient checkpointing to save memory (trades compute for memory)")

# save
parser.add_argument("--last_ckpt",type=str,default="/dn_raw_DnCNN_syn_e47.pth",help="the ckpt of last net")
parser.add_argument("--resume", type=str, choices=("continue", "new"), default="new",help="continue to train model")
parser.add_argument("--save_prefix", type=str, default="dn_raw_DnCNN_syn_e",help="prefix added to all ckpt to be saved")
parser.add_argument("--log_dir", type=str, default='./logs_s1', help='path of log files')
parser.add_argument("--save_every", type=int, default=100, help="Number of training steps too log psnr and perform")
parser.add_argument("--save_every_epochs", type=int, default=1, help="Number of training epchs to save state")

parser.add_argument("--learning_rate_dtcn", type=float, default=1e-4, help="the initial learning rate")
parser.add_argument("--decay_rate", type=float, default=0.5, help="the decay rate of lr rate")
parser.add_argument("--epoch", type=int, default=200, help="number of epochs the model needs to run")
parser.add_argument("--steps", type=str, default="100,180", help="schedule steps,use comma(,) between numbers")

parser.add_argument("--save_path", type=str, default='./denoise_last_ckpt/',help="prefix added to all ckpt to be saved")

opt = parser.parse_args()

# Normalize noise between [0, 1]
steps = opt.steps
steps = steps.split(',')
opt.steps = [int(eval(step)) for step in steps]

opt.sd_channel_mults = tuple(int(val) for val in opt.sd_channel_mults.split(',') if val)


