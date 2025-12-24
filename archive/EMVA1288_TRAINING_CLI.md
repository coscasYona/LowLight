# EMVA1288 Training CLI - From Scratch

## Complete Training Command (Recommended)

```bash
python stg2_emva1288_train.py \
    --trainset_path /workspace/data/SID/Sony \
    --train_list /workspace/data/SID/Sony_train_list.txt \
    --use_sid_raw \
    --emva_camera_type SonyA7S2 \
    --emva_noise_code prq \
    --sd_attn_type linear \
    --sd_scheduler ddim \
    --epoch 500 \
    --batch_size 8 \
    --load_thread 8 \
    --patch_size 512 \
    --sd_base_channels 32 \
    --sd_channel_mults 1,2,4 \
    --sd_num_steps 4 \
    --sd_time_embed_dim 64 \
    --sd_cond_dim 64 \
    --learning_rate_dtcn 1e-4 \
    --loss_scale 10.0 \
    --use_grad_clip \
    --grad_clip_max 1.0 \
    --l1_weight 0.8 \
    --gradient_weight 0.1 \
    --save_path ./runs/emva1288_from_scratch/ \
    --save_prefix emva1288_epoch_ \
    --save_every_epochs 1 \
    --resume new \
    --skip_eval
```

## Memory-Efficient Training (Lower GPU Memory)

```bash
python stg2_emva1288_train.py \
    --trainset_path /workspace/data/SID/Sony \
    --train_list /workspace/data/SID/Sony_train_list.txt \
    --use_sid_raw \
    --emva_camera_type SonyA7S2 \
    --emva_noise_code prq \
    --sd_attn_type channel \
    --sd_scheduler ddim \
    --use_gradient_checkpointing \
    --epoch 500 \
    --batch_size 16 \
    --load_thread 8 \
    --patch_size 256 \
    --sd_base_channels 16 \
    --sd_channel_mults 1,2 \
    --sd_num_steps 4 \
    --learning_rate_dtcn 1e-4 \
    --loss_scale 10.0 \
    --use_grad_clip \
    --l1_weight 0.8 \
    --gradient_weight 0.1 \
    --save_path ./runs/emva1288_memory_efficient/ \
    --save_prefix emva1288_epoch_ \
    --save_every_epochs 1 \
    --resume new \
    --skip_eval
```

## Maximum Sharpness Training (Higher L1 + Gradient Weight)

```bash
python stg2_emva1288_train.py \
    --trainset_path /workspace/data/SID/Sony \
    --train_list /workspace/data/SID/Sony_train_list.txt \
    --use_sid_raw \
    --emva_camera_type SonyA7S2 \
    --emva_noise_code prq \
    --sd_attn_type linear \
    --sd_scheduler ddim \
    --epoch 500 \
    --batch_size 8 \
    --load_thread 8 \
    --patch_size 512 \
    --sd_base_channels 32 \
    --sd_channel_mults 1,2,4 \
    --sd_num_steps 4 \
    --learning_rate_dtcn 1e-4 \
    --loss_scale 10.0 \
    --use_grad_clip \
    --l1_weight 0.9 \
    --gradient_weight 0.2 \
    --save_path ./runs/emva1288_sharp/ \
    --save_prefix emva1288_epoch_ \
    --save_every_epochs 1 \
    --resume new \
    --skip_eval
```

## Training with Validation (Full Training)

```bash
python stg2_emva1288_train.py \
    --trainset_path /workspace/data/SID/Sony \
    --train_list /workspace/data/SID/Sony_train_list.txt \
    --use_sid_raw \
    --eval_dir /workspace/data/SID/Sony \
    --emva_camera_type SonyA7S2 \
    --emva_noise_code prq \
    --sd_attn_type linear \
    --sd_scheduler ddim \
    --epoch 500 \
    --batch_size 8 \
    --load_thread 8 \
    --patch_size 512 \
    --sd_base_channels 32 \
    --sd_channel_mults 1,2,4 \
    --sd_num_steps 4 \
    --learning_rate_dtcn 1e-4 \
    --loss_scale 10.0 \
    --use_grad_clip \
    --l1_weight 0.8 \
    --gradient_weight 0.1 \
    --save_path ./runs/emva1288_full/ \
    --save_prefix emva1288_epoch_ \
    --save_every_epochs 1 \
    --resume new
```

## Training with Both SID and Fuji Datasets

```bash
python stg2_emva1288_train.py \
    --trainset_path /workspace/data/SID/Sony \
    --train_list /workspace/data/SID/Sony_train_list.txt \
    --use_sid_raw \
    --fuji_trainset_path /workspace/data/SID/Fuji \
    --fuji_train_list /workspace/data/SID/Fuji_train_list.txt \
    --use_fuji_raw \
    --emva_camera_type SonyA7S2 \
    --emva_noise_code prq \
    --sd_attn_type linear \
    --sd_scheduler ddim \
    --epoch 500 \
    --batch_size 8 \
    --load_thread 8 \
    --patch_size 512 \
    --sd_base_channels 32 \
    --sd_channel_mults 1,2,4 \
    --sd_num_steps 4 \
    --learning_rate_dtcn 1e-4 \
    --loss_scale 10.0 \
    --use_grad_clip \
    --l1_weight 0.8 \
    --gradient_weight 0.1 \
    --save_path ./runs/emva1288_combined/ \
    --save_prefix emva1288_epoch_ \
    --save_every_epochs 1 \
    --resume new \
    --skip_eval
```

## Parameter Explanations

### Required Arguments
- `--trainset_path`: Path to training dataset root directory (e.g., `/workspace/data/SID/Sony`)
- `--train_list`: Path to training list file (e.g., `/workspace/data/SID/Sony_train_list.txt`)
- `--use_sid_raw`: Flag to use SID RAW directory structure (short/long folders)

### EMVA1288 Specific
- `--emva_camera_type`: Camera type for noise model (`SonyA7S2`, `IMX686`, `NikonD850`, etc.)
- `--emva_noise_code`: Noise components (`prq` = Poisson shot + row + quantization)

### Architecture
- `--sd_attn_type`: `linear` (recommended, O(n) memory) or `channel` (most efficient, O(C) memory)
- `--sd_scheduler`: `ddim` (faster, deterministic) or `ddpm` (original, stochastic)
- `--sd_base_channels`: Base number of channels (16-32 typical)
- `--sd_channel_mults`: Channel multipliers, comma-separated (e.g., `1,2,4` or `1,2`)
- `--sd_num_steps`: Number of diffusion steps (4-50 typical)
- `--use_gradient_checkpointing`: Enable for memory savings (slower training)

### Training Parameters
- `--epoch`: Number of training epochs (500+ recommended)
- `--batch_size`: Batch size (8-16 for linear attention, 16-32 for channel attention)
- `--patch_size`: Training patch size (256-512 typical)
- `--learning_rate_dtcn`: Learning rate (default: 1e-4)
- `--load_thread`: Number of data loading threads (8 recommended)

### Loss Function Parameters (New)
- `--loss_scale`: Loss scaling factor to increase gradient strength (default: 10.0)
- `--use_grad_clip`: Enable gradient clipping for stability
- `--grad_clip_max`: Maximum gradient norm (default: 1.0)
- `--l1_weight`: Weight for L1 loss in hybrid MSE+L1 (default: 0.8, higher = sharper)
- `--gradient_weight`: Weight for edge-preserving gradient loss (default: 0.1, higher = sharper edges)

### Save/Resume
- `--save_path`: Directory to save checkpoints
- `--save_prefix`: Prefix for checkpoint filenames
- `--save_every_epochs`: Save checkpoint every N epochs (1 = every epoch)
- `--resume`: `new` for from-scratch training, `continue` to resume
- `--skip_eval`: Skip validation during training (faster)

## Quick Start

For a quick from-scratch training with good defaults:

```bash
python stg2_emva1288_train.py \
    --trainset_path /workspace/data/SID/Sony \
    --train_list /workspace/data/SID/Sony_train_list.txt \
    --use_sid_raw \
    --emva_camera_type SonyA7S2 \
    --emva_noise_code prq \
    --sd_attn_type linear \
    --sd_scheduler ddim \
    --epoch 500 \
    --batch_size 8 \
    --load_thread 8 \
    --patch_size 512 \
    --sd_base_channels 32 \
    --sd_channel_mults 1,2,4 \
    --loss_scale 10.0 \
    --use_grad_clip \
    --l1_weight 0.8 \
    --gradient_weight 0.1 \
    --save_path ./runs/emva1288/ \
    --save_prefix emva1288_epoch_ \
    --resume new \
    --skip_eval
```

## Tips

1. **Start with linear attention** - Best balance of memory and performance
2. **Use loss_scale 10.0** - Helps with small gradients when loss converges quickly
3. **Enable gradient clipping** - Stabilizes training with loss scaling
4. **L1 weight 0.8** - Good balance for sharp images without artifacts
5. **Gradient weight 0.1-0.2** - Preserves edges, adjust if too sharp/blurry
6. **Use DDIM scheduler** - Faster inference, deterministic results
7. **Patch size 512** - Better for detail preservation, 256 if memory constrained
8. **Skip eval during training** - Faster training, run validation separately

