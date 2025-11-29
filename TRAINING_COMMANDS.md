# Training Commands - Complete CLI Examples

## Basic Training with Linear Attention (Recommended)

```bash
python stg2_denoise_train.py \
    --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
    --use_sid_raw \
    --sd_attn_type linear \
    --sd_scheduler ddim \
    --epoch 500 \
    --batch_size 8 \
    --load_thread 8 \
    --patch_size 128 \
    --sd_base_channels 16 \
    --sd_channel_mults 1,2 \
    --save_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/runs/sid_try/ \
    --save_prefix sid_try_epoch_ \
    --resume new \
    --skip_eval
```

## Maximum Memory Efficiency (Channel Attention + Gradient Checkpointing)

```bash
python stg2_denoise_train.py \
    --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
    --use_sid_raw \
    --sd_attn_type channel \
    --sd_scheduler ddim \
    --use_gradient_checkpointing \
    --epoch 500 \
    --batch_size 16 \
    --load_thread 8 \
    --patch_size 128 \
    --sd_base_channels 16 \
    --sd_channel_mults 1,2 \
    --save_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/runs/sid_try/ \
    --save_prefix sid_try_epoch_ \
    --resume new \
    --skip_eval
```

## Full Training with Validation (Linear Attention)

```bash
python stg2_denoise_train.py \
    --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
    --use_sid_raw \
    --eval_dir /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --sd_attn_type linear \
    --sd_scheduler ddim \
    --epoch 500 \
    --batch_size 8 \
    --load_thread 8 \
    --patch_size 128 \
    --sd_base_channels 16 \
    --sd_channel_mults 1,2 \
    --sd_num_steps 4 \
    --sd_time_embed_dim 64 \
    --sd_cond_dim 64 \
    --save_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/runs/sid_try/ \
    --save_prefix sid_try_epoch_ \
    --save_every_epochs 1 \
    --learning_rate_dtcn 1e-4 \
    --resume new
```

## Training with Both SID and Fuji Datasets (Linear Attention)

```bash
python stg2_denoise_train.py \
    --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
    --use_sid_raw \
    --fuji_trainset_path /path/to/fuji/dataset \
    --fuji_train_list /path/to/fuji_train.txt \
    --use_fuji_raw \
    --sd_attn_type linear \
    --sd_scheduler ddim \
    --epoch 500 \
    --batch_size 8 \
    --load_thread 8 \
    --patch_size 128 \
    --sd_base_channels 16 \
    --sd_channel_mults 1,2 \
    --save_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/runs/combined_try/ \
    --save_prefix combined_epoch_ \
    --resume new \
    --skip_eval
```

## Resume Training

```bash
python stg2_denoise_train.py \
    --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
    --use_sid_raw \
    --sd_attn_type linear \
    --epoch 500 \
    --batch_size 8 \
    --load_thread 8 \
    --patch_size 128 \
    --sd_base_channels 16 \
    --sd_channel_mults 1,2 \
    --save_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/runs/sid_try/ \
    --save_prefix sid_try_epoch_ \
    --resume continue \
    --skip_eval
```

## Parameter Descriptions

### Required Arguments
- `--trainset_path`: Path to training dataset root directory
- `--train_list`: Path to training list file (for SID RAW format)
- `--use_sid_raw`: Flag to use SID RAW directory structure

### Architecture Arguments
- `--sd_attn_type`: Attention type - `linear` (recommended, default), `channel` (most efficient)
- `--sd_scheduler`: Diffusion scheduler - `ddpm` (original), `ddim` (faster, deterministic)
- `--use_gradient_checkpointing`: Enable for additional memory savings (slower training)

### Training Arguments
- `--epoch`: Number of training epochs
- `--batch_size`: Batch size (can increase with efficient attention)
- `--patch_size`: Patch size for training
- `--sd_base_channels`: Base number of channels
- `--sd_channel_mults`: Channel multipliers (comma-separated, e.g., "1,2" or "1,2,4")
- `--learning_rate_dtcn`: Learning rate (default: 1e-4)
- `--save_every_epochs`: Save checkpoint every N epochs

### Path Arguments
- `--save_path`: Directory to save checkpoints
- `--save_prefix`: Prefix for checkpoint filenames
- `--resume`: `new` for new training, `continue` to resume

### Optional Arguments
- `--skip_eval`: Skip validation during training (faster)
- `--load_thread`: Number of data loading threads
- `--eval_dir`: Path to evaluation dataset (if not skipping eval)

## Memory Usage Comparison

| Configuration | Approx. GPU Memory | Batch Size |
|--------------|-------------------|------------|
| Linear attention | ~2GB | 8-16 |
| Channel attention | ~1GB | 16-32 |
| Channel + checkpointing | ~0.5GB | 32+ |

## Tips

1. **Start with linear attention** - Best balance of memory and performance
2. **Use DDIM scheduler** - Faster inference, deterministic results
3. **Increase batch size** - With efficient attention, you can use larger batches
4. **Enable gradient checkpointing** - If still running out of memory
5. **Use channel attention** - For maximum memory savings (may need fine-tuning)

