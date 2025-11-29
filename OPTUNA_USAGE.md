# Optuna Hyperparameter Optimization Guide

## Overview

The `stg2_denoise_optuna.py` script wraps the training process with Optuna to automatically search for optimal hyperparameters. It optimizes:

- Learning rate
- Batch size
- Base channels
- Channel multipliers
- Number of diffusion steps
- Time embedding dimension
- Condition embedding dimension
- Patch size
- Attention type (linear/channel)
- Scheduler (ddpm/ddim)

## Basic Usage

### Simple Optimization (10 trials, 10 epochs each)

```bash
python stg2_denoise_optuna.py \
    --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
    --use_sid_raw \
    --n_trials 50 \
    --n_epochs_per_trial 10 \
    --save_path ./runs/optuna/
```

### With Pruning (Stop Unpromising Trials Early)

```bash
python stg2_denoise_optuna.py \
    --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
    --use_sid_raw \
    --n_trials 100 \
    --n_epochs_per_trial 20 \
    --pruning \
    --save_path ./runs/optuna/
```

### Resume Existing Study

```bash
python stg2_denoise_optuna.py \
    --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
    --use_sid_raw \
    --study_name denoise_optimization \
    --load_study \
    --n_trials 50 \
    --save_path ./runs/optuna/
```

## Arguments

### Optuna-Specific Arguments

- `--n_trials`: Number of optimization trials (default: 50)
- `--n_epochs_per_trial`: Number of epochs to train per trial (default: 10)
- `--study_name`: Name of the Optuna study (default: 'denoise_optimization')
- `--storage`: Storage URL for study (default: sqlite:///{study_name}.db)
- `--load_study`: Load existing study instead of creating new one
- `--pruning`: Enable pruning to stop unpromising trials early
- `--metric`: Metric to optimize - 'loss', 'psnr', or 'ssim' (default: 'loss')
- `--direction`: Optimization direction - 'minimize' or 'maximize' (default: 'minimize')

### Required Training Arguments

- `--trainset_path`: Path to training dataset root directory
- `--train_list`: Path to training list file (for SID RAW format)
- `--use_sid_raw`: Flag to use SID RAW directory structure

### Optional Training Arguments

All other training arguments from `stg2_denoise_train.py` can be passed through.

## Hyperparameter Search Space

The optimization searches over:

| Parameter | Range/Options |
|-----------|---------------|
| Learning Rate | 1e-5 to 1e-3 (log scale) |
| Batch Size | 4 to 32 (step 2) |
| Base Channels | [8, 16, 32, 64] |
| Channel Multipliers | (1,2), (1,2,4), (1,2,4,8) |
| Num Steps | 2 to 8 |
| Time Embed Dim | [32, 64, 128] |
| Cond Embed Dim | [32, 64, 128] |
| Patch Size | [64, 128, 256] |
| Attention Type | ['linear', 'channel'] |
| Scheduler | ['ddpm', 'ddim'] |

## Output

After optimization completes:

1. **Best Parameters JSON**: `{save_path}/best_params.json`
   - Contains best hyperparameters found
   - Best objective value
   - Trial number

2. **Optimization History Plot**: `{save_path}/optimization_history.html`
   - Interactive plot showing optimization progress
   - Requires plotly (included in requirements)

3. **Parameter Importance Plot**: `{save_path}/param_importances.html`
   - Shows which hyperparameters are most important
   - Helps understand what matters most

4. **Study Database**: `{study_name}.db`
   - SQLite database with all trial results
   - Can be loaded later to resume or analyze

5. **Trial Checkpoints**: `{save_path}/optuna_trial_{N}/`
   - Individual checkpoints for each trial
   - TensorBoard logs for each trial

## Using Best Parameters

After optimization, use the best parameters for full training:

```bash
# Read best_params.json and extract parameters, then:
python stg2_denoise_train.py \
    --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
    --use_sid_raw \
    --learning_rate_dtcn {best_lr} \
    --batch_size {best_batch_size} \
    --sd_base_channels {best_base_channels} \
    --sd_channel_mults {best_channel_mults} \
    --sd_num_steps {best_num_steps} \
    --sd_time_embed_dim {best_time_embed_dim} \
    --sd_cond_dim {best_cond_dim} \
    --patch_size {best_patch_size} \
    --sd_attn_type {best_attn_type} \
    --sd_scheduler {best_scheduler} \
    --epoch 500 \
    --save_path ./runs/final_training/
```

## Tips

1. **Start Small**: Begin with `--n_trials 20 --n_epochs_per_trial 5` to test
2. **Use Pruning**: Enable `--pruning` to save time on bad trials
3. **Increase Gradually**: Once you find promising regions, increase trials and epochs
4. **Monitor Progress**: Check TensorBoard logs for each trial
5. **Resume Studies**: Use `--load_study` to continue optimization
6. **Parallel Trials**: Optuna can run trials in parallel (see Optuna documentation)

## Example Workflow

1. **Quick Search** (20 trials, 5 epochs each):
   ```bash
   python stg2_denoise_optuna.py --trainset_path ... --train_list ... --use_sid_raw \
       --n_trials 20 --n_epochs_per_trial 5 --pruning
   ```

2. **Refined Search** (50 trials, 10 epochs each):
   ```bash
   python stg2_denoise_optuna.py --trainset_path ... --train_list ... --use_sid_raw \
       --n_trials 50 --n_epochs_per_trial 10 --pruning --load_study
   ```

3. **Final Training** with best parameters:
   ```bash
   python stg2_denoise_train.py ... [best parameters from JSON]
   ```

## Advanced: Custom Search Space

To modify the search space, edit `stg2_denoise_optuna.py` and adjust the `trial.suggest_*` calls in the `objective()` function.

