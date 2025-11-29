# Optuna Hyperparameter Optimization - CLI Commands

## Quick Start

### Basic Optimization (SID Dataset)

```bash
python stg2_denoise_optuna.py \
    --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
    --use_sid_raw \
    --n_trials 50 \
    --n_epochs_per_trial 10 \
    --study_name denoise_optimization_v2 \
    --save_path ./runs/optuna/ \
    --pruning
```

### With Fuji 2025 Dataset

```bash
python stg2_denoise_optuna.py \
    --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
    --use_sid_raw \
    --fuji_trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Fuji2025 \
    --fuji_train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Fuji2025/Fuji_train_list.txt \
    --use_fuji_raw \
    --n_trials 50 \
    --n_epochs_per_trial 10 \
    --study_name denoise_optimization_v2 \
    --save_path ./runs/optuna/ \
    --pruning
```

## Full Command with All Options

```bash
python stg2_denoise_optuna.py \
    --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
    --use_sid_raw \
    --n_trials 100 \
    --n_epochs_per_trial 20 \
    --study_name denoise_optimization_v2 \
    --storage sqlite:///optuna_studies.db \
    --pruning \
    --metric loss \
    --direction minimize \
    --save_path ./runs/optuna/ \
    --load_thread 8
```

## Quick Test (1 trial, 1 epoch)

```bash
python stg2_denoise_optuna.py \
    --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
    --use_sid_raw \
    --n_trials 1 \
    --n_epochs_per_trial 1 \
    --study_name test_optuna \
    --save_path ./runs/test_optuna/
```

## Resume Existing Study

```bash
python stg2_denoise_optuna.py \
    --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
    --use_sid_raw \
    --n_trials 50 \
    --n_epochs_per_trial 10 \
    --study_name denoise_optimization_v2 \
    --load_study \
    --save_path ./runs/optuna/ \
    --pruning
```

## Arguments Reference

### Required Arguments
- `--trainset_path`: Path to training dataset root directory
- `--train_list`: Path to training list file (for SID RAW format)
- `--use_sid_raw`: Flag to use SID RAW directory structure

### Optuna Arguments
- `--n_trials`: Number of optimization trials (default: 50)
- `--n_epochs_per_trial`: Number of epochs per trial (default: 10)
- `--study_name`: Name of the Optuna study (default: 'denoise_optimization')
- `--storage`: Storage URL (default: sqlite:///{study_name}.db)
- `--load_study`: Load existing study instead of creating new one
- `--pruning`: Enable pruning (stop unpromising trials early)
- `--metric`: Metric to optimize - 'loss', 'psnr', or 'ssim' (default: 'loss')
- `--direction`: Optimization direction - 'minimize' or 'maximize' (default: 'minimize')

### Optional Training Arguments
- `--save_path`: Base save path for checkpoints and results (default: './runs/optuna/')
- `--load_thread`: Number of data loader threads (default: 8)
- `--fuji_trainset_path`: Path to Fuji 2025 training dataset
- `--fuji_train_list`: Path to Fuji 2025 training list file
- `--use_fuji_raw`: Flag to use Fuji 2025 RAW format

## Example Workflows

### 1. Quick Exploration (10 trials, 5 epochs each)
```bash
python stg2_denoise_optuna.py \
    --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
    --use_sid_raw \
    --n_trials 10 \
    --n_epochs_per_trial 5 \
    --study_name quick_explore \
    --save_path ./runs/optuna/ \
    --pruning
```

### 2. Full Optimization (100 trials, 20 epochs each)
```bash
python stg2_denoise_optuna.py \
    --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
    --use_sid_raw \
    --n_trials 100 \
    --n_epochs_per_trial 20 \
    --study_name full_optimization \
    --save_path ./runs/optuna/ \
    --pruning
```

### 3. Memory-Efficient (Channel Attention)
```bash
python stg2_denoise_optuna.py \
    --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
    --use_sid_raw \
    --n_trials 50 \
    --n_epochs_per_trial 10 \
    --study_name optuna_channel \
    --save_path ./runs/optuna/ \
    --pruning \
    --sd_attn_type channel
```

## Output Files

After optimization completes, you'll find:

- `{save_path}/best_params.json` - Best hyperparameters found
- `{save_path}/optimization_history.html` - Interactive optimization plot
- `{save_path}/param_importances.html` - Parameter importance plot (if 2+ trials)
- `{study_name}.db` - SQLite database with all trial results
- `{save_path}/optuna_trial_{N}/` - Individual trial checkpoints and logs

## Troubleshooting

### If you get "incompatible parameter format" error:

```bash
# Option 1: Use a new study name
--study_name denoise_optimization_v2

# Option 2: Delete old database
rm denoise_optimization.db

# Option 3: Check study details
python migrate_optuna_study.py --study_name denoise_optimization
```

### View study information:

```bash
python migrate_optuna_study.py --study_name denoise_optimization --list
```

