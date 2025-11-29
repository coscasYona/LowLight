# Optuna Hyperparameter Optimization - Quick Start

## Installation

First, install Optuna:
```bash
pip install optuna plotly
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Basic Example

```bash
python stg2_denoise_optuna.py \
    --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
    --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
    --use_sid_raw \
    --n_trials 50 \
    --n_epochs_per_trial 10 \
    --pruning \
    --save_path ./runs/optuna/
```

## What It Does

1. **Searches hyperparameter space** automatically
2. **Trains multiple trials** with different configurations
3. **Prunes unpromising trials** early (if enabled)
4. **Tracks all results** in SQLite database
5. **Finds best parameters** and saves them to JSON
6. **Generates visualizations** (optimization history, parameter importance)

## Output Files

- `best_params.json` - Best hyperparameters found
- `optimization_history.html` - Interactive optimization plot
- `param_importances.html` - Parameter importance plot
- `{study_name}.db` - SQLite database with all trials
- `optuna_trial_{N}/` - Individual trial checkpoints and logs

## Next Steps

After optimization, use the best parameters from `best_params.json` for full training with `stg2_denoise_train.py`.

See `OPTUNA_USAGE.md` for detailed documentation.

