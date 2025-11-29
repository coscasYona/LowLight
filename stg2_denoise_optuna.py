#!/usr/bin/env python3
"""
Simple Optuna wrapper for hyperparameter optimization.
Wraps the existing stg2_denoise_train.py script.
"""

import os
import sys
import argparse
import json
import optuna
from optuna.trial import TrialState

# Prevent stg2_denoise_options from parsing sys.argv
_original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]
import stg2_denoise_train as train_module
sys.argv = _original_argv


def objective(trial, base_args):
    """Optuna objective function - simple wrapper around training."""
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 4, 32, step=2)
    sd_base_channels = trial.suggest_categorical('sd_base_channels', [8, 16, 32, 64])
    channel_mults_str = trial.suggest_categorical('sd_channel_mults', ['1,2', '1,2,4', '1,2,4,8'])
    channel_mults = tuple(int(x) for x in channel_mults_str.split(','))
    sd_num_steps = trial.suggest_int('sd_num_steps', 2, 8)
    patch_size = trial.suggest_categorical('patch_size', [64, 128, 256])
    attn_type = trial.suggest_categorical('sd_attn_type', ['linear', 'channel'])
    
    # Update args
    args = argparse.Namespace(**vars(base_args))
    args.learning_rate_dtcn = learning_rate
    args.batch_size = batch_size
    args.sd_base_channels = sd_base_channels
    args.sd_channel_mults = channel_mults
    args.sd_num_steps = sd_num_steps
    args.patch_size = patch_size
    args.sd_attn_type = attn_type
    args.resume = 'new'
    args.save_every_epochs = 1
    
    # Unique save path for this trial
    trial_id = trial.number
    args.save_path = os.path.join(base_args.save_path, f'trial_{trial_id}')
    args.save_prefix = f'trial_{trial_id}_epoch_'
    
    print(f"\nTrial {trial_id}: lr={learning_rate:.6f}, bs={batch_size}, ch={sd_base_channels}, mults={channel_mults}, steps={sd_num_steps}, patch={patch_size}, attn={attn_type}")
    
    try:
        # Run training and capture final loss
        final_loss = train_module.main(args)
        return final_loss if final_loss is not None else float('inf')
    except Exception as e:
        print(f"Trial {trial_id} failed: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')


def main():
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Optimization')
    
    # Optuna args
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--study_name', type=str, default='denoise_optuna', help='Study name')
    parser.add_argument('--pruning', action='store_true', help='Enable pruning')
    
    # Training args (required)
    parser.add_argument('--trainset_path', type=str, required=True)
    parser.add_argument('--train_list', type=str, required=True)
    parser.add_argument('--use_sid_raw', action='store_true')
    parser.add_argument('--save_path', type=str, default='./runs/optuna/')
    parser.add_argument('--epoch', type=int, default=10, help='Epochs per trial')
    
    # Parse and separate
    optuna_args, remaining = parser.parse_known_args()
    
    # Create base args with defaults
    base_args = argparse.Namespace()
    base_args.trainset_path = optuna_args.trainset_path
    base_args.train_list = optuna_args.train_list
    base_args.use_sid_raw = optuna_args.use_sid_raw
    base_args.save_path = optuna_args.save_path
    base_args.epoch = optuna_args.epoch
    base_args.skip_eval = True  # Skip validation for speed
    base_args.load_thread = 8
    base_args.in_channels = 4
    base_args.out_channels = 4
    base_args.sd_time_embed_dim = 64
    base_args.sd_cond_dim = 64
    base_args.sd_scheduler = 'ddim'
    base_args.use_gradient_checkpointing = False
    base_args.use_fuji_raw = False
    base_args.fuji_trainset_path = None
    base_args.fuji_train_list = None
    
    # Parse remaining args
    i = 0
    while i < len(remaining):
        if remaining[i].startswith('--'):
            key = remaining[i][2:].replace('-', '_')
            if i + 1 < len(remaining) and not remaining[i + 1].startswith('--'):
                value = remaining[i + 1]
                try:
                    if '.' in value:
                        setattr(base_args, key, float(value))
                    else:
                        setattr(base_args, key, int(value))
                except ValueError:
                    setattr(base_args, key, value)
                i += 2
            else:
                setattr(base_args, key, True)
                i += 1
        else:
            i += 1
    
    # Create study
    storage = f"sqlite:///{optuna_args.study_name}.db"
    try:
        study = optuna.create_study(
            study_name=optuna_args.study_name,
            storage=storage,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner() if optuna_args.pruning else None,
        )
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(study_name=optuna_args.study_name, storage=storage)
        print(f"Loaded existing study: {optuna_args.study_name}")
    
    print(f"\nStarting optimization: {optuna_args.n_trials} trials")
    print(f"Study: {optuna_args.study_name}\n")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_args),
        n_trials=optuna_args.n_trials,
        show_progress_bar=True,
    )
    
    # Print results
    print(f"\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.6f}")
    print(f"  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best params
    os.makedirs(optuna_args.save_path, exist_ok=True)
    with open(os.path.join(optuna_args.save_path, 'best_params.json'), 'w') as f:
        json.dump({'best_value': trial.value, 'best_params': trial.params}, f, indent=2)
    
    print(f"\nBest parameters saved to: {os.path.join(optuna_args.save_path, 'best_params.json')}")


if __name__ == '__main__':
    main()
