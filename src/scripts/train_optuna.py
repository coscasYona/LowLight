#!/usr/bin/env python3
"""
Optuna hyperparameter optimization wrapper for EMVA 1288 training.
Wraps the Hydra-based train.py script for hyperparameter sweeps.
"""

import os
import sys
import json
import argparse
import optuna
from optuna.trial import TrialState

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Hydra utilities
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import pytorch_lightning as pl

# Import training function
from train import train_with_config


def objective(trial, base_config_overrides, config_path, config_name):
    """Optuna objective function - wraps training with suggested hyperparameters."""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 1, 3, step=1)  # Limited for laptop constraints
    base_channels = trial.suggest_categorical('base_channels', [32, 64, 128])
    
    # Channel multipliers as string (Optuna doesn't support lists in categorical)
    channel_mults_options = ['1,2', '1,2,4', '1,2,4,8']
    channel_mults_str = trial.suggest_categorical('channel_mults', channel_mults_options)
    # Convert to list for display and Hydra
    channel_mults = [int(x) for x in channel_mults_str.split(',')]
    
    num_steps = trial.suggest_int('num_steps', 4, 64)
    patch_size = trial.suggest_categorical('patch_size', [128, 256, 512])
    attn_type = trial.suggest_categorical('attn_type', ['linear', 'channel'])
    scheduler = trial.suggest_categorical('scheduler', ['ddpm', 'ddim'])
    
    # Optional: suggest loss weights
    l1_weight = trial.suggest_float('l1_weight', 0.5, 1.0)
    gradient_weight = trial.suggest_float('gradient_weight', 0.01, 0.1)
    
    # Build Hydra config overrides for this trial
    trial_overrides = base_config_overrides.copy()
    # Format channel_mults as Hydra list syntax
    channel_mults_hydra = '[' + ','.join(map(str, channel_mults)) + ']'
    trial_overrides.extend([
        f'training.learning_rate={learning_rate}',
        f'training.batch_size={batch_size}',
        f'model.base_channels={base_channels}',
        f'model.channel_mults={channel_mults_hydra}',
        f'model.num_steps={num_steps}',
        f'model.attn_type={attn_type}',
        f'model.scheduler={scheduler}',
        f'data.patch_size={patch_size}',
        f'training.l1_weight={l1_weight}',
        f'training.gradient_weight={gradient_weight}',
    ])
    
    # Force single GPU for Optuna trials (multi-GPU requires torchrun, not compatible with Optuna)
    trial_overrides.append('hardware.devices=1')
    
    # Unique save path for this trial
    trial_id = trial.number
    trial_overrides.append(f'paths.save_dir=./checkpoints/optuna_trial_{trial_id}')
    trial_overrides.append(f'paths.log_dir=./logs/optuna_trial_{trial_id}')
    
    # Reduce epochs for faster sweeps (can be overridden via base_overrides)
    has_epochs_override = any('training.epochs=' in override for override in trial_overrides)
    if not has_epochs_override:
        trial_overrides.append('training.epochs=20')  # Default for sweeps
    
    print(f"\nTrial {trial_id}: lr={learning_rate:.6f}, bs={batch_size}, "
          f"ch={base_channels}, mults={channel_mults}, steps={num_steps}, "
          f"patch={patch_size}, attn={attn_type}, sched={scheduler}")
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra with config directory
    with initialize_config_dir(config_dir=config_path, version_base=None):
        # Compose config with overrides
        cfg = compose(config_name=config_name, overrides=trial_overrides)
        
        try:
            # Run training and capture final loss
            final_loss = train_with_config(cfg)
            return final_loss if final_loss is not None and final_loss != float('inf') else float('inf')
        except Exception as e:
            print(f"Trial {trial_id} failed: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')
        finally:
            # Clean up Hydra instance
            GlobalHydra.instance().clear()


def main():
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Optimization for EMVA1288 Training')
    
    # Optuna args
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--study_name', type=str, default='emva1288_optuna', help='Study name')
    parser.add_argument('--storage', type=str, default=None, 
                       help='Storage URL (default: databases/optuna/production/{study_name}.db)')
    parser.add_argument('--test', action='store_true', 
                       help='Use test database folder instead of production')
    parser.add_argument('--pruning', action='store_true', help='Enable pruning')
    
    # Training config overrides (passed to Hydra)
    parser.add_argument('--config-path', type=str, default=None,
                       help='Path to config directory (default: ../configs relative to script)')
    parser.add_argument('--config-name', type=str, default='train',
                       help='Config name (default: train)')
    parser.add_argument('--data', type=str, default=None,
                       help='Data config override (e.g., sid, eld)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model config override (e.g., emva1288, unet)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs per trial (overrides default sweep epochs)')
    
    # Parse args
    args, remaining = parser.parse_known_args()
    
    # Build base config overrides from args
    base_overrides = []
    if args.data:
        base_overrides.append(f'data={args.data}')
    if args.model:
        base_overrides.append(f'model={args.model}')
    if args.epochs:
        base_overrides.append(f'training.epochs={args.epochs}')
    
    # Parse remaining args as Hydra overrides
    i = 0
    while i < len(remaining):
        if remaining[i].startswith('--'):
            # Convert --arg=value or --arg value to Hydra format
            if '=' in remaining[i]:
                key, value = remaining[i][2:].split('=', 1)
                base_overrides.append(f'{key.replace("-", "_")}={value}')
                i += 1
            elif i + 1 < len(remaining) and not remaining[i + 1].startswith('--'):
                key = remaining[i][2:].replace('-', '_')
                value = remaining[i + 1]
                base_overrides.append(f'{key}={value}')
                i += 2
            else:
                key = remaining[i][2:].replace('-', '_')
                base_overrides.append(f'{key}=true')
                i += 1
        else:
            i += 1
    
    # Determine config path
    if args.config_path:
        config_path = os.path.abspath(args.config_path)
    else:
        # Default: ../configs relative to script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '..', 'configs')
        config_path = os.path.abspath(config_path)
    
    # Create study with organized database location
    if args.storage:
        storage = args.storage
    else:
        # Use organized folder structure
        db_folder = 'databases/optuna/test' if args.test else 'databases/optuna/production'
        os.makedirs(db_folder, exist_ok=True)
        storage = f"sqlite:///{db_folder}/{args.study_name}.db"
    
    try:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner() if args.pruning else None,
        )
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(study_name=args.study_name, storage=storage)
        print(f"Loaded existing study: {args.study_name}")
    
    print(f"\nStarting optimization: {args.n_trials} trials")
    print(f"Study: {args.study_name}")
    print(f"Database: {storage}")
    print(f"Config path: {config_path}")
    print(f"Config name: {args.config_name}")
    if base_overrides:
        print(f"Base overrides: {base_overrides}")
    print()
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_overrides, config_path, args.config_name),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.6f}")
    print(f"  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print(f"{'='*60}")
    
    # Save results to files
    save_path = './checkpoints/optuna'
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Save best params JSON
    best_params_path = os.path.join(save_path, f'{args.study_name}_best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_value': trial.value,
            'best_params': trial.params,
            'study_name': args.study_name,
            'storage': storage,
            'n_trials': args.n_trials,
            'best_trial_number': trial.number
        }, f, indent=2)
    
    print(f"\nBest parameters saved to: {best_params_path}")
    
    # Database is the primary storage - all results are in SQLite
    db_path = storage.replace('sqlite:///', '')
    print(f"\n{'='*60}")
    print(f"Optuna SQLite Database (PRIMARY STORAGE):")
    print(f"  Location: {db_path}")
    print(f"  Study: {args.study_name}")
    print(f"\nTo view results in Optuna Dashboard, run:")
    print(f"  optuna dashboard --storage {storage}")
    print(f"\nOr load in Python:")
    print(f"  import optuna")
    print(f"  study = optuna.load_study(study_name='{args.study_name}', storage='{storage}')")
    print(f"{'='*60}")
    
    # Print command to use best params
    print(f"\nTo use best parameters, run training with:")
    print(f"  python src/scripts/train.py \\")
    for key, value in trial.params.items():
        if key == 'channel_mults':
            # Format as Hydra list
            if isinstance(value, list):
                value_str = '[' + ','.join(map(str, value)) + ']'
            else:
                value_str = value
            print(f"    model.{key}={value_str} \\")
        elif key in ['learning_rate', 'batch_size', 'l1_weight', 'gradient_weight']:
            print(f"    training.{key}={value} \\")
        elif key in ['base_channels', 'num_steps', 'attn_type', 'scheduler']:
            print(f"    model.{key}={value} \\")
        elif key == 'patch_size':
            print(f"    data.{key}={value} \\")
    print()


if __name__ == '__main__':
    main()

