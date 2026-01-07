#!/usr/bin/env python3
"""
Optuna hyperparameter optimization wrapper for EMVA 1288 training.
Wraps the Hydra-based train.py script for hyperparameter sweeps.
"""

import os
import sys
import json
import argparse
import subprocess
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


def objective(trial, base_config_overrides, config_path, config_name, use_torchrun, num_gpus):
    """Optuna objective function - wraps training with suggested hyperparameters."""
    
    # ==========================================================================
    # FIXED ARCHITECTURE PARAMETERS (not searched by Optuna)
    # These are fixed to ensure consistent model size (~25M params, ~100MB)
    # and fair comparison between trials
    # ==========================================================================
    base_channels = 64          # Fixed: good balance of capacity and speed
    channel_mults = [1, 2, 4, 8]  # Fixed: full depth for quality
    num_steps = 8               # Fixed: good balance for DDIM scheduler
    patch_size = 256            # Fixed: reasonable training patch size
    attn_type = 'linear'        # Fixed: memory-efficient attention
    scheduler = 'ddim'          # Fixed: faster inference with deterministic sampling
    
    # ==========================================================================
    # OPTIMIZED HYPERPARAMETERS (searched by Optuna)
    # Focus on training dynamics rather than architecture
    # ==========================================================================
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 1, 4, step=1)  # Hardware-dependent
    
    # Loss weights - critical for balancing reconstruction vs edge preservation
    l1_weight = trial.suggest_float('l1_weight', 0.3, 0.9)
    gradient_weight = trial.suggest_float('gradient_weight', 0.01, 0.15)
    
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
    
    # Unique save path for this trial (use absolute paths for torchrun compatibility)
    trial_id = trial.number
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    trial_overrides.append(f'paths.save_dir={workspace_root}/checkpoints/optuna_trial_{trial_id}')
    trial_overrides.append(f'paths.log_dir={workspace_root}/logs/optuna_trial_{trial_id}')
    
    # Set epochs for sweeps (can be overridden via base_overrides or --epochs)
    has_epochs_override = any('training.epochs=' in override for override in trial_overrides)
    if not has_epochs_override:
        trial_overrides.append('training.epochs=50')  # Default for sweeps (enough for convergence signal)
    
    print(f"\nTrial {trial_id}: lr={learning_rate:.6f}, bs={batch_size}, "
          f"l1_w={l1_weight:.3f}, grad_w={gradient_weight:.3f} "
          f"[fixed: ch={base_channels}, mults={channel_mults}, steps={num_steps}]")
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra with config directory
    with initialize_config_dir(config_dir=config_path, version_base=None):
        # Compose config with overrides
        cfg = compose(config_name=config_name, overrides=trial_overrides)
        
        try:
            if use_torchrun:
                # Use torchrun to launch training with multi-GPU support
                # Use train_distributed.py which doesn't use Hydra's main decorator
                script_dir = os.path.dirname(os.path.abspath(__file__))
                train_script = os.path.join(script_dir, 'train_distributed.py')
                
                # Serialize config to JSON to pass to distributed script
                config_json = json.dumps(OmegaConf.to_container(cfg, resolve=True))
                
                # Build command - train_distributed.py reads config from env var
                # Use a unique master port per trial to avoid conflicts
                master_port = 29500 + (trial_id % 100)  # Port range 29500-29599
                cmd = [
                    'torchrun',
                    '--nproc_per_node', str(num_gpus),
                    '--nnodes', '1',
                    '--node_rank', '0',
                    '--master_addr', 'localhost',
                    '--master_port', str(master_port),
                    train_script
                ]
                
                # Set environment to mark that we're running under torchrun
                # Override CUDA_VISIBLE_DEVICES to use all GPUs for training
                env = os.environ.copy()
                env['USE_TORCHRUN'] = '1'
                env['TORCHRUN_NUM_GPUS'] = str(num_gpus)
                env['TRAIN_CONFIG_JSON'] = config_json  # Pass config via environment
                # Enable detailed error reporting for torchrun
                os.makedirs(cfg.paths.save_dir, exist_ok=True)
                env['TORCHELASTIC_ERROR_FILE'] = os.path.join(cfg.paths.save_dir, f'torchrun_error_rank_{{rank}}.log')
                # Set CUDA_VISIBLE_DEVICES in subprocess to use all GPUs for distributed training
                # This allows torchrun to access all GPUs even if parent process has limited visibility
                if num_gpus == 4:
                    env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
                else:
                    # For other numbers, use first N GPUs
                    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(num_gpus)))
                print(f"Launching torchrun with CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
                
                # Run training via torchrun
                try:
                    result = subprocess.run(
                        cmd,
                        env=env,
                        cwd=os.path.dirname(script_dir),  # Run from src/ directory
                        capture_output=False,  # Show output in real-time
                        check=False
                    )
                except KeyboardInterrupt:
                    # User interrupted - mark trial as interrupted
                    print(f"\nTrial {trial_id} interrupted by user")
                    trial.set_user_attr('status', 'interrupted')
                    raise  # Re-raise to let Optuna handle it properly
                
                if result.returncode != 0:
                    # Check if it was interrupted (SIGINT = -2 on Unix)
                    if result.returncode == -2 or result.returncode == 130:  # SIGINT
                        print(f"Trial {trial_id} was interrupted")
                        trial.set_user_attr('status', 'interrupted')
                        raise KeyboardInterrupt("Trial interrupted")
                    
                    # Check for error log files
                    error_log_pattern = os.path.join(cfg.paths.save_dir, 'torchrun_error_rank_*.log')
                    import glob
                    error_logs = glob.glob(error_log_pattern)
                    if error_logs:
                        print(f"\nTrial {trial_id} failed. Error logs found:")
                        for log_file in error_logs:
                            print(f"  - {log_file}")
                            try:
                                with open(log_file, 'r') as f:
                                    error_content = f.read()
                                    if error_content:
                                        print(f"    Error content:\n{error_content[:500]}")  # First 500 chars
                            except:
                                pass
                    
                    print(f"Trial {trial_id} failed with return code {result.returncode}")
                    print(f"Check logs in: {cfg.paths.save_dir}")
                    return float('inf')
                
                # Read the final loss from the file written by train.py
                loss_file = os.path.join(cfg.paths.save_dir, 'final_loss.txt')
                if os.path.exists(loss_file):
                    try:
                        with open(loss_file, 'r') as f:
                            final_loss = float(f.read().strip())
                        print(f"Trial {trial_id} completed with loss: {final_loss:.6f}")
                        return final_loss
                    except Exception as e:
                        print(f"Failed to read loss file {loss_file}: {e}")
                        return float('inf')
                else:
                    print(f"Loss file not found: {loss_file}")
                    return float('inf')
            else:
                # Direct call (single GPU or no torchrun)
                final_loss = train_with_config(cfg)
                return final_loss if final_loss is not None and final_loss != float('inf') else float('inf')
        except KeyboardInterrupt:
            # Re-raise KeyboardInterrupt to allow Optuna to handle it gracefully
            print(f"\nTrial {trial_id} interrupted by user")
            raise
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
    
    # Multi-GPU configuration
    parser.add_argument('--use-torchrun', action='store_true', default=True,
                       help='Use torchrun for multi-GPU training (default: True)')
    parser.add_argument('--num-gpus', type=int, default=4,
                       help='Number of GPUs to use for training (default: 4)')
    
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
    
    # Note: We use n_jobs=1 to ensure Optuna runs trials sequentially
    # This prevents Optuna from trying to use multiple GPUs for parallel trials
    # When launching torchrun, we'll set CUDA_VISIBLE_DEVICES in the subprocess
    # to use all 4 GPUs for distributed training
    
    print(f"\nStarting optimization: {args.n_trials} trials")
    print(f"Study: {args.study_name}")
    print(f"Database: {storage}")
    print(f"Config path: {config_path}")
    print(f"Config name: {args.config_name}")
    if base_overrides:
        print(f"Base overrides: {base_overrides}")
    print(f"Using torchrun: {args.use_torchrun}")
    if args.use_torchrun:
        print(f"Number of GPUs per trial: {args.num_gpus}")
        print("Note: Optuna runs trials sequentially (n_jobs=1)")
        print("      torchrun will use all GPUs for each trial")
    print()
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_overrides, config_path, args.config_name, 
                               args.use_torchrun, args.num_gpus),
        n_trials=args.n_trials,
        show_progress_bar=True,
        n_jobs=1,  # Run trials sequentially (one at a time)
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

