#!/usr/bin/env python3
"""
Optuna hyperparameter optimization wrapper for EMVA 1288 training.
Wraps the Hydra-based train.py script for hyperparameter sweeps.

Uses torchrun to launch each trial with multi-GPU support.
"""

import os
import sys
import json
import argparse
import subprocess
import re
import socket
import signal
import optuna
from optuna.trial import TrialState

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Hydra utilities
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import pytorch_lightning as pl


def _find_free_port():
    """Find a free port for torchrun master."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _kill_process_group(process):
    """Kill the entire process group to ensure all GPU workers are terminated."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass  # Process already dead
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # Force kill if SIGTERM didn't work
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        process.wait()


def _run_training_with_torchrun(trial_id, cfg, config_path, config_name, trial_overrides):
    """
    Run training as a subprocess using torchrun for multi-GPU support.
    Returns the final validation loss.
    """
    # Find script path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, 'train.py')
    
    # Get number of GPUs from config
    n_gpus = cfg.hardware.devices if hasattr(cfg.hardware, 'devices') else 4
    
    # Find a free port for this trial
    master_port = _find_free_port()
    
    # Build torchrun command
    cmd = [
        'torchrun',
        f'--nproc_per_node={n_gpus}',
        f'--master_port={master_port}',
        '--standalone',
        train_script,
        f'--config-path={config_path}',
        f'--config-name={config_name}',
    ]
    
    # Add all overrides
    for override in trial_overrides:
        cmd.append(override)
    
    print(f"\nTrial {trial_id}: Running with torchrun on {n_gpus} GPUs...")
    # Print key hyperparameters being passed
    param_summary = [o for o in trial_overrides if any(k in o for k in ['learning_rate', 'batch_size', 'base_channels', 'num_steps'])]
    print(f"  Params: {param_summary}")
    
    # Run subprocess with real-time output streaming
    # Use start_new_session=True to create a new process group for proper cleanup
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            cwd=os.path.dirname(os.path.dirname(script_dir)),  # Run from project root
            start_new_session=True,  # Create new process group for clean termination
        )
        
        # Collect output while streaming to console
        output_lines = []
        for line in process.stdout:
            print(line, end='')  # Print in real-time
            output_lines.append(line)
        
        # Wait for process to complete
        return_code = process.wait()
        stdout = ''.join(output_lines)
        
        # Check return code
        if return_code != 0:
            print(f"Trial {trial_id} failed with return code {return_code}")
            return 0.0  # Return 0 PSNR for failed trials (we're maximizing)
        
        # Try to find final PSNR in output (from validation metrics)
        # Look for pattern like "[Validation] Epoch X: PSNR=XX.XX, SSIM=X.XXXX"
        psnr_matches = re.findall(r'\[Validation\] Epoch \d+: PSNR=([\d.]+)', stdout)
        if psnr_matches:
            final_psnr = float(psnr_matches[-1])  # Take the last (best) epoch's PSNR
            print(f"Trial {trial_id}: Final PSNR = {final_psnr:.2f} dB")
            return final_psnr
        
        # Alternative: look for any PSNR value
        psnr_matches = re.findall(r'PSNR[=:\s]+([\d.]+)', stdout)
        if psnr_matches:
            final_psnr = float(psnr_matches[-1])
            print(f"Trial {trial_id}: Final PSNR (alt) = {final_psnr:.2f} dB")
            return final_psnr
        
        print(f"Trial {trial_id}: Could not parse PSNR from output")
        return 0.0  # Return 0 PSNR for unparseable trials
        
    except KeyboardInterrupt:
        print(f"\nTrial {trial_id}: Interrupted by user, killing all GPU processes...")
        # Kill the entire process group (all torchrun workers)
        _kill_process_group(process)
        raise
    except Exception as e:
        print(f"Trial {trial_id}: Error running subprocess: {e}")
        _kill_process_group(process)
        return 0.0  # Return 0 PSNR for failed trials


def objective(trial, base_config_overrides, config_path, config_name, study_name):
    """Optuna objective function - wraps training with suggested hyperparameters."""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 1, 3, step=1)  # Limited for GPU memory
    base_channels = trial.suggest_categorical('base_channels', [32, 64, 128])
    
    # Channel multipliers as string (Optuna doesn't support lists in categorical)
    channel_mults_options = ['1,2', '1,2,4', '1,2,4,8']
    channel_mults_str = trial.suggest_categorical('channel_mults', channel_mults_options)
    # Convert to list for display and Hydra
    channel_mults = [int(x) for x in channel_mults_str.split(',')]
    
    # More diffusion steps = better quality but slower inference
    # DDIM can work well with 20-100 steps, DDPM needs more (100-1000)
    num_steps = trial.suggest_int('num_steps', 20, 100)
    patch_size = 512  # Fixed patch size for consistency
    attn_type = trial.suggest_categorical('attn_type', ['linear', 'channel'])
    scheduler = trial.suggest_categorical('scheduler', ['ddpm', 'ddim'])
    
    # Loss weights - search around legacy values that worked (0.8, 0.1)
    l1_weight = trial.suggest_float('l1_weight', 0.7, 0.95)  # Legacy: 0.8
    gradient_weight = trial.suggest_float('gradient_weight', 0.05, 0.15)  # Legacy: 0.1
    
    # Hybrid x0 loss: DISABLED - causes training instability
    # Using standard HybridDiffusionLoss which matches legacy behavior
    use_hybrid_x0_loss = False
    ssim_weight = 0.3  # Not used when hybrid loss is disabled
    
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
        f'training.use_hybrid_x0_loss={use_hybrid_x0_loss}',
        f'training.ssim_weight={ssim_weight}',
    ])
    
    # Unique save path for this trial - organized by study name
    # Structure: checkpoints/{study_name}/trial_{trial_id}/
    trial_id = trial.number
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    trial_overrides.append(f'paths.save_dir={project_root}/checkpoints/{study_name}/trial_{trial_id}')
    trial_overrides.append(f'paths.log_dir={project_root}/logs/{study_name}/trial_{trial_id}')
    
    # Hardware configuration - use all availabl e GPUs
    trial_overrides.extend([
        'hardware.accelerator=gpu',
        'hardware.devices=4',  # Multi-GPU training per trial
    ])
    
    # Prevent Hydra from changing working directory (keeps paths consistent)
    trial_overrides.append(f'hydra.run.dir={project_root}')
    
    # Reduce epochs for faster sweeps (can be overridden via base_overrides)
    has_epochs_override = any('training.epochs=' in override for override in trial_overrides)
    if not has_epochs_override:
        trial_overrides.append('training.epochs=300')  # Default for sweeps
    
    print(f"\nTrial {trial_id}: lr={learning_rate:.6f}, bs={batch_size}, "
          f"ch={base_channels}, mults={channel_mults}, steps={num_steps}, "
          f"attn={attn_type}, sched={scheduler}")
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra with config directory to get base config
    with initialize_config_dir(config_dir=config_path, version_base=None):
        # Compose config with overrides (for _run_training_with_torchrun to read)
        cfg = compose(config_name=config_name, overrides=trial_overrides)
        
        try:
            # Run training via torchrun subprocess for multi-GPU support
            final_psnr = _run_training_with_torchrun(
                trial_id, cfg, config_path, config_name, trial_overrides
            )
            return final_psnr if final_psnr is not None and final_psnr > 0 else 0.0
        except Exception as e:
            print(f"Trial {trial_id} failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0  # Return 0 PSNR for failed trials
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
    
    # Check if database file exists
    db_path = storage.replace('sqlite:///', '')
    if os.path.exists(db_path):
        print(f"Database file exists at {db_path}, attempting to load existing study...")
    
    try:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction='maximize',  # Maximize PSNR
            pruner=optuna.pruners.MedianPruner() if args.pruning else None,
        )
        print(f"Created new study: {args.study_name}")
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(study_name=args.study_name, storage=storage)
        completed_trials = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        best_trial = study.best_trial if completed_trials > 0 else None
        print(f"✓ Loaded existing study: {args.study_name}")
        print(f"  Current number of trials: {len(study.trials)}")
        if best_trial:
            print(f"  Best trial so far: {best_trial.number} with PSNR: {best_trial.value:.2f} dB")
    
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
        lambda trial: objective(trial, base_overrides, config_path, args.config_name, args.study_name),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Best trial (maximizing PSNR):")
    trial = study.best_trial
    print(f"  PSNR: {trial.value:.2f} dB")
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
            'best_psnr_db': trial.value,
            'objective': 'maximize_psnr',
            'best_params': trial.params,
            'study_name': args.study_name,
            'storage': storage,
            'n_trials': args.n_trials,
            'best_trial_number': trial.number
        }, f, indent=2)
    
    print(f"\nBest parameters saved to: {best_params_path}")
    
    # Database is the primary storage - all results are in SQLite
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
