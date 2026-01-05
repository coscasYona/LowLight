#!/usr/bin/env python3
"""
Distributed training entry point for torchrun.
This script bypasses Hydra's main decorator to work with torchrun.
"""

import os
import sys
import json

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from train import train_with_config


def main():
    """Main entry point for distributed training via torchrun."""
    
    # Get rank info for logging
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    
    try:
        # Check if config is passed via environment variable (from Optuna)
        config_json = os.environ.get('TRAIN_CONFIG_JSON', None)
        
        if config_json:
            # Load config from JSON (set by Optuna)
            config_dict = json.loads(config_json)
            cfg = OmegaConf.create(config_dict)
        else:
            # Fallback: try to get config from command line args
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument('--config-json', type=str, help='Config as JSON string')
            parser.add_argument('--config-file', type=str, help='Path to config file')
            args, remaining = parser.parse_known_args()
            
            if args.config_json:
                config_dict = json.loads(args.config_json)
                cfg = OmegaConf.create(config_dict)
            elif args.config_file:
                cfg = OmegaConf.load(args.config_file)
            else:
                # Try to use Hydra with command line overrides
                script_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(script_dir, '..', 'configs')
                config_path = os.path.abspath(config_path)
                
                # Parse remaining args as Hydra overrides
                overrides = []
                for arg in remaining:
                    if '=' in arg:
                        overrides.append(arg)
                
                GlobalHydra.instance().clear()
                with initialize_config_dir(config_dir=config_path, version_base=None):
                    cfg = compose(config_name='train', overrides=overrides)
        
        # Run training
        final_loss = train_with_config(cfg)
        return final_loss if final_loss is not None and final_loss != float('inf') else float('inf')
    except Exception as e:
        # Print error with full traceback for debugging
        print(f"\n{'='*60}")
        print(f"ERROR in train_distributed.py (rank {local_rank}/{world_size}):")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        # Re-raise to let torchrun handle it
        raise
    finally:
        GlobalHydra.instance().clear()


if __name__ == "__main__":
    main()
