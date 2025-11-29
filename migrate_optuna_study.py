#!/usr/bin/env python3
"""
Helper script to migrate or clean old Optuna studies with incompatible parameter formats.

Old studies used list/tuple format for 'sd_channel_mults' (e.g., [1, 2, 4]),
but the new code requires string format (e.g., '1,2,4').
"""

import sys
import optuna
import argparse


def check_study_compatibility(study_name, storage):
    """Check if a study has incompatible parameter formats."""
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        has_incompatible = False
        
        for trial in study.trials:
            if trial.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]:
                if 'sd_channel_mults' in trial.params:
                    val = trial.params['sd_channel_mults']
                    if isinstance(val, (list, tuple)):
                        has_incompatible = True
                        print(f"  Trial {trial.number}: incompatible format - {val} (type: {type(val).__name__})")
                        break
        
        return has_incompatible, study
    except Exception as e:
        print(f"Error loading study: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description='Check or clean Optuna studies with incompatible formats')
    parser.add_argument('--study_name', type=str, default='denoise_optimization',
                       help='Name of the study to check')
    parser.add_argument('--storage', type=str, default=None,
                       help='Storage URL (default: sqlite:///{study_name}.db)')
    parser.add_argument('--delete', action='store_true',
                       help='Delete the study if incompatible (use with caution!)')
    parser.add_argument('--list', action='store_true',
                       help='List all studies in the database')
    
    args = parser.parse_args()
    
    if args.storage:
        storage = args.storage
    else:
        storage = f"sqlite:///{args.study_name}.db"
    
    if args.list:
        # List all studies
        try:
            import sqlite3
            db_path = storage.replace('sqlite:///', '')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT study_name FROM studies")
            studies = cursor.fetchall()
            conn.close()
            
            if studies:
                print(f"Studies in {db_path}:")
                for (name,) in studies:
                    print(f"  - {name}")
            else:
                print(f"No studies found in {db_path}")
        except Exception as e:
            print(f"Error listing studies: {e}")
        return
    
    print(f"Checking study: {args.study_name}")
    print(f"Storage: {storage}\n")
    
    has_incompatible, study = check_study_compatibility(args.study_name, storage)
    
    if study is None:
        print("Could not load study.")
        return
    
    if has_incompatible:
        print(f"\n❌ Study '{args.study_name}' has incompatible parameter format!")
        print(f"   Found {len(study.trials)} trials with old list/tuple format.")
        print(f"\n   To fix:")
        print(f"   1. Use a different study name in your optimization script")
        print(f"   2. Delete this study: python migrate_optuna_study.py --study_name {args.study_name} --delete")
        print(f"   3. Or manually delete: rm {storage.replace('sqlite:///', '')}")
        
        if args.delete:
            import os
            db_path = storage.replace('sqlite:///', '')
            if os.path.exists(db_path):
                os.remove(db_path)
                print(f"\n✓ Deleted database: {db_path}")
            else:
                print(f"\n✗ Database file not found: {db_path}")
    else:
        print(f"✓ Study '{args.study_name}' is compatible!")
        print(f"   Found {len(study.trials)} trials")


if __name__ == '__main__':
    main()

