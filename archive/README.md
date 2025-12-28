# Archive - Old/Unused Files

This folder contains old and unused files from before the codebase refactoring. All code has been moved to the `src/` folder.

## Contents

### Old Python Scripts
- `dataset_loader_sid.py` - Old SID dataset loader
- `dataset_loader.py` - Old dataset loader
- `dataset_preparation.py` - Old dataset preparation script
- `stg2_denoise_*.py` - Old training/test scripts
- `stg2_emva1288_train.py.legacy` - Legacy training script
- `migrate_optuna_study.py` - Optuna migration script

### Old Directories
- `net/` - Old network/model definitions
- `util/` - Old utility functions
- `utils/` - Old utility functions (duplicate)
- `data_process/` - Old data processing code
- `databases/` - Old Optuna databases (if moved)
- `dataset/` - Old dataset code

### Old Output/Checkpoint Folders
- `runs/` - Old training runs/outputs
- `denoise_last_ckpt/` - Old checkpoint folder

### Old Scripts
- `docker-run.sh` - Old Docker run script
- `run.sh` - Old run script

### Old Documentation
- `ARCHITECTURE_IMPROVEMENTS.md`
- `DOCKER.md`
- `EMVA1288_TRAINING_CLI.md`
- `EMVA1288_TRAINING.md`
- `METRICS_IMPLEMENTATION.md`
- `OPTUNA_*.md` - Various Optuna documentation files
- `TRAINING_COMMANDS.md`

### Test/Data Files
- `templet.ARW` - Test RAW image file
- `test_epoch_psnr_*.mat` - Old test result files

## Note

These files are kept for reference and will be deleted in future commits once training is confirmed to be working well with the new refactored codebase.

