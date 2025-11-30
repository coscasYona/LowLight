# Optuna Databases

This directory contains all Optuna optimization study databases organized by type.

## Directory Structure

```
databases/
└── optuna/
    ├── production/    # Production optimization studies
    └── test/          # Test/experimental studies
```

## Production Databases

- `denoise_optimization.db` - Main optimization study
- `denoise_optimization_v2.db` - Version 2 optimization study
- `denoise_optuna.db` - General Optuna study

## Test Databases

- `test_*.db` - Various test and experimental studies

## Usage

When running Optuna optimization, specify the database path:

```bash
# For production studies
python stg2_denoise_optuna.py --study_name denoise_optimization \
  --storage sqlite:///databases/optuna/production/denoise_optimization.db

# For test studies
python stg2_denoise_optuna.py --study_name test_study \
  --storage sqlite:///databases/optuna/test/test_study.db
```

## Accessing Databases

### Using Optuna CLI

```bash
# View study
optuna study optimize databases/optuna/production/denoise_optimization.db

# Dashboard
optuna dashboard --storage sqlite:///databases/optuna/production/denoise_optimization.db
```

### Using Python

```python
import optuna

study = optuna.load_study(
    study_name='denoise_optimization',
    storage='sqlite:///databases/optuna/production/denoise_optimization.db'
)
```

## Notes

- Production databases are tracked in git (if not too large)
- Test databases are ignored by git (see .gitignore)
- All databases use SQLite format
- Databases can be safely moved/copied as they are self-contained

