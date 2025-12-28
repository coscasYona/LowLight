# Optuna Database Logging

## Yes, Everything is Logged to a Database!

Optuna automatically logs **everything** to a SQLite database by default. Here's what gets stored:

## Database Location

By default, the database is saved as:
```
{study_name}.db
```

For example: `denoise_optimization.db`

You can specify a custom location with `--storage sqlite:///path/to/optuna.db`

## What Gets Logged

### 1. **Trial Parameters** (Hyperparameters)
All suggested hyperparameters for each trial:
- `learning_rate`
- `batch_size`
- `sd_base_channels`
- `sd_channel_mults`
- `sd_num_steps`
- `sd_time_embed_dim`
- `sd_cond_dim`
- `patch_size`
- `sd_attn_type`
- `sd_scheduler`

### 2. **Trial Results** (Objective Values)
- Final objective value (loss/metric)
- Trial state (COMPLETE, PRUNED, FAIL)
- Trial number
- Start/end timestamps

### 3. **Training Metrics** (User Attributes)
Comprehensive metrics stored for each trial:
- `final_loss` - Final training loss
- `min_loss` - Minimum loss during training
- `max_loss` - Maximum loss during training
- `loss_history` - Complete loss history (all epochs)
- `loss_trend` - Whether loss is decreasing/increasing
- `loss_std` - Standard deviation of losses
- `loss_improvement` - Total improvement from first to last epoch
- `num_epochs` - Number of epochs trained
- `batch_size` - Batch size used
- `learning_rate` - Learning rate used
- `sd_base_channels` - Base channels
- `sd_channel_mults` - Channel multipliers
- `sd_num_steps` - Diffusion steps
- `sd_time_embed_dim` - Time embedding dimension
- `sd_cond_dim` - Condition embedding dimension
- `patch_size` - Patch size
- `attn_type` - Attention type
- `scheduler` - Scheduler type
- `save_path` - Where trial checkpoints are saved
- `trial_id` - Trial identifier

### 4. **Intermediate Values** (For Pruning)
- Loss at each epoch (for pruning decisions)
- Step number

## Database Schema

The SQLite database contains several tables:
- `studies` - Study metadata
- `trials` - Trial information
- `trial_params` - Hyperparameter values
- `trial_values` - Objective values
- `trial_user_attrs` - Custom attributes (all our metrics)
- `trial_intermediate_values` - Intermediate results

## Accessing the Database

### Using Optuna API

```python
import optuna

# Load study
study = optuna.load_study(
    study_name='denoise_optimization',
    storage='sqlite:///denoise_optimization.db'
)

# Get all trials
for trial in study.trials:
    print(f"Trial {trial.number}:")
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
    print(f"  Attributes: {trial.user_attrs}")
    print(f"  State: {trial.state}")
```

### Using SQLite Directly

```bash
sqlite3 denoise_optimization.db

# View all trials
SELECT trial_id, number, value, state FROM trials;

# View trial parameters
SELECT * FROM trial_params WHERE trial_id = 0;

# View custom attributes (metrics)
SELECT * FROM trial_user_attrs WHERE trial_id = 0;

# View intermediate values
SELECT * FROM trial_intermediate_values WHERE trial_id = 0;
```

## Benefits

1. **Persistent Storage** - All results saved automatically
2. **Resume Optimization** - Continue from where you left off
3. **Analysis** - Query and analyze all trials
4. **Reproducibility** - Complete record of all experiments
5. **No Data Loss** - Even if script crashes, data is saved

## Example Queries

### Find Best Trial
```sql
SELECT trial_id, value FROM trials 
WHERE state = 'COMPLETE' 
ORDER BY value ASC 
LIMIT 1;
```

### Get All Loss Histories
```sql
SELECT trial_id, key, value FROM trial_user_attrs 
WHERE key = 'loss_history';
```

### Find Trials with Decreasing Loss
```sql
SELECT trial_id FROM trial_user_attrs 
WHERE key = 'loss_trend' AND value = 'decreasing';
```

## Summary

**Yes, everything is logged to the database automatically!** You don't need to do anything special - Optuna handles it all. The database contains:

✅ All hyperparameters for every trial
✅ All objective values (losses/metrics)
✅ Complete training metrics (loss history, trends, etc.)
✅ Trial states and timestamps
✅ Intermediate values for pruning

This makes it easy to:
- Resume optimization
- Analyze results
- Compare trials
- Find best configurations
- Reproduce experiments

