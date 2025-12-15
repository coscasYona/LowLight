# Image Quality Metrics Implementation

## Overview
This document describes the implementation of PSNR and SNR metrics for denoising evaluation across training, validation, and testing.

## Architecture

### 1. Metrics Module (`util/metrics.py`)
**Purpose**: Centralized metric calculations using torchmetrics

**Components**:
- `ImageQualityMetrics`: Class for computing PSNR and SNR metrics
- `log_metrics_to_tensorboard()`: Function to log metrics to TensorBoard
- `print_metrics_summary()`: Function to print metrics to console

**Metrics Computed**:
- **PSNR_Noisy_vs_Clean**: PSNR between noisy input and clean reference
- **PSNR_Denoised_vs_Clean**: PSNR between denoised output and clean reference
- **PSNR_Improvement**: Difference in dB (Denoised - Noisy)
- **SNR_Noisy_vs_Clean**: SNR of noisy input
- **SNR_Denoised_vs_Clean**: SNR of denoised output
- **SNR_Enhancement_dB**: SNR improvement in dB
- **SNR_Enhancement_Ratio**: Multiplicative improvement factor (e.g., 3.5x)

### 2. Logging Integration (`util/util.py`)

**Modified Functions**:

#### `log_validation_images()`
- Added parameter: `compute_metrics=True` (enabled by default for validation)
- Automatically computes and logs metrics when called
- Logs to TensorBoard under `Validation/` prefix

#### `log_training_images()`
- Added parameter: `compute_metrics=False` (disabled by default for training)
- Can optionally compute metrics if enabled
- Logs to TensorBoard under `Train/` prefix

### 3. Usage in Training Code

**No changes needed in training files!**

The training code (`stg2_emva1288_train.py`) already calls:
```python
util.log_validation_images(
    writer=writer,
    epoch=epoch,
    model=dn_model,
    image_data=val_batch_data,
    save_path=args.save_path
)
```

This automatically:
1. Generates denoised images
2. Computes all PSNR/SNR metrics
3. Logs to TensorBoard
4. Prints summary to console

## Example Output

### Console Output
```
[Validation] PSNR - Noisy: 25.34 dB | Denoised: 32.18 dB | Improvement: 6.84 dB
[Validation] SNR - Noisy: 18.21 dB | Denoised: 24.53 dB | Enhancement: 6.32 dB (4.27x)
```

### TensorBoard Metrics
Under the `Validation/` tab:
- PSNR_Noisy_vs_Clean
- PSNR_Denoised_vs_Clean
- PSNR_Improvement
- SNR_Noisy_vs_Clean
- SNR_Denoised_vs_Clean
- SNR_Enhancement_dB
- SNR_Enhancement_Ratio

## Benefits

1. **Separation of Concerns**: Metrics logic is separate from training/testing code
2. **Reusability**: Same metrics used across train/validation/test
3. **Professional**: Uses industry-standard torchmetrics library
4. **Maintainable**: Changes to metrics only need to happen in one place
5. **Configurable**: Can enable/disable metrics per context

## Future Extensions

### For Test Files
To add metrics to test files (e.g., `stg2_denoise_test_SID.py`), you can:

1. **Option A**: Use the metrics module directly
```python
from util.metrics import ImageQualityMetrics, log_metrics_to_tensorboard, print_metrics_summary

metrics_calculator = ImageQualityMetrics(device='cuda')
metrics = metrics_calculator.compute_all_metrics(clean, noisy, denoised)
log_metrics_to_tensorboard(writer, epoch, metrics, prefix='Test_SID')
print_metrics_summary(metrics, prefix='Test_SID')
```

2. **Option B**: Modify test files to store images and call logging functions
```python
# After generating denoised images in test loop
test_image_data = {
    'img_gt': clean,
    'noisy_state': noisy,
    # ... other required fields
}
util.log_validation_images(writer, epoch, model, test_image_data, prefix='Test')
```

## Dependencies

- `torchmetrics`: For PSNR and SNR calculations
- Installed via: `pip install torchmetrics`
