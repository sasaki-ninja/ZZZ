# B-spline Dataset Evaluation Guide

## Quick Answer to Your Question

When you have **multiple datasets and B-spline coefficients**, choose the most accurate dataset using these methods in order of priority:

1. **Cross-validation error** (most important)
2. **RMSE** (interpretable error metric)
3. **R²** (variance explained)
4. **AIC/BIC** (balance fit vs complexity)
5. **Residual analysis** (check assumptions)

## Files in This Guide

1. **`dataset_accuracy_evaluation.md`** - Complete theoretical guide and methods
2. **`dataset_evaluation_example.py`** - Practical implementation example
3. **`requirements_evaluation.txt`** - Required Python packages
4. **`B-spline_Dataset_Evaluation_Guide.md`** - This summary file

## Quick Start

### Step 1: Install Dependencies
```bash
pip install numpy pandas matplotlib scipy scikit-learn
# or
pip install -r requirements_evaluation.txt
```

### Step 2: Run the Example
```bash
python dataset_evaluation_example.py
```

### Step 3: Adapt to Your Data
Replace the sample data generation with your actual datasets:

```python
# Your actual datasets
datasets = [
    (x1_data, y1_data),  # Dataset 1
    (x2_data, y2_data),  # Dataset 2
    (x3_data, y3_data),  # Dataset 3
]

dataset_names = ['Your_Dataset_1', 'Your_Dataset_2', 'Your_Dataset_3']

# Evaluate and select best
results = evaluate_dataset_accuracy(datasets, dataset_names)
best_dataset, ranked_results = select_best_dataset(results)
```

## Key Evaluation Metrics

| Metric | Purpose | Best Value |
|--------|---------|------------|
| **CV Error** | Generalization performance | Lower is better |
| **RMSE** | Interpretable error | Lower is better |
| **R²** | Variance explained | Higher is better (>0.8 is good) |
| **AIC/BIC** | Model complexity balance | Lower is better |
| **Residual p-value** | Model assumptions | >0.05 is good |

## Decision Framework

The algorithm ranks datasets using weighted scores:
- Cross-validation error: 30% weight
- RMSE: 20% weight  
- R²: 20% weight
- AIC: 15% weight
- BIC: 15% weight

## Common Scenarios

### Scenario 1: Clean vs Noisy Data
- **Clean data** typically wins with low CV error and high R²
- **Noisy data** has high RMSE and poor residual distribution

### Scenario 2: Large vs Small Datasets
- **Larger datasets** generally more reliable
- **Smaller datasets** can be good if high quality (low noise)

### Scenario 3: Regular vs Sparse Sampling
- **Regular sampling** usually better for B-splines
- **Sparse sampling** acceptable if points well-distributed

## Interpreting Results

### Good Dataset Characteristics:
- CV error < 0.1 (depends on data scale)
- R² > 0.8
- Normal residual distribution (p > 0.05)
- AIC/BIC reasonable relative to other datasets

### Red Flags:
- Very high CV error
- R² < 0.5
- Non-normal residuals (p < 0.05)
- Obvious outliers or patterns in residual plots

## Customization Options

### Adjust Evaluation Criteria
Modify the weights in `select_best_dataset()`:
```python
weights = {
    'CV_Error_rank': 0.4,    # Increase if generalization is critical
    'RMSE_rank': 0.3,        # Increase if interpretability matters
    'R²_rank': 0.2,          # Standard weight
    'AIC_rank': 0.05,        # Decrease if complexity less important
    'BIC_rank': 0.05         # Decrease if complexity less important
}
```

### Change B-spline Parameters
Modify spline degree and smoothing:
```python
# In evaluate_dataset_accuracy()
spline_degree = 3  # Change to 1 (linear), 2 (quadratic), etc.
tck = splrep(data_x, data_y, s=0.1, k=spline_degree)  # Add smoothing
```

### Add Custom Metrics
Extend the evaluation with domain-specific metrics:
```python
def custom_metric(original_data, fitted_spline, x_points):
    # Your custom evaluation logic
    return metric_value
```

## For Weather/Climate Data

Since your project appears to involve weather prediction:

1. **Temporal considerations**: Use time-series cross-validation
2. **Spatial considerations**: Check spatial autocorrelation
3. **Physical constraints**: Ensure B-splines respect physical limits
4. **Seasonal patterns**: Consider seasonal decomposition

## Troubleshooting

### Import Errors
```bash
# Install missing packages
pip install numpy pandas matplotlib scipy scikit-learn
```

### Memory Issues with Large Datasets
```python
# Use sampling for evaluation
sample_size = min(10000, len(data_x))
idx = np.random.choice(len(data_x), sample_size, replace=False)
data_x_sample = data_x[idx]
data_y_sample = data_y[idx]
```

### Convergence Issues
```python
# Add smoothing to B-spline fitting
tck = splrep(data_x, data_y, s=0.1, k=spline_degree)  # s > 0 for smoothing
```

## Next Steps

1. **Run the example** to understand the workflow
2. **Adapt to your data** by replacing sample datasets
3. **Customize weights** based on your priorities
4. **Add domain-specific metrics** if needed
5. **Validate results** by comparing with known good datasets

The best dataset will be the one that balances low prediction error with good generalization performance while maintaining reasonable model complexity.