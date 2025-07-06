# Dataset Accuracy Evaluation for B-spline Fitting

## Overview
When working with multiple datasets and B-spline equations, choosing the most accurate dataset is crucial for reliable curve fitting and prediction. Here are proven methods to evaluate dataset accuracy.

## 1. Statistical Error Metrics

### Mean Squared Error (MSE)
```python
import numpy as np
from scipy.interpolate import BSpline

def calculate_mse(original_data, fitted_spline, x_points):
    """Calculate MSE between original data and fitted B-spline"""
    predicted = fitted_spline(x_points)
    return np.mean((original_data - predicted) ** 2)
```

### Root Mean Squared Error (RMSE)
```python
def calculate_rmse(original_data, fitted_spline, x_points):
    """Calculate RMSE - more interpretable than MSE"""
    mse = calculate_mse(original_data, fitted_spline, x_points)
    return np.sqrt(mse)
```

### Mean Absolute Error (MAE)
```python
def calculate_mae(original_data, fitted_spline, x_points):
    """Calculate MAE - robust to outliers"""
    predicted = fitted_spline(x_points)
    return np.mean(np.abs(original_data - predicted))
```

### R-squared (Coefficient of Determination)
```python
def calculate_r_squared(original_data, fitted_spline, x_points):
    """Calculate R² - proportion of variance explained"""
    predicted = fitted_spline(x_points)
    ss_res = np.sum((original_data - predicted) ** 2)
    ss_tot = np.sum((original_data - np.mean(original_data)) ** 2)
    return 1 - (ss_res / ss_tot)
```

## 2. Cross-Validation Methods

### K-Fold Cross-Validation
```python
from sklearn.model_selection import KFold

def cross_validate_spline(data_x, data_y, k_folds=5, spline_degree=3):
    """Perform k-fold cross-validation for B-spline fitting"""
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    errors = []
    
    for train_idx, test_idx in kf.split(data_x):
        # Split data
        x_train, x_test = data_x[train_idx], data_x[test_idx]
        y_train, y_test = data_y[train_idx], data_y[test_idx]
        
        # Fit B-spline on training data
        from scipy.interpolate import splrep, splev
        tck = splrep(x_train, y_train, s=0, k=spline_degree)
        
        # Predict on test data
        y_pred = splev(x_test, tck)
        
        # Calculate error
        error = np.mean((y_test - y_pred) ** 2)
        errors.append(error)
    
    return np.mean(errors), np.std(errors)
```

### Leave-One-Out Cross-Validation
```python
from sklearn.model_selection import LeaveOneOut

def loo_cross_validate_spline(data_x, data_y, spline_degree=3):
    """Perform leave-one-out cross-validation"""
    loo = LeaveOneOut()
    errors = []
    
    for train_idx, test_idx in loo.split(data_x):
        x_train, x_test = data_x[train_idx], data_x[test_idx]
        y_train, y_test = data_y[train_idx], data_y[test_idx]
        
        # Fit and predict
        from scipy.interpolate import splrep, splev
        tck = splrep(x_train, y_train, s=0, k=spline_degree)
        y_pred = splev(x_test, tck)
        
        error = (y_test - y_pred) ** 2
        errors.extend(error)
    
    return np.mean(errors)
```

## 3. Information Criteria

### Akaike Information Criterion (AIC)
```python
def calculate_aic(data_y, fitted_spline, x_points, num_coefficients):
    """Calculate AIC - balances fit quality and model complexity"""
    n = len(data_y)
    predicted = fitted_spline(x_points)
    mse = np.mean((data_y - predicted) ** 2)
    
    # AIC = 2k - 2ln(L), where L is likelihood
    # For normal distribution: AIC = n*ln(MSE) + 2k
    aic = n * np.log(mse) + 2 * num_coefficients
    return aic
```

### Bayesian Information Criterion (BIC)
```python
def calculate_bic(data_y, fitted_spline, x_points, num_coefficients):
    """Calculate BIC - stronger penalty for model complexity"""
    n = len(data_y)
    predicted = fitted_spline(x_points)
    mse = np.mean((data_y - predicted) ** 2)
    
    # BIC = ln(n)*k - 2ln(L)
    # For normal distribution: BIC = n*ln(MSE) + ln(n)*k
    bic = n * np.log(mse) + np.log(n) * num_coefficients
    return bic
```

## 4. Residual Analysis

### Residual Distribution Analysis
```python
import matplotlib.pyplot as plt
from scipy import stats

def analyze_residuals(original_data, fitted_spline, x_points):
    """Analyze residuals for model quality assessment"""
    predicted = fitted_spline(x_points)
    residuals = original_data - predicted
    
    # Check normality of residuals
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    
    # Check for patterns in residuals
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(predicted, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=20, density=True, alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Residual Distribution')
    
    plt.subplot(1, 3, 3)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals)
    }
```

## 5. Dataset Comparison Framework

### Comprehensive Dataset Evaluation
```python
def evaluate_dataset_accuracy(datasets, dataset_names, spline_degree=3):
    """
    Comprehensive evaluation of multiple datasets for B-spline fitting
    
    Parameters:
    - datasets: list of tuples [(x1, y1), (x2, y2), ...]
    - dataset_names: list of dataset names
    - spline_degree: degree of B-spline
    
    Returns:
    - DataFrame with comparison metrics
    """
    import pandas as pd
    from scipy.interpolate import splrep, BSpline
    
    results = []
    
    for i, (data_x, data_y) in enumerate(datasets):
        # Fit B-spline
        tck = splrep(data_x, data_y, s=0, k=spline_degree)
        spline = BSpline(*tck)
        
        # Calculate metrics
        mse = calculate_mse(data_y, spline, data_x)
        rmse = calculate_rmse(data_y, spline, data_x)
        mae = calculate_mae(data_y, spline, data_x)
        r2 = calculate_r_squared(data_y, spline, data_x)
        
        # Cross-validation
        cv_error, cv_std = cross_validate_spline(data_x, data_y, k_folds=5, spline_degree=spline_degree)
        
        # Information criteria
        num_coefficients = len(tck[1])  # Number of B-spline coefficients
        aic = calculate_aic(data_y, spline, data_x, num_coefficients)
        bic = calculate_bic(data_y, spline, data_x, num_coefficients)
        
        # Residual analysis
        residual_stats = analyze_residuals(data_y, spline, data_x)
        
        results.append({
            'Dataset': dataset_names[i],
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'CV_Error': cv_error,
            'CV_Std': cv_std,
            'AIC': aic,
            'BIC': bic,
            'Num_Coefficients': num_coefficients,
            'Residual_Normality_p': residual_stats['shapiro_p'],
            'Data_Points': len(data_x)
        })
    
    return pd.DataFrame(results)
```

## 6. Decision Guidelines

### Ranking Criteria
1. **Primary Metrics** (in order of importance):
   - **Cross-validation error**: Most reliable for generalization
   - **RMSE**: Interpretable and penalizes large errors
   - **R²**: Proportion of variance explained

2. **Secondary Metrics**:
   - **AIC/BIC**: Balance between fit and complexity
   - **MAE**: Robust to outliers
   - **Residual analysis**: Check model assumptions

### Selection Strategy
```python
def select_best_dataset(evaluation_results):
    """
    Select best dataset based on multiple criteria
    
    Parameters:
    - evaluation_results: DataFrame from evaluate_dataset_accuracy
    
    Returns:
    - Best dataset name and ranking explanation
    """
    # Normalize metrics (lower is better for error metrics)
    normalized_df = evaluation_results.copy()
    
    # For error metrics (lower is better)
    error_metrics = ['MSE', 'RMSE', 'MAE', 'CV_Error', 'AIC', 'BIC']
    for metric in error_metrics:
        if metric in normalized_df.columns:
            normalized_df[f'{metric}_rank'] = normalized_df[metric].rank(ascending=True)
    
    # For goodness metrics (higher is better)
    goodness_metrics = ['R²']
    for metric in goodness_metrics:
        if metric in normalized_df.columns:
            normalized_df[f'{metric}_rank'] = normalized_df[metric].rank(ascending=False)
    
    # Calculate weighted score
    weights = {
        'CV_Error_rank': 0.3,    # Highest weight
        'RMSE_rank': 0.2,
        'R²_rank': 0.2,
        'AIC_rank': 0.15,
        'BIC_rank': 0.15
    }
    
    normalized_df['Weighted_Score'] = 0
    for metric, weight in weights.items():
        if metric in normalized_df.columns:
            normalized_df['Weighted_Score'] += weight * normalized_df[metric]
    
    # Best dataset has lowest weighted score
    best_idx = normalized_df['Weighted_Score'].idxmin()
    best_dataset = normalized_df.loc[best_idx, 'Dataset']
    
    return best_dataset, normalized_df.sort_values('Weighted_Score')
```

## 7. Practical Implementation Example

```python
# Example usage
def main():
    # Assume you have multiple datasets
    datasets = [
        (x1_data, y1_data),  # Dataset 1
        (x2_data, y2_data),  # Dataset 2
        (x3_data, y3_data),  # Dataset 3
    ]
    
    dataset_names = ['Dataset_A', 'Dataset_B', 'Dataset_C']
    
    # Evaluate all datasets
    results = evaluate_dataset_accuracy(datasets, dataset_names)
    
    # Select best dataset
    best_dataset, ranked_results = select_best_dataset(results)
    
    print(f"Best dataset: {best_dataset}")
    print("\nRanked results:")
    print(ranked_results[['Dataset', 'CV_Error', 'RMSE', 'R²', 'AIC', 'Weighted_Score']])
    
    return best_dataset, results

if __name__ == "__main__":
    best_dataset, evaluation_results = main()
```

## 8. Key Recommendations

1. **Always use cross-validation** - it's the most reliable method for assessing generalization
2. **Consider multiple metrics** - don't rely on a single measure
3. **Check residuals** - ensure model assumptions are met
4. **Balance complexity** - use AIC/BIC to avoid overfitting
5. **Domain knowledge** - incorporate subject-matter expertise
6. **Data quality** - check for outliers, missing values, and noise
7. **Sufficient sample size** - ensure adequate data for reliable B-spline fitting

The best dataset will typically have:
- Low cross-validation error
- Good R² value (>0.8 for strong relationships)
- Normally distributed residuals
- Reasonable balance between fit quality and model complexity