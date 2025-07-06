#!/usr/bin/env python3
"""
Example script demonstrating how to evaluate multiple datasets for B-spline accuracy.
This script shows how to apply the methods from dataset_accuracy_evaluation.md

Installation requirements:
pip install numpy pandas matplotlib scipy scikit-learn

Or install from the requirements file:
pip install -r requirements_evaluation.txt
"""

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.interpolate import splrep, splev, BSpline
    from sklearn.model_selection import KFold
    from scipy import stats
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install required packages:")
    print("pip install numpy pandas matplotlib scipy scikit-learn")
    print("or run: pip install -r requirements_evaluation.txt")
    exit(1)

# Import the evaluation functions (from the guide)
def calculate_mse(original_data, fitted_spline, x_points):
    """Calculate MSE between original data and fitted B-spline"""
    predicted = fitted_spline(x_points)
    return np.mean((original_data - predicted) ** 2)

def calculate_rmse(original_data, fitted_spline, x_points):
    """Calculate RMSE - more interpretable than MSE"""
    mse = calculate_mse(original_data, fitted_spline, x_points)
    return np.sqrt(mse)

def calculate_mae(original_data, fitted_spline, x_points):
    """Calculate MAE - robust to outliers"""
    predicted = fitted_spline(x_points)
    return np.mean(np.abs(original_data - predicted))

def calculate_r_squared(original_data, fitted_spline, x_points):
    """Calculate R² - proportion of variance explained"""
    predicted = fitted_spline(x_points)
    ss_res = np.sum((original_data - predicted) ** 2)
    ss_tot = np.sum((original_data - np.mean(original_data)) ** 2)
    return 1 - (ss_res / ss_tot)

def cross_validate_spline(data_x, data_y, k_folds=5, spline_degree=3):
    """Perform k-fold cross-validation for B-spline fitting"""
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    errors = []
    
    for train_idx, test_idx in kf.split(data_x):
        x_train, x_test = data_x[train_idx], data_x[test_idx]
        y_train, y_test = data_y[train_idx], data_y[test_idx]
        
        # Fit B-spline on training data
        tck = splrep(x_train, y_train, s=0, k=spline_degree)
        
        # Predict on test data
        y_pred = splev(x_test, tck)
        
        # Calculate error
        error = np.mean((y_test - y_pred) ** 2)
        errors.append(error)
    
    return np.mean(errors), np.std(errors)

def calculate_aic(data_y, fitted_spline, x_points, num_coefficients):
    """Calculate AIC - balances fit quality and model complexity"""
    n = len(data_y)
    predicted = fitted_spline(x_points)
    mse = np.mean((data_y - predicted) ** 2)
    
    # AIC = n*ln(MSE) + 2k
    aic = n * np.log(mse) + 2 * num_coefficients
    return aic

def calculate_bic(data_y, fitted_spline, x_points, num_coefficients):
    """Calculate BIC - stronger penalty for model complexity"""
    n = len(data_y)
    predicted = fitted_spline(x_points)
    mse = np.mean((data_y - predicted) ** 2)
    
    # BIC = n*ln(MSE) + ln(n)*k
    bic = n * np.log(mse) + np.log(n) * num_coefficients
    return bic

def analyze_residuals(original_data, fitted_spline, x_points, dataset_name="Dataset"):
    """Analyze residuals for model quality assessment"""
    predicted = fitted_spline(x_points)
    residuals = original_data - predicted
    
    # Check normality of residuals
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    
    # Create residual plots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(predicted, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{dataset_name}: Residuals vs Predicted')
    
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=20, density=True, alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title(f'{dataset_name}: Residual Distribution')
    
    plt.subplot(1, 3, 3)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'{dataset_name}: Q-Q Plot')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals)
    }

def evaluate_dataset_accuracy(datasets, dataset_names, spline_degree=3):
    """
    Comprehensive evaluation of multiple datasets for B-spline fitting
    """
    results = []
    
    for i, (data_x, data_y) in enumerate(datasets):
        print(f"Evaluating {dataset_names[i]}...")
        
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
        residual_stats = analyze_residuals(data_y, spline, data_x, dataset_names[i])
        
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

def select_best_dataset(evaluation_results):
    """
    Select best dataset based on multiple criteria
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

def generate_sample_datasets():
    """
    Generate sample datasets for demonstration
    """
    np.random.seed(42)
    
    # Generate x values
    x = np.linspace(0, 10, 100)
    
    # True function (for reference)
    true_func = lambda x: 2 * np.sin(x) + 0.5 * x
    
    # Dataset 1: Clean data with minimal noise
    y1 = true_func(x) + np.random.normal(0, 0.1, len(x))
    
    # Dataset 2: Noisy data
    y2 = true_func(x) + np.random.normal(0, 0.5, len(x))
    
    # Dataset 3: Data with outliers
    y3 = true_func(x) + np.random.normal(0, 0.2, len(x))
    # Add some outliers
    outlier_idx = np.random.choice(len(x), 10, replace=False)
    y3[outlier_idx] += np.random.normal(0, 2, 10)
    
    # Dataset 4: Sparse data (fewer points)
    sparse_idx = np.random.choice(len(x), 30, replace=False)
    sparse_idx.sort()
    x4 = x[sparse_idx]
    y4 = true_func(x4) + np.random.normal(0, 0.15, len(x4))
    
    return [
        (x, y1),
        (x, y2), 
        (x, y3),
        (x4, y4)
    ], ['Clean_Data', 'Noisy_Data', 'Outlier_Data', 'Sparse_Data']

def plot_datasets(datasets, dataset_names):
    """
    Plot all datasets for visual comparison
    """
    plt.figure(figsize=(15, 10))
    
    for i, (data_x, data_y) in enumerate(datasets):
        plt.subplot(2, 2, i+1)
        plt.scatter(data_x, data_y, alpha=0.6, s=20)
        plt.title(f'{dataset_names[i]} (n={len(data_x)})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function demonstrating dataset evaluation workflow
    """
    print("=== B-spline Dataset Accuracy Evaluation Example ===\n")
    
    # Generate sample datasets
    print("1. Generating sample datasets...")
    datasets, dataset_names = generate_sample_datasets()
    
    # Plot datasets
    print("2. Plotting datasets for visual inspection...")
    plot_datasets(datasets, dataset_names)
    
    # Evaluate all datasets
    print("3. Evaluating datasets...")
    results = evaluate_dataset_accuracy(datasets, dataset_names, spline_degree=3)
    
    # Display results
    print("\n4. Evaluation Results:")
    print("="*80)
    print(results[['Dataset', 'RMSE', 'R²', 'CV_Error', 'AIC', 'BIC', 'Data_Points']].round(4))
    
    # Select best dataset
    print("\n5. Selecting best dataset...")
    best_dataset, ranked_results = select_best_dataset(results)
    
    print(f"\nBest dataset: {best_dataset}")
    print("\nRanked results:")
    print("="*80)
    print(ranked_results[['Dataset', 'CV_Error', 'RMSE', 'R²', 'AIC', 'Weighted_Score']].round(4))
    
    # Summary insights
    print("\n6. Summary Insights:")
    print("="*80)
    clean_data = results[results['Dataset'] == 'Clean_Data']
    noisy_data = results[results['Dataset'] == 'Noisy_Data']
    outlier_data = results[results['Dataset'] == 'Outlier_Data']
    sparse_data = results[results['Dataset'] == 'Sparse_Data']
    
    print(f"• Clean data has lowest CV error: {clean_data['CV_Error'].values[0]:.4f}")
    print(f"• Noisy data has highest RMSE: {noisy_data['RMSE'].values[0]:.4f}")
    print(f"• Outlier data has poor R²: {outlier_data['R²'].values[0]:.4f}")
    print(f"• Sparse data has fewer points but may still be good: {sparse_data['Data_Points'].values[0]} points")
    
    # Recommendations
    print("\n7. Recommendations:")
    print("="*80)
    best_result = ranked_results.iloc[0]
    print(f"• Use '{best_result['Dataset']}' for your B-spline fitting")
    print(f"• Cross-validation error: {best_result['CV_Error']:.4f}")
    print(f"• Expected accuracy (R²): {best_result['R²']:.4f}")
    print(f"• Model complexity (AIC): {best_result['AIC']:.2f}")
    
    if best_result['Residual_Normality_p'] > 0.05:
        print("• Residuals are normally distributed (good model fit)")
    else:
        print("• Residuals may not be normally distributed (check model assumptions)")
    
    return best_dataset, results

if __name__ == "__main__":
    best_dataset, evaluation_results = main()