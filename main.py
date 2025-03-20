import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import BART-related classes
from bayestree import DataTrain, DataVal, BartOptions, McmcOptions, BartMcmc


def load_and_prepare_data():

    data = pd.read_csv("data/data_final.csv", sep=";")
    
    # Replace NaN values in ln_distdt with 0
    data['ln_distdt'] = data['ln_distdt'].fillna(0)
    
    # Filter where pi != 0
    data = data[data['pi'] != 0]
    
    # Include existing spatial columns
    spatial_cols = ['x', 'y', 'distdt', 'ln_distdt']
    
    # Include borough as categorical variable if available
    borough_dummies = pd.DataFrame()
    if 'borough' in data.columns:
        data['borough'] = data['borough'].astype('category')
        # Create dummy variables for borough
        borough_dummies = pd.get_dummies(data['borough'], prefix='borough')
    else:
        print("Variable 'borough' not found in the data")
    
    # Base features (non-spatial)
    feature_cols = ['ln_fi', 'ln_fri', 'ln_fli', 'ln_pi', 'traffic_10000', 'ln_cti', 'ln_cli', 'ln_cri',
                  'total_lane', 'avg_crossw', 'tot_road_w', 'tot_crossw',
                  'commercial', 'number_of_', 'of_exclusi', 'curb_exten',
                  'median', 'all_pedest', 'half_phase', 'new_half_r',
                  'any_ped_pr', 'ped_countd', 'lt_restric', 'lt_prot_re',
                  'lt_protect', 'any_exclus', 'all_red_an', 'green_stra',
                  'parking']
    
    # Check which columns actually exist in the data
    feature_cols = [col for col in feature_cols if col in data.columns]
    spatial_cols = [col for col in spatial_cols if col in data.columns]
    
    # Combine base features with spatial variables
    X_base = data[feature_cols].fillna(0)
    X_spatial = data[spatial_cols].fillna(0)
    
    # Calculate additional spatial features if x and y are available
    if all(col in X_spatial.columns for col in ['x', 'y']):
        # 1. Normalize x and y coordinates
        X_spatial['x_scaled'] = stats.zscore(X_spatial['x'])
        X_spatial['y_scaled'] = stats.zscore(X_spatial['y'])
        
        # 2. Calculate distance to city centroid
        x_center = X_spatial['x'].mean()
        y_center = X_spatial['y'].mean()
        X_spatial['dist_to_center'] = np.sqrt((X_spatial['x'] - x_center)**2 + (X_spatial['y'] - y_center)**2)
        
        # 3. Interaction between coordinates
        X_spatial['xy_interaction'] = X_spatial['x_scaled'] * X_spatial['y_scaled']
    else:
        print("Coordinates x and y not found in the data")
    
    # Combine all features
    X_features = pd.concat([X_base, X_spatial], axis=1)
    
    # Add borough dummy variables if available
    if not borough_dummies.empty:
        X_features = pd.concat([X_features, borough_dummies], axis=1)
    
    return {
        'X': X_features, 
        'y': data['acc'], 
        'int_no': data['int_no'],
        'pi': data['pi']
    }


def run_bart_model(X_train, y_train, X_test, y_test):
    # Convert to numpy arrays with explicit float type
    X_train_np = X_train.values.astype(np.float64)
    X_test_np = X_test.values.astype(np.float64)
    y_train_np = y_train.values.astype(np.float64)
    y_test_np = y_test.values.astype(np.float64)
    
    # Prepare data for BART
    data_train = DataTrain(y_train_np, X_train_np)
    data_test = DataVal(y_test_np, X_test_np)
    
    # Set BART options
    bart_options = BartOptions(data_train)
    mcmc_options = McmcOptions(nChain=2, nBurn=1000, nSample=1000, nThin=5)
    
    # Run BART
    bart_mcmc = BartMcmc(mcmc_options, bart_options, data_train, data_test)
    results = bart_mcmc.estimate()
    
    # Print results
    print("RMSE (train):", np.sqrt(np.mean((y_train - results.y_hat_mean)**2)))
    print("RMSE (test):", np.sqrt(np.mean((y_test - results.y_pred_mean)**2)))
    
    return results


def run_cross_validation(X, y, k=5):
    # Create fold indices
    np.random.seed(42)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Initialize lists to store results
    results_list = []
    all_metrics = pd.DataFrame(columns=['fold', 'mae', 'mse', 'rmse'])
    
    # For each fold
    for i, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"\n========== Fold {i} ==========")
        
        # Create training and test sets
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        # Run BART model
        fold_results = run_bart_model(X_train, y_train, X_test, y_test)
        
        # Debugging: Check the shapes
        print(f"y_test shape: {y_test.shape}, y_pred_mean shape: {fold_results.y_pred_mean.shape}")
        
        # Ensure the prediction length matches the test set length
        if len(y_test) != len(fold_results.y_pred_mean):
            print("Mismatch in prediction length and test set length.")
            continue  # Skip this fold or handle the mismatch appropriately
        
        # Store results
        results_list.append(fold_results)
        
        # Add evaluation metrics to dataframe
        all_metrics = pd.concat([all_metrics, pd.DataFrame({
            'fold': [i],
            'mae': [mean_absolute_error(y_test, fold_results.y_pred_mean)],
            'mse': [mean_squared_error(y_test, fold_results.y_pred_mean)],
            'rmse': [np.sqrt(mean_squared_error(y_test, fold_results.y_pred_mean))]
        })], ignore_index=True)
        
        # Display fold results
        print(f"Fold {i} - MAE: {mean_absolute_error(y_test, fold_results.y_pred_mean)}")
        print(f"Fold {i} - MSE: {mean_squared_error(y_test, fold_results.y_pred_mean)}")
        print(f"Fold {i} - RMSE: {np.sqrt(mean_squared_error(y_test, fold_results.y_pred_mean))}")
    
    # Calculate averages
    avg_metrics = all_metrics[['mae', 'mse', 'rmse']].mean()
    
    # Display results for all folds
    print("\n========== Results by fold ==========")
    print(all_metrics)
    
    # Display averages
    print("\n========== Average metrics ==========")
    print(f"Average MAE: {avg_metrics['mae']}")
    print(f"Average MSE: {avg_metrics['mse']}")
    print(f"Average RMSE: {avg_metrics['rmse']}")
    
    # Create a visualization of fold results
    plt.figure(figsize=(12, 8))
    all_metrics_long = pd.melt(all_metrics, id_vars=['fold'], value_vars=['mae', 'mse', 'rmse'])
    
    # Create a multi-panel plot
    g = sns.FacetGrid(all_metrics_long, col='variable', col_wrap=3, height=4)
    g.map(sns.barplot, 'fold', 'value')
    g.set_axis_labels('Fold', 'Value')
    g.set_titles('{col_name}')
    plt.tight_layout()
    
    return {
        'results_per_fold': results_list,
        'metrics_per_fold': all_metrics,
        'average_metrics': avg_metrics
    }


def main():
    data = load_and_prepare_data()
    
    # Run 5-fold cross validation
    cv_results = run_cross_validation(data['X'], data['y'], k=5)
    
    # Return results
    return cv_results


# Run the main script
if __name__ == "__main__":
    results = main()