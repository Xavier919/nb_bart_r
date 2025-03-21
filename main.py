import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
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
                    'pi', 'fi', 'fli', 'fri', 'fti', 'cli', 'cri', 'cti','total_lane', 'avg_crossw', 'tot_road_w', 'tot_crossw',
                    'commercial', 'number_of_', 'of_exclusi', 'curb_exten', 'median', 'all_pedest', 'half_phase', 'new_half_r',
                    'any_ped_pr', 'ped_countd', 'lt_restric', 'lt_prot_re', 'lt_protect', 'any_exclus', 'all_red_an', 'green_stra',
                    'parking', 'north_veh', 'north_ped', 'east_veh', 'east_ped', 'south_veh', 'south_ped', 'west_veh', 'west_ped']
    

    # Check which columns actually exist in the data
    feature_cols = [col for col in feature_cols if col in data.columns]
    spatial_cols = [col for col in spatial_cols if col in data.columns]
    
    # Combine base features with spatial variables
    X_base = data[feature_cols].fillna(0)
    X_spatial = data[spatial_cols].fillna(0)
    
    # Apply quadratic transformation to specified variables
    for col in ['ln_cti', 'ln_cri', 'ln_cli', 'ln_pi', 'ln_fri', 'ln_fli', 'ln_fi']:
        if col in X_base.columns:
            X_base[f'{col}_squared'] = X_base[col] ** 2

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
        'X_non_spatial': X_base,  # Add non-spatial features separately
        'X_spatial': X_spatial,   # Add spatial features separately
        'y': data['acc'], 
        'int_no': data['int_no'],
        'pi': data['pi']
    }


def select_features_with_lasso(X, y):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit Lasso model with cross-validation to find optimal alpha
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_scaled, y)
    
    # Select features with non-zero coefficients
    selected_features = X.columns[lasso.coef_ != 0]
    print(f"Selected {len(selected_features)} features: {list(selected_features)}")
    
    return selected_features


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


def save_results(int_no, y_true, y_pred, pi, model_name):
    results_path = "results"
    os.makedirs(results_path, exist_ok=True)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'int_no': int_no,
        'true_counts': y_true,
        'pred_counts': y_pred,
        'true_rate': y_true / pi,
        'pred_rate': y_pred / pi,
        'pi': pi
    })
    
    # Save prediction summary
    results.to_csv(os.path.join(results_path, f"{model_name}_cv_results.csv"), index=False)
    
    # Create and save intersection ranking
    rankings = results[['int_no', 'pred_counts']].sort_values(by='pred_counts', ascending=False)
    rankings['ranking'] = range(1, len(rankings) + 1)
    rankings.to_csv(os.path.join(results_path, f"{model_name}_intersections_cv_ranking.csv"), index=False)


def run_cross_validation(X_non_spatial, X_spatial, y, int_no, pi, k=5):
    # Create fold indices
    np.random.seed(42)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Initialize lists to store results
    results_list = []
    all_metrics = pd.DataFrame(columns=['fold', 'mae', 'mse', 'rmse'])
    combined_results = pd.DataFrame()  # New DataFrame to store combined results
    
    # For each fold
    for i, (train_idx, test_idx) in enumerate(kf.split(X_non_spatial), 1):
        print(f"\n========== Fold {i} ==========")
        
        # Create training and test sets
        X_train_non_spatial = X_non_spatial.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test_non_spatial = X_non_spatial.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        # Select features using Lasso on non-spatial features
        selected_features = select_features_with_lasso(X_train_non_spatial, y_train)
        X_train_selected = X_train_non_spatial[selected_features]
        X_test_selected = X_test_non_spatial[selected_features]
        
        # Combine selected non-spatial features with all spatial features
        X_train_combined = pd.concat([X_train_selected, X_spatial.iloc[train_idx]], axis=1)
        X_test_combined = pd.concat([X_test_selected, X_spatial.iloc[test_idx]], axis=1)
        
        # Run BART model with combined features
        fold_results = run_bart_model(X_train_combined, y_train, X_test_combined, y_test)
        
        # Debugging: Check the shapes
        print(f"y_test shape: {y_test.shape}, y_pred_mean shape: {fold_results.y_pred_mean.shape}")
        
        # Ensure the prediction length matches the test set length
        if len(y_test) != len(fold_results.y_pred_mean):
            print("Mismatch in prediction length and test set length.")
            continue  # Skip this fold or handle the mismatch appropriately
        
        # Store results
        results_list.append(fold_results)
        
        # Combine results for this fold
        fold_results_df = pd.DataFrame({
            'int_no': int_no.iloc[test_idx],
            'true_counts': y_test,
            'pred_counts': fold_results.y_pred_mean,
            'true_rate': y_test / pi.iloc[test_idx],
            'pred_rate': fold_results.y_pred_mean / pi.iloc[test_idx],
            'pi': pi.iloc[test_idx]
        })
        combined_results = pd.concat([combined_results, fold_results_df], ignore_index=True)
        
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
    g.map(sns.barplot, 'fold', 'value', order=sorted(all_metrics_long['fold'].unique()))
    g.set_axis_labels('Fold', 'Value')
    g.set_titles('{col_name}')
    plt.tight_layout()
    
    # Save combined results
    combined_results.to_csv(os.path.join("results", "bart_combined_cv_results.csv"), index=False)
    
    # Create and save combined intersection ranking
    combined_rankings = combined_results[['int_no', 'pred_counts']].sort_values(by='pred_counts', ascending=False)
    combined_rankings['ranking'] = range(1, len(combined_rankings) + 1)
    combined_rankings.to_csv(os.path.join("results", "bart_combined_intersections_cv_ranking.csv"), index=False)
    
    return {
        'results_per_fold': results_list,
        'metrics_per_fold': all_metrics,
        'average_metrics': avg_metrics
    }


def main():
    data = load_and_prepare_data()
    
    # Run 5-fold cross validation
    cv_results = run_cross_validation(data['X_non_spatial'], data['X_spatial'], data['y'], data['int_no'], data['pi'], k=5)
    
    # Return results
    return cv_results


# Run the main script
if __name__ == "__main__":
    results = main()