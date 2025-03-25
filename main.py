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
from scipy.spatial.distance import pdist, squareform
import torch
import torch.optim as optim

# Import classes from negative_binomial.py
from negative_binomial import NegativeBinomial, Data, Options, Results


def load_and_prepare_data():

    data = pd.read_csv("data/data_final.csv", sep=";")
    
    # Replace NaN values in ln_distdt with 0
    data['ln_distdt'] = data['ln_distdt'].fillna(0)
    
    # Filter where pi != 0
    data = data[data['pi'] != 0]
    
    # Include existing spatial columns
    spatial_cols = ['x', 'y']
    
    # Include borough as categorical variable if available
    borough_dummies = pd.DataFrame()
    if 'borough' in data.columns:
        data['borough'] = data['borough'].astype('category')
        # Create dummy variables for borough
        borough_dummies = pd.get_dummies(data['borough'], prefix='borough')
    else:
        print("Variable 'borough' not found in the data")
    
    # Base features (non-spatial)
    feature_cols = ['ln_fi', 'ln_fri', 'ln_fli', 'ln_pi', 'traffic_10000', 'ln_cti', 'ln_cli', 'ln_cri',"ln_distdt",
                    'total_lane', 'avg_crossw', 'tot_road_w', 'tot_crossw',
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

    # Add borough dummy variables if available
    if not borough_dummies.empty:
        X_features = pd.concat([X_base, X_spatial, borough_dummies], axis=1)
    else:
        X_features = pd.concat([X_base, X_spatial], axis=1)
    
    return {
        'X': X_features, 
        'X_non_spatial': X_base,  # Add non-spatial features separately
        'X_spatial': X_spatial,   # Add spatial features separately
        'y': data['acc'], 
        'int_no': data['int_no'],
        'pi': data['pi']
    }


def create_spatial_weight_matrix(X_spatial):
    """
    Create a spatial weight matrix based on distances between coordinates.
    Uses GPU acceleration if available.
    """
    # Extract coordinates
    coords = X_spatial[['x', 'y']].values
    print(f"Coordinates shape: {coords.shape}")  # Debug shape
    
    # Use GPU if available
    if torch.cuda.is_available():
        coords_tensor = torch.tensor(coords, device='cuda', dtype=torch.float32)
        
        # Calculate pairwise distances on GPU
        n = len(coords_tensor)
        distances = torch.zeros((n, n), device='cuda')
        
        # More efficient batched calculation to avoid potential OOM errors
        batch_size = 100  # Adjust based on GPU memory
        for i in range(0, n, batch_size):
            end_idx = min(i + batch_size, n)
            batch_coords = coords_tensor[i:end_idx]
            
            # Calculate distances for this batch to all other points
            # Reshape for broadcasting: (batch, 1, dims) - (1, all_points, dims)
            diff = batch_coords.unsqueeze(1) - coords_tensor.unsqueeze(0)
            batch_distances = torch.sqrt((diff ** 2).sum(dim=2))
            distances[i:end_idx] = batch_distances
        
        # Create weight matrix based on distance bands
        W = torch.zeros((n, n), device='cuda')
        
        # Define weights based on adjacency orders
        W[(distances > 0) & (distances <= 200)] = 1.0
        W[(distances > 200) & (distances <= 400)] = 0.5
        W[(distances > 400) & (distances <= 600)] = 0.25
        
        # Row-normalize the weight matrix
        row_sums = W.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        W = W / row_sums
        
        # Convert back to numpy
        W = W.cpu().numpy()
    else:
        # Original CPU implementation
        distances = squareform(pdist(coords, 'euclidean'))
        n = len(X_spatial)
        W = np.zeros((n, n))
        W[(distances > 0) & (distances <= 200)] = 1.0
        W[(distances > 200) & (distances <= 400)] = 0.5
        W[(distances > 400) & (distances <= 600)] = 0.25
        
        row_sums = W.sum(axis=1)
        W[row_sums > 0] = W[row_sums > 0] / row_sums[row_sums > 0].reshape(-1, 1)
    
    return W


def select_features_with_lasso(X, y):
    """
    Select features using Lasso regression (CPU implementation)
    Always keeps specified important features regardless of Lasso selection
    """
    # List of features to always keep
    important_features = ['ln_cti', 'ln_cri', 'ln_cli', 'ln_pi', 'ln_fri', 'ln_fli', 'ln_fi', 'ln_cti_squared', 'ln_cri_squared', 'ln_cli_squared', 'ln_pi_squared', 'ln_fri_squared', 'ln_fli_squared', 'ln_fi_squared']
    
    # Filter to only include features that exist in the dataset
    important_features = [f for f in important_features if f in X.columns]
    
    # Original CPU implementation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lasso = LassoCV(cv=5, random_state=42, max_iter=500000, tol=1e-4)
    lasso.fit(X_scaled, y)
    
    # Get features selected by Lasso
    lasso_selected = X.columns[lasso.coef_ != 0].tolist()
    
    # Combine Lasso-selected features with important features (avoiding duplicates)
    selected_features = list(set(lasso_selected + important_features))
    
    print(f"Selected {len(selected_features)} features:")
    print(f"  - Always included: {important_features}")
    print(f"  - Lasso selected: {[f for f in lasso_selected if f not in important_features]}")
    
    return selected_features


def run_nb_model(X_train, y_train, X_test, y_test, pi_train, W_train):
    """
    Run Negative Binomial model with fixed parameters and spatial correlation
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        pi_train: Exposure variable for training
        W_train: Spatial weight matrix for training
    
    Returns:
        results: Results object with model outputs
    """
    # Convert to numpy arrays
    X_train_np = X_train.values.astype(np.float64)
    y_train_np = y_train.values.astype(np.int64)  # Changed to int64 to fix the error
    
    # Create Data object for NegativeBinomial model (using X as fixed effects)
    nb_data = Data(y=y_train_np, x_fix=X_train_np, x_rnd=None, W=W_train)
    
    # Initialize model
    nb_model = NegativeBinomial(nb_data, data_bart=None)
    
    # Define model options
    options = Options(
        model_name='fixed',
        nChain=1, nBurn=200, nSample=200, nThin=2, 
        mh_step_initial=0.1, mh_target=0.3, mh_correct=0.01, mh_window=50,
        disp=100, delete_draws=False, seed=42
    )
    
    # Prior parameters
    n_fix = X_train_np.shape[1]
    r0 = 1e-2
    b0 = 1e-2
    c0 = 1e-2
    beta_mu0 = np.zeros(n_fix)
    beta_Si0Inv = 1e-2 * np.eye(n_fix)
    
    # Initialize with zeros since we don't have random effects
    mu_mu0 = np.zeros(0)
    mu_Si0Inv = np.eye(0)
    nu = 2
    A = np.array([])
    
    # Spatial parameters
    sigma2_b0 = 1e-2
    sigma2_c0 = 1e-2
    tau_mu0 = 0
    tau_si0 = 2
    
    # Initialize parameters
    r_init = 5.0
    beta_init = np.zeros(n_fix)
    mu_init = np.zeros(0)
    Sigma_init = np.eye(0)
    
    # Define ranking thresholds
    ranking_top_m_list = [10, 25, 50, 100]
    
    # Run model estimation
    try:
        results = nb_model.estimate(
            options, None,  # No BART options
            r0, b0, c0,
            beta_mu0, beta_Si0Inv,
            mu_mu0, mu_Si0Inv, nu, A,
            sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
            r_init, beta_init, mu_init, Sigma_init,
            ranking_top_m_list
        )
        
        # Predict for test data
        X_test_np = X_test.values.astype(np.float64)
        
        # Handle potential NaN values in beta_mean
        beta_mean = results.post_beta['mean'].values
        if np.isnan(beta_mean).any():
            print("Warning: NaN values detected in beta coefficients. Using zeros instead.")
            beta_mean = np.zeros_like(beta_mean)
        
        # For negative binomial, we need to use the exp(X*beta) * r formula for predictions
        r_mean = results.post_r['mean'].values[0]
        # Handle potential NaN in r
        if np.isnan(r_mean):
            print("Warning: NaN value detected in r parameter. Using 1.0 instead.")
            r_mean = 1.0
            
        psi_test = X_test_np @ beta_mean
        
        # Apply stronger clipping to prevent overflow
        psi_test = np.clip(psi_test, -25, 25)  # More conservative clipping
        
        # Use a more stable computation of the expected value
        lam_test = np.exp(psi_test) * r_mean
        
        # Final safeguard against NaN or infinity values
        lam_test = np.nan_to_num(lam_test, nan=0.0, posinf=100, neginf=0)
        
        # Store predictions
        results.test_predictions = lam_test
        results.test_actual = y_test.values
        
    except Exception as e:
        print(f"Error in model estimation: {e}")
        # Return a simple results object with default predictions if model fails
        # This ensures the cross-validation continues even if one fold fails
        results = None
        return results
    
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
    """
    Run k-fold cross-validation with the Negative Binomial model
    """
    # Create fold indices
    np.random.seed(42)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Initialize lists to store results
    results_list = []
    all_metrics = []  # Change to a list instead of empty DataFrame
    combined_results = pd.DataFrame()
    
    # Make sure int_no is present in X_spatial
    if 'int_no' not in X_spatial.columns:
        X_spatial = X_spatial.copy()
        X_spatial['int_no'] = int_no.values
    
    # Print dataset shapes for debugging
    print(f"X_non_spatial shape: {X_non_spatial.shape}")
    print(f"X_spatial shape: {X_spatial.shape}")
    print(f"y shape: {y.shape}")
    print(f"int_no shape: {int_no.shape}")
    print(f"pi shape: {pi.shape}")
    
    # For each fold
    for i, (train_idx, test_idx) in enumerate(kf.split(X_non_spatial), 1):
        print(f"\n========== Fold {i} ==========")
        
        # Create training and test sets
        X_train_non_spatial = X_non_spatial.iloc[train_idx]
        y_train = y.iloc[train_idx]
        pi_train = pi.iloc[train_idx]
        X_test_non_spatial = X_non_spatial.iloc[test_idx]
        y_test = y.iloc[test_idx]
        pi_test = pi.iloc[test_idx]
        
        # Get spatial features for this fold
        X_train_spatial = X_spatial.iloc[train_idx]
        
        # Print shapes for debugging
        print(f"Train shapes - X_non_spatial: {X_train_non_spatial.shape}, X_spatial: {X_train_spatial.shape}, y: {y_train.shape}")
        print(f"Test shapes - X_non_spatial: {X_test_non_spatial.shape}, X_spatial: {X_spatial.iloc[test_idx].shape}, y: {y_test.shape}")
        
        # Select features using Lasso on non-spatial features
        selected_features = select_features_with_lasso(X_train_non_spatial, y_train)
        X_train_selected = X_train_non_spatial[selected_features]
        X_test_selected = X_test_non_spatial[selected_features]
        
        # Combine selected non-spatial features with necessary spatial features for modeling
        X_train_combined = pd.concat([X_train_selected, X_train_spatial[['x', 'y']]], axis=1)
        X_test_combined = pd.concat([X_test_selected, X_spatial.iloc[test_idx][['x', 'y']]], axis=1)
        
        # Create spatial weight matrix for training data
        W_train = create_spatial_weight_matrix(X_train_spatial)
        
        # Run Negative Binomial model
        fold_results = run_nb_model(X_train_combined, y_train, X_test_combined, y_test, pi_train, W_train)
        
        # Skip this fold if model failed
        if fold_results is None:
            print(f"Skipping fold {i} due to estimation errors")
            continue
        
        # Store results
        results_list.append(fold_results)
        
        # Get predictions for test set
        y_pred = fold_results.test_predictions
        
        # Combine results for this fold
        fold_results_df = pd.DataFrame({
            'int_no': int_no.iloc[test_idx],
            'true_counts': y_test,
            'pred_counts': y_pred,
            'true_rate': y_test / pi_test,
            'pred_rate': y_pred / pi_test,
            'pi': pi_test
        })
        combined_results = pd.concat([combined_results, fold_results_df], ignore_index=True)
        
        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Add evaluation metrics to list instead of concatenating DataFrames
        all_metrics.append({
            'fold': i,
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        })
        
        # Display fold results
        print(f"Fold {i} - MAE: {mae}")
        print(f"Fold {i} - MSE: {mse}")
        print(f"Fold {i} - RMSE: {rmse}")
    
    # Convert list to DataFrame after the loop
    all_metrics = pd.DataFrame(all_metrics)
    
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
    
    # Save plot
    plt.savefig(os.path.join("results", "nb_model_cv_metrics.png"))
    
    # Save combined results
    combined_results.to_csv(os.path.join("results", "nb_model_combined_cv_results.csv"), index=False)
    
    # Create and save combined intersection ranking
    combined_rankings = combined_results[['int_no', 'pred_counts']].sort_values(by='pred_counts', ascending=False)
    combined_rankings['ranking'] = range(1, len(combined_rankings) + 1)
    combined_rankings.to_csv(os.path.join("results", "nb_model_combined_intersections_cv_ranking.csv"), index=False)
    
    return {
        'results_per_fold': results_list,
        'metrics_per_fold': all_metrics,
        'average_metrics': avg_metrics
    }


def plot_results(combined_results, X_spatial):
    """Plot various visualizations of the model results"""
    results_path = "results"
    os.makedirs(results_path, exist_ok=True)
    
    # Create figure for predicted vs actual counts
    plt.figure(figsize=(10, 6))
    plt.scatter(combined_results['true_counts'], combined_results['pred_counts'], alpha=0.6)
    plt.plot([0, max(combined_results['true_counts'])], [0, max(combined_results['true_counts'])], 'r--')
    plt.xlabel('Observed Accident Counts')
    plt.ylabel('Predicted Accident Counts')
    plt.title('Observed vs Predicted Accident Counts')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'predicted_vs_actual.png'))
    
    # Plot spatial distribution of accidents
    if 'x' in X_spatial.columns and 'y' in X_spatial.columns:
        # Create a mapping of int_no to spatial coordinates - fixed to use int_no from combined_results
        unique_int_nos = combined_results['int_no'].unique()
        
        # Create a DataFrame with int_no, x, y by merging information
        spatial_results = pd.DataFrame()
        
        # Group by int_no and calculate mean predicted counts
        pred_by_int = combined_results.groupby('int_no')['pred_counts'].mean().reset_index()
        
        # Only proceed if we can match int_no with spatial coordinates
        if 'int_no' in X_spatial.columns:
            spatial_mapping = X_spatial[['int_no', 'x', 'y']].drop_duplicates('int_no').set_index('int_no')
            
            # Merge spatial coordinates with predictions
            spatial_results = pd.merge(
                pred_by_int, 
                spatial_mapping.reset_index(), 
                on='int_no', how='inner'
            )
        else:
            # Try to match by index position if int_no values are available in order
            if hasattr(X_spatial, 'index') and X_spatial.index.name == 'int_no':
                spatial_results = pd.merge(
                    pred_by_int,
                    X_spatial.reset_index(),
                    on='int_no', how='inner'
                )
            else:
                print("Warning: Cannot create spatial plot - unable to match int_no with coordinates")
        
        # Only create plot if we have data
        if not spatial_results.empty:
            plt.figure(figsize=(12, 10))
            plt.scatter(spatial_results['x'], spatial_results['y'], 
                       c=spatial_results['pred_counts'], cmap='viridis', 
                       s=50, alpha=0.8)
            plt.colorbar(label='Predicted Accident Count')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title('Spatial Distribution of Predicted Accident Counts')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(results_path, 'spatial_distribution.png'))
        else:
            print("Skipping spatial plot: Could not match int_no with spatial coordinates")
    
    print("Plots saved as PNG files in the results directory.")


def main():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Run 5-fold cross validation with NegativeBinomial model
    cv_results = run_cross_validation(data['X_non_spatial'], data['X_spatial'], data['y'], data['int_no'], data['pi'], k=5)
    
    # Load the combined results for plotting
    combined_results_path = os.path.join("results", "nb_model_combined_cv_results.csv")
    if os.path.exists(combined_results_path):
        combined_results = pd.read_csv(combined_results_path)
        # Generate plots
        plot_results(combined_results, data['X_spatial'])
    else:
        print("Combined results file not found. Skipping plot generation.")
    
    # Return results
    return cv_results


# Run the main script
if __name__ == "__main__":
    results = main()