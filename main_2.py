import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from scipy.special import loggamma
from scipy.sparse.linalg import expm
import warnings
import traceback
warnings.filterwarnings('ignore')

class NegativeBinomialBART:
    """
    Combined model using Negative Binomial regression with BART component
    for modeling accident counts at intersections.
    
    This model handles both spatial and non-spatial features, incorporating
    a negative binomial likelihood for count data, a BART component for
    flexible modeling of feature relationships, and random effects for boroughs.
    """
    
    def __init__(self, n_trees=50, max_depth=3, n_samples=1000, n_burn=500):
        """
        Initialize the NegativeBinomialBART model.
        
        Parameters:
        -----------
        n_trees : int
            Number of trees in the BART component
        max_depth : int
            Maximum depth of each tree
        n_samples : int
            Number of posterior samples after burn-in
        n_burn : int
            Number of burn-in samples
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_samples = n_samples
        self.n_burn = n_burn
        self.model = None
        self.trace = None
        
    def _create_spatial_weight_matrix(self, X_spatial):
        """
        Create a spatial weight matrix based on coordinates.
        
        Parameters:
        -----------
        X_spatial : DataFrame
            Spatial features including x, y coordinates
            
        Returns:
        --------
        W : ndarray
            Spatial weight matrix
        """
        # Extract coordinates
        if 'x' in X_spatial.columns and 'y' in X_spatial.columns:
            coords = X_spatial[['x', 'y']].values
            n = coords.shape[0]
            
            # Calculate distances between all pairs of points
            W = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
                        # Using inverse distance weighting with cutoff
                        if dist < 1000:  # More restrictive cutoff (was 5000)
                            W[i, j] = 1 / dist
            
            # Row-normalize the weight matrix
            row_sums = W.sum(axis=1)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            W = W / row_sums[:, np.newaxis]
            
            return W
        else:
            print("Coordinates x and y not found in spatial features")
            # Return identity matrix if no coordinates
            return np.eye(X_spatial.shape[0])
    
    def build_model(self, X_non_spatial, X_spatial, y, borough_ids=None):
        """
        Build the PyMC model combining negative binomial, BART components,
        and borough-level random effects.
        
        Parameters:
        -----------
        X_non_spatial : DataFrame
            Non-spatial features
        X_spatial : DataFrame
            Spatial features (should include x, y coordinates)
        y : Series
            Accident counts
        borough_ids : Series, optional
            Borough identifiers for each intersection
        """
        # Don't scale features for the main model
        X_non_spatial_model = X_non_spatial.copy()
        X_spatial_model = X_spatial.copy()
        
        # Store features for later prediction
        self.X_non_spatial_features = X_non_spatial.columns.tolist()
        self.X_spatial_features = X_spatial.columns.tolist()
        
        # Store original data ranges to set appropriate priors
        self.x_ranges = {
            'non_spatial': {col: (X_non_spatial[col].min(), X_non_spatial[col].max()) 
                            for col in X_non_spatial.columns},
            'spatial': {col: (X_spatial[col].min(), X_spatial[col].max()) 
                       for col in X_spatial.columns}
        }
        
        # Prepare spatial weight matrix based on x, y coordinates
        W = self._create_spatial_weight_matrix(X_spatial)
        
        # Store dimensions for later use
        self.n_samples_data = len(y)
        self.n_non_spatial = X_non_spatial.shape[1]
        self.n_spatial = X_spatial.shape[1]
        
        # Store original feature names
        self.non_spatial_features = X_non_spatial.columns.tolist()
        self.spatial_features = X_spatial.columns.tolist()
        
        # Determine number of boroughs if borough_ids provided
        if borough_ids is not None:
            self.borough_ids = borough_ids
            self.n_boroughs = len(np.unique(borough_ids))
        else:
            self.borough_ids = None
            self.n_boroughs = 0
        
        # Build the model
        with pm.Model() as self.model:
            # Use much tighter priors when not scaling
            # Priors should be appropriate to the original data scale
            beta = pm.Normal('beta', mu=0, sigma=0.1, shape=self.n_non_spatial)
            
            # Fixed effects component (no scaling)
            fixed_effects = pm.math.dot(X_non_spatial_model, beta)
            
            # BART component for spatial features (x, y coordinates)
            if X_spatial.shape[1] > 0:  # Only if we have spatial features
                sigma_bart = pm.HalfNormal('sigma_bart', sigma=0.1)
                
                # Smaller length scale for unscaled data
                length_scale = pm.Gamma('length_scale', alpha=3, beta=3)
                
                # Use fewer components to prevent overfitting
                n_gp_components = min(5, self.n_spatial)
                gp_weights = pm.Normal('gp_weights', mu=0, sigma=0.05/length_scale, 
                                      shape=(n_gp_components, self.n_spatial))
                
                # Feature expansions on unscaled data
                f_raw = pm.math.dot(X_spatial_model, gp_weights.T)
                
                # Apply nonlinearity for tree-like behavior
                f_nonlinear = pm.math.tanh(f_raw)
                
                # Use much smaller scaling for the BART component
                f = sigma_bart * pm.math.sum(f_nonlinear, axis=1) / pt.sqrt(n_gp_components)
                
                # Apply mixing weight to BART component
                mix_bart = pm.Beta('mix_bart', alpha=1, beta=10)
                bart_effect = mix_bart * f
            else:
                bart_effect = 0
                
            # Smaller spatial component variance
            phi_sigma = pm.HalfNormal('phi_sigma', sigma=0.1)
            
            # Create proper spatial correlation structure if we have a spatial weight matrix
            if W is not None and not np.array_equal(W, np.eye(self.n_samples_data)):
                # Spatial autocorrelation parameter
                rho = pm.Uniform('rho', lower=-0.99, upper=0.99)
                
                # Calculate spatial effects with proper CAR structure
                I_minus_rho_W = pt.eye(self.n_samples_data) - rho * W
                Q = pt.dot(I_minus_rho_W.T, I_minus_rho_W) / phi_sigma**2
                
                # Spatial random effects with proper CAR structure
                phi = pm.MvNormal('phi', mu=0, tau=Q, shape=self.n_samples_data)
                
                # Apply mixing weight to spatial component
                mix_spatial = pm.Beta('mix_spatial', alpha=1, beta=10)
                spatial_effect = mix_spatial * phi
            else:
                spatial_effect = 0
                
            # Add borough-level random effects if borough_ids provided
            if self.borough_ids is not None and self.n_boroughs > 0:
                # Standard deviation for borough random effects
                sigma_borough = pm.HalfNormal('sigma_borough', sigma=0.1)
                
                # Borough-level random effects
                borough_effects = pm.Normal('borough_effects', mu=0, sigma=sigma_borough, 
                                           shape=self.n_boroughs)
                
                # Apply borough effects using borough_ids as indices
                borough_effect = borough_effects[self.borough_ids]
            else:
                borough_effect = 0
                
            # Combine all components
            log_rate = fixed_effects + spatial_effect + bart_effect + borough_effect
            
            # Tighter clipping
            log_rate = pm.math.clip(log_rate, -4, 4)
            
            mu = pm.math.exp(log_rate)
            
            # Negative Binomial likelihood parameters
            alpha = pm.Gamma('alpha', alpha=1, beta=0.1)  # Dispersion parameter
            
            # Likelihood
            y_obs = pm.NegativeBinomial('y_obs', mu=mu, alpha=alpha, observed=y)
        
        return self.model
    
    def fit(self, X_non_spatial, X_spatial, y, borough_ids=None):
        """
        Fit the model using MCMC sampling.
        
        Parameters:
        -----------
        X_non_spatial : DataFrame
            Non-spatial features
        X_spatial : DataFrame
            Spatial features
        y : Series
            Accident counts
        borough_ids : Series, optional
            Borough identifiers for each intersection
            
        Returns:
        --------
        self : NegativeBinomialBART
            Fitted model
        """
        # Build model if not already built
        if self.model is None:
            self.build_model(X_non_spatial, X_spatial, y, borough_ids)
        
        # Sample from the posterior
        with self.model:
            self.trace = pm.sample(
                draws=self.n_samples,
                tune=self.n_burn,
                chains=1,
                cores=2,
                return_inferencedata=True
            )
        
        return self
    
    def predict(self, X_non_spatial_new, X_spatial_new, borough_ids_new=None):
        """
        Predict accident counts for new data.
        
        Parameters:
        -----------
        X_non_spatial_new : DataFrame
            New non-spatial features
        X_spatial_new : DataFrame
            New spatial features
        borough_ids_new : Series, optional
            Borough identifiers for new intersections
            
        Returns:
        --------
        y_pred : ndarray
            Predicted accident counts
        """
        if self.trace is None:
            raise ValueError("Model must be fit before making predictions")
        
        # No need to scale the new data
        X_non_spatial_model = X_non_spatial_new.copy()
        X_spatial_model = X_spatial_new.copy()
        
        # Get posterior samples
        beta_samples = self.trace.posterior['beta'].values.reshape(
            -1, self.n_non_spatial)
        
        # Compute with unscaled data
        fixed_effects = np.dot(X_non_spatial_model, beta_samples.T)
        
        # Approximate spatial effects
        phi_samples = self.trace.posterior['phi'].values.reshape(-1, self.n_samples_data)
        # Use the mean spatial effect as an approximation for new locations
        phi_mean = np.mean(phi_samples, axis=0).mean()
        
        # Approximate BART component for new data
        f_sigma_samples = self.trace.posterior['sigma_bart'].values.flatten()
        gp_weights_samples = self.trace.posterior['gp_weights'].values.reshape(
            -1, 
            self.trace.posterior['gp_weights'].shape[-2], 
            self.trace.posterior['gp_weights'].shape[-1]
        )
        
        # Calculate f for each posterior sample
        n_posterior = len(f_sigma_samples)
        mu = np.zeros((X_non_spatial_new.shape[0], n_posterior))
        
        # Apply mixing weights from posterior
        mix_bart_samples = self.trace.posterior['mix_bart'].values.flatten()
        mix_spatial_samples = self.trace.posterior['mix_spatial'].values.flatten()
        
        for i in range(n_posterior):
            # Compute f for this posterior sample
            f_raw = np.dot(X_spatial_model, gp_weights_samples[i].T)
            
            # Apply nonlinearity as in the model
            f_nonlinear = np.tanh(f_raw)
            
            # Scale consistently with model building
            n_gp_components = f_nonlinear.shape[1]
            f = f_sigma_samples[i] * f_nonlinear.sum(axis=1) / np.sqrt(n_gp_components)
            
            # Apply mixing weights
            log_rate = fixed_effects[:, i] + mix_spatial_samples[i] * phi_mean + mix_bart_samples[i] * f
            
            # Use the SAME clipping range as in build_model
            log_rate = np.clip(log_rate, -4, 4)
            mu[:, i] = np.exp(log_rate)
        
        # Take the mean prediction across posterior samples
        y_pred = np.mean(mu, axis=1)
        
        return y_pred
    
    def get_feature_importance(self):
        """
        Calculate feature importance based on the posterior distribution of coefficients.
        
        Returns:
        --------
        importance_df : DataFrame
            Feature importance metrics
        """
        if self.trace is None:
            raise ValueError("Model must be fit before calculating feature importance")
        
        # Create a combined feature importance for both types of features
        feature_importance = []
        
        # Extract beta coefficients for non-spatial features
        if 'beta' in self.trace.posterior:
            beta_samples = self.trace.posterior['beta'].values.reshape(-1, self.n_non_spatial)
            
            # Calculate importance metrics for non-spatial features
            for i, feature in enumerate(self.non_spatial_features):
                importance = {
                    'feature': feature,
                    'type': 'non-spatial',
                    'mean': np.abs(beta_samples[:, i]).mean(),
                    'std': np.abs(beta_samples[:, i]).std(),
                    'prob_nonzero': np.mean(np.abs(beta_samples[:, i]) > 0.1)
                }
                feature_importance.append(importance)
        
        # For spatial features, we need to use gp_weights instead of sigma_bart
        if 'gp_weights' in self.trace.posterior:
            # Extract gp_weights which are the actual weights for spatial features
            gp_weights_samples = self.trace.posterior['gp_weights'].values
            
            # Reshape to get samples × components × features
            n_samples = gp_weights_samples.shape[0] * gp_weights_samples.shape[1]
            n_components = gp_weights_samples.shape[2]
            n_features = gp_weights_samples.shape[3]
            
            # Reshape to combine samples and chains
            gp_weights_flat = gp_weights_samples.reshape(-1, n_components, n_features)
            
            # Average importance across GP components
            feature_importance_values = np.abs(gp_weights_flat).mean(axis=1)  # Average across components
            
            # Calculate importance metrics for spatial features
            for i, feature in enumerate(self.spatial_features):
                if i < feature_importance_values.shape[1]:  # Ensure index is valid
                    importance = {
                        'feature': feature,
                        'type': 'spatial',
                        'mean': feature_importance_values[:, i].mean(),
                        'std': feature_importance_values[:, i].std(),
                        'prob_nonzero': np.mean(feature_importance_values[:, i] > 0.1)
                    }
                    feature_importance.append(importance)
        
        # Convert to DataFrame and sort
        importance_df = pd.DataFrame(feature_importance)
        if not importance_df.empty:
            importance_df = importance_df.sort_values('mean', ascending=False)
        
        return importance_df
    
    def plot_spatial_effects(self):
        """
        Plot the spatial random effects.
        
        Returns:
        --------
        fig : matplotlib Figure
            Plot of spatial effects
        """
        if self.trace is None:
            raise ValueError("Model must be fit before plotting spatial effects")
        
        # Extract spatial effects
        phi_samples = self.trace.posterior['phi'].values.reshape(-1, self.n_samples_data)
        phi_mean = np.mean(phi_samples, axis=0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # If we have x, y coordinates, we can plot spatial effects on a map
        if 'x' in self.spatial_features and 'y' in self.spatial_features:
            sc = ax.scatter(
                self.scaler_spatial.inverse_transform(
                    np.zeros((self.n_samples_data, self.n_spatial)))[:, 
                    self.spatial_features.index('x')],
                self.scaler_spatial.inverse_transform(
                    np.zeros((self.n_samples_data, self.n_spatial)))[:, 
                    self.spatial_features.index('y')],
                c=phi_mean,
                cmap='coolwarm',
                alpha=0.7,
                s=50
            )
            plt.colorbar(sc, ax=ax, label='Spatial effect')
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.set_title('Spatial Random Effects')
        else:
            # Simple histogram if no coordinates
            ax.hist(phi_mean, bins=30)
            ax.set_xlabel('Spatial effect')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Spatial Random Effects')
        
        return fig

def load_and_prepare_data():
    """
    Load and prepare data for modeling, using intersection coordinates for spatial modeling
    and borough as a grouping variable for random effects.
    """
    data = pd.read_csv("data/data_final.csv", sep=";")
    
    # Base features (non-spatial)
    feature_cols = ['ln_fi', 'ln_fri', 'ln_fli', 'ln_pi', 'traffic_10000', 'ln_cti', 'ln_cli', 'ln_cri',
                    'pi', 'fi', 'fli', 'fri', 'fti', 'cli', 'cri', 'cti', 'total_lane', 'avg_crossw', 
                    'tot_road_w', 'tot_crossw', 'commercial', 'number_of_', 'of_exclusi', 'curb_exten', 
                    'median', 'all_pedest', 'half_phase', 'new_half_r', 'any_ped_pr', 'ped_countd', 
                    'lt_restric', 'lt_prot_re', 'lt_protect', 'any_exclus', 'all_red_an', 'green_stra',
                    'parking', 'north_veh', 'north_ped', 'east_veh', 'east_ped', 'south_veh', 'south_ped', 
                    'west_veh', 'west_ped']

    # Check which columns actually exist in the data
    feature_cols = [col for col in feature_cols if col in data.columns]
    
    # Combine base features
    X_base = data[feature_cols].fillna(0)
    
    # Apply quadratic transformation to specified variables
    for col in ['ln_cti', 'ln_cri', 'ln_cli', 'ln_pi', 'ln_fri', 'ln_fli', 'ln_fi']:
        if col in X_base.columns:
            X_base[f'{col}_squared'] = X_base[col] ** 2

    # Use x and y coordinates for spatial modeling at intersection level
    if 'x' in data.columns and 'y' in data.columns:
        X_spatial = data[['x', 'y']].copy()
    else:
        print("Coordinates 'x' and 'y' not found in the data, using empty spatial features")
        X_spatial = pd.DataFrame(index=data.index)
    
    # Extract borough information for random effects
    if 'borough' in data.columns:
        borough_ids = data['borough'].astype('category').cat.codes
        borough_names = data['borough'].astype('category').cat.categories
    else:
        print("Borough information not found, using dummy borough IDs")
        borough_ids = np.zeros(len(data))
        borough_names = ["Unknown"]
    
    return {
        'X_non_spatial': X_base,
        'X_spatial': X_spatial,
        'y': data['acc'], 
        'int_no': data['int_no'],
        'borough_ids': borough_ids,
        'borough_names': borough_names
    }

def select_features_with_lasso(X, y):
    """
    Select features using Lasso regression.
    
    Parameters:
    -----------
    X : DataFrame
        Features
    y : Series
        Target variable
        
    Returns:
    --------
    selected_features : list
        List of selected feature names
    """
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

def run_cross_validation(X_non_spatial, X_spatial, y, int_no, borough_ids, k=5):
    """
    Run k-fold cross-validation for the NegativeBinomialBART model.
    
    Parameters:
    -----------
    X_non_spatial : DataFrame
        Non-spatial features
    X_spatial : DataFrame
        Spatial features (x, y coordinates)
    y : Series
        Accident counts
    int_no : Series
        Intersection IDs
    borough_ids : Series
        Borough identifiers for each intersection
    k : int
        Number of folds
        
    Returns:
    --------
    cv_results : dict
        Cross-validation results
    """
    # Create fold indices
    np.random.seed(42)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Initialize lists to store results
    all_metrics = pd.DataFrame(columns=['fold', 'mae', 'mse', 'rmse'])
    combined_results = pd.DataFrame()
    feature_importance_results = []
    
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
        
        # Prepare spatial features
        X_train_spatial = X_spatial.iloc[train_idx]
        X_test_spatial = X_spatial.iloc[test_idx]
        
        # Fix borough IDs to ensure consistent mapping
        # This is the key fix: we need to remap the borough IDs for each fold
        train_boroughs = borough_ids.iloc[train_idx].unique()
        borough_id_map = {old_id: new_id for new_id, old_id in enumerate(train_boroughs)}
        
        # Remap the borough IDs for train and test
        borough_ids_train_remapped = borough_ids.iloc[train_idx].map(borough_id_map)
        
        # For test set, handle boroughs not seen in training
        # Map unseen boroughs to a special value (-1) and we'll handle it in prediction
        borough_ids_test_remapped = borough_ids.iloc[test_idx].map(
            lambda x: borough_id_map.get(x, -1)
        )
        
        # Initialize model with appropriate parameters
        model = NegativeBinomialBART(
            n_trees=50,
            max_depth=2,
            n_samples=3,  # Small number for testing
            n_burn=3      # Small number for testing
        )
        
        # Fit the model with remapped borough random effects
        try:
            # Use the fit method with remapped borough IDs
            model.fit(
                X_train_selected, 
                X_train_spatial,
                y_train,
                borough_ids=borough_ids_train_remapped
            )
            
            # Store the borough mapping for prediction
            model.borough_id_map = borough_id_map
            
            # Modify the predict method call to handle unseen boroughs
            y_pred = predict_with_borough_handling(
                model,
                X_test_selected,
                X_test_spatial,
                borough_ids_test_remapped
            )
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Add metrics to dataframe
            all_metrics = pd.concat([all_metrics, pd.DataFrame({
                'fold': [i],
                'mae': [mae],
                'mse': [mse],
                'rmse': [rmse]
            })], ignore_index=True)
            
            # Store results for this fold
            fold_results = pd.DataFrame({
                'int_no': int_no.iloc[test_idx],
                'true_counts': y_test,
                'pred_counts': y_pred
            })
            combined_results = pd.concat([combined_results, fold_results], ignore_index=True)
            
            # Get feature importance
            feature_importance = model.get_feature_importance()
            feature_importance['fold'] = i
            feature_importance_results.append(feature_importance)
        
            # Display fold results
            print(f"Fold {i} - MAE: {mae}")
            print(f"Fold {i} - MSE: {mse}")
            print(f"Fold {i} - RMSE: {rmse}")
        
        except Exception as e:
            print(f"Error in fold {i}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            continue
    
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
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Save combined results
    combined_results.to_csv(os.path.join("results", "nb_bart_combined_cv_results.csv"), index=False)
    
    # Create and save combined intersection ranking
    combined_rankings = combined_results[['int_no', 'pred_counts']].sort_values(by='pred_counts', ascending=False)
    combined_rankings['ranking'] = range(1, len(combined_rankings) + 1)
    combined_rankings.to_csv(os.path.join("results", "nb_bart_combined_intersections_cv_ranking.csv"), index=False)
    
    # Combine feature importance from all folds
    feature_importance_df = pd.concat(feature_importance_results, ignore_index=True)
    feature_importance_summary = feature_importance_df.groupby('feature').agg({
        'mean': 'mean',
        'std': 'mean',
        'prob_nonzero': 'mean'
    }).sort_values('mean', ascending=False)
    
    feature_importance_summary.to_csv(os.path.join("results", "nb_bart_feature_importance.csv"))
    
    return {
        'metrics_per_fold': all_metrics,
        'average_metrics': avg_metrics,
        'combined_results': combined_results,
        'feature_importance': feature_importance_summary
    }

# New helper function to handle predictions with unmapped boroughs
def predict_with_borough_handling(model, X_non_spatial_new, X_spatial_new, borough_ids_new):
    """
    Wrapper around the predict method to handle unseen boroughs.
    
    Parameters:
    -----------
    model : NegativeBinomialBART
        Fitted model
    X_non_spatial_new : DataFrame
        New non-spatial features
    X_spatial_new : DataFrame
        New spatial features
    borough_ids_new : Series
        Borough identifiers for new intersections, may include -1 for unseen boroughs
        
    Returns:
    --------
    y_pred : ndarray
        Predicted accident counts
    """
    # Replace unseen boroughs (marked as -1) with the mean borough effect
    # The actual implementation depends on how you want to handle unseen boroughs
    
    # One approach: replace -1 values with a random valid borough ID
    if -1 in borough_ids_new.values:
        # Get a valid borough ID to use as replacement (e.g., the most common)
        if hasattr(model, 'borough_ids') and model.borough_ids is not None:
            valid_borough = model.borough_ids.mode().iloc[0]
            # Replace -1 with valid borough ID
            borough_ids_new = borough_ids_new.replace(-1, valid_borough)
        else:
            # If no borough information, just set to 0
            borough_ids_new = borough_ids_new.replace(-1, 0)
    
    # Make prediction with handled borough IDs
    return model.predict(X_non_spatial_new, X_spatial_new, borough_ids_new=borough_ids_new)

def main():
    """
    Main function to run the analysis.
    
    Returns:
    --------
    cv_results : dict
        Cross-validation results
    """
    print("Loading and preparing data...")
    data = load_and_prepare_data()
    
    print("\nRunning cross-validation...")
    cv_results = run_cross_validation(
        data['X_non_spatial'],
        data['X_spatial'],
        data['y'],
        data['int_no'],
        data['borough_ids'],
        k=5
    )
    
    print("\nAnalysis complete. Results saved to the 'results' directory.")
    return cv_results

if __name__ == "__main__":
    results = main()