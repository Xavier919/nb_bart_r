import pandas as pd
import numpy as np
import os
import sys
import time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families.family import NegativeBinomial
from sklearn.linear_model import LassoCV
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

#####################################
# Data Processing Functions
#####################################

def load_and_prepare_data():
    """Load and prepare the Montreal intersection data for modeling"""
    data = pd.read_csv("data/data_final.csv", sep=";")
    
    # Replace NaN values in ln_distdt with 0
    data['ln_distdt'] = data['ln_distdt'].fillna(0)
    
    # Filter where pi != 0
    data = data[data['pi'] != 0]
    
    # First correct borough names with encoding issues
    borough_corrections = {
        '?le-Bizard-Sainte-GeneviÞve': 'Île-Bizard-Sainte-Geneviève',
        'C¶te-Saint-Luc': 'Côte-Saint-Luc',
        'C¶te-des-Neiges-Notre-Dame-de-Graces': 'Côte-des-Neiges-Notre-Dame-de-Grâce',
        'MontrÚal-Est': 'Montréal-Est',
        'MontrÚal-Nord': 'Montréal-Nord',
        'Pointe-aux-Trembles-RiviÞres-des-Prairies': 'Rivière-des-Prairies-Pointe-aux-Trembles',
        'St-LÚonard': 'Saint-Léonard'
    }
    
    # Then group boroughs into zones
    borough_zones = {
        # Zone ouest
        'Kirkland': 'Zone ouest',
        'Beaconsfield': 'Zone ouest',
        'Île-Bizard-Sainte-Geneviève': 'Zone ouest',
        'Pierrefonds-Roxboro': 'Zone ouest',
        'Dollard-des-Ormeaux': 'Zone ouest',
        'Dorval': 'Zone ouest',
        
        # Zone est
        'Rivière-des-Prairies-Pointe-aux-Trembles': 'Zone est',
        'Montréal-Est': 'Zone est',
        'Anjou': 'Zone est',
        
        # Zone centre
        'Outremont': 'Zone centre',
        'Mont-Royal': 'Zone centre',
        
        # Zone sud
        'Sud-Ouest': 'Zone sud',
        'Côte-Saint-Luc': 'Zone sud',
        'Verdun': 'Zone sud',
        'Lasalle': 'Zone sud',
        'Lachine': 'Zone sud',
        
        # Zone centre-sud
        'Côte-des-Neiges-Notre-Dame-de-Grâce': 'Zone centre-sud',
        'Hampstead': 'Zone centre-sud',
        'Westmount': 'Zone centre-sud'
    }
    
    # Apply corrections if borough column exists
    if 'borough' in data.columns:
        # First fix encoding issues
        data['borough'] = data['borough'].replace(borough_corrections)
        # Then group into zones
        data['borough'] = data['borough'].replace(borough_zones)
    
    # Include existing spatial columns
    spatial_cols = ['x', 'y']
    
    # Base features (non-spatial)
    feature_cols = ['ln_fi', 'ln_fri', 'ln_fli', 'ln_pi', 'ln_cti', 'ln_cli', 'ln_cri', 'ln_distdt',
                    'fi', 'fri', 'fli', 'pi', 'cti', 'cli', 'cri', 'distdt',
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
    
    # Create full dataset
    full_data = pd.concat([
        data[['int_no', 'acc', 'pi', 'borough']], 
        X_base, 
        X_spatial
    ], axis=1)
    
    return full_data


def create_spatial_weights(data, k_neighbors=10):
    """
    Create spatial weights based on k-nearest neighbors.
    Returns a normalized spatial weights matrix.
    """
    # Extract coordinates
    coords = data[['x', 'y']].values
    n = len(coords)
    
    # Create distance matrix
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # Create weight matrix
    W = np.zeros((n, n))
    
    # Create weights that decay with distance
    for i in range(n):
        for j, idx in enumerate(indices[i][1:]):  # skip the first which is self
            # Inverse distance weighting
            W[i, idx] = 1.0 / max(distances[i, j+1], 0.0001)  # Avoid division by zero
    
    # Row-normalize the weights
    row_sums = W.sum(axis=1)
    W = W / row_sums[:, np.newaxis]
    
    # Replace NaNs with zeros
    W = np.nan_to_num(W)
    
    return W


def calculate_spatial_lag(data, variable, W):
    """
    Calculate the spatial lag of a variable using weights matrix W.
    """
    return W @ data[variable].values


def select_features_with_lasso(X, y, max_features=20):
    """
    Select most important features using Lasso regression
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run Lasso with cross-validation
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_scaled, y)
    
    # Get features selected by Lasso
    coef = pd.Series(lasso.coef_, index=X.columns)
    selected_features = coef[coef != 0].abs().sort_values(ascending=False)
    
    # Limit to max number of features
    if len(selected_features) > max_features:
        selected_features = selected_features.iloc[:max_features]
    
    print(f"Selected {len(selected_features)} features")
    return selected_features.index.tolist()


#####################################
# Spatial Negative Binomial Model
#####################################

class SpatialMixedNegativeBinomial:
    """
    A spatial mixed-effects negative binomial model.
    Uses a two-stage approach combining statsmodels GLM with random effects.
    """
    
    def __init__(self, data=None, formula=None, random_effect=None, exposure=None, spatial_vars=None):
        """
        Initialize the model.
        
        Args:
            data: DataFrame containing the data
            formula: Model formula as a string
            random_effect: Column name for random effects
            exposure: Column name for exposure variable (not used for modeling counts directly)
            spatial_vars: List of spatial variables (x, y)
        """
        self.data = data
        self.formula = formula
        self.random_effect = random_effect
        self.exposure = None  # Not using exposure for modeling counts
        self.spatial_vars = spatial_vars
        self.model = None
        self.results = None
        self.W = None
        self.random_effects = None
        self.random_effects_variance = None
        
    def add_spatial_lags(self, data, variables, W=None):
        """
        Add spatial lags of specified variables to the dataset.
        
        Args:
            data: DataFrame to modify
            variables: List of variables to create spatial lags for
            W: Weight matrix (if None, will be calculated)
            
        Returns:
            Modified DataFrame with spatial lag variables
        """
        data_copy = data.copy()
        
        # Create weight matrix if not provided
        if W is None:
            W = create_spatial_weights(data_copy[self.spatial_vars])
            self.W = W
        
        # Create spatial lags
        for var in variables:
            if var in data_copy.columns:
                data_copy[f'sp_lag_{var}'] = calculate_spatial_lag(data_copy, var, W)
        
        return data_copy
    
    def fit(self, add_spatial_lags_for=None):
        """
        Fit the model to the data using a two-stage approach for mixed effects.
        
        Args:
            add_spatial_lags_for: List of variables to create spatial lags for before fitting
            
        Returns:
            Model fit results
        """
        data_to_use = self.data.copy()
        
        # Add spatial lags if requested
        if add_spatial_lags_for:
            data_to_use = self.add_spatial_lags(data_to_use, add_spatial_lags_for)
        
        # No offset term - we're modeling counts directly
        full_formula = self.formula
        
        # Step 1: Estimate random effects using MixedLM
        if self.random_effect:
            print(f"Estimating random effects for {self.random_effect}...")
            
            # Extract fixed effects part of the formula (remove response variable)
            fixed_effects_formula = full_formula.split('~')[1].strip()
            
            # Create design matrices for fixed and random effects
            import patsy
            y, X = patsy.dmatrices(full_formula, data_to_use, return_type='dataframe')
            
            # Create groups for random effects
            groups = data_to_use[self.random_effect]
            
            # Fit mixed linear model to estimate random effects
            # We use log(acc + 1) as response for the random effects estimation
            data_to_use['log_acc'] = np.log1p(data_to_use['acc'])
            re_formula = f"log_acc ~ {fixed_effects_formula}"
            y_re, X_re = patsy.dmatrices(re_formula, data_to_use, return_type='dataframe')
            
            # Create exog_re matrix for random effects (intercept only)
            exog_re = np.ones((len(data_to_use), 1))
            
            mixed_model = sm.MixedLM(
                endog=y_re,
                exog=X_re,
                groups=groups,
                exog_re=exog_re  # Use exog_re instead of re_formula
            )
            
            mixed_results = mixed_model.fit()
            print("Random effects model summary:")
            print(mixed_results.summary())
            
            # Extract random effects
            self.random_effects = mixed_results.random_effects
            self.random_effects_variance = mixed_results.cov_re.iloc[0, 0]
            
            # Create random effect adjustment for the negative binomial model
            data_to_use['random_effect_adj'] = 0.0
            
            for group in self.random_effects:
                mask = data_to_use[self.random_effect] == group
                data_to_use.loc[mask, 'random_effect_adj'] = self.random_effects[group].iloc[0]
            
            # Remove the random effect term from the formula for the NB model
            # since we're now accounting for it via the adjustment
            import re
            pattern = r'\+\s*C\(' + self.random_effect + r'\)'
            full_formula = re.sub(pattern, '', full_formula)
            
            # Instead of using offset, we'll adjust the response variable
            # We'll create an adjusted count based on the random effects
            data_to_use['adj_acc'] = data_to_use['acc'] * np.exp(-data_to_use['random_effect_adj'])
            
            # Update formula to use adjusted response
            full_formula = full_formula.replace('acc ~', 'adj_acc ~')
            
            print(f"Adjusted formula with random effects: {full_formula}")
        
        # Step 2: Fit negative binomial model with random effects adjustment
        print(f"Fitting negative binomial model with formula: {full_formula}")
        
        nb_model = smf.glm(
            formula=full_formula,
            data=data_to_use,
            family=sm.families.NegativeBinomial(alpha=1.0)
        )
        
        self.model = nb_model
        self.results = nb_model.fit()
        
        # Store the data with random effect adjustments for prediction
        self.data_with_re = data_to_use
        
        return self.results
    
    def predict(self, newdata=None, add_spatial_lags=True):
        """
        Make predictions from the fitted model, accounting for random effects.
        
        Args:
            newdata: New data for prediction (if None, uses training data)
            add_spatial_lags: Whether to add spatial lags to newdata
            
        Returns:
            Array of predicted values
        """
        if self.results is None:
            raise ValueError("Model must be fitted before prediction")
        
        if newdata is None:
            # Use the data with random effects already computed
            if hasattr(self, 'data_with_re'):
                newdata = self.data_with_re.copy()
            else:
                newdata = self.data.copy()
        else:
            # For new data, we need to add random effects
            newdata = newdata.copy()
            if self.random_effect and hasattr(self, 'random_effects'):
                newdata['random_effect_adj'] = 0.0
                
                # Apply random effects from training data to matching groups in new data
                for group in self.random_effects:
                    mask = newdata[self.random_effect] == group
                    if mask.any():
                        newdata.loc[mask, 'random_effect_adj'] = self.random_effects[group].iloc[0]
                    else:
                        # For groups not seen in training, use 0 (no random effect)
                        # Alternative: could use average of random effects or draw from distribution
                        pass
        
        # Add spatial lags if the model used them
        if add_spatial_lags and 'sp_lag_' in str(self.formula):
            # Get list of variables with spatial lags from formula
            import re
            sp_lag_vars = re.findall(r'sp_lag_(\w+)', self.formula)
            newdata = self.add_spatial_lags(newdata, sp_lag_vars, self.W)
        
        # Predict counts and adjust for random effects
        if 'adj_acc' in str(self.formula):
            # If we used adjusted counts, we need to reverse the adjustment
            raw_predictions = self.results.predict(newdata)
            predictions = raw_predictions * np.exp(newdata['random_effect_adj'])
        else:
            predictions = self.results.predict(newdata)
        
        return predictions
    
    def summary(self):
        """Print model summary including random effects information"""
        if self.results is None:
            print("Model has not been fitted yet")
            return
        
        print(self.results.summary())
        
        # Print random effects information if available
        if hasattr(self, 'random_effects') and self.random_effects is not None:
            print("\nRandom Effects Summary:")
            print(f"Random Effects Variance: {self.random_effects_variance:.4f}")
            
            # Create a DataFrame of random effects for better display
            re_df = pd.DataFrame({
                'Group': list(self.random_effects.keys()),
                'Random Effect': [self.random_effects[g].iloc[0] for g in self.random_effects]
            })
            re_df = re_df.sort_values('Random Effect', ascending=False)
            print("\nRandom Effects by Group:")
            print(re_df)
        
        # Print performance metrics
        y_true = self.data['acc']
        y_pred = self.predict()
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        print(f"\nPerformance Metrics:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        return {'mae': mae, 'rmse': rmse}


#####################################
# Cross-Validation and Evaluation
#####################################

def run_cross_validation(data, formula, k=5, random_effect=None, exposure=None):
    """
    Run k-fold cross-validation on the spatial mixed negative binomial model.
    
    Args:
        data: Full dataset
        formula: Model formula
        k: Number of folds for cross-validation
        random_effect: Column name for random effects
        exposure: Column name for exposure variable (not used when modeling counts)
        
    Returns:
        Cross-validation results
    """
    np.random.seed(42)
    
    # Create KFold object
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Initialize lists to store results
    fold_results = []
    test_actual = []
    test_predicted = []
    
    # For each fold
    for i, (train_idx, test_idx) in enumerate(kf.split(data), 1):
        print(f"\n========== Fold {i}/{k} ==========")
        
        # Split data into train and testf
        train_data = data.iloc[train_idx].copy()
        test_data = data.iloc[test_idx].copy()
        
        # Create and fit model
        model = SpatialMixedNegativeBinomial(
            data=train_data,
            formula=formula,
            random_effect=random_effect,
            exposure=None,  # Not using exposure
            spatial_vars=['x', 'y']
        )
        
        # Get variables that might need spatial lags from formula
        import re
        potential_vars = re.findall(r'(?<!\w)(\w+)(?!\w)', formula)
        spatial_lag_vars = [var for var in potential_vars if var in train_data.columns 
                          and var not in ['acc', 'pi', 'int_no', 'borough']]
        
        # Fit model
        results = model.fit(add_spatial_lags_for=spatial_lag_vars)
        
        # Make predictions
        train_preds = model.predict(train_data)
        test_preds = model.predict(test_data)
        
        # Calculate metrics
        train_mae = mean_absolute_error(train_data['acc'], train_preds)
        train_mse = mean_squared_error(train_data['acc'], train_preds)
        train_rmse = np.sqrt(train_mse)
        
        test_mae = mean_absolute_error(test_data['acc'], test_preds)
        test_mse = mean_squared_error(test_data['acc'], test_preds)
        test_rmse = np.sqrt(test_mse)
        
        print(f"Train MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}")
        
        # Store results
        fold_results.append({
            'fold': i,
            'train_mae': train_mae,
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'test_mae': test_mae,
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'model': model
        })
        
        # Collect predictions
        test_actual.extend(test_data['acc'].values)
        test_predicted.extend(test_preds)
    
    # Calculate overall CV metrics
    overall_mae = mean_absolute_error(test_actual, test_predicted)
    overall_mse = mean_squared_error(test_actual, test_predicted)
    overall_rmse = np.sqrt(overall_mse)
    
    print(f"\n========== Overall CV Results ==========")
    print(f"Overall MAE: {overall_mae:.4f}")
    print(f"Overall MSE: {overall_mse:.4f}")
    print(f"Overall RMSE: {overall_rmse:.4f}")
    
    return {
        'fold_results': fold_results,
        'overall_mae': overall_mae,
        'overall_mse': overall_mse,
        'overall_rmse': overall_rmse,
        'test_actual': test_actual,
        'test_predicted': test_predicted
    }


#####################################
# Visualization Functions
#####################################

def plot_spatial_correlation(data, W, results_path="results"):
    """
    Create a network visualization showing spatial relationships.
    
    Args:
        data: DataFrame with spatial coordinates
        W: Spatial weights matrix
        results_path: Where to save the visualization
    """
    os.makedirs(results_path, exist_ok=True)
    
    # Get strong connections (higher than threshold)
    threshold = np.percentile(W[W > 0], 90)  # Top 10% of non-zero weights
    
    # Create edge list with weights above threshold
    edges = []
    for i in range(W.shape[0]):
        for j in range(W.shape[0]):
            if W[i, j] > threshold:
                edges.append((i, j, W[i, j]))
    
    # Set node positions based on coordinates
    pos = {i: (data['x'].iloc[i], data['y'].iloc[i]) for i in range(len(data))}
    
    # Create a simple map visualization
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Draw a scatter plot of the intersections in grey with lower alpha
    ax.scatter(data['x'], data['y'], s=10, c='grey', alpha=0.5, label='Intersections')
    
    # Draw lines between correlated intersections
    # Sort edges by weight to draw strongest correlations last (on top)
    sorted_edges = sorted(edges, key=lambda x: x[2])
    
    # Create a colormap - yellow to red
    cmap = plt.cm.YlOrRd
    max_weight = max([e[2] for e in sorted_edges])
    min_weight = min([e[2] for e in sorted_edges])
    
    # Draw edges with thinner width
    for u, v, w in sorted_edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        # Normalize weight for color mapping
        norm_weight = w / max_weight
        ax.plot([x1, x2], [y1, y2], '-', 
                 linewidth=2,  # Reduced multiplier
                 alpha=0.5,
                 color=cmap(norm_weight))
    
    # Add a colorbar legend - specify the axis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Correlation Strength')
    
    # Add a legend for the intersections
    ax.legend(loc='upper right')
    
    ax.set_title('Spatial Correlation Map (Yellow to Red = Weak to Strong Correlation)')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(alpha=0.3)
    plt.savefig(os.path.join(results_path, 'spatial_correlation_map.png'), dpi=300)
    plt.close()


def plot_predicted_vs_actual(actual, predicted, results_path="results"):
    """
    Create scatter plots comparing predicted vs actual accident counts.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        results_path: Path to save the visualization
    """
    os.makedirs(results_path, exist_ok=True)
    
    # Create DataFrame
    results_df = pd.DataFrame({'actual': actual, 'predicted': predicted})
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(results_df['actual'], results_df['predicted'], 
                alpha=0.6, edgecolors='k', linewidths=0.5)
    
    # Add perfect prediction line
    max_val = max(results_df['actual'].max(), results_df['predicted'].max()) * 1.1
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        results_df['actual'], results_df['predicted'])
    plt.plot(
        [0, max_val], 
        [intercept, intercept + slope * max_val], 
        'b-', 
        label=f'Regression line (R² = {r_value**2:.3f})'
    )
    
    # Add labels and title
    plt.xlabel('Actual Accident Count')
    plt.ylabel('Predicted Accident Count')
    plt.title('Predicted vs Actual Accident Counts')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Set equal aspect ratio
    plt.axis('equal')
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'predicted_vs_actual.png'), dpi=300)
    plt.close()
    
    # Create residual plot
    plt.figure(figsize=(10, 8))
    residuals = results_df['predicted'] - results_df['actual']
    plt.scatter(results_df['actual'], residuals, alpha=0.6, edgecolors='k', linewidths=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    
    # Add labels and title
    plt.xlabel('Actual Accident Count')
    plt.ylabel('Residual (Predicted - Actual)')
    plt.title('Residual Plot')
    plt.grid(alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'residual_plot.png'), dpi=300)
    plt.close()


def plot_accident_heatmap(data, results_path="results"):
    """
    Create a heatmap visualization of predicted accidents using x-y coordinates.
    
    Args:
        data: DataFrame with x, y coordinates and accident counts/predictions
        results_path: Path to save the visualization
    """
    os.makedirs(results_path, exist_ok=True)
    
    plt.figure(figsize=(14, 12))
    
    # Create scatter plot with points colored by accident count
    scatter = plt.scatter(
        data['x'], 
        data['y'],
        c=data['predicted'],
        cmap='YlOrRd',
        alpha=0.7,
        s=30,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Predicted Accident Count')
    
    # Add title and labels
    plt.title('Spatial Distribution of Predicted Accidents')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # Add grid for reference
    plt.grid(alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'accident_prediction_heatmap.png'), dpi=300)
    plt.close()


def plot_model_coefficients(model_results, results_path="results"):
    """
    Create a clear, formatted table of model coefficients and save to CSV.
    Also creates a coefficient plot with confidence intervals.
    
    Args:
        model_results: Model results object from statsmodels
        results_path: Path to save results
    """
    if model_results is None:
        print("No model results available")
        return
    
    # Create coefficient table
    coef_df = pd.DataFrame({
        'Estimate': model_results.params,
        'Std. Error': model_results.bse,
        'Lower 95% CI': model_results.conf_int()[0],
        'Upper 95% CI': model_results.conf_int()[1],
        'P-Value': model_results.pvalues
    })
    
    # Add a column for significance
    coef_df['Significant'] = coef_df['P-Value'] < 0.05
    
    # Remove intercept for plotting
    if 'Intercept' in coef_df.index:
        plot_df = coef_df.drop('Intercept')
    else:
        plot_df = coef_df.copy()
    
    # Sort by absolute effect size
    plot_df = plot_df.sort_values(by='Estimate', key=abs, ascending=False)
    
    # Print nicely formatted table to console
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.width', 120)      # Wider display
    pd.set_option('display.precision', 4)    # Show 4 decimal places
    
    print("\n=== Model Coefficients ===")
    print(coef_df[['Estimate', 'Std. Error', 'Lower 95% CI', 'Upper 95% CI', 'Significant']])
    
    # Save to CSV
    os.makedirs(results_path, exist_ok=True)
    coef_df.to_csv(os.path.join(results_path, "model_coefficients.csv"))
    print(f"Coefficient table saved to {os.path.join(results_path, 'model_coefficients.csv')}")
    
    # Create coefficient plot
    plt.figure(figsize=(12, max(8, len(plot_df) * 0.3)))
    
    # Plot coefficients and confidence intervals
    y_pos = np.arange(len(plot_df))
    
    # Plot each point separately with appropriate color based on significance
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        color = 'darkred' if row['Significant'] else 'darkblue'
        plt.errorbar(
            x=row['Estimate'],
            y=i,
            xerr=[[row['Estimate'] - row['Lower 95% CI']], 
                  [row['Upper 95% CI'] - row['Estimate']]],
            fmt='o',
            capsize=5,
            color=color,
            alpha=0.7
        )
    
    # Add vertical line at zero
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.yticks(y_pos, plot_df.index)
    plt.xlabel('Coefficient Estimate')
    plt.title('Model Coefficients with 95% Confidence Intervals')
    plt.grid(axis='x', alpha=0.3)
    
    # Add legend
    plt.plot([], [], 'o', color='darkred', label='Significant (p<0.05)')
    plt.plot([], [], 'o', color='darkblue', label='Not significant')
    plt.legend(loc='best')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "coefficient_plot.png"), dpi=300)
    print(f"Coefficient plot saved to {os.path.join(results_path, 'coefficient_plot.png')}")


#####################################
# Main Function
#####################################

def main():
    """Main function to run the entire workflow"""
    # Load and prepare data
    print("Loading and preparing data...")
    data = load_and_prepare_data()
    
    # Define non-feature columns
    non_feature_cols = ['int_no', 'acc', 'pi', 'borough']
    
    # Get features
    feature_cols = [col for col in data.columns if col not in non_feature_cols]
    X = data[feature_cols]
    y = data['acc']
    
    selected_features = select_features_with_lasso(X, y, max_features=40)
    
    # Create spatial weights matrix
    print("\nCreating spatial weights matrix...")
    W = create_spatial_weights(data[['x', 'y']])
    
    # Build formula with selected features
    formula_terms = []
    for feature in selected_features:
        formula_terms.append(feature)
    
    # Add borough as a categorical variable
    if 'borough' in data.columns:
        formula_terms.append("C(borough)")
    
    # Create formula string
    formula = "acc ~ " + " + ".join(formula_terms)
    
    # First run full model on entire dataset
    print("\n========== Full Model on Entire Dataset ==========")
    model = SpatialMixedNegativeBinomial(
        data=data,
        formula=formula,
        random_effect='borough' if 'borough' in data.columns else None,
        exposure=None,
        spatial_vars=['x', 'y']
    )
    
    # Add spatial lags for important variables
    spatial_lag_vars = ['acc', 'ln_pi', 'ln_cti']
    full_results = model.fit(add_spatial_lags_for=spatial_lag_vars)
    
    # Print model summary
    model.summary()
    
    # Plot model coefficients
    plot_model_coefficients(model.results, results_path="results")
    
    # Create spatial correlation visualization
    plot_spatial_correlation(data, W)
    
    # Run cross-validation
    print("\n========== Cross-Validation ==========")
    cv_results = run_cross_validation(
        data=data,
        formula=formula,
        k=5,
        random_effect='borough' if 'borough' in data.columns else None,
        exposure=None
    )
    
    # Plot cross-validation results
    plot_predicted_vs_actual(
        cv_results['test_actual'], 
        cv_results['test_predicted'],
        results_path="results/cv"
    )
    
    # Create DataFrame with CV results
    cv_data = data.copy()
    # Map predictions back to original data
    cv_pred_map = {}
    for i, idx in enumerate(data.index):
        if i < len(cv_results['test_predicted']):
            cv_pred_map[idx] = cv_results['test_predicted'][i]
    
    cv_data['predicted'] = cv_data.index.map(cv_pred_map)
    cv_data = cv_data.dropna(subset=['predicted'])
    
    # Create heatmap from CV results
    plot_accident_heatmap(cv_data, results_path="results/cv")
    
    print("\nProcessing complete. Results saved to 'results' directory.")
    return {
        'full_model': model,
        'cv_results': cv_results
    }


# Run the main script when this file is executed
if __name__ == "__main__":
    main()
