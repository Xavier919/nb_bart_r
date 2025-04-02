#install inla package

# Install CARBayes package first
library(spdep)
library(dplyr)
library(ggplot2)
library(Matrix)
library(FNN)
library(caret)
library(glmnet)
library(R6)
library(CARBayes)
library(MASS)
library(conflicted)
library(INLA)

# Resolve the select conflict explicitly
conflict_prefer("select", "dplyr")
cat("Using dplyr::select:", identical(select, dplyr::select), "\n")


#####################################
# Data Processing Functions
#####################################

load_and_prepare_data <- function() {
  # Load and prepare the Montreal intersection data for modeling
  data <- read.csv("data/data_final.csv", sep = ";")
  
  # Replace NaN values in ln_distdt with 0
  data$ln_distdt[is.na(data$ln_distdt)] <- 0
  
  # Filter where pi != 0
  data <- data[data$pi != 0, ]
  
  # First correct borough names with encoding issues
  borough_corrections <- list(
    '?le-Bizard-Sainte-GeneviÞve' = 'Île-Bizard-Sainte-Geneviève',
    'C¶te-Saint-Luc' = 'Côte-Saint-Luc',
    'C¶te-des-Neiges-Notre-Dame-de-Graces' = 'Côte-des-Neiges-Notre-Dame-de-Grâce',
    'MontrÚal-Est' = 'Montréal-Est',
    'MontrÚal-Nord' = 'Montréal-Nord',
    'Pointe-aux-Trembles-RiviÞres-des-Prairies' = 'Rivière-des-Prairies-Pointe-aux-Trembles',
    'St-LÚonard' = 'Saint-Léonard'
  )
  
  # Then group boroughs into zones
  borough_zones <- list(
    # Zone ouest
    'Kirkland' = 'Zone ouest',
    'Beaconsfield' = 'Zone ouest',
    'Île-Bizard-Sainte-Geneviève' = 'Zone ouest',
    'Pierrefonds-Roxboro' = 'Zone ouest',
    'Dollard-des-Ormeaux' = 'Zone ouest',
    'Dorval' = 'Zone ouest',
    
    # Zone est
    'Rivière-des-Prairies-Pointe-aux-Trembles' = 'Zone est',
    'Montréal-Est' = 'Zone est',
    'Anjou' = 'Zone est',
    
    # Zone centre
    'Outremont' = 'Zone centre',
    'Mont-Royal' = 'Zone centre',
    
    # Zone sud
    'Sud-Ouest' = 'Zone sud',
    'Côte-Saint-Luc' = 'Zone sud',
    'Verdun' = 'Zone sud',
    'Lasalle' = 'Zone sud',
    'Lachine' = 'Zone sud',
    
    # Zone centre-sud
    'Côte-des-Neiges-Notre-Dame-de-Grâce' = 'Zone centre-sud',
    'Hampstead' = 'Zone centre-sud',
    'Westmount' = 'Zone centre-sud'
  )
  
  # Apply corrections if borough column exists
  if ("borough" %in% colnames(data)) {
    # First fix encoding issues
    for (old_name in names(borough_corrections)) {
      data$borough[data$borough == old_name] <- borough_corrections[[old_name]]
    }
    
    # Then group into zones
    for (borough in names(borough_zones)) {
      data$borough[data$borough == borough] <- borough_zones[[borough]]
    }
  }
  
  # Include existing spatial columns
  spatial_cols <- c('x', 'y')
  
  # Base features (non-spatial)
  feature_cols <- c('ln_fi', 'ln_fri', 'ln_fli', 'ln_pi', 'ln_cti', 'ln_cli', 'ln_cri', 'ln_distdt',
                   'fi', 'fri', 'fli', 'pi', 'cti', 'cli', 'cri', 'distdt',
                   'total_lane', 'avg_crossw', 'tot_road_w', 'tot_crossw',
                   'commercial', 'number_of_', 'of_exclusi', 'curb_exten', 'median', 'all_pedest', 'half_phase', 'new_half_r',
                   'any_ped_pr', 'ped_countd', 'lt_restric', 'lt_prot_re', 'lt_protect', 'any_exclus', 'all_red_an', 'green_stra',
                   'parking', 'north_veh', 'north_ped', 'east_veh', 'east_ped', 'south_veh', 'south_ped', 'west_veh', 'west_ped')
  
  # Check which columns actually exist in the data
  feature_cols <- feature_cols[feature_cols %in% colnames(data)]
  spatial_cols <- spatial_cols[spatial_cols %in% colnames(data)]
  
  # Combine base features with spatial variables
  X_base <- data[, feature_cols]
  X_base[is.na(X_base)] <- 0
  X_spatial <- data[, spatial_cols]
  X_spatial[is.na(X_spatial)] <- 0
  
  # Apply quadratic transformation to specified variables
  quad_vars <- c('ln_cti', 'ln_cri', 'ln_cli', 'ln_pi', 'ln_fri', 'ln_fli', 'ln_fi')
  for (col in quad_vars) {
    if (col %in% colnames(X_base)) {
      X_base[[paste0(col, "_squared")]] <- X_base[[col]]^2
    }
  }
  
  # Create full dataset 
  id_cols <- c('int_no', 'acc', 'pi', 'borough')
  id_cols <- id_cols[id_cols %in% colnames(data)]
  
  # Combine all columns into full_data
  full_data <- cbind(data[, id_cols], X_base, X_spatial)
  
  # Add spatial index needed for INLA
  full_data$idx_spatial <- 1:nrow(full_data)
  # Ensure borough is a factor for INLA random effect
  if ("borough" %in% colnames(full_data)) {
      full_data$borough <- as.factor(full_data$borough)
  }
  
  return(full_data)
}

create_spatial_weights <- function(data, k_neighbors = 3) {
  # Create spatial weights based on k-nearest neighbors
  # Returns a symmetric spatial weights matrix (required for CARBayes)
  
  # Extract coordinates
  coords <- as.matrix(data[, c('x', 'y')])
  n <- nrow(coords)
  
  # Create distance matrix using k-nearest neighbors
  nn <- get.knn(coords, k = k_neighbors + 1)
  
  # Create initial weight matrix
  W <- matrix(0, nrow = n, ncol = n)
  
  # Create weights that decay with distance
  for (i in 1:n) {
    # Skip the first neighbor (self)
    for (j in 2:(k_neighbors + 1)) {
      idx <- nn$nn.index[i, j]
      dist <- nn$nn.dist[i, j]
      # Inverse distance weighting
      W[i, idx] <- 1.0 / max(dist, 0.0001)
    }
  }
  
  # Make it symmetric by taking the average of W and its transpose
  # This ensures perfect symmetry for CARBayes
  W_symmetric <- (W + t(W)) / 2
  
  # Verify symmetry
  is_symmetric <- all(abs(W_symmetric - t(W_symmetric)) < 1e-10)
  cat("Spatial weights matrix symmetry check:", ifelse(is_symmetric, "PASSED", "FAILED"), "\n")
  
  return(W_symmetric)
}

calculate_spatial_lag <- function(data, variable, W) {
  # Calculate the spatial lag of a variable using weights matrix W
  return(W %*% data[[variable]])
}

select_features_with_lasso <- function(X, y, max_features = 30) {
  # Select most important features using Lasso regression
  
  # Convert to matrix
  X_matrix <- as.matrix(X)
  
  # Scale features
  X_scaled <- scale(X_matrix)
  
  # Run Lasso with cross-validation
  set.seed(42)
  lasso_cv <- cv.glmnet(X_scaled, y, alpha = 1, family = "poisson")
  
  # Fit model with optimal lambda
  lasso_model <- glmnet(X_scaled, y, alpha = 1, lambda = lasso_cv$lambda.min)
  
  # Get features selected by Lasso
  coef_matrix <- as.matrix(coef(lasso_model))
  selected_features <- coef_matrix[-1, 1]  # Exclude intercept
  selected_features <- selected_features[selected_features != 0]
  
  # Sort by absolute value
  selected_features <- selected_features[order(abs(selected_features), decreasing = TRUE)]
  
  # Limit to max number of features
  if (length(selected_features) > max_features) {
    selected_features <- selected_features[1:max_features]
  }
  
  cat("Selected", length(selected_features), "features\n")
  return(names(selected_features))
}

#####################################
# Spatial Negative Binomial Model - REMOVED R6 CLASS
#####################################

# The SpatialMixedNegativeBinomial R6 class is removed.
# We will use INLA directly.

#####################################
# Cross-Validation and Evaluation - Using INLA
#####################################

run_cross_validation_inla <- function(data, selected_features, k = 5, random_effect_col = "borough", spatial_coords = c('x', 'y')) {
  set.seed(42)

  # Create fold indices
  folds <- createFolds(data$acc, k = k, returnTrain = TRUE)

  # Initialize lists to store results
  fold_results <- list()
  test_actual <- list()
  test_predicted <- list()

  # Create spatial weights matrix ONCE for the full dataset
  coords <- as.matrix(data[, spatial_coords])
  # Ensure data passed has standard names if function expects them
  W_full <- create_spatial_weights(data.frame(x=coords[,1], y=coords[,2]), k_neighbors=3)
  W_sparse_full <- Matrix(W_full, sparse = TRUE)

  # Ensure W_sparse_full is a valid graph structure for INLA (e.g., symmetric)
  # INLA::inla.read.graph can be used for checks/conversion if needed.
  # For now, assume create_spatial_weights returns a suitable symmetric matrix.

  # Define priors once
  pc_prec_unstructured <- list(prior = 'pc.prec', param = c(1, 0.01))
  pc_phi <- list(prior = 'pc', param = c(0.5, 0.5))
  pc_prec_borough <- list(prior = 'pc.prec', param = c(1, 0.01))
  nb_family_control <- list(hyper = list(theta = list(prior = "loggamma", param = c(1, 0.01))))
  
  # --- Define initial values for control.mode ---
  # Order: NB theta (log), BYM2 prec (log), BYM2 phi (logit), IID prec (log)
  initial_theta_values <- c(log(5), 0, 0, 0) 
  inla_control_mode <- list(theta = initial_theta_values, restart = TRUE)
  # --------------------------------------------

  # For each fold
  for (i in 1:k) {
    cat("\n========== Fold", i, "/", k, "==========\n")

    # Split data into train and test
    train_indices <- folds[[i]]
    test_indices <- setdiff(1:nrow(data), train_indices)

    # Create response variable for this fold (NA for test set)
    y_cv <- data$acc
    y_cv[test_indices] <- NA

    # Create a temporary data copy for this fold's specific response
    data_cv <- data
    data_cv$y_cv <- y_cv # Add the response vector with NAs
    
    # --- SAFETY CHECK: Ensure idx_spatial is not in selected_features for the formula ---
    features_for_formula <- setdiff(selected_features, "idx_spatial")
    if (length(features_for_formula) == 0) {
        # Handle case where only idx_spatial might have been selected (unlikely but possible)
        # Option 1: Add intercept only model
         formula_str <- "y_cv ~ 1"
        # Option 2: Skip fold / add specific handling
        # For now, adding intercept and the random effects
    } else {
         formula_str <- paste("y_cv ~", paste(features_for_formula, collapse = " + "))
    }
    # ----------------------------------------------------------------------------------

    # Prepare INLA formula using the temporary response 'y_cv'
    # Append random effects to the formula string
    formula_str <- paste(
        formula_str, # Base formula (intercept or features)
        "+ f(idx_spatial, model = 'bym2', graph = W_sparse_full, scale.model = TRUE, constr = TRUE, hyper = list(prec = pc_prec_unstructured, phi = pc_phi))",
        "+ f(", random_effect_col, ", model = 'iid', hyper = list(prec = pc_prec_borough))"
    )
    inla_formula <- as.formula(formula_str)

    cat("Fitting INLA model for fold", i, "...\n")
    # Fit INLA model
    inla_model <- try(inla(inla_formula,
                           data = data_cv, # Use the data with y_cv
                           family = "nbinomial",
                           control.predictor = list(compute = TRUE, link = 1), # Compute predictions for NAs in y_cv
                           control.compute = list(dic = TRUE, waic = TRUE, config = TRUE),
                           control.family = nb_family_control, # Use corrected control list
                           control.mode = inla_control_mode, # Use updated control.mode
                           # Remove verbose=TRUE for cleaner output, use inla.debug=TRUE if needed
                           ),
                      silent = TRUE)

    if (inherits(inla_model, "try-error")) {
        cat("ERROR fitting INLA model for fold", i, ":", attr(inla_model, "condition")$message, "\n")
        fold_results[[i]] <- list(fold = i, train_mae = NA, train_rmse = NA, test_mae = NA, test_rmse = NA, model = NULL)
        test_actual[[i]] <- data$acc[test_indices] # Get actual values from original data
        test_predicted[[i]] <- rep(NA, length(test_indices))
        next # Skip to next fold
    }

    cat("Model fitting complete for fold", i, ".\n")

    # Extract predictions (mean of the posterior predictive distribution)
    # Fitted values are available for all points via summary.fitted.values
    predicted_mean <- inla_model$summary.fitted.values$mean

    # Separate train and test predictions based on original indices
    train_preds <- predicted_mean[train_indices]
    test_preds <- predicted_mean[test_indices]

    # Get actual values
    train_actual <- data$acc[train_indices]
    test_actual_fold <- data$acc[test_indices]

    # Calculate metrics (ensure no NAs in actuals/preds being compared)
    train_mae <- mean(abs(train_actual - train_preds), na.rm = TRUE)
    train_rmse <- sqrt(mean((train_actual - train_preds)^2, na.rm = TRUE))

    test_mae <- mean(abs(test_actual_fold - test_preds), na.rm = TRUE)
    test_rmse <- sqrt(mean((test_actual_fold - test_preds)^2, na.rm = TRUE))

    cat("Fold", i, "- Train MAE:", format(train_mae, digits = 4), "RMSE:", format(train_rmse, digits = 4), "\n")
    cat("Fold", i, "- Test MAE:", format(test_mae, digits = 4), "RMSE:", format(test_rmse, digits = 4), "\n")

    # Store results
    fold_results[[i]] <- list(
      fold = i,
      train_mae = train_mae,
      train_rmse = train_rmse,
      test_mae = test_mae,
      test_rmse = test_rmse,
      model_summary = summary(inla_model)
    )

    # Collect predictions and actual values for overall metrics
    test_actual[[i]] <- test_actual_fold
    test_predicted[[i]] <- test_preds
  }

  # Combine all predictions
  test_actual_combined <- unlist(test_actual)
  test_predicted_combined <- unlist(test_predicted)

  # Calculate overall CV metrics
  overall_mae <- mean(abs(test_actual_combined - test_predicted_combined), na.rm = TRUE)
  overall_rmse <- sqrt(mean((test_actual_combined - test_predicted_combined)^2, na.rm = TRUE))

  cat("\n========== Overall CV Results ==========\n")
  cat("Overall MAE:", format(overall_mae, digits = 4), "\n")
  cat("Overall RMSE:", format(overall_rmse, digits = 4), "\n")

  return(list(
    fold_results = fold_results,
    overall_mae = overall_mae,
    overall_rmse = overall_rmse,
    test_actual = test_actual_combined,
    test_predicted = test_predicted_combined
  ))
}


#####################################
# Feature Selection and Visualization
#####################################

select_and_visualize_features <- function(data, response_var = "acc", max_features = 20) {
  # Extract features and response
  y <- data[[response_var]]
  
  # Exclude non-feature columns, including the spatial index
  exclude_cols <- c('int_no', 'acc', 'pi', 'borough', 'x', 'y', 'idx_spatial') # Added 'idx_spatial' here
  feature_cols <- setdiff(colnames(data), exclude_cols)
  X <- data[, feature_cols]
  
  # Ensure X does not contain only NA/empty columns if some features were dropped
  X <- X[, colSums(is.na(X)) < nrow(X)] 
  
  # Select features using LASSO
  selected_features <- select_features_with_lasso(X, y, max_features)
  
  # Create a formula with selected features
  formula_str <- paste(response_var, "~", paste(selected_features, collapse = " + "))
  cat("Suggested formula based on LASSO feature selection:\n")
  cat(formula_str, "\n\n")
  
  # Visualize feature importance
  if (requireNamespace("ggplot2", quietly = TRUE) && length(selected_features) > 0) { # Check if features selected
    # Get coefficients from LASSO
    X_matrix <- as.matrix(X[, selected_features, drop = FALSE]) # Use only selected features for scaling/fitting
    X_scaled <- scale(X_matrix)
    
    # Handle cases where scaling produces NaN (e.g., zero variance columns, although unlikely with LASSO selection)
    X_scaled[is.na(X_scaled)] <- 0 
    
    # Ensure response y has the same length as rows in X_scaled
    if(nrow(X_scaled) != length(y)) {
        stop("Mismatch between selected features matrix rows and response variable length.")
    }

    lasso_cv <- cv.glmnet(X_scaled, y, alpha = 1, family = "poisson")
    lasso_model <- glmnet(X_scaled, y, alpha = 1, lambda = lasso_cv$lambda.min)
    coef_matrix <- as.matrix(coef(lasso_model))
    
    # Create data frame for plotting
    coef_df <- data.frame(
      Feature = rownames(coef_matrix)[-1],  # Exclude intercept
      Coefficient = coef_matrix[-1, 1]
    )
    coef_df <- coef_df[coef_df$Coefficient != 0, ]
    
    if(nrow(coef_df) > 0) { # Check if any non-zero coefficients exist
        coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), ]
        coef_df$Feature <- factor(coef_df$Feature, levels = coef_df$Feature[order(abs(coef_df$Coefficient))])
        
        # Plot
        p <- ggplot(coef_df[1:min(nrow(coef_df), max_features), ], 
                    aes(x = reorder(Feature, abs(Coefficient)), y = Coefficient)) +
          geom_bar(stat = "identity", fill = "steelblue") +
          coord_flip() +
          labs(title = "Feature Importance from LASSO", 
               x = "Features", 
               y = "Coefficient Value") +
          theme_minimal()
        
        print(p)
    } else {
        cat("No features selected by LASSO with non-zero coefficients.\n")
    }
  }
  
  return(list(
    selected_features = selected_features,
    formula = formula_str
  ))
}

# === Main Script Execution ===

# First, load and prepare the data
full_data <- load_and_prepare_data()

# Run feature selection and visualization
feature_results <- select_and_visualize_features(full_data, response_var = "acc", max_features = 30)

# Use the features from feature selection
selected_features <- feature_results$selected_features

# --- Option 1: Run Cross-Validation using INLA ---
cat("\n--- Running Cross-Validation with INLA ---\n")
cv_results_inla <- run_cross_validation_inla(
  data = full_data,
  selected_features = selected_features,
  k = 5, # 5-fold cross-validation
  random_effect_col = "borough", # Using borough as random effect
  spatial_coords = c('x', 'y')
)

# --- Option 2: Fit a Single INLA Model on Full Data ---
cat("\n--- Fitting Single INLA Model on Full Data ---\n")

# Define priors (reuse or redefine if needed)
pc_prec_unstructured <- list(prior = 'pc.prec', param = c(1, 0.01))
pc_phi <- list(prior = 'pc', param = c(0.5, 0.5))
pc_prec_borough <- list(prior = 'pc.prec', param = c(1, 0.01))
nb_family_control <- list(hyper = list(theta = list(prior = "loggamma", param = c(1, 0.01))))

# --- Define initial values for control.mode ---
initial_theta_values_full <- c(log(5), 0, 0, 0) # Same as in CV
inla_control_mode_full <- list(theta = initial_theta_values_full, restart = TRUE)
# --------------------------------------------

# Create spatial weights matrix for the full dataset
W_full <- create_spatial_weights(full_data[, c('x','y')]) # Pass only coords if function expects that
W_sparse_full <- Matrix(W_full, sparse = TRUE)

# --- SAFETY CHECK: Ensure idx_spatial is not in selected_features for the formula ---
features_for_formula_full <- setdiff(selected_features, "idx_spatial")
if (length(features_for_formula_full) == 0) {
    base_formula_part <- "1" # Intercept only if no other features
} else {
    base_formula_part <- paste(features_for_formula_full, collapse = " + ")
}
# ----------------------------------------------------------------------------------

# Define INLA formula for the full model with corrected hyperparams
inla_formula_full <- as.formula(paste(
  "acc ~", base_formula_part, # Use the filtered features part
  "+ f(idx_spatial, model = 'bym2', graph = W_sparse_full, scale.model = TRUE, constr = TRUE, hyper = list(prec = pc_prec_unstructured, phi = pc_phi))", # Corrected: prec.unstructured -> prec
  "+ f(borough, model = 'iid', hyper = list(prec = pc_prec_borough))" # Corrected hyper
))

# Fit the final INLA model
final_inla_model <- try(inla(inla_formula_full,
                           data = full_data,
                           family = "nbinomial",
                           control.predictor = list(compute = TRUE, link=1),
                           control.compute = list(dic = TRUE, waic = TRUE, config = TRUE), # Added config=TRUE back for potential debugging
                           control.family = nb_family_control, # Use corrected control list
                           control.mode = inla_control_mode_full), # Use updated control.mode
                      silent = TRUE)


if (inherits(final_inla_model, "try-error")) {
    cat("ERROR fitting final INLA model:", attr(final_inla_model, "condition")$message, "\n")
} else {
    cat("\n--- Final INLA Model Summary ---\n")
    print(summary(final_inla_model))

    # Extract predictions
    predictions_inla <- final_inla_model$summary.fitted.values$mean # Mean of posterior predictive

    # Calculate overall MAE/RMSE for the full model fit (in-sample)
    mae_full <- mean(abs(full_data$acc - predictions_inla), na.rm = TRUE)
    rmse_full <- sqrt(mean((full_data$acc - predictions_inla)^2, na.rm = TRUE))

    cat("\nIn-Sample Metrics (Full Model):\n")
    cat("MAE:", format(mae_full, digits = 4), "\n")
    cat("RMSE:", format(rmse_full, digits = 4), "\n")

    # You can access specific parts of the summary, e.g.:
    # final_inla_model$summary.fixed (Fixed effects)
    # final_inla_model$summary.hyperpar (Hyperparameters for random effects and NB dispersion)
    # final_inla_model$dic$dic (DIC value)
    # final_inla_model$waic$waic (WAIC value)
}