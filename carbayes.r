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
# Spatial Negative Binomial Model - Fixed Version
#####################################

SpatialMixedNegativeBinomial <- R6::R6Class(
  "SpatialMixedNegativeBinomial",
  
  public = list(
    data = NULL,
    formula = NULL,
    random_effect = NULL,
    exposure = NULL,
    spatial_vars = NULL,
    model = NULL,
    results = NULL,
    W = NULL,
    W_list = NULL,
    data_with_re = NULL,
    
    initialize = function(data = NULL, formula = NULL, random_effect = NULL, 
                          exposure = NULL, spatial_vars = NULL) {
      self$data <- data
      self$formula <- formula
      self$random_effect <- random_effect
      self$exposure <- NULL  # Not using exposure for modeling counts
      self$spatial_vars <- spatial_vars
    },
    
    create_W_list = function(W) {
      # This is the problematic function - rewrite to avoid recursion
      # Convert W matrix to neighborhood list for CARBayes
      n <- nrow(W)
      W_list <- list()
      W_list$n <- n
      
      # Create neighborhood structure (adjacency) - using vectorized operations where possible
      W_list$adj <- vector("list", n)
      W_list$weights <- vector("list", n)
      W_list$num <- numeric(n)
      
      # Process in chunks to avoid stack issues
      chunk_size <- 100
      n_chunks <- ceiling(n / chunk_size)
      
      for (chunk in 1:n_chunks) {
        start_idx <- (chunk - 1) * chunk_size + 1
        end_idx <- min(chunk * chunk_size, n)
        
        for (i in start_idx:end_idx) {
          # Find neighbors (non-zero weights)
          neighbors <- which(W[i, ] > 0)
          W_list$adj[[i]] <- neighbors
          W_list$weights[[i]] <- W[i, neighbors]
          W_list$num[i] <- length(neighbors)
        }
        
        # Free up memory after each chunk
        gc()
      }
      
      # Total number of edges
      W_list$sumjk <- sum(W_list$num)
      
      return(W_list)
    },
    
    fit = function() {
      data_to_use <- self$data
      
      # Create spatial weights structure for CARBayes
      if (is.null(self$W)) {
        self$W <- create_spatial_weights(data_to_use[, self$spatial_vars])
      }
      
      # Convert W matrix to CARBayes format - potential stack overflow point
      # Increase R's C stack limit before this operation
      options(expressions = 500000)  # Increase expression evaluation limit
      memory.limit(size = 1000000)      # Increase memory limit if possible
      
      # Process W matrix to reduce density if needed
      if (mean(self$W > 0) > 0.1) {
        cat("High density W matrix, thresholding to reduce density...\n")
        # Keep only the strongest connections
        threshold <- quantile(self$W[self$W > 0], 0.5)  # Keep top 50% of weights
        self$W[self$W < threshold] <- 0
      }
      
      self$W_list <- self$create_W_list(self$W)
      
      # Extract parts of the formula
      formula_parts <- strsplit(self$formula, "~")[[1]]
      response_var <- trimws(formula_parts[1])
      predictors <- trimws(formula_parts[2])
      
      # Add random effect if specified
      if (!is.null(self$random_effect) && self$random_effect %in% colnames(data_to_use)) {
        cat("Using", self$random_effect, "as a random effect in CARBayes model\n")
        
        # Convert random effect to factor if not already
        if (!is.factor(data_to_use[[self$random_effect]])) {
          data_to_use[[self$random_effect]] <- as.factor(data_to_use[[self$random_effect]])
        }
        
        # Include random effect in formula
        predictors <- paste(predictors, "+", self$random_effect)
      }
      
      # Reconstruct formula
      full_formula <- as.formula(paste(response_var, "~", predictors))
      
      cat("Fitting CARBayes model with formula:", deparse(full_formula), "\n")
      

      model <- S.CARleroux(
        formula = full_formula,
        family = "poisson",  
        data = data_to_use,
        W = self$W,
        burnin = 2000,      
        n.sample = 5000,    
        thin = 2,           
        rho = 0.5,          
        verbose = TRUE
      )
      
      self$model <- model
      self$results <- model
      self$data_with_re <- data_to_use
      
      cat("Model fitting complete.\n")
      
      return(self$results)
    },
    
    predict = function(newdata = NULL) {
      if (is.null(self$results)) {
        stop("Model must be fitted before prediction")
      }
      
      # Defensive check for missing samples
      if (is.null(self$results$samples) || is.null(self$results$samples$beta)) {
        cat("Warning: Model samples are not available for prediction\n")
        return(NULL)
      }
      
      if (is.null(newdata)) {
        # Use the data with random effects already computed
        if (!is.null(self$data_with_re)) {
          newdata <- self$data_with_re
        } else {
          newdata <- self$data
        }
      }
      
      # Extract formula and create model matrix for prediction
      tryCatch({
        # Get the formula used for fitting
        formula_parts <- strsplit(self$formula, "~")[[1]]
        predictors <- trimws(formula_parts[2])
        
        # Add fixed effect for borough if it was used as a random effect
        if (!is.null(self$random_effect) && self$random_effect %in% colnames(newdata)) {
          if (!grepl(self$random_effect, predictors)) {
            predictors <- paste(predictors, "+", self$random_effect)
          }
        }
        
        X_formula <- as.formula(paste("~", predictors))
        
        # Create model matrix - handle factor variables properly
        X <- try(model.matrix(X_formula, data = newdata), silent = TRUE)
        
        if (inherits(X, "try-error")) {
          cat("Error creating model matrix. Trying simpler approach...\n")
          # Try a simpler approach - extract variable names and create matrix manually
          pred_vars <- strsplit(predictors, "\\+")[[1]]
          pred_vars <- trimws(pred_vars)
          
          # Remove any complex terms (interactions, etc.)
          simple_vars <- pred_vars[!grepl(":", pred_vars) & !grepl("\\*", pred_vars)]
          
          # Create a formula with just simple terms
          simple_formula <- as.formula(paste("~", paste(simple_vars, collapse = " + ")))
          X <- model.matrix(simple_formula, data = newdata)
        }
        
        # Get posterior means of fixed effects
        beta <- self$results$samples$beta
        beta_mean <- apply(beta, 2, mean)
        
        # Debug information
        cat("Model has", length(beta_mean), "coefficients\n")
        cat("Design matrix has", ncol(X), "columns\n")
        
        if (!is.null(colnames(beta)) && !is.null(colnames(X))) {
          cat("Model coefficient names:", paste(colnames(beta)[1:min(5, length(colnames(beta)))], collapse=", "), "...\n")
          cat("Design matrix column names:", paste(colnames(X)[1:min(5, ncol(X))], collapse=", "), "...\n")
        } else {
          # Fix for missing column names
          if (is.null(colnames(beta)) && ncol(beta) == length(beta_mean)) {
            cat("Adding default names to coefficients...\n")
            colnames(beta) <- paste0("beta", 1:ncol(beta))
          }
          
          if (is.null(colnames(X))) {
            cat("Adding default names to design matrix...\n")
            colnames(X) <- paste0("X", 1:ncol(X))
          }
        }
        
        # Check that dimensions match
        if (length(beta_mean) != ncol(X)) {
          cat("WARNING: Dimension mismatch in prediction.\n")
          cat("Model has", length(beta_mean), "coefficients, design matrix has", ncol(X), "columns.\n")
          
          # If we have more coefficients than predictors, truncate the coefficients
          if (length(beta_mean) > ncol(X)) {
            cat("Truncating coefficients to match design matrix dimensions.\n")
            beta_mean <- beta_mean[1:ncol(X)]
          } else {
            # If we have more predictors than coefficients, truncate the design matrix
            cat("Truncating design matrix to match coefficient dimensions.\n")
            X <- X[, 1:length(beta_mean)]
          }
        }
        
        # Linear predictor
        eta <- X %*% beta_mean
        
        # Predictions based on Poisson model
        lambda <- exp(eta)
        
        # Random effects (if available)
        if (!is.null(self$results$samples$phi)) {
          phi_mean <- apply(self$results$samples$phi, 2, mean)
          
          # For in-sample predictions
          if (nrow(newdata) == length(phi_mean)) {
            lambda <- lambda * exp(phi_mean)
          } else {
            cat("Warning: Out-of-sample prediction doesn't include random effects\n")
          }
        }
        
        # Return predictions
        return(lambda)
      }, error = function(e) {
        cat("Error in prediction:", e$message, "\n")
        return(rep(NA, nrow(newdata)))
      })
    },
    
    summary = function() {
      if (is.null(self$results)) {
        cat("Model has not been fitted yet\n")
        return(NULL)
      }
      
      cat("\n=== CARBayes Model Summary ===\n")
      
      # Display basic model info
      cat("Model:", ifelse(!is.null(self$results$modelname), self$results$modelname, "Not available"), "\n")
      cat("Family:", ifelse(!is.null(self$results$family), self$results$family, "Not available"), "\n")
      
      # Display fixed effects (with credible intervals) - with better error handling
      cat("\nFixed Effects:\n")
      tryCatch({
        if (!is.null(self$results$summary.results) && is.list(self$results$summary.results) && 
            !is.null(self$results$summary.results$beta)) {
          
          # Add proper names to the coefficients if missing
          if (is.null(rownames(self$results$summary.results$beta))) {
            # Extract variable names from formula
            formula_parts <- strsplit(self$formula, "~")[[1]]
            predictors <- trimws(formula_parts[2])
            pred_vars <- c("(Intercept)", strsplit(predictors, "\\+")[[1]])
            pred_vars <- trimws(pred_vars)
            
            # Check if we have the right number of variables
            if (length(pred_vars) == nrow(self$results$summary.results$beta)) {
              rownames(self$results$summary.results$beta) <- pred_vars
            } else {
              rownames(self$results$summary.results$beta) <- paste0("var", 1:nrow(self$results$summary.results$beta))
            }
          }
          
          print(self$results$summary.results$beta)
        } else if (!is.null(self$results$samples) && !is.null(self$results$samples$beta)) {
          # Compute summary stats manually from MCMC samples
          beta_samples <- self$results$samples$beta
          beta_means <- apply(beta_samples, 2, mean)
          beta_sds <- apply(beta_samples, 2, sd)
          beta_quants <- apply(beta_samples, 2, quantile, probs = c(0.025, 0.5, 0.975))
          
          # Create a summary data frame similar to what CARBayes would produce
          beta_summary <- data.frame(
            Mean = beta_means,
            SD = beta_sds,
            `2.5%` = beta_quants[1,],
            `50%` = beta_quants[2,],
            `97.5%` = beta_quants[3,]
          )
          
          # Add variable names if available
          if (is.null(rownames(beta_summary)) && !is.null(colnames(beta_samples))) {
            rownames(beta_summary) <- colnames(beta_samples)
          } else if (is.null(rownames(beta_summary))) {
            # Extract variable names from formula
            formula_parts <- strsplit(self$formula, "~")[[1]]
            predictors <- trimws(formula_parts[2])
            pred_vars <- c("(Intercept)", strsplit(predictors, "\\+")[[1]])
            pred_vars <- trimws(pred_vars)
            
            # Check if we have the right number of variables
            if (length(pred_vars) == nrow(beta_summary)) {
              rownames(beta_summary) <- pred_vars
            } else {
              rownames(beta_summary) <- paste0("var", 1:nrow(beta_summary))
            }
          }
          
          print(beta_summary)
        } else {
          cat("No fixed effects information available in the expected format\n")
          
          # Debug: print what is available
          cat("Available objects in results:\n")
          print(names(self$results))
          
          if (!is.null(self$results$samples)) {
            cat("Available objects in samples:\n")
            print(names(self$results$samples))
          }
        }
      }, error = function(e) {
        cat("Error displaying fixed effects:", e$message, "\n")
        cat("Attempting to extract information directly from samples...\n")
        
        if (!is.null(self$results$samples) && !is.null(self$results$samples$beta)) {
          # Try a simpler approach to display coefficient estimates
          beta_means <- colMeans(self$results$samples$beta)
          cat("Coefficient estimates (means of posterior samples):\n")
          print(beta_means)
        }
      })
      
      # Display hyperparameters with better error handling
      cat("\nHyperparameters:\n")
      tryCatch({
        # Try to extract nu2 (spatial variance)
        if (!is.null(self$results$samples$nu2)) {
          cat("Spatial Variance (nu2):", mean(self$results$samples$nu2), "\n")
        }
        
        # Try to extract rho (spatial autocorrelation)
        if (!is.null(self$results$samples$rho)) {
          rho_mean <- mean(self$results$samples$rho)
          if (!is.na(rho_mean)) {
            cat("Spatial Autocorrelation (rho):", rho_mean, "\n")
          } else {
            cat("Spatial Autocorrelation (rho): Not available\n")
          }
        } else {
          cat("Spatial Autocorrelation (rho): Not available\n")
        }
        
        # Display negative binomial dispersion if available
        if (!is.null(self$results$samples$theta)) {
          cat("Negative Binomial Dispersion:", mean(self$results$samples$theta), "\n")
        }
      }, error = function(e) {
        cat("Error displaying hyperparameters:", e$message, "\n")
      })
      
      # Display DIC for model comparison
      tryCatch({
        if (!is.null(self$results$modelfit)) {
          cat("\nDIC:", self$results$modelfit[1], "\n")
          cat("p.d:", self$results$modelfit[2], "\n")
        }
      }, error = function(e) {
        cat("Error displaying model fit statistics:", e$message, "\n")
      })
      
      # Calculate prediction metrics
      tryCatch({
        y_true <- as.numeric(self$data[[strsplit(self$formula, "~")[[1]][1]]])
        y_pred <- as.numeric(self$predict())
        
        # Check if predictions were successful
        if (all(is.na(y_pred))) {
          cat("\nCould not calculate prediction metrics - all predictions are NA\n")
        } else {
          mae <- mean(abs(y_true - y_pred), na.rm = TRUE)
          rmse <- sqrt(mean((y_true - y_pred)^2, na.rm = TRUE))
          
          cat("\nPrediction Metrics:\n")
          cat("MAE:", format(mae, digits = 4), "\n")
          cat("RMSE:", format(rmse, digits = 4), "\n")
          
          return(list(mae = mae, rmse = rmse))
        }
      }, error = function(e) {
        cat("Error calculating prediction metrics:", e$message, "\n")
        return(NULL)
      })
    }
  )
)

#####################################
# Cross-Validation and Evaluation
#####################################

run_cross_validation <- function(data, response_var = "acc", k = 5, random_effect = NULL, exposure = NULL, max_features = 30) {
  set.seed(42)
  
  # Create fold indices
  folds <- createFolds(data$acc, k = k, returnTrain = TRUE)
  
  # Initialize lists to store results
  fold_results <- list()
  test_actual <- list()
  test_predicted <- list()
  
  # For each fold
  for (i in 1:k) {
    cat("\n========== Fold", i, "/", k, "==========\n")
    
    # Split data into train and test
    train_data <- data[folds[[i]], ]
    test_data <- data[-folds[[i]], ]
    
    # Perform feature selection on training data only
    cat("Performing feature selection on training data...\n")
    exclude_cols <- c('int_no', 'acc', 'pi', 'borough', 'x', 'y')
    feature_cols <- setdiff(colnames(train_data), exclude_cols)
    X_train <- train_data[, feature_cols]
    y_train <- train_data[[response_var]]
    
    # Select features using LASSO
    selected_features <- select_features_with_lasso(X_train, y_train, max_features)
    
    # Create formula with selected features
    formula <- paste(response_var, "~", paste(selected_features, collapse = " + "))
    cat("Selected formula for this fold:", formula, "\n")
    
    # Create and fit model with fold-specific features
    model <- SpatialMixedNegativeBinomial$new(
      data = train_data,
      formula = formula,
      random_effect = random_effect,
      exposure = NULL,
      spatial_vars = c('x', 'y')
    )
    
    # Fit model
    results <- model$fit()
    
    # Make predictions
    train_preds <- model$predict(train_data)
    test_preds <- model$predict(test_data)
    
    # Handle potential NA predictions
    if (all(is.na(train_preds))) {
      cat("WARNING: All training predictions are NA, using mean of response\n")
      train_preds <- rep(mean(train_data$acc), nrow(train_data))
    }
    
    if (all(is.na(test_preds))) {
      cat("WARNING: All test predictions are NA, using mean of training response\n")
      test_preds <- rep(mean(train_data$acc), nrow(test_data))
    }
    
    # Calculate metrics
    train_mae <- mean(abs(train_data$acc - train_preds), na.rm = TRUE)
    train_mse <- mean((train_data$acc - train_preds)^2, na.rm = TRUE)
    train_rmse <- sqrt(train_mse)
    
    test_mae <- mean(abs(test_data$acc - test_preds), na.rm = TRUE)
    test_mse <- mean((test_data$acc - test_preds)^2, na.rm = TRUE)
    test_rmse <- sqrt(test_mse)
    
    cat("Train MAE:", format(train_mae, digits = 4), 
        "MSE:", format(train_mse, digits = 4), 
        "RMSE:", format(train_rmse, digits = 4), "\n")
    cat("Test MAE:", format(test_mae, digits = 4), 
        "MSE:", format(test_mse, digits = 4), 
        "RMSE:", format(test_rmse, digits = 4), "\n")
    
    # Store results
    fold_results[[i]] <- list(
      fold = i,
      train_mae = train_mae,
      train_mse = train_mse,
      train_rmse = train_rmse,
      test_mae = test_mae,
      test_mse = test_mse,
      test_rmse = test_rmse,
      model = model,
      selected_features = selected_features,
      formula = formula
    )
    
    # Collect predictions and actual values
    test_actual[[i]] <- test_data$acc
    test_predicted[[i]] <- test_preds
  }
  
  # Combine all predictions
  test_actual_combined <- unlist(test_actual)
  test_predicted_combined <- unlist(test_predicted)
  
  # Calculate overall CV metrics
  overall_mae <- mean(abs(test_actual_combined - test_predicted_combined), na.rm = TRUE)
  overall_mse <- mean((test_actual_combined - test_predicted_combined)^2, na.rm = TRUE)
  overall_rmse <- sqrt(overall_mse)
  
  cat("\n========== Overall CV Results ==========\n")
  cat("Overall MAE:", format(overall_mae, digits = 4), "\n")
  cat("Overall MSE:", format(overall_mse, digits = 4), "\n")
  cat("Overall RMSE:", format(overall_rmse, digits = 4), "\n")
  
  # Analyze feature consistency across folds
  cat("\n========== Feature Selection Consistency ==========\n")
  all_features <- unique(unlist(lapply(fold_results, function(x) x$selected_features)))
  feature_counts <- sapply(all_features, function(feat) {
    sum(sapply(fold_results, function(x) feat %in% x$selected_features))
  })
  
  feature_consistency <- data.frame(
    Feature = names(feature_counts),
    Count = feature_counts,
    Percentage = 100 * feature_counts / k
  )
  feature_consistency <- feature_consistency[order(feature_consistency$Count, decreasing = TRUE), ]
  print(feature_consistency)
  
  return(list(
    fold_results = fold_results,
    overall_mae = overall_mae,
    overall_mse = overall_mse,
    overall_rmse = overall_rmse,
    test_actual = test_actual_combined,
    test_predicted = test_predicted_combined,
    feature_consistency = feature_consistency
  ))
}

#####################################
# Feature Selection and Visualization
#####################################

select_and_visualize_features <- function(data, response_var = "acc", max_features = 20) {
  # Extract features and response
  y <- data[[response_var]]
  
  # Exclude non-feature columns
  exclude_cols <- c('int_no', 'acc', 'pi', 'borough', 'x', 'y')
  feature_cols <- setdiff(colnames(data), exclude_cols)
  X <- data[, feature_cols]
  
  # Select features using LASSO
  selected_features <- select_features_with_lasso(X, y, max_features)
  
  # Create a formula with selected features
  formula_str <- paste(response_var, "~", paste(selected_features, collapse = " + "))
  cat("Suggested formula based on LASSO feature selection:\n")
  cat(formula_str, "\n\n")
  
  # Visualize feature importance
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    # Get coefficients from LASSO
    X_matrix <- as.matrix(X)
    X_scaled <- scale(X_matrix)
    lasso_cv <- cv.glmnet(X_scaled, y, alpha = 1, family = "poisson")
    lasso_model <- glmnet(X_scaled, y, alpha = 1, lambda = lasso_cv$lambda.min)
    coef_matrix <- as.matrix(coef(lasso_model))
    
    # Create data frame for plotting
    coef_df <- data.frame(
      Feature = rownames(coef_matrix)[-1],  # Exclude intercept
      Coefficient = coef_matrix[-1, 1]
    )
    coef_df <- coef_df[coef_df$Coefficient != 0, ]
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
  }
  
  return(list(
    selected_features = selected_features,
    formula = formula_str
  ))
}

# Now call the functions defined in the script
# First, load and prepare the data
full_data <- load_and_prepare_data()

# Run the model with cross-validation that includes per-fold feature selection
cv_results <- run_cross_validation(
  data = full_data, 
  response_var = "acc",
  k = 5,  # 5-fold cross-validation
  random_effect = "borough",  # Using borough as random effect
  max_features = 30  # Maximum number of features to select in each fold
)