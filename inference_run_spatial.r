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
library(readr)
library(Metrics)
library(tidyr)
library(sp)
library(sf)
library(mapview)
library(reshape2)
library(viridis)

conflict_prefer("select", "dplyr")

load_and_prepare_data <- function() {
  data <- read.csv("data/data_final.csv", sep = ";")
  data$ln_distdt[is.na(data$ln_distdt)] <- 0
  data <- data[data$pi != 0, ]
  
  borough_corrections <- list(
    '?le-Bizard-Sainte-GeneviÞve' = 'Île-Bizard-Sainte-Geneviève',
    'C¶te-Saint-Luc' = 'Côte-Saint-Luc',
    'C¶te-des-Neiges-Notre-Dame-de-Graces' = 'Côte-des-Neiges-Notre-Dame-de-Grâce',
    'MontrÚal-Est' = 'Montréal-Est',
    'MontrÚal-Nord' = 'Montréal-Nord',
    'Pointe-aux-Trembles-RiviÞres-des-Prairies' = 'Rivière-des-Prairies-Pointe-aux-Trembles',
    'St-LÚonard' = 'Saint-Léonard'
  )
  
  borough_zones <- list(
    'Kirkland' = 'Zone ouest', 'Beaconsfield' = 'Zone ouest',
    'Île-Bizard-Sainte-Geneviève' = 'Zone ouest', 'Pierrefonds-Roxboro' = 'Zone ouest',
    'Dollard-des-Ormeaux' = 'Zone ouest', 'Dorval' = 'Zone ouest',
    'Rivière-des-Prairies-Pointe-aux-Trembles' = 'Zone est', 'Montréal-Est' = 'Zone est',
    'Anjou' = 'Zone est', 'Outremont' = 'Zone centre', 'Mont-Royal' = 'Zone centre',
    'Sud-Ouest' = 'Zone sud', 'Côte-Saint-Luc' = 'Zone sud', 'Verdun' = 'Zone sud',
    'Lasalle' = 'Zone sud', 'Lachine' = 'Zone sud', 'Côte-des-Neiges-Notre-Dame-de-Grâce' = 'Zone centre-sud',
    'Hampstead' = 'Zone centre-sud', 'Westmount' = 'Zone centre-sud'
  )
  
  if ("borough" %in% colnames(data)) {
    for (old_name in names(borough_corrections)) {
      data$borough[data$borough == old_name] <- borough_corrections[[old_name]]
    }
    for (borough in names(borough_zones)) {
      data$borough[data$borough == borough] <- borough_zones[[borough]]
    }
    data$borough <- as.factor(data$borough)
  }
  
  spatial_cols <- c('x', 'y')
  feature_cols <- c('ln_fi', 'ln_fri', 'ln_fli', 'ln_pi', 'ln_cti', 'ln_cli', 'ln_cri', 'ln_distdt',
                   'total_lane', 'avg_crossw', 'tot_road_w', 'tot_crossw', 'commercial', 'number_of_', 'of_exclusi', 
                   'curb_exten', 'median', 'all_pedest', 'half_phase', 'new_half_r', 'any_ped_pr', 'ped_countd', 'lt_restric',
                   'lt_prot_re', 'lt_protect', 'any_exclus', 'all_red_an', 'green_stra', 'parking')
  
  feature_cols <- feature_cols[feature_cols %in% colnames(data)]
  spatial_cols <- spatial_cols[spatial_cols %in% colnames(data)]
  
  X_base <- data[, feature_cols]
  X_base[is.na(X_base)] <- 0
  X_spatial <- data[, spatial_cols]
  X_spatial[is.na(X_spatial)] <- 0
  
  quad_vars <- c('ln_cti', 'ln_cri', 'ln_cli', 'ln_pi', 'ln_fri', 'ln_fli', 'ln_fi')
  for (col in quad_vars) {
    if (col %in% colnames(X_base)) {
      X_base[[paste0(col, "_squared")]] <- X_base[[col]]^2
    }
  }
  
  id_cols <- c('int_no', 'acc', 'borough')
  id_cols <- id_cols[id_cols %in% colnames(data)]
  
  full_data <- cbind(data[, id_cols], X_base, X_spatial)
  
  full_data$idx_spatial <- 1:nrow(full_data)
  
  return(full_data)
}

create_spatial_weights <- function(data, k_neighbors = 10) {
  coords <- as.matrix(data[, c('x', 'y')])
  n <- nrow(coords)
  
  nn <- get.knn(coords, k = k_neighbors + 1)
  W <- matrix(0, nrow = n, ncol = n)
  
  for (i in 1:n) {
    for (j in 2:(k_neighbors + 1)) {
      idx <- nn$nn.index[i, j]
      # Use binary weights (1 if in k-neighbors, 0 otherwise)
      W[i, idx] <- 1
    }
  }
  
  # Symmetrize by taking maximum between W and t(W)
  # This ensures that if either point considers the other a neighbor, they are connected
  W_symmetric <- pmax(W, t(W))
  return(W_symmetric)
}

select_features_with_lasso <- function(X, y, max_features = 30) {
  X_matrix <- as.matrix(X)
  X_scaled <- scale(X_matrix)
  
  set.seed(42)
  lasso_cv <- cv.glmnet(X_scaled, y, alpha = 1, family = "poisson")
  lasso_model <- glmnet(X_scaled, y, alpha = 1, lambda = lasso_cv$lambda.min)
  
  coef_matrix <- as.matrix(coef(lasso_model))
  selected_features <- coef_matrix[-1, 1]
  selected_features <- selected_features[selected_features != 0]
  selected_features <- selected_features[order(abs(selected_features), decreasing = TRUE)]
  
  if (length(selected_features) > max_features) {
    selected_features <- selected_features[1:max_features]
  }
  
  return(names(selected_features))
}

carbayes_model <- R6::R6Class(
  "carbayes_model",
  
  public = list(
    data = NULL,
    formula = NULL,
    response_var = NULL, # Added to store response variable name
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
      # Extract and store response variable name from formula
      if (!is.null(formula)) {
          self$response_var <- all.vars(as.formula(formula))[1]
      }
      self$random_effect <- random_effect
      self$exposure <- NULL # Assuming exposure is not used based on previous code
      self$spatial_vars <- spatial_vars
    },
    
    create_W_list = function(W) {
      n <- nrow(W)
      W_list <- list()
      W_list$n <- n
      
      W_list$adj <- vector("list", n)
      W_list$weights <- vector("list", n)
      W_list$num <- numeric(n)
      
      chunk_size <- 100
      n_chunks <- ceiling(n / chunk_size)
      
      for (chunk in 1:n_chunks) {
        start_idx <- (chunk - 1) * chunk_size + 1
        end_idx <- min(chunk * chunk_size, n)
        
        for (i in start_idx:end_idx) {
          neighbors <- which(W[i, ] > 0)
          W_list$adj[[i]] <- neighbors
          W_list$weights[[i]] <- W[i, neighbors]
          W_list$num[i] <- length(neighbors)
        }
        
        gc()
      }
      
      W_list$sumjk <- sum(W_list$num)
      return(W_list)
    },
    
    fit = function() {
      data_to_use <- self$data
      
      if (is.null(self$W)) {
        self$W <- create_spatial_weights(data_to_use[, self$spatial_vars])
      }
      
      options(expressions = 500000)
      memory.limit(size = 1000000)
      
      if (mean(self$W > 0) > 0.1) {
        threshold <- quantile(self$W[self$W > 0], 0.5)
        self$W[self$W < threshold] <- 0
      }
      
      self$W_list <- self$create_W_list(self$W)

      # Use the formula directly as passed during initialization
      # It was already adjusted in the main loop if the random effect was problematic
      full_formula <- as.formula(self$formula)

      # Ensure the random effect column is a factor IF it's actually in the formula
      # The check for levels already happened outside
      # self$random_effect here is the name passed during init (might be NULL now)
      if (!is.null(self$random_effect) && self$random_effect %in% all.vars(full_formula) && self$random_effect %in% colnames(data_to_use)) {
          if (!is.factor(data_to_use[[self$random_effect]])) {
              # Use levels from the original data if possible, otherwise just convert
               if(!is.null(self$data) && self$random_effect %in% colnames(self$data) && is.factor(self$data[[self$random_effect]])) {
                   data_to_use[[self$random_effect]] <- factor(data_to_use[[self$random_effect]], levels=levels(self$data[[self$random_effect]]))
               } else {
                   data_to_use[[self$random_effect]] <- as.factor(data_to_use[[self$random_effect]])
               }
          }
      }

      # MODIFICATION: Add random effect to formula if not already included
      if (!is.null(self$random_effect) && self$random_effect %in% colnames(data_to_use)) {
        # Check if random effect is already in the formula
        formula_terms <- attr(terms(full_formula), "term.labels")
        if (!self$random_effect %in% formula_terms) {
          # Add random effect to formula
          formula_str <- as.character(full_formula)
          rhs <- formula_str[3]  # Right-hand side of formula
          new_rhs <- paste(rhs, "+", self$random_effect)
          new_formula_str <- paste(formula_str[2], formula_str[1], new_rhs)
          full_formula <- as.formula(new_formula_str)
          cat("Added random effect '", self$random_effect, "' to CARBayes formula\n", sep="")
        }
      }

      model <- S.CARleroux(
        formula = full_formula, # Use the modified formula with random effect
        family = "poisson",
        data = data_to_use, # data_to_use has factor conversion if needed
        W = self$W,
        burnin = 2000,
        n.sample = 5000,
        thin = 2,
        #rho = 0.8,
        verbose = TRUE
      )

      self$model <- model
      self$results <- model
      # Store data_to_use which might have the factor conversion
      self$data_with_re <- data_to_use # This holds the data used for fitting

      return(self$results)
    },
    
    predict = function(newdata = NULL) {
        if (is.null(self$results)) {
            stop("Model must be fitted before prediction")
        }

        # Use the formula stored from the fitted model object
        fit_formula <- self$results$formula
        predict_on_train <- FALSE # Flag to track prediction type

        if (is.null(newdata)) {
            # Predicting on training data
            newdata <- self$data_with_re # Use the data exactly as used for fitting
            predict_on_train <- TRUE
        } else {
            # Predicting on new data (e.g., test set)
            # Ensure factor levels in newdata match training data IF the factor was used in the fit_formula
            if (!is.null(self$random_effect) && self$random_effect %in% all.vars(fit_formula) && self$random_effect %in% colnames(newdata)) {
                 if (!is.null(self$data_with_re) && is.factor(self$data_with_re[[self$random_effect]])) {
                     original_levels <- levels(self$data_with_re[[self$random_effect]])
                     # Allow new levels in prediction, but map known levels
                     current_levels_new <- levels(factor(newdata[[self$random_effect]]))
                     combined_levels <- union(original_levels, current_levels_new)
                     newdata[[self$random_effect]] <- factor(newdata[[self$random_effect]], levels = combined_levels)
                 } else {
                      if(!is.factor(newdata[[self$random_effect]])) {
                          newdata[[self$random_effect]] <- as.factor(newdata[[self$random_effect]])
                      }
                 }
            }
             # Ensure any other factors used as predictors also have consistent levels
             fixed_vars <- all.vars(fit_formula)[-1] # Get predictor names
             fixed_vars <- setdiff(fixed_vars, self$random_effect) # Exclude random effect name
             for(fv in fixed_vars) {
                 if(fv %in% colnames(self$data_with_re) && is.factor(self$data_with_re[[fv]])) {
                     if(fv %in% colnames(newdata)) {
                         original_levels <- levels(self$data_with_re[[fv]])
                         current_levels_new <- levels(factor(newdata[[fv]]))
                         combined_levels <- union(original_levels, current_levels_new)
                         newdata[[fv]] <- factor(newdata[[fv]], levels=combined_levels)
                     }
                 }
             }
        }

        # Construct model matrix based on the *actual* formula used for fitting
        # Extract RHS terms safely
        rhs_terms <- attr(terms(fit_formula), "term.labels")
        if (length(rhs_terms) == 0) {
             X_formula <- ~ 1
        } else {
             X_formula <- reformulate(termlabels = rhs_terms)
        }

        # Use na.action=na.pass to keep rows with NAs
        X <- try(model.matrix(X_formula, data = newdata, na.action=na.pass), silent = TRUE)

        if (inherits(X, "try-error")) {
            stop("Failed to create model matrix for prediction. Error: ", attr(X, "condition")$message)
        }

        beta <- self$results$samples$beta
        beta_mean <- apply(beta, 2, mean) # Names might be generic like var1, var2...

        # --- Positional Alignment ---
        # Get the expected column names from the model matrix
        matrix_colnames <- colnames(X)
        n_matrix_cols <- ncol(X)
        n_coeffs <- length(beta_mean)

        if (n_matrix_cols != n_coeffs) {
            # If counts don't match, something is fundamentally wrong.
            print("--- Alignment Failure: Dimension Mismatch ---")
            print("Model Matrix Columns:")
            print(matrix_colnames)
            print(paste("Number of matrix columns:", n_matrix_cols))
            print("Fitted Coefficient Names (potentially generic):")
            print(names(beta_mean))
            print(paste("Number of coefficients:", n_coeffs))
            print("Original Formula:")
            print(fit_formula)
            print("---------------------------------------------")
            stop("Dimension mismatch between model matrix and coefficients. Cannot align by position.")
        } else {
            # If counts match, assume order is correct and assign matrix names to coefficients
            # This forces alignment based on position.
            names(beta_mean) <- matrix_colnames
            # No need to reorder X or beta_mean further if we assume the initial order was correct.
            message("Aligned coefficients to model matrix columns by position.")
        }
        # --- End Positional Alignment ---

        # Check for NA coefficients (shouldn't happen, but safety check)
        if(any(is.na(beta_mean))) {
            stop("NA values found in coefficients after alignment.")
        }
        # Check dimensions again after alignment (should always match now if we passed the check)
        if(ncol(X) != length(beta_mean)) {
             # This check is somewhat redundant now but kept as a safeguard
             print("--- Dimension Mismatch After Positional Alignment (Internal Error) ---")
             print("Final X columns:")
             print(colnames(X))
             print("Final beta_mean names:")
             print(names(beta_mean))
             print("----------------------------------------")
             stop("Dimension mismatch after attempting positional alignment. This should not happen.")
        }

        # Perform matrix multiplication, handling potential NAs in X
        eta <- rep(NA_real_, nrow(X)) # Initialize eta with NA
        valid_rows <- apply(X, 1, function(row) !any(is.na(row))) # Find rows in X without NAs
        if (any(valid_rows)) {
             # Ensure beta_mean is a column vector for matrix multiplication
             beta_vector <- matrix(beta_mean, ncol = 1)
             # Use drop=FALSE to handle single-column X correctly
             eta[valid_rows] <- X[valid_rows, , drop = FALSE] %*% beta_vector
        }

        # --- Spatial random effects (phi) ---
        eta_spatial <- rep(0, nrow(newdata)) # Initialize spatial effect contribution
        if (!is.null(self$results$samples$phi)) {
            phi_mean <- apply(self$results$samples$phi, 2, mean)

            if (predict_on_train) {
                # If predicting on training data, apply phi directly
                if(length(phi_mean) == nrow(self$data_with_re)) {
                    eta_spatial <- phi_mean
                    message("Applied fitted spatial effects (phi) for training data prediction.")
                } else {
                    warning("Could not apply phi to training data due to length mismatch.")
                }
            } else {
                # If predicting on new data, use k-nearest neighbors with inverse distance weighting
                if (!is.null(self$data_with_re) && all(self$spatial_vars %in% colnames(self$data_with_re)) &&
                    all(self$spatial_vars %in% colnames(newdata))) {

                    coords_train <- as.matrix(self$data_with_re[, self$spatial_vars])
                    coords_test <- as.matrix(newdata[, self$spatial_vars])

                    # Check for NAs in coordinates
                    valid_train_coords <- !apply(coords_train, 1, anyNA)
                    valid_test_coords <- !apply(coords_test, 1, anyNA)

                    if (sum(valid_train_coords) > 0 && sum(valid_test_coords) > 0) {
                        # Find k nearest *valid* training points for each *valid* test point
                        k_interp <- 10 # Number of neighbors for interpolation
                        nn_result <- FNN::get.knnx(
                            coords_train[valid_train_coords, , drop = FALSE],
                            coords_test[valid_test_coords, , drop = FALSE],
                            k = k_interp
                        )

                        # Map relative indices back to original training data indices
                        original_train_indices_matrix <- matrix(which(valid_train_coords)[nn_result$nn.index],
                                                                nrow = nrow(nn_result$nn.index),
                                                                ncol = k_interp)

                        # Ensure phi_mean corresponds to the original training data order
                        if(length(phi_mean) == nrow(self$data_with_re)) {
                            # Initialize eta_spatial for all test points (including those with NA coords)
                            eta_spatial <- rep(0, nrow(newdata))
                            # Get indices of test points with valid coordinates
                            valid_test_indices <- which(valid_test_coords)

                            # Loop through valid test points to calculate IDW
                            for (j in 1:length(valid_test_indices)) {
                                test_idx <- valid_test_indices[j] # Original index in newdata
                                neighbor_orig_indices <- original_train_indices_matrix[j, ]
                                distances <- nn_result$nn.dist[j, ]

                                # Handle zero distances - assign infinite weight (effectively becomes nearest neighbor)
                                zero_dist_idx <- which(distances < 1e-9) # Use a small tolerance
                                if (length(zero_dist_idx) > 0) {
                                    # If one or more points have zero distance, use the effect of the first one found
                                    eta_spatial[test_idx] <- phi_mean[neighbor_orig_indices[zero_dist_idx[1]]]
                                } else {
                                    # Apply inverse distance weighting
                                    weights <- 1 / distances
                                    weights <- weights / sum(weights) # Normalize weights

                                    # Weighted average of spatial effects
                                    eta_spatial[test_idx] <- sum(phi_mean[neighbor_orig_indices] * weights, na.rm = TRUE) # Add na.rm for safety
                                }
                            }
                            cat("Applied inverse distance weighted spatial effect interpolation (k=", k_interp, ") for CARBayes prediction.\n")
                        } else {
                            warning("Could not apply IDW spatial effect: Mismatch between spatial effects length and training data size.")
                        }
                    } else {
                        warning("Could not apply IDW spatial effect: No valid coordinates found in training or test data.")
                    }
                } else {
                    warning("Could not apply IDW spatial effect: Coordinate columns missing or training data unavailable.")
                }
            }
        } else {
            warning("Spatial random effect samples (phi) not found in CARBayes results.")
        }
        # --- End Spatial random effects ---

        # Add spatial effect to linear predictor (eta)
        # Ensure dimensions match before adding, handle NAs in eta
        valid_eta <- !is.na(eta)
        if(length(eta_spatial) == length(eta)) {
            eta[valid_eta] <- eta[valid_eta] + eta_spatial[valid_eta]
        } else if (length(eta_spatial) > 0 && length(valid_eta) > 0) {
            # Fallback if lengths differ but some valid eta exist (should not happen ideally)
             warning("Length mismatch between eta and spatial effects, attempting partial addition.")
             min_len <- min(length(eta_spatial), sum(valid_eta))
             eta[which(valid_eta)[1:min_len]] <- eta[which(valid_eta)[1:min_len]] + eta_spatial[1:min_len]
        } else {
             warning("Could not add spatial effects due to length mismatch or NA eta.")
        }


        lambda <- exp(eta) # NAs in eta will result in NAs in lambda

        # Impute remaining NAs in lambda (e.g., from NA in X or failed spatial effect)
        na_lambda <- is.na(lambda)
        if(any(na_lambda)) {
            # Use mean of training response for imputation
            # Use the stored response_var name
            if(!is.null(self$data_with_re) && !is.null(self$response_var) && self$response_var %in% colnames(self$data_with_re)) {
                 mean_response <- mean(self$data_with_re[[self$response_var]], na.rm = TRUE)
                 lambda[na_lambda] <- mean_response
                 cat("Imputed", sum(na_lambda), "NA predictions in CARBayes with training mean response.\n")
            } else {
                 lambda[na_lambda] <- 1 # Fallback if training response unavailable
                 cat("Imputed", sum(na_lambda), "NA predictions in CARBayes with 1 (training mean unavailable).\n")
            }
        }

        return(lambda)
    },
    
    summary = function() {
      if (is.null(self$results) || is.null(self$response_var) || is.null(self$data_with_re)) {
        return(list(mae = NA, rmse = NA)) # Return NA if essential components are missing
      }

      # Use the stored response variable name and the data used for fitting
      response_col <- self$response_var
      if (!response_col %in% colnames(self$data_with_re)) {
          warning(paste("Response variable", response_col, "not found in stored data for summary."))
          return(list(mae = NA, rmse = NA))
      }

      y_true <- as.numeric(self$data_with_re[[response_col]])
      # Predict on the training data used for fitting
      y_pred <- as.numeric(self$predict(newdata = NULL)) # newdata=NULL predicts on self$data_with_re

      # Ensure lengths match before calculating metrics
      if (length(y_true) != length(y_pred)) {
          warning("Length mismatch between true values and predictions in summary. Cannot calculate metrics.")
          return(list(mae = NA, rmse = NA))
      }

      mae <- mean(abs(y_true - y_pred), na.rm = TRUE)
      rmse <- sqrt(mean((y_true - y_pred)^2, na.rm = TRUE))

      return(list(mae = mae, rmse = rmse))
    }
  )
)

# Simplified function to train and evaluate CARBayes on the whole dataset
train_and_evaluate_carbayes <- function(data, response_var = "acc", random_effect = "borough", max_features = 30) {
  cat("--- Preparing Data and Selecting Features ---\n")
  # Exclude identifier and spatial columns for feature selection
  exclude_cols <- c('int_no', 'acc', 'pi', 'borough', 'x', 'y', 'idx_spatial') # Removed district/part IDs
  feature_cols <- setdiff(colnames(data), exclude_cols)

  # Ensure feature columns actually exist after exclusions
  feature_cols <- feature_cols[feature_cols %in% colnames(data)]
  if (length(feature_cols) == 0) {
      cat("Warning: No feature columns found. Using intercept only.\n")
      X_data <- matrix(1, nrow = nrow(data), ncol = 1) # Intercept only
      colnames(X_data) <- "(Intercept)"
      selected_features <- character(0) # No features selected
  } else {
      X_data <- data[, feature_cols, drop = FALSE] # Use drop=FALSE for single column case
      y_data <- data[[response_var]]

      # Handle potential missing values in features before LASSO
      X_data[is.na(X_data)] <- 0

      selected_features <- select_features_with_lasso(X_data, y_data, max_features)
      cat("Selected features via LASSO:", paste(selected_features, collapse=", "), "\n")
  }

  # Create formula string
  if (length(selected_features) > 0) {
      formula_str_base <- paste(selected_features, collapse = " + ")
  } else {
      formula_str_base <- "1" # Intercept only if no features selected
  }
  formula_str <- paste(response_var, "~", formula_str_base)
  formula <- as.formula(formula_str)
  cat("Using formula:", formula_str, "\n")


  # --- Train CARBayes model on the entire dataset ---
  cat("--- Training CARBayes Model on Full Dataset ---\n")
  # Create spatial weights from the full dataset
  if(nrow(data) > 3 && !any(is.na(data$x)) && !any(is.na(data$y))) {
      W_full <- create_spatial_weights(data[, c('x', 'y')], k_neighbors = 10)
  } else {
      cat("Error: Insufficient data or NA coordinates in the dataset for CARBayes weights. Aborting.\n")
      return(NULL) # Cannot proceed without weights
  }

  carbayes <- carbayes_model$new(
    data = data, # Use full data
    formula = formula_str,
    random_effect = random_effect,
    exposure = NULL,
    spatial_vars = c('x', 'y')
  )
  # Explicitly set the full weights before fitting
  carbayes$W <- W_full

  carbayes_results <- try(carbayes$fit(), silent = TRUE)

  if (inherits(carbayes_results, "try-error")) {
      cat("ERROR fitting CARBayes model:", attr(carbayes_results, "condition")$message, "\n")
      train_preds <- rep(NA, nrow(data))
      train_mae <- NA; train_mse <- NA; train_rmse <- NA
      model_summary_stats <- NULL
  } else {
      # Predict on the training data itself
      cat("--- Predicting on Training Data ---\n")
      train_preds <- carbayes$predict(newdata = NULL) # Predict on training data

      # Calculate metrics on the training set
      actual_values <- data[[response_var]]
      train_mae <- mean(abs(actual_values - train_preds), na.rm = TRUE)
      train_mse <- mean((actual_values - train_preds)^2, na.rm = TRUE)
      train_rmse <- sqrt(train_mse)
      cat(sprintf("CARBayes Training: MAE = %.4f, RMSE = %.4f\n", train_mae, train_rmse))
      # Get model summary (coefficients etc.)
      model_summary_stats <- summary(carbayes$results)
  }

  # Return results
  return(list(
    model_object = carbayes, # Return the fitted model object
    results_summary = model_summary_stats, # Summary from CARBayes fit
    performance = list(
        mae = train_mae,
        mse = train_mse,
        rmse = train_rmse
    ),
    predictions = data.frame(
        int_no = data$int_no,
        actual = data[[response_var]],
        predicted = train_preds
    ),
    selected_features = selected_features,
    formula = formula_str
  ))
}


# Generate figures from CARBayes results (No longer CV)
generate_carbayes_figures <- function(carbayes_output, full_data) {
  # --- Initial Checks and Data Prep ---
  if (is.null(carbayes_output) || is.null(carbayes_output$model_object) || is.null(carbayes_output$predictions)) {
      warning("CARBayes output, model object, or predictions are missing. Cannot generate figures.")
      return(list()) # Return empty list
  }

  model_results <- carbayes_output$model_object$results
  predictions_df <- carbayes_output$predictions

  # Ensure full_data has the necessary columns
  if (!all(c("x", "y", "acc", "int_no") %in% colnames(full_data))) {
      stop("`full_data` must contain 'x', 'y', 'acc', and 'int_no' columns.")
  }
   # Ensure predictions_df has necessary columns
  if (!all(c("int_no", "actual", "predicted") %in% colnames(predictions_df))) {
      stop("`predictions_df` must contain 'int_no', 'actual', and 'predicted' columns.")
  }

  # Merge predictions with full data coordinates (using int_no)
  plot_data <- merge(full_data[, c("int_no", "x", "y")], predictions_df, by = "int_no", all.x = TRUE) # Use all.x=TRUE to keep all locations

  # Calculate residuals
  plot_data$residuals <- plot_data$actual - plot_data$predicted

  # Extract mean spatial random effects (phi) if available
  phi_mean <- NULL
  if (!is.null(model_results$samples$phi)) {
      phi_samples <- model_results$samples$phi
      # Ensure phi corresponds to the original data order used for fitting
      # Assuming the order in samples$phi matches the order in model_object$data_with_re
      if (!is.null(carbayes_output$model_object$data_with_re) && ncol(phi_samples) == nrow(carbayes_output$model_object$data_with_re)) {
          phi_mean <- apply(phi_samples, 2, mean)
          # Create a temporary df with phi and the index used for fitting
          phi_df <- data.frame(
              idx_spatial = carbayes_output$model_object$data_with_re$idx_spatial, # Assuming idx_spatial was in the fitted data
              phi_mean = phi_mean
          )
          # Merge phi into plot_data using the original index
          plot_data <- merge(plot_data, phi_df, by.x = "int_no", by.y="idx_spatial", all.x = TRUE) # Adjust merge keys if needed
          cat("Mean spatial random effects (phi) extracted and merged.\n")
      } else {
          warning("Mismatch between phi samples columns and fitted data rows, or fitted data missing. Skipping phi map.")
      }
  } else {
      warning("Spatial random effects (phi) not found in model results. Skipping phi map.")
  }

  # Remove rows with NA coordinates for spatial plots
  plot_data_spatial <- plot_data[!is.na(plot_data$x) & !is.na(plot_data$y), ]

  # Initialize list to store plots
  plots_list <- list()

  # --- Figure 1: Map of Predicted Accidents (REMOVED) ---
  # Code for map_plot has been removed

  # --- Figure 2: Predicted vs Actual / Observed vs Fitted (Existing, slightly modified) ---
  scatter_plot <- NULL
  plot_data_scatter <- plot_data[!is.na(plot_data$actual) & !is.na(plot_data$predicted), ]
  if(nrow(plot_data_scatter) > 0) {
      scatter_plot <- ggplot(plot_data_scatter, aes(x = predicted, y = actual)) + # Swapped axes to match Observed vs Fitted convention
        geom_point(alpha = 0.5) +
        geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
        geom_smooth(method = "lm", color = "blue", se = TRUE) +
        theme_classic() +
        theme(
          panel.background = element_rect(fill = "white", color = NA),
          plot.background = element_rect(fill = "white", color = NA),
          legend.background = element_rect(fill = "white", color = NA)
        ) +
        labs(title = "Observed vs. Fitted Accidents (Full Dataset)", # Updated title
             subtitle = "CARBayes Model Training Results", # Updated subtitle
             x = "Fitted Accidents (Predicted)", # Updated label
             y = "Observed Accidents (Actual)") # Updated label
      plots_list$scatter_obs_fitted <- scatter_plot

      # Calculate correlation
      correlation <- tryCatch(cor(plot_data_scatter$actual, plot_data_scatter$predicted), error = function(e) NA)
      if(!is.na(correlation)) {
          cat(sprintf("\nCorrelation between observed and fitted values (Training): %.4f\n", correlation))
      } else {
          cat("\nCould not calculate correlation between observed and fitted values.\n")
      }
  } else {
       warning("No valid actual/predicted values for scatter plot.")
  }

  # --- Figure 3: Spatial Weights Visualization ---
  weights_plot <- NULL
  W_matrix <- carbayes_output$model_object$W
  coords_for_weights <- full_data[, c("int_no", "x", "y")] # Use original full_data coords

  if (!is.null(W_matrix) && nrow(coords_for_weights) > 0 && nrow(W_matrix) == nrow(coords_for_weights)) {
      tryCatch({
          # Assign int_no as names if possible (assuming W rows match coords rows)
          rownames(W_matrix) <- coords_for_weights$int_no
          colnames(W_matrix) <- coords_for_weights$int_no

          # Convert W matrix to a long format data frame for links
          W_df <- melt(as.matrix(W_matrix), varnames = c("from_int", "to_int"), value.name = "weight")

          # Filter out zero weights and self-loops
          links <- W_df[W_df$weight > 1e-9 & W_df$from_int != W_df$to_int, ] # Use tolerance for floating point

          # Merge with coordinates
          links <- merge(links, coords_for_weights, by.x = "from_int", by.y = "int_no")
          names(links)[names(links) == "x"] <- "x_start"
          names(links)[names(links) == "y"] <- "y_start"
          links <- merge(links, coords_for_weights, by.x = "to_int", by.y = "int_no")
          names(links)[names(links) == "x"] <- "x_end"
          names(links)[names(links) == "y"] <- "y_end"

          # Create the plot
          weights_plot <- ggplot() +
            geom_segment(data = links, aes(x = x_start, y = y_start, xend = x_end, yend = y_end, color = weight), alpha = 0.5) +
            geom_point(data = coords_for_weights, aes(x = x, y = y), size = 0.5, color = "black") +
            scale_color_viridis_c(option = "plasma", name = "Spatial Weight\n(Binary weights)") +
            labs(title = "Intersection Map with Spatial Links",
                 subtitle = paste("Links based on k=", carbayes_output$model_object$W_list$num[1] %||% "unknown", " nearest neighbors"), # Attempt to get k
                 x = "X Coordinate", y = "Y Coordinate") +
            theme_minimal() +
            theme(aspect.ratio = 1, plot.background = element_rect(fill = "white", color = NA))
          plots_list$map_weights <- weights_plot
      }, error = function(e) {
          cat("Error creating spatial weights plot:", e$message, "\n")
      })
  } else {
      warning("W matrix or coordinates missing/mismatched for spatial weights plot.")
  }

  # --- Figure 4: Coefficient Plot ---
  coef_plot <- NULL
  if (!is.null(model_results$summary.results)) {
      tryCatch({
          summary_df <- as.data.frame(model_results$summary.results)
          summary_df$Parameter <- rownames(summary_df)

          # Filter out non-coefficient parameters (adjust regex as needed)
          coef_df <- summary_df[!grepl("^(tau2|rho|deviance|loglikelihood|phi\\[)", summary_df$Parameter, ignore.case = TRUE), ]

          # Check for standard columns and rename
          if (all(c("Mean", "2.5%", "97.5%") %in% colnames(coef_df))) {
              names(coef_df)[names(coef_df) == "Mean"] <- "Estimate"
              names(coef_df)[names(coef_df) == "2.5%"] <- "LowerCI"
              names(coef_df)[names(coef_df) == "97.5%"] <- "UpperCI"
          } else if (all(c("Median", "2.5%", "97.5%") %in% colnames(coef_df))) {
              names(coef_df)[names(coef_df) == "Median"] <- "Estimate"
              names(coef_df)[names(coef_df) == "2.5%"] <- "LowerCI"
              names(coef_df)[names(coef_df) == "97.5%"] <- "UpperCI"
              cat("Using Median for point estimate in coefficient plot.\n")
          } else {
              warning("Could not find standard columns for estimates and CIs in summary.results. Coefficient plot might be incorrect.")
              coef_df$Estimate <- NA; coef_df$LowerCI <- NA; coef_df$UpperCI <- NA # Add placeholders
          }

          # Ensure numeric types
          coef_df$Estimate <- as.numeric(coef_df$Estimate)
          coef_df$LowerCI <- as.numeric(coef_df$LowerCI)
          coef_df$UpperCI <- as.numeric(coef_df$UpperCI)

          # Create the plot
          coef_plot <- ggplot(coef_df, aes(x = Estimate, y = reorder(Parameter, Estimate))) +
            geom_pointrange(aes(xmin = LowerCI, xmax = UpperCI)) +
            geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
            labs(title = "CARBayes Model Coefficients",
                 subtitle = "Posterior Mean/Median and 95% Credible Intervals",
                 x = "Coefficient Estimate", y = "Parameter") +
            theme_minimal() +
            theme(axis.text.y = element_text(size = 8), plot.background = element_rect(fill = "white", color = NA))
          plots_list$plot_coefficients <- coef_plot
      }, error = function(e) {
          cat("Error creating coefficient plot:", e$message, "\n")
      })
  } else {
      warning("Model summary results not found for coefficient plot.")
  }

  # --- Figure 5: Residual Map ---
  residual_map <- NULL
  plot_data_res <- plot_data_spatial[!is.na(plot_data_spatial$residuals), ]
  if(nrow(plot_data_res) > 0) {
      tryCatch({
          sp_residuals <- plot_data_res
          coordinates(sp_residuals) <- ~ x + y
          sf_residuals <- st_as_sf(sp_residuals)

          max_abs_res <- max(abs(sf_residuals$residuals), na.rm = TRUE)

          residual_map <- ggplot() +
            geom_sf(data = sf_residuals, aes(color = residuals), size = 1) +
            scale_color_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0,
                                  limit = c(-max_abs_res, max_abs_res), name = "Residuals\n(Obs - Pred)") +
            theme_classic() +
            theme(panel.background = element_rect(fill = "white", color = NA),
                  plot.background = element_rect(fill = "white", color = NA),
                  legend.background = element_rect(fill = "white", color = NA)) +
            labs(title = "Map of Model Residuals",
                 subtitle = "Blue = Over-prediction, Red = Under-prediction")
          plots_list$map_residuals <- residual_map
      }, error = function(e) {
          cat("Error creating residual map:", e$message, "\n")
      })
  } else {
      warning("No valid residuals for residual map.")
  }

  # --- Figure 6: Spatial Random Effects (Phi) Map ---
  phi_map <- NULL
  plot_data_phi <- plot_data_spatial[!is.na(plot_data_spatial$phi_mean), ]
  if(nrow(plot_data_phi) > 0) {
      tryCatch({
          sp_phi <- plot_data_phi
          coordinates(sp_phi) <- ~ x + y
          sf_phi <- st_as_sf(sp_phi)

          phi_map <- ggplot() +
            geom_sf(data = sf_phi, aes(color = phi_mean), size = 1) +
            scale_color_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0, name = "Mean Spatial\nEffect (Phi)") +
            theme_classic() +
            theme(panel.background = element_rect(fill = "white", color = NA),
                  plot.background = element_rect(fill = "white", color = NA),
                  legend.background = element_rect(fill = "white", color = NA)) +
            labs(title = "Map of Spatial Random Effects (Posterior Mean Phi)",
                 subtitle = "Shows underlying spatial pattern")
          plots_list$map_phi <- phi_map
      }, error = function(e) {
          cat("Error creating phi map:", e$message, "\n")
      })
  } else {
      warning("No valid phi values for phi map.")
  }

  # --- Figure 7: Residuals vs. Fitted Plot ---
  res_fitted_plot <- NULL
  plot_data_resfit <- plot_data[!is.na(plot_data$residuals) & !is.na(plot_data$predicted), ]
  if(nrow(plot_data_resfit) > 0) {
      res_fitted_plot <- ggplot(plot_data_resfit, aes(x = predicted, y = residuals)) +
        geom_point(alpha = 0.5) +
        geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
        geom_smooth(method = "loess", color = "blue", se = FALSE) + # Added loess smoother
        theme_classic() +
        theme(panel.background = element_rect(fill = "white", color = NA),
              plot.background = element_rect(fill = "white", color = NA)) +
        labs(title = "Residuals vs. Fitted Values",
             subtitle = "Should show random scatter around zero",
             x = "Fitted Values (Predicted)",
             y = "Residuals (Observed - Fitted)")
      plots_list$plot_res_fitted <- res_fitted_plot
  } else {
      warning("No valid residuals/fitted values for residuals vs. fitted plot.")
  }

  # --- Figure 8: Moran's I Test and Plot for Residuals ---
  # Note: Moran plot is saved directly, not returned as ggplot object
  random_effect_name <- carbayes_output$model_object$random_effect
  moran_plot_path <- file.path("results", paste0("moran_plot_residuals_", 
                                                ifelse(is.null(random_effect_name), 
                                                      "without_re", "with_re"), 
                                                ".png"))
  plots_list$moran_plot_path <- NULL # Initialize path as NULL

  if (!is.null(W_matrix) && nrow(plot_data_res) > 0) { # Use plot_data_res which has NA residuals removed
      residuals_vec <- plot_data_res$residuals
      # Ensure W corresponds to the data points with non-NA residuals
      # This requires careful indexing if plot_data_res is a subset of original data
      # Assuming W_matrix rows/cols correspond to full_data rows
      # Find indices of plot_data_res within full_data (if int_no is unique identifier)
      valid_indices <- match(plot_data_res$int_no, full_data$int_no)
      valid_indices <- valid_indices[!is.na(valid_indices)] # Remove NAs if any int_no wasn't found

      if(length(valid_indices) == length(residuals_vec) && length(valid_indices) > 1) {
          W_subset <- W_matrix[valid_indices, valid_indices]

          if (!isSymmetric(W_subset)) {
              warning("W subset is not symmetric, symmetrizing for Moran's I.")
              W_subset <- (W_subset + t(W_subset)) / 2
          }

          no_neighbor_rows <- which(rowSums(W_subset) == 0)
          listw_obj <- NULL
          residuals_vec_moran <- residuals_vec

          if (length(no_neighbor_rows) > 0) {
              warning(paste("Found", length(no_neighbor_rows), "locations with no neighbors in W subset. Excluding for Moran's I."), call. = FALSE)
              if(length(no_neighbor_rows) < nrow(W_subset)) {
                  residuals_vec_moran <- residuals_vec[-no_neighbor_rows]
                  W_subset_moran <- W_subset[-no_neighbor_rows, -no_neighbor_rows]
                  listw_obj <- tryCatch(mat2listw(W_subset_moran, style = "W"), error = function(e) {cat("Error creating listw (subset):", e$message, "\n"); NULL})
              } else {
                  cat("All points are isolates after subsetting. Cannot perform Moran's I.\n")
              }
          } else {
              listw_obj <- tryCatch(mat2listw(W_subset, style = "W"), error = function(e) {cat("Error creating listw (full subset):", e$message, "\n"); NULL})
          }

          if (!is.null(listw_obj)) {
              # Perform Moran's I test
              moran_test_result <- tryCatch(
                  moran.test(residuals_vec_moran, listw = listw_obj, randomisation = TRUE),
                  error = function(e) {cat("Error during Moran's I test:", e$message, "\n"); NULL}
              )
              if (!is.null(moran_test_result)) {
                  cat("\nMoran's I Test for Model Residuals:\n")
                  print(moran_test_result)
                  plots_list$moran_test_result <- moran_test_result # Store test result
              }

              # Create results directory if it doesn't exist (for Moran plot)
              if (!dir.exists("results")) {
                  dir.create("results")
              }
              
              # Create Moran scatter plot
              cat("\nGenerating Moran Scatter Plot for Residuals...\n")
              png(moran_plot_path, width = 6, height = 6, units = "in", res = 300)
              tryCatch({
                  moran.plot(residuals_vec_moran, listw = listw_obj,
                             main = "Moran Scatter Plot for Model Residuals",
                             xlab = "Residuals", ylab = "Spatially Lagged Residuals")
              }, error = function(e) {
                  cat("Error generating Moran plot:", e$message, "\n")
                  plot(1, type="n", axes=FALSE, xlab="", ylab=""); title("Moran Plot Failed") # Blank plot on error
              }, finally = {
                  dev.off()
              })
              cat(paste("Saved Moran scatter plot to", moran_plot_path, "\n"))
              plots_list$moran_plot_path <- moran_plot_path # Store path if saved
          } else {
              cat("Skipping Moran's I test and plot due to issues creating listw object.\n")
          }
      } else {
           warning("Index mismatch or insufficient data for Moran's I calculation.")
      }
  } else {
      warning("W matrix or residuals missing for Moran's I test/plot.")
  }

  # Return the list of ggplot objects and other info
  # Keep original prediction data frame as well
  plots_list$predictions_data <- plot_data # Include data used for plots

  return(plots_list)
}

# Main execution
cat("Loading and preparing data...\n")
full_data <- load_and_prepare_data()

# Ensure necessary columns exist before running
required_cols <- c("acc", "borough", "x", "y", "int_no")
if (!all(required_cols %in% colnames(full_data))) {
    stop("Missing required columns in full_data: ", paste(setdiff(required_cols, colnames(full_data)), collapse=", "))
}

# Add this function before the run_and_compare_models function
save_plot_safe <- function(plot, filename, width = 8, height = 6, ...) {
  if (is.null(plot)) {
    warning(paste("Cannot save plot to", filename, "- plot object is NULL"))
    return(FALSE)
  }
  
  # Create results directory if it doesn't exist
  if (!dir.exists("results")) {
    dir.create("results")
  }
  
  # Full path for saving
  filepath <- file.path("results", filename)
  
  # Try to save the plot
  result <- tryCatch({
    ggsave(filepath, plot, width = width, height = height, ...)
    TRUE
  }, error = function(e) {
    warning(paste("Failed to save plot to", filepath, ":", e$message))
    FALSE
  })
  
  if (result) {
    cat(paste("Saved plot to", filepath, "\n"))
  }
  
  return(result)
}

# Define a function to run both models and compare them
run_and_compare_models <- function(data, response_var = "acc", random_effect = "borough", max_features = 40) {
  cat("\n==== Running Models With and Without Random Effect ====\n")
  
  # First run model WITH random effect
  cat("\n--- Training CARBayes Model WITH Random Effect ---\n")
  carbayes_with_re <- train_and_evaluate_carbayes(
    data = data,
    response_var = response_var,
    random_effect = random_effect,
    max_features = max_features
  )
  
  # Then run model WITHOUT random effect
  cat("\n--- Training CARBayes Model WITHOUT Random Effect ---\n")
  carbayes_without_re <- train_and_evaluate_carbayes(
    data = data,
    response_var = response_var,
    random_effect = NULL, # Set to NULL to exclude random effect
    max_features = max_features
  )
  
  # Generate figures for model WITH random effect
  cat("\n--- Generating Figures for Model WITH Random Effect ---\n")
  figures_with_re <- generate_carbayes_figures(carbayes_with_re, data)
  
  # Generate figures for model WITHOUT random effect
  cat("\n--- Generating Figures for Model WITHOUT Random Effect ---\n")
  figures_without_re <- generate_carbayes_figures(carbayes_without_re, data)
  
  # Save figures for model WITH random effect
  cat("\n--- Saving Figures for Model WITH Random Effect ---\n")

  # Create results directory if it doesn't exist
  if (!dir.exists("results")) {
    dir.create("results")
  }

  # Save plots for model WITH random effect
  save_plot_safe(figures_with_re$scatter_obs_fitted, "observed_vs_fitted_with_re.png", width = 8, height = 6)
  save_plot_safe(figures_with_re$map_weights, "spatial_weights_map_with_re.png", width = 10, height = 10)
  save_plot_safe(figures_with_re$plot_coefficients, "coefficients_plot_with_re.png", 
            width = 8, height = max(4, length(figures_with_re$plot_coefficients$data$Parameter) * 0.3), 
            limitsize = FALSE)
  save_plot_safe(figures_with_re$map_residuals, "residuals_map_with_re.png", width = 10, height = 8)
  save_plot_safe(figures_with_re$map_phi, "phi_map_with_re.png", width = 10, height = 8)
  save_plot_safe(figures_with_re$plot_res_fitted, "residuals_vs_fitted_with_re.png", width = 8, height = 6)
  
  # Save plots for model WITHOUT random effect
  cat("\n--- Saving Figures for Model WITHOUT Random Effect ---\n")
  save_plot_safe(figures_without_re$scatter_obs_fitted, "observed_vs_fitted_without_re.png", width = 8, height = 6)
  save_plot_safe(figures_without_re$map_weights, "spatial_weights_map_without_re.png", width = 10, height = 10)
  save_plot_safe(figures_without_re$plot_coefficients, "coefficients_plot_without_re.png", 
            width = 8, height = max(4, length(figures_without_re$plot_coefficients$data$Parameter) * 0.3), 
            limitsize = FALSE)
  save_plot_safe(figures_without_re$map_residuals, "residuals_map_without_re.png", width = 10, height = 8)
  save_plot_safe(figures_without_re$map_phi, "phi_map_without_re.png", width = 10, height = 8)
  save_plot_safe(figures_without_re$plot_res_fitted, "residuals_vs_fitted_without_re.png", width = 8, height = 6)
  
  # Ensure Moran plots are generated for both models
  # For model WITH random effect
  if (is.null(figures_with_re$moran_plot_path)) {
    cat("\n--- Generating Moran Plot for Model WITH Random Effect ---\n")
    # Create Moran plot for model WITH random effect
    create_moran_plot(carbayes_with_re, data, "moran_plot_residuals_with_re.png")
  } else {
    cat("\nMoran plot for model WITH random effect already generated at:", figures_with_re$moran_plot_path, "\n")
  }
  
  # For model WITHOUT random effect
  if (is.null(figures_without_re$moran_plot_path)) {
    cat("\n--- Generating Moran Plot for Model WITHOUT Random Effect ---\n")
    # Create Moran plot for model WITHOUT random effect
    create_moran_plot(carbayes_without_re, data, "moran_plot_residuals_without_re.png")
  } else {
    cat("\nMoran plot for model WITHOUT random effect already generated at:", figures_without_re$moran_plot_path, "\n")
  }
  
  # Compare model performance
  cat("\n==== Model Comparison ====\n")
  
  # Performance metrics
  cat("\n--- Performance Metrics ---\n")
  metrics_df <- data.frame(
    Model = c("With Random Effect", "Without Random Effect"),
    MAE = c(carbayes_with_re$performance$mae, carbayes_without_re$performance$mae),
    RMSE = c(carbayes_with_re$performance$rmse, carbayes_without_re$performance$rmse)
  )
  print(metrics_df)
  
  # Print spatial parameters (rho and tau2)
  cat("\n--- Spatial Parameters Estimates ---\n")
  
  # For model WITH random effect
  cat("Model WITH Random Effect:\n")
  if (!is.null(carbayes_with_re$model_object$results$summary.results)) {
    summary_with_re <- as.data.frame(carbayes_with_re$model_object$results$summary.results)
    
    # Extract rho (spatial autocorrelation parameter)
    if ("rho" %in% rownames(summary_with_re)) {
      rho_with_re <- summary_with_re["rho", ]
      cat("rho (spatial autocorrelation):\n")
      print(rho_with_re)
      cat("\n")
    } else {
      cat("rho parameter not found in model results\n")
    }
    
    # Extract tau2 (variance parameter)
    if ("tau2" %in% rownames(summary_with_re)) {
      tau2_with_re <- summary_with_re["tau2", ]
      cat("tau2 (variance parameter):\n")
      print(tau2_with_re)
      cat("\n")
    } else {
      cat("tau2 parameter not found in model results\n")
    }
    
    # Print interpretation of rho
    if ("rho" %in% rownames(summary_with_re)) {
      rho_mean <- summary_with_re["rho", "Mean"]
      cat("Interpretation of rho (", rho_mean, "):\n", sep="")
      if (rho_mean > 0.8) {
        cat("- Strong spatial autocorrelation (rho > 0.8)\n")
        cat("- The spatial component is highly important in this model\n")
      } else if (rho_mean > 0.5) {
        cat("- Moderate spatial autocorrelation (0.5 < rho < 0.8)\n")
        cat("- The spatial component plays a significant role in this model\n")
      } else if (rho_mean > 0.2) {
        cat("- Weak spatial autocorrelation (0.2 < rho < 0.5)\n")
        cat("- The spatial component has some influence in this model\n")
      } else {
        cat("- Very weak or no spatial autocorrelation (rho < 0.2)\n")
        cat("- The spatial component has minimal influence in this model\n")
      }
    }
  } else {
    cat("Summary results not available for model WITH random effect\n")
  }
  
  # For model WITHOUT random effect
  cat("\nModel WITHOUT Random Effect:\n")
  if (!is.null(carbayes_without_re$model_object$results$summary.results)) {
    summary_without_re <- as.data.frame(carbayes_without_re$model_object$results$summary.results)
    
    # Extract rho (spatial autocorrelation parameter)
    if ("rho" %in% rownames(summary_without_re)) {
      rho_without_re <- summary_without_re["rho", ]
      cat("rho (spatial autocorrelation):\n")
      print(rho_without_re)
      cat("\n")
    } else {
      cat("rho parameter not found in model results\n")
    }
    
    # Extract tau2 (variance parameter)
    if ("tau2" %in% rownames(summary_without_re)) {
      tau2_without_re <- summary_without_re["tau2", ]
      cat("tau2 (variance parameter):\n")
      print(tau2_without_re)
      cat("\n")
    } else {
      cat("tau2 parameter not found in model results\n")
    }
    
    # Print interpretation of rho
    if ("rho" %in% rownames(summary_without_re)) {
      rho_mean <- summary_without_re["rho", "Mean"]
      cat("Interpretation of rho (", rho_mean, "):\n", sep="")
      if (rho_mean > 0.8) {
        cat("- Strong spatial autocorrelation (rho > 0.8)\n")
        cat("- The spatial component is highly important in this model\n")
      } else if (rho_mean > 0.5) {
        cat("- Moderate spatial autocorrelation (0.5 < rho < 0.8)\n")
        cat("- The spatial component plays a significant role in this model\n")
      } else if (rho_mean > 0.2) {
        cat("- Weak spatial autocorrelation (0.2 < rho < 0.5)\n")
        cat("- The spatial component has some influence in this model\n")
      } else {
        cat("- Very weak or no spatial autocorrelation (rho < 0.2)\n")
        cat("- The spatial component has minimal influence in this model\n")
      }
    }
  } else {
    cat("Summary results not available for model WITHOUT random effect\n")
  }
  
  # Moran's I test results
  cat("\n--- Moran's I Test Results ---\n")
  cat("Model WITH Random Effect:\n")
  if (!is.null(figures_with_re$moran_test_result)) {
    print(figures_with_re$moran_test_result)
  } else {
    cat("Moran's I test result not available\n")
  }
  
  cat("\nModel WITHOUT Random Effect:\n")
  if (!is.null(figures_without_re$moran_test_result)) {
    print(figures_without_re$moran_test_result)
  } else {
    cat("Moran's I test result not available\n")
  }
  
  # DIC comparison
  cat("\n--- DIC Comparison ---\n")
  if (!is.null(carbayes_with_re$model_object$results$modelfit) && 
      !is.null(carbayes_without_re$model_object$results$modelfit)) {
    dic_with_re <- carbayes_with_re$model_object$results$modelfit["DIC"]
    dic_without_re <- carbayes_without_re$model_object$results$modelfit["DIC"]
    
    cat("DIC (with RE):", dic_with_re, "\n")
    cat("DIC (without RE):", dic_without_re, "\n")
    cat("DIC difference (without - with):", dic_without_re - dic_with_re, "\n")
    
    if (dic_with_re < dic_without_re) {
      cat("Conclusion: The model WITH random effect has lower DIC and is preferred.\n")
    } else if (dic_without_re < dic_with_re) {
      cat("Conclusion: The model WITHOUT random effect has lower DIC and is preferred.\n")
    } else {
      cat("Conclusion: Both models have similar DIC values.\n")
    }
  } else {
    cat("Cannot compare DIC: Values not available\n")
  }
  
  # Return both model outputs and figures
  return(list(
    with_re = carbayes_with_re,
    without_re = carbayes_without_re,
    figures_with_re = figures_with_re,
    figures_without_re = figures_without_re
  ))
}

# Add a dedicated function to create Moran plots
create_moran_plot <- function(model_output, full_data, filename) {
  if (is.null(model_output) || is.null(model_output$model_object) || is.null(model_output$predictions)) {
    warning("Model output, model object, or predictions are missing. Cannot generate Moran plot.")
    return(NULL)
  }
  
  # Create results directory if it doesn't exist
  if (!dir.exists("results")) {
    dir.create("results")
  }
  
  # Full path for saving
  moran_plot_path <- file.path("results", filename)
  
  # Get predictions and merge with spatial data
  predictions_df <- model_output$predictions
  plot_data <- merge(full_data[, c("int_no", "x", "y")], predictions_df, by = "int_no", all.x = TRUE)
  
  # Calculate residuals
  plot_data$residuals <- plot_data$actual - plot_data$predicted
  
  # Remove rows with NA coordinates or residuals
  plot_data_res <- plot_data[!is.na(plot_data$x) & !is.na(plot_data$y) & !is.na(plot_data$residuals), ]
  
  # Get spatial weights matrix
  W_matrix <- model_output$model_object$W
  
  if (!is.null(W_matrix) && nrow(plot_data_res) > 0) {
    residuals_vec <- plot_data_res$residuals
    
    # Find indices of plot_data_res within full_data
    valid_indices <- match(plot_data_res$int_no, full_data$int_no)
    valid_indices <- valid_indices[!is.na(valid_indices)] # Remove NAs if any int_no wasn't found

    if(length(valid_indices) == length(residuals_vec) && length(valid_indices) > 1) {
      W_subset <- W_matrix[valid_indices, valid_indices]
      
      if (!isSymmetric(W_subset)) {
        warning("W subset is not symmetric, symmetrizing for Moran's I.")
        W_subset <- (W_subset + t(W_subset)) / 2
      }
      
      no_neighbor_rows <- which(rowSums(W_subset) == 0)
      listw_obj <- NULL
      residuals_vec_moran <- residuals_vec

      if (length(no_neighbor_rows) > 0) {
        warning(paste("Found", length(no_neighbor_rows), "locations with no neighbors in W subset. Excluding for Moran's I."), call. = FALSE)
        if(length(no_neighbor_rows) < nrow(W_subset)) {
          residuals_vec_moran <- residuals_vec[-no_neighbor_rows]
          W_subset_moran <- W_subset[-no_neighbor_rows, -no_neighbor_rows]
          listw_obj <- tryCatch(mat2listw(W_subset_moran, style = "W"), error = function(e) {cat("Error creating listw (subset):", e$message, "\n"); NULL})
        } else {
          cat("All points are isolates after subsetting. Cannot perform Moran's I.\n")
        }
      } else {
        listw_obj <- tryCatch(mat2listw(W_subset, style = "W"), error = function(e) {cat("Error creating listw (full subset):", e$message, "\n"); NULL})
      }

      if (!is.null(listw_obj)) {
        # Perform Moran's I test
        moran_test_result <- tryCatch(
          moran.test(residuals_vec_moran, listw = listw_obj, randomisation = TRUE),
          error = function(e) {cat("Error during Moran's I test:", e$message, "\n"); NULL}
        )
        if (!is.null(moran_test_result)) {
          cat("\nMoran's I Test Result:\n")
          print(moran_test_result)
        }
        
        # Create Moran scatter plot
        cat("\nGenerating Moran Scatter Plot...\n")
        png(moran_plot_path, width = 6, height = 6, units = "in", res = 300)
        tryCatch({
          moran.plot(residuals_vec_moran, listw = listw_obj,
                     main = "Moran Scatter Plot for Model Residuals",
                     xlab = "Residuals", ylab = "Spatially Lagged Residuals")
        }, error = function(e) {
          cat("Error generating Moran plot:", e$message, "\n")
          plot(1, type="n", axes=FALSE, xlab="", ylab=""); title("Moran Plot Failed") # Blank plot on error
        }, finally = {
          dev.off()
        })
        cat(paste("Saved Moran scatter plot to", moran_plot_path, "\n"))
        return(moran_plot_path)
      } else {
        cat("Skipping Moran's I test and plot due to issues creating listw object.\n")
      }
    } else {
      warning("Index mismatch or insufficient data for Moran's I calculation.")
    }
  } else {
    warning("W matrix or residuals missing for Moran's I test/plot.")
  }
  
  return(NULL)
}

# Run both models and compare them
model_comparison <- run_and_compare_models(
  data = full_data,
  response_var = "acc",
  random_effect = "borough",
  max_features = 40
)

cat("\nScript finished.\n")