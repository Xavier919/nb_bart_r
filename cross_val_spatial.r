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
library(spatialreg)
library(spgwr)
library(sp)
library(sf)
library(mapview)
library(INLA)

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
      dist <- nn$nn.dist[i, j]
      W[i, idx] <- 1.0 / max(dist, 0.0001)
    }
  }
  
  W_symmetric <- (W + t(W)) / 2
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

make_spatial_weights_for_sem <- function(sp_data, k) {
  id <- row.names(as(sp_data, "data.frame"))
  neighbours <- knn2nb(knearneigh(coordinates(sp_data), k = k), row.names = id)
  listw <- nb2listw(neighbours, style = "B")
  return(listw)
}

carbayes_model <- R6::R6Class(
  "carbayes_model",
  
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
      self$exposure <- NULL
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

      model <- S.CARleroux(
        formula = full_formula, # Use the formula passed in
        family = "poisson",
        data = data_to_use, # data_to_use has factor conversion if needed
        W = self$W,
        burnin = 2000,
        n.sample = 10000,
        thin = 4,
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
            if(!is.null(self$data_with_re) && response_var %in% colnames(self$data_with_re)) {
                 mean_response <- mean(self$data_with_re[[response_var]], na.rm = TRUE)
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
      if (is.null(self$results)) {
        return(NULL)
      }
      
      y_true <- as.numeric(self$data[[strsplit(self$formula, "~")[[1]][1]]])
      y_pred <- as.numeric(self$predict())
      
      mae <- mean(abs(y_true - y_pred), na.rm = TRUE)
      rmse <- sqrt(mean((y_true - y_pred)^2, na.rm = TRUE))
      
      return(list(mae = mae, rmse = rmse))
    }
  )
)

# Function to train the SEM Poisson model (Modified for Block CV)
train_poisson_sem <- function(formula, train_data, val_data, fold_weights) {
  # Step 1: Fit a standard Poisson model first on training data
  poisson_model <- glm(formula, data = train_data, family = poisson(link = "log"))

  # Step 2: Extract residuals from the training model
  residuals_train <- residuals(poisson_model, type = "deviance")

  # Step 3: Fit spatial error model to the training residuals using training weights
  train_data$residuals <- residuals_train
  # Use tryCatch for robustness if errorsarlm fails
  sem_residuals_model <- tryCatch({
      errorsarlm(as.formula("residuals ~ 1"), data = train_data, listw = fold_weights, zero.policy = TRUE)
  }, error = function(e) {
      cat("Warning: errorsarlm failed. Proceeding without spatial error adjustment for this fold.\n")
      # print(e) # Optional: print the error message
      return(NULL)
  })

  # Step 4: Predict on validation data using only the fixed effects part
  # The spatial error term is not directly predictable on a spatially separate block
  # without specific train-to-test weights or assumptions.
  poisson_predictions <- predict(poisson_model, newdata = val_data, type = "response")

  # Optional: Add a constant adjustment based on the mean fitted spatial error on train?
  # This is a simplification. A more advanced approach would be needed for true spatial prediction.
  # if (!is.null(sem_residuals_model)) {
  #   mean_spatial_effect <- mean(fitted(sem_residuals_model))
  #   adjusted_predictions <- poisson_predictions * exp(mean_spatial_effect)
  # } else {
  #   adjusted_predictions <- poisson_predictions
  # }
  # For simplicity and clarity of block CV, let's return only poisson predictions
  adjusted_predictions <- poisson_predictions

  return(adjusted_predictions)
}

# Combined cross-validation function for CARBayes, SEM, and INLA models
# MODIFIED for Stratified 5-Fold CV across districts
run_combined_cross_validation <- function(data, response_var = "acc", random_effect = "borough", max_features = 30) {
  set.seed(42)
  k <- 5 # Define the number of folds

  # --- Define District Mapping (Used for Stratification) ---
  # This mapping uses the final borough/zone names present *AFTER* load_and_prepare_data
  borough_to_district <- list(
    `1` = c('Zone est', 'Montréal-Nord', 'Saint-Léonard', 'Mercier-Hochelaga-Maisonneuve'),
    `2` = c('Zone centre', 'Rosemont-La-Petite-Patrie', 'Villeray-Saint-Michel-Parc-Extension'),
    `3` = c('Plateau-Mont-Royal', 'Ville-Marie'),
    `4` = c('Zone ouest', 'Ahuntsic-Cartierville', 'Saint-Laurent'),
    `5` = c('Zone sud', 'Zone centre-sud')
  )
  num_districts <- length(borough_to_district) # Should also be 5 in this case

  if (!random_effect %in% colnames(data)) {
      stop(paste("Error: The specified random effect column '", random_effect, "' (used for district assignment) does not exist in the data.", sep=""))
  }
  # Ensure the column is a factor AFTER potential modifications in load_and_prepare_data
  if (!is.factor(data[[random_effect]])) {
      data[[random_effect]] <- as.factor(data[[random_effect]])
      cat("Converting borough/zone column to factor for district assignment.\n")
  }

  data$district_id <- NA_integer_ # Initialize district ID column

  # Assign district IDs based on the potentially modified borough/zone names
  for (district_num in names(borough_to_district)) {
      boroughs_in_district <- borough_to_district[[district_num]]
      data$district_id[data[[random_effect]] %in% boroughs_in_district] <- as.integer(district_num)
  }

  # Check for unassigned data points
  unassigned_count <- sum(is.na(data$district_id))
  if (unassigned_count > 0) {
      unassigned_boroughs <- unique(data[[random_effect]][is.na(data$district_id)])
      cat(sprintf("Warning: %d data points could not be assigned to a district. Unassigned boroughs/zones: %s\n",
                  unassigned_count, paste(unassigned_boroughs, collapse=", ")))
      cat("Removing unassigned data points from the analysis.\n")
      data <- data[!is.na(data$district_id), ]
      if(nrow(data) == 0) stop("No data remaining after removing unassigned points.")
  }
  data$district_id <- as.factor(data$district_id) # Convert to factor
  cat("District assignments created with the following distribution:\n")
  print(table(data$district_id))
  # ------------------------------------

  # --- Create Stratified Folds (Parts) ---
  cat("\nCreating stratified 5-fold assignments (parts)...\n")
  set.seed(123) # Seed for reproducibility of fold assignment
  data <- data %>%
    group_by(district_id) %>%
    # Randomly assign each intersection within a district to one of 5 parts
    mutate(part_id = sample(cut(seq(1,n()), breaks = k, labels = FALSE))) %>%
    # Alternative using ntile after shuffling:
    # mutate(part_id = ntile(runif(n()), k)) %>%
    ungroup()

  data$part_id <- as.factor(data$part_id) # Ensure part_id is a factor
  cat("Distribution of intersections across parts (folds):\n")
  print(table(data$part_id))
  cat("Distribution of intersections across parts within districts:\n")
  print(table(District = data$district_id, Part = data$part_id))
  # ------------------------------------


  # --- Generate Map of Folds/Parts (Optional Visualization) ---
  cat("\nGenerating map of cross-validation part assignments...\n")

  # Create results directory if it doesn't exist
  dir.create("results", showWarnings = FALSE)

  # --- FIX: Ensure unique column names before plotting ---
  if(any(duplicated(colnames(data)))) {
      cat("WARNING: Duplicate column names found in 'data' before plotting. Keeping first instance...\n")
      duplicate_names <- colnames(data)[duplicated(colnames(data))]
      cat("Duplicate names found:", paste(unique(duplicate_names), collapse=", "), "\n")
      data <- data[, !duplicated(colnames(data))] # Keep only the first column for any duplicated name
      cat("Duplicate columns handled.\n")
  }
  # --- END FIX ---

  # Ensure ggplot2 is loaded
  if (!require(ggplot2)) {
    install.packages("ggplot2")
    library(ggplot2)
  }

  # Use the 'data' object directly as it now contains the part_id
  if ("part_id" %in% colnames(data) && "x" %in% colnames(data) && "y" %in% colnames(data)) {

    # Ensure part_id is a factor for discrete coloring
    if (!is.factor(data$part_id)) {
      data$part_id <- as.factor(data$part_id)
    }

    part_map_plot <- ggplot(data, aes(x = x, y = y, color = part_id)) + # Use 'data' directly, color by part_id
      geom_point(size = 0.5, alpha = 0.7) +
      scale_color_viridis_d(name = "CV Part (Fold)") + # Update legend title
      labs(
        title = "Map of Intersections by Stratified CV Part Assignment", # Update title
        x = "X Coordinate",
        y = "Y Coordinate"
      ) +
      theme_minimal() +
      theme(
         plot.title = element_text(hjust = 0.5),
         legend.position = "bottom",
         plot.background = element_rect(fill = "white", colour = NA),
         panel.background = element_rect(fill = "white", colour = "grey90")
      ) +
      coord_fixed()

    print(part_map_plot) # Print the plot

    # Save the plot
    ggsave("results/cv_stratified_part_map_assignment.png", part_map_plot, width = 8, height = 8) # Updated path
    cat("Part assignment map saved as 'results/cv_stratified_part_map_assignment.png'\n")

  } else {
    cat("Could not generate part map: Required columns ('x', 'y', 'part_id') not found in the data after handling duplicates.\n")
  }
  # --- End Map Generation ---


  # Initialize results for all models
  carbayes_fold_results <- list()
  sem_fold_results <- list()
  inla_fold_results <- list()

  carbayes_test_actual <- list()
  carbayes_test_predicted <- list()
  carbayes_test_ids <- list()
  sem_test_actual <- list()
  sem_test_predicted <- list()
  inla_test_actual <- list()
  inla_test_predicted <- list()

  # --- Preparations needed across folds ---
  pc_prec_unstructured <- list(prior = 'pc.prec', param = c(1, 0.01))
  pc_phi <- list(prior = 'pc', param = c(0.5, 0.5))
  pc_prec_borough <- list(prior = 'pc.prec', param = c(1, 0.01))
  nb_family_control <- list(hyper = list(theta = list(prior = "loggamma", param = c(1, 0.01))))
  initial_theta_values <- c(log(5), 0, 0, 0)
  inla_control_mode <- list(theta = initial_theta_values, restart = TRUE)
  # ---------------------------------------

  feature_formulas <- list()

  # --- Stratified 5-Fold Cross-Validation Loop ---
  for (i in 1:k) { # Iterate through the 5 folds/parts
    cat(sprintf("\n========== Processing Fold %d (Testing Part %d) ==========\n", i, i))
    # Test set = all intersections assigned to part 'i'
    test_idx <- which(data$part_id == i)
    # Training set = all intersections NOT assigned to part 'i'
    train_idx <- which(data$part_id != i)

    # Handle cases where a fold might be empty (unlikely with stratification but good practice)
    if (length(train_idx) == 0 || length(test_idx) == 0) {
        cat("Skipping fold", i, "due to empty train or test set.\n")
        # Assign NAs appropriately if skipping
        carbayes_fold_results[[i]] <- list(fold = i, test_mae = NA, test_mse = NA, test_rmse = NA)
        sem_fold_results[[i]] <- list(fold = i, test_mae = NA, test_mse = NA, test_rmse = NA)
        inla_fold_results[[i]] <- list(fold = i, test_mae = NA, test_mse = NA, test_rmse = NA)
        carbayes_test_actual[[i]] <- numeric(0)
        carbayes_test_predicted[[i]] <- numeric(0)
        carbayes_test_ids[[i]] <- numeric(0)
        sem_test_actual[[i]] <- numeric(0)
        sem_test_predicted[[i]] <- numeric(0)
        inla_test_actual[[i]] <- numeric(0)
        inla_test_predicted[[i]] <- numeric(0)
        next
    }

    train_data <- data[train_idx, ]
    test_data <- data[test_idx, ]

    # --- Feature Selection (using LASSO on training data) ---
    # Exclude part_id and district_id as well
    exclude_cols <- c('int_no', 'acc', 'pi', 'borough', 'x', 'y', 'idx_spatial', 'district_id', 'part_id')
    feature_cols <- setdiff(colnames(train_data), exclude_cols)

    # Ensure feature columns actually exist after exclusions
    feature_cols <- feature_cols[feature_cols %in% colnames(train_data)]
    if (length(feature_cols) == 0) {
        cat("Warning: No feature columns found for fold", i, ". Using intercept only.\n")
        X_train <- matrix(1, nrow = nrow(train_data), ncol = 1) # Intercept only
        colnames(X_train) <- "(Intercept)"
        selected_features <- character(0) # No features selected
    } else {
        X_train <- train_data[, feature_cols, drop = FALSE] # Use drop=FALSE for single column case
        y_train <- train_data[[response_var]]

        # Handle potential missing values in features before LASSO
        X_train[is.na(X_train)] <- 0

        selected_features <- select_features_with_lasso(X_train, y_train, max_features)
    }

    # Create formula string (base for all models)
    if (length(selected_features) > 0) {
        formula_str_base <- paste(selected_features, collapse = " + ")
    } else {
        formula_str_base <- "1" # Intercept only if no features selected
    }
    formula_str <- paste(response_var, "~", formula_str_base)
    feature_formulas[[i]] <- formula_str
    formula <- as.formula(formula_str)


    # --- Train CARBayes model ---
    cat("--- Training CARBayes Model ---\n")
    # Create spatial weights ONLY from training data
    # Check for sufficient points and valid coordinates before creating weights
    if(nrow(train_data) > 3 && !any(is.na(train_data$x)) && !any(is.na(train_data$y))) {
        W_train_car <- create_spatial_weights(train_data[, c('x', 'y')], k_neighbors = 3)
    } else {
        cat("Warning: Insufficient data or NA coordinates in training set for CARBayes weights. Skipping CARBayes for fold", i, "\n")
        W_train_car <- NULL # Set weights to NULL
    }

    # Only proceed if weights could be created
    if (!is.null(W_train_car)) {
        carbayes <- carbayes_model$new(
          data = train_data, # Use training data
          formula = formula_str,
          random_effect = random_effect, # Still pass random effect if needed by model internals
          exposure = NULL,
          spatial_vars = c('x', 'y')
        )
        # Explicitly set the training weights before fitting
        carbayes$W <- W_train_car

        carbayes_results <- try(carbayes$fit(), silent = TRUE)

        if (inherits(carbayes_results, "try-error")) {
            cat("ERROR fitting CARBayes model for fold", i, ":", attr(carbayes_results, "condition")$message, "\n")
            carbayes_test_preds <- rep(NA, nrow(test_data))
            carbayes_test_mae <- NA; carbayes_test_mse <- NA; carbayes_test_rmse <- NA
        } else {
            # Predict on the test data (which is now mixed spatially)
            carbayes_test_preds <- carbayes$predict(test_data)

            # Calculate metrics for CARBayes on the test set
            carbayes_test_mae <- mean(abs(test_data[[response_var]] - carbayes_test_preds), na.rm = TRUE)
            carbayes_test_mse <- mean((test_data[[response_var]] - carbayes_test_preds)^2, na.rm = TRUE)
            carbayes_test_rmse <- sqrt(carbayes_test_mse)
            cat(sprintf("CARBayes Fold %d: Test MAE = %.4f, RMSE = %.4f\n", i, carbayes_test_mae, carbayes_test_rmse))
        }
    } else {
        # Handle case where weights couldn't be created
        carbayes_results <- NULL # Ensure results object is NULL
        carbayes_test_preds <- rep(NA, nrow(test_data))
        carbayes_test_mae <- NA; carbayes_test_mse <- NA; carbayes_test_rmse <- NA
    }


    # --- Train SEM Poisson model ---
    cat("--- Training SEM Poisson Model ---\n")
    # Create spatial data objects for train/test
    # Check for valid coordinates before creating spatial objects
    if(!any(is.na(train_data$x)) && !any(is.na(train_data$y)) && nrow(train_data) > 0) {
        train_sp <- train_data
        coordinates(train_sp) <- ~ x + y
    } else {
        cat("Warning: NA coordinates or empty training data for SEM spatial object creation. Skipping SEM for fold", i, "\n")
        train_sp <- NULL
    }
    if(!any(is.na(test_data$x)) && !any(is.na(test_data$y)) && nrow(test_data) > 0) {
        test_sp <- test_data
        coordinates(test_sp) <- ~ x + y
    } else {
        cat("Warning: NA coordinates or empty test data for SEM spatial object creation. Skipping SEM prediction for fold", i, "\n")
        test_sp <- NULL
    }


    # Create spatial weights ONLY from training data for SEM model fitting
    if (!is.null(train_sp) && nrow(train_sp) > 3) { # Need k+1 points for k=3 neighbors
        w_train_sem <- tryCatch(make_spatial_weights_for_sem(train_sp, k = 3),
                                error = function(e) {
                                    cat("Warning: Could not create SEM weights for fold", i, ":", e$message, "\n")
                                    return(NULL)
                                })
    } else {
        w_train_sem <- NULL
    }

    # Only proceed if weights and spatial objects are valid
    if (!is.null(w_train_sem) && !is.null(train_sp) && !is.null(test_sp)) {
        sem_test_preds <- try(train_poisson_sem(
          formula = formula, # Use the formula object based on selected features
          train_data = train_sp,
          val_data = test_sp,
          fold_weights = w_train_sem # Use only training weights
        ), silent = TRUE)

        if (inherits(sem_test_preds, "try-error") || any(is.na(sem_test_preds)) || length(sem_test_preds) != nrow(test_data)) {
            cat("ERROR or NA/length mismatch in SEM model prediction for fold", i, "\n")
            sem_test_mae <- NA; sem_test_mse <- NA; sem_test_rmse <- NA
            # Ensure preds vector has correct length even if prediction failed
            sem_test_preds <- rep(NA, nrow(test_data))
        } else {
            # Calculate metrics for SEM Poisson
            sem_test_mae <- mean(abs(test_data[[response_var]] - sem_test_preds), na.rm = TRUE)
            sem_test_mse <- mean((test_data[[response_var]] - sem_test_preds)^2, na.rm = TRUE)
            sem_test_rmse <- sqrt(sem_test_mse)
            cat(sprintf("SEM Fold %d: Test MAE = %.4f, RMSE = %.4f\n", i, sem_test_mae, sem_test_rmse))
        }
    } else {
        # Handle case where SEM couldn't be run
        sem_test_preds <- rep(NA, nrow(test_data))
        sem_test_mae <- NA; sem_test_mse <- NA; sem_test_rmse <- NA
    }


    # --- Train INLA model ---
    cat("--- Training INLA Model ---\n")
    # Prepare training data for INLA
    train_data_inla <- train_data
    # Create an index relative to the training data for the spatial effect
    train_data_inla$idx_train_spatial <- 1:nrow(train_data_inla)

    # Create sparse spatial weights matrix ONLY from training data
    # Check for sufficient points and valid coordinates
    W_sparse_train_inla <- NULL # Initialize as NULL
    if(nrow(train_data_inla) > 3 && !any(is.na(train_data_inla$x)) && !any(is.na(train_data_inla$y))) {
        coords_train_inla <- as.matrix(train_data_inla[, c('x', 'y')])
        W_dense_train_inla <- tryCatch(create_spatial_weights(data.frame(x=coords_train_inla[,1], y=coords_train_inla[,2]), k_neighbors=3),
                                       error = function(e) {
                                           cat("Warning: Could not create dense weights for INLA in fold", i, ":", e$message, "\n")
                                           return(NULL)
                                       })
        if (!is.null(W_dense_train_inla)) {
            W_sparse_train_inla <- Matrix(W_dense_train_inla, sparse = TRUE)
            # Ensure the graph matrix dimensions match the training data size
            if(nrow(W_sparse_train_inla) != nrow(train_data_inla)) {
                 cat("Warning: Dimension mismatch between training data and spatial weights matrix for INLA. Skipping INLA for fold", i, "\n")
                 W_sparse_train_inla <- NULL # Invalidate weights
            }
        }
    } else {
        cat("Warning: Insufficient data or NA coordinates in training set for INLA weights. Skipping INLA for fold", i, "\n")
    }


    # Only proceed if INLA weights are valid
    if (!is.null(W_sparse_train_inla)) {
        # Construct INLA formula using training index and weights
        # Ensure random_effect column exists and is a factor in train_data_inla
         if (!is.factor(train_data_inla[[random_effect]])) {
              # Use levels from the full data factor to handle all possible levels
              if(is.factor(data[[random_effect]])) {
                  train_data_inla[[random_effect]] <- factor(train_data_inla[[random_effect]], levels = levels(data[[random_effect]]))
              } else { # Fallback if original wasn't factor
                  train_data_inla[[random_effect]] <- factor(train_data_inla[[random_effect]])
              }
         }
         # Ensure test data also has factor levels consistent with training data
         if(is.factor(data[[random_effect]])) {
             all_levels <- levels(data[[random_effect]])
             test_data[[random_effect]] <- factor(test_data[[random_effect]], levels = all_levels)
         } else { # Fallback
             test_data[[random_effect]] <- factor(test_data[[random_effect]], levels = levels(train_data_inla[[random_effect]]))
         }


        inla_formula_str <- paste(
            response_var, "~", formula_str_base, # Use base formula (features or intercept)
            "+ f(idx_train_spatial, model = 'bym2', graph = W_sparse_train_inla, scale.model = TRUE, constr = TRUE, hyper = list(prec = pc_prec_unstructured, phi = pc_phi))",
            "+ f(", random_effect, ", model = 'iid', hyper = list(prec = pc_prec_borough))"
        )
        inla_formula <- as.formula(inla_formula_str)

        # Fit INLA model ONLY on training data
        inla_model <- try(inla(inla_formula,
                               data = train_data_inla, # Use training data
                               family = "nbinomial",
                               # Add compute=TRUE for predictor to get spatial effects easily
                               control.predictor = list(compute = TRUE, link = 1),
                               control.compute = list(dic = TRUE, waic = TRUE, config = TRUE, return.marginals.predictor=FALSE), # Save memory
                               control.family = nb_family_control,
                               control.mode = inla_control_mode,
                               # verbose = TRUE # Keep FALSE for cleaner CV output
                               ),
                          silent = TRUE)

        if (inherits(inla_model, "try-error")) {
            cat("ERROR fitting INLA model for fold", i, ":", attr(inla_model, "condition")$message, "\n")
            inla_test_preds <- rep(NA, nrow(test_data))
            inla_test_mae <- NA; inla_test_mse <- NA; inla_test_rmse <- NA
        } else {
            # --- Predict INLA manually for the test set ---
            # Build model matrix for fixed effects for the test data
            fixed_effects_formula <- as.formula(paste("~", formula_str_base))
            # Ensure factor levels in test_data match those used in training for model.matrix
            for(col in selected_features) {
                 if(is.factor(train_data_inla[[col]])) {
                      # Use levels from the full data factor if available
                      if(is.factor(data[[col]])) {
                          test_data[[col]] <- factor(test_data[[col]], levels=levels(data[[col]]))
                      } else {
                          test_data[[col]] <- factor(test_data[[col]], levels=levels(train_data_inla[[col]]))
                      }
                 }
            }
            # Random effect factor levels already handled before INLA formula creation

            X_test <- model.matrix(fixed_effects_formula, data = test_data)

            # Get mean coefficients for fixed effects
            beta_mean <- inla_model$summary.fixed$mean

            # Ensure alignment between coefficients and model matrix columns
            if (length(beta_mean) != ncol(X_test)) {
                 # This might happen if LASSO selected 0 features -> intercept only
                 if ("(Intercept)" %in% colnames(X_test) && "(Intercept)" %in% names(beta_mean)) {
                     common_cols <- intersect(colnames(X_test), names(beta_mean))
                     X_test <- X_test[, common_cols, drop = FALSE]
                     beta_mean <- beta_mean[common_cols]
                 } else {
                     cat("Warning: Mismatch between INLA fixed effects and test data model matrix columns. Prediction might be inaccurate.\n")
                     # Attempt to align, otherwise predict NA
                     common_cols <- intersect(colnames(X_test), names(beta_mean))
                     if(length(common_cols) > 0) {
                        X_test <- X_test[, common_cols, drop = FALSE]
                        beta_mean <- beta_mean[common_cols]
                     } else {
                        X_test <- matrix(0, nrow=nrow(test_data), ncol=0) # Predict NA later
                     }
                 }
            } else {
                 # Ensure order matches if names are present
                 if(all(colnames(X_test) %in% names(beta_mean))) {
                     beta_mean <- beta_mean[colnames(X_test)]
                 } else {
                     cat("Warning: Column names mismatch between INLA fixed effects and test matrix, assuming order is correct.\n")
                 }
            }

            # Calculate fixed effects contribution to linear predictor
            if(ncol(X_test) > 0 && length(beta_mean) == ncol(X_test)) { # Check alignment before multiplication
                eta_fixed <- X_test %*% beta_mean
            } else if ("(Intercept)" %in% names(beta_mean) && ncol(X_test) == 1 && colnames(X_test) == "(Intercept)") {
                 # Handle intercept-only case explicitly
                 eta_fixed <- X_test %*% beta_mean["(Intercept)"]
            } else if (length(selected_features) == 0 && "(Intercept)" %in% names(beta_mean)) {
                 # Handle case where LASSO selected 0 features, formula was ~ 1
                 eta_fixed <- rep(beta_mean["(Intercept)"], nrow(test_data))
            }
             else {
                cat("Warning: Could not align fixed effects for INLA prediction in fold", i, ". Setting fixed effect contribution to 0.\n")
                eta_fixed <- rep(0, nrow(test_data))
            }


            # Get mean random effects for borough
            re_summary <- inla_model$summary.random[[random_effect]]
            re_map <- setNames(re_summary$mean, re_summary$ID)
            eta_random <- re_map[as.character(test_data[[random_effect]])]
            eta_random[is.na(eta_random)] <- 0 # Use 0 for levels not in training summary

            # --- Approximate Spatial Effect using Nearest Neighbor ---
            eta_spatial_approx <- rep(0, nrow(test_data)) # Initialize with 0
            if (!is.null(inla_model$summary.random$idx_train_spatial) &&
                !any(is.na(train_data_inla$x)) && !any(is.na(train_data_inla$y)) &&
                !any(is.na(test_data$x)) && !any(is.na(test_data$y))) {
                # Get coordinates of training and test data
                coords_train <- train_data_inla[, c("x", "y")]
                coords_test <- test_data[, c("x", "y")]

                # Find the index of the nearest training point for each test point
                nn_result <- FNN::get.knnx(coords_train, coords_test, k = 1)
                nearest_train_indices <- nn_result$nn.index[, 1]

                # Get the fitted spatial effects (combined BYM2 effect) for training points
                spatial_effects_train <- inla_model$summary.random$idx_train_spatial$mean

                # Ensure lengths match before indexing
                if(length(spatial_effects_train) == nrow(train_data_inla)) {
                     # Map the nearest training index to its spatial effect
                     eta_spatial_approx <- spatial_effects_train[nearest_train_indices]
                     eta_spatial_approx[is.na(eta_spatial_approx)] <- 0 # Handle potential NAs
                     cat("Applied nearest neighbor spatial effect approximation for INLA prediction.\n")
                } else {
                     warning("Could not apply nearest neighbor spatial effect: Mismatch between spatial effects length and training data size.")
                }
            } else {
                 warning("Spatial random effect 'idx_train_spatial' not found, or NA coordinates present. Skipping NN spatial effect approximation.")
            }
            # --- End Spatial Effect Approximation ---


            # Combine fixed, random (borough), and approximated spatial effects
            eta_test <- eta_fixed + eta_random + eta_spatial_approx # Add spatial approx

            # Apply inverse link function (exponential for Poisson/NB log link)
            inla_test_preds <- exp(eta_test)
            # Impute NAs with mean of training response if any step failed leading to NA eta
            na_preds <- is.na(inla_test_preds)
            if(any(na_preds)) {
                inla_test_preds[na_preds] <- mean(train_data_inla[[response_var]], na.rm = TRUE)
                cat("Imputed", sum(na_preds), "NA predictions in INLA with training mean.\n")
            }


            # Calculate metrics for INLA
            inla_test_mae <- mean(abs(test_data[[response_var]] - inla_test_preds), na.rm = TRUE)
            inla_test_mse <- mean((test_data[[response_var]] - inla_test_preds)^2, na.rm = TRUE)
            inla_test_rmse <- sqrt(inla_test_mse)
            cat(sprintf("INLA Fold %d: Test MAE = %.4f, RMSE = %.4f\n", i, inla_test_mae, inla_test_rmse))
        }
    } else {
        # Handle case where INLA weights were invalid
        inla_model <- NULL # Ensure model object is NULL
        inla_test_preds <- rep(NA, nrow(test_data))
        inla_test_mae <- NA; inla_test_mse <- NA; inla_test_rmse <- NA
    }


    # --- Store results for the fold ---
    carbayes_fold_results[[i]] <- list(
      fold = i, # Use fold number
      test_mae = carbayes_test_mae, test_mse = carbayes_test_mse, test_rmse = carbayes_test_rmse,
      model_summary = if (!is.null(carbayes_results)) summary(carbayes$results) else NULL, # Check if results exist
      selected_features = selected_features, formula = formula_str
    )

    sem_fold_results[[i]] <- list(
      fold = i, # Use fold number
      test_mae = sem_test_mae, test_mse = sem_test_mse, test_rmse = sem_test_rmse,
      selected_features = selected_features, formula = formula_str
    )

    inla_fold_results[[i]] <- list(
      fold = i, # Use fold number
      test_mae = inla_test_mae, test_mse = inla_test_mse, test_rmse = inla_test_rmse,
      model_summary = if (!is.null(inla_model) && !inherits(inla_model, "try-error")) summary(inla_model) else NULL, # Check if model exists and is valid
      selected_features = selected_features, formula = formula_str
    )

    # Store actuals and predictions for overall calculation
    carbayes_test_actual[[i]] <- test_data[[response_var]]
    carbayes_test_predicted[[i]] <- carbayes_test_preds
    carbayes_test_ids[[i]] <- test_data$int_no # Store IDs from test_data
    sem_test_actual[[i]] <- test_data[[response_var]]
    sem_test_predicted[[i]] <- sem_test_preds
    inla_test_actual[[i]] <- test_data[[response_var]]
    inla_test_predicted[[i]] <- inla_test_preds
  } # End of stratified fold loop

  # --- Combine results and calculate overall metrics ---
  # Filter out potential NULLs if folds were skipped
  carbayes_test_actual <- carbayes_test_actual[!sapply(carbayes_test_actual, is.null)]
  carbayes_test_predicted <- carbayes_test_predicted[!sapply(carbayes_test_predicted, is.null)]
  carbayes_test_ids <- carbayes_test_ids[!sapply(carbayes_test_ids, is.null)]
  sem_test_actual <- sem_test_actual[!sapply(sem_test_actual, is.null)]
  sem_test_predicted <- sem_test_predicted[!sapply(sem_test_predicted, is.null)]
  inla_test_actual <- inla_test_actual[!sapply(inla_test_actual, is.null)]
  inla_test_predicted <- inla_test_predicted[!sapply(inla_test_predicted, is.null)]

  carbayes_test_actual_combined <- unlist(carbayes_test_actual)
  carbayes_test_predicted_combined <- unlist(carbayes_test_predicted)
  carbayes_test_ids_combined <- unlist(carbayes_test_ids)
  sem_test_actual_combined <- unlist(sem_test_actual)
  sem_test_predicted_combined <- unlist(sem_test_predicted)
  inla_test_actual_combined <- unlist(inla_test_actual)
  inla_test_predicted_combined <- unlist(inla_test_predicted)

  # Calculate overall metrics for CARBayes
  carbayes_overall_mae <- mean(abs(carbayes_test_actual_combined - carbayes_test_predicted_combined), na.rm = TRUE)
  carbayes_overall_mse <- mean((carbayes_test_actual_combined - carbayes_test_predicted_combined)^2, na.rm = TRUE)
  carbayes_overall_rmse <- sqrt(carbayes_overall_mse)

  # Calculate overall metrics for SEM Poisson
  sem_overall_mae <- mean(abs(sem_test_actual_combined - sem_test_predicted_combined), na.rm = TRUE)
  sem_overall_mse <- mean((sem_test_actual_combined - sem_test_predicted_combined)^2, na.rm = TRUE)
  sem_overall_rmse <- sqrt(sem_overall_mse)

  # Calculate overall metrics for INLA
  inla_overall_mae <- mean(abs(inla_test_actual_combined - inla_test_predicted_combined), na.rm = TRUE)
  inla_overall_mse <- mean((inla_test_actual_combined - inla_test_predicted_combined)^2, na.rm = TRUE)
  inla_overall_rmse <- sqrt(inla_overall_mse)


  # Feature consistency analysis (based on LASSO selection)
  valid_carbayes_results <- carbayes_fold_results[!sapply(carbayes_fold_results, function(x) is.null(x$selected_features))]
  all_features <- unique(unlist(lapply(valid_carbayes_results, function(x) x$selected_features)))
  if (length(all_features) > 0 && length(valid_carbayes_results) > 0) { # Check if valid results exist
      feature_counts <- sapply(all_features, function(feat) {
        sum(sapply(valid_carbayes_results, function(x) feat %in% x$selected_features))
      })

      feature_consistency <- data.frame(
        Feature = names(feature_counts),
        Count = feature_counts,
        Percentage = 100 * feature_counts / length(valid_carbayes_results) # Use count of valid folds
      )
      feature_consistency <- feature_consistency[order(feature_consistency$Count, decreasing = TRUE), ]
  } else {
       feature_consistency <- data.frame(Feature=character(), Count=integer(), Percentage=numeric())
       cat("No features were consistently selected across folds.\n")
  }


  return(list(
    # Return data with district and part assignments
    data_with_assignments = data, # Renamed from data_with_folds
    carbayes_results = list(
      fold_results = carbayes_fold_results,
      overall_mae = carbayes_overall_mae, overall_mse = carbayes_overall_mse, overall_rmse = carbayes_overall_rmse,
      test_ids = carbayes_test_ids_combined, # Combined IDs from all test folds
      test_actual = carbayes_test_actual_combined,
      test_predicted = carbayes_test_predicted_combined
    ),
    sem_results = list(
      fold_results = sem_fold_results,
      overall_mae = sem_overall_mae, overall_mse = sem_overall_mse, overall_rmse = sem_overall_rmse,
      test_actual = sem_test_actual_combined, test_predicted = sem_test_predicted_combined
    ),
    inla_results = list(
      fold_results = inla_fold_results,
      overall_mae = inla_overall_mae, overall_mse = inla_overall_mse, overall_rmse = inla_overall_rmse,
      test_actual = inla_test_actual_combined, test_predicted = inla_test_predicted_combined
    ),
    feature_consistency = feature_consistency,
    feature_formulas = feature_formulas
  ))
}

# Generate figures from cross-validation results (Modified for Stratified CV)
generate_cv_figures <- function(cv_results, data_with_assignments) { # Changed argument name
  # Figure 1: Map of predicted accidents (using CARBayes results as example)

  # Ensure data_with_assignments has the necessary columns
  if (!all(c("x", "y", "acc", "int_no") %in% colnames(data_with_assignments))) {
      stop("`data_with_assignments` must contain 'x', 'y', 'acc', and 'int_no' columns.")
  }
  # part_id is useful for context but not strictly needed for matching predictions here

  all_predictions <- data.frame(
    x = data_with_assignments$x,
    y = data_with_assignments$y,
    actual = data_with_assignments$acc,
    int_no = data_with_assignments$int_no, # Keep int_no for matching
    # part_id = data_with_assignments$part_id, # Optional: keep for context
    predicted = NA_real_ # Initialize with numeric NA
  )

  # Get combined predictions and corresponding IDs from CARBayes results
  pred_ids <- cv_results$carbayes_results$test_ids
  pred_values <- cv_results$carbayes_results$test_predicted

  # Handle case where predictions might be NULL or empty
  if (is.null(pred_ids) || is.null(pred_values) || length(pred_ids) == 0 || length(pred_values) == 0) {
      warning("No CARBayes predictions available to generate figures.")
      return(list(map_plot = NULL, scatter_plot = NULL, predictions = all_predictions)) # Return empty predictions df
  }


  # Create a lookup table: prediction value for each int_no
  prediction_lookup <- setNames(pred_values, pred_ids)

  # Match predictions back to the all_predictions dataframe using int_no
  match_indices <- match(all_predictions$int_no, pred_ids)

  # Assign predictions where a match was found
  valid_match_indices <- !is.na(match_indices)
  all_predictions$predicted[valid_match_indices] <- pred_values[match_indices[valid_match_indices]]


  # Remove any rows where predictions are still NA (e.g., if a fold failed or matching issue)
  plot_data <- all_predictions[!is.na(all_predictions$predicted), ]

  if(nrow(plot_data) == 0) {
      warning("No valid predictions found after matching to generate plots.")
      return(list(map_plot = NULL, scatter_plot = NULL, predictions = plot_data))
  }

  # Create spatial points data frame for mapping
  # Check if coordinates are valid before creating spatial object
  if(any(is.na(plot_data$x)) || any(is.na(plot_data$y))) {
      warning("Missing coordinates found in data for plotting. Removing these points.")
      plot_data <- plot_data[!is.na(plot_data$x) & !is.na(plot_data$y), ]
  }

  if(nrow(plot_data) == 0) {
      warning("No valid data points remaining after removing missing coordinates.")
       return(list(map_plot = NULL, scatter_plot = NULL, predictions = plot_data))
  }

  sp_predictions <- plot_data
  map_plot <- NULL # Initialize map_plot as NULL
  tryCatch({
      coordinates(sp_predictions) <- ~ x + y
      # Convert to sf for better mapping
      sf_predictions <- st_as_sf(sp_predictions)

      # Create map of predicted accidents
      map_plot <- ggplot() +
        geom_sf(data = sf_predictions, aes(color = predicted), size = 1) +
        scale_color_viridis_c(name = "Predicted\nAccidents") +
        theme_classic() +
        theme(
          panel.background = element_rect(fill = "white", color = NA),
          plot.background = element_rect(fill = "white", color = NA),
          legend.background = element_rect(fill = "white", color = NA),
          legend.position = "right"
        ) +
        labs(title = "Map of Predicted Accidents (Stratified 5-Fold CV)", # Updated title
             subtitle = "CARBayes Model Cross-Validation Results")
  }, error = function(e){
       cat("Error creating spatial map plot:", e$message, "\n")
       # map_plot remains NULL
  })


  # Figure 2: Predicted vs Actual values
  scatter_plot <- ggplot(plot_data, aes(x = actual, y = predicted)) +
    geom_point(alpha = 0.5) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    geom_smooth(method = "lm", color = "blue", se = TRUE) +
    theme_classic() +
    theme(
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      legend.background = element_rect(fill = "white", color = NA)
    ) +
    labs(title = "Predicted vs Actual Accidents (Stratified 5-Fold CV)", # Updated title
         subtitle = "CARBayes Model Cross-Validation Results",
         x = "Actual Accidents",
         y = "Predicted Accidents")

  # Calculate correlation
  correlation <- tryCatch(cor(plot_data$actual, plot_data$predicted), error = function(e) NA)
  if(!is.na(correlation)) {
      cat(sprintf("\nCorrelation between actual and predicted values: %.4f\n", correlation))
  } else {
      cat("\nCould not calculate correlation between actual and predicted values.\n")
  }


  return(list(map_plot = map_plot, scatter_plot = scatter_plot, predictions = plot_data))
}

# Main execution
cat("Loading and preparing data...\n")
full_data <- load_and_prepare_data()

# Ensure necessary columns exist before running CV
required_cols <- c("acc", "borough", "x", "y", "int_no")
if (!all(required_cols %in% colnames(full_data))) {
    stop("Missing required columns in full_data: ", paste(setdiff(required_cols, colnames(full_data)), collapse=", "))
}


cat("Running combined stratified 5-fold cross-validation...\n") # Updated message
# k=5 is now defined inside the function
cv_results <- run_combined_cross_validation(
  data = full_data,
  response_var = "acc",
  random_effect = "borough", # Used for district stratification
  max_features = 40
)

# Retrieve the data with fold assignments for figure generation
data_with_assignments <- cv_results$data_with_assignments # Renamed variable

# Print Results (Ensure results exist before accessing)
cat("\n==== CARBayes Model Results ====\n")
if (!is.null(cv_results$carbayes_results) && length(cv_results$carbayes_results$fold_results) > 0) {
    cat("Metrics by fold:\n") # Updated label
    for (i in 1:length(cv_results$carbayes_results$fold_results)) {
      fold <- cv_results$carbayes_results$fold_results[[i]]
      if (!is.null(fold)) {
           # Use %||% NA to handle potential NA metrics within a fold result
           cat(sprintf("Fold %d: Test MAE = %.4f, Test MSE = %.4f, Test RMSE = %.4f\n",
                    fold$fold, fold$test_mae %||% NA, fold$test_mse %||% NA, fold$test_rmse %||% NA))
      }
    }

    # Calculate mean metrics, handling potential NAs from failed folds
    carbayes_mean_mae <- mean(sapply(cv_results$carbayes_results$fold_results, function(x) x$test_mae), na.rm = TRUE)
    carbayes_mean_mse <- mean(sapply(cv_results$carbayes_results$fold_results, function(x) x$test_mse), na.rm = TRUE)
    carbayes_mean_rmse <- mean(sapply(cv_results$carbayes_results$fold_results, function(x) x$test_rmse), na.rm = TRUE)

    cat("\nCARBayes Mean performance across folds:\n")
    cat(sprintf("Mean MAE: %.4f\n", carbayes_mean_mae))
    cat(sprintf("Mean MSE: %.4f\n", carbayes_mean_mse))
    cat(sprintf("Mean RMSE: %.4f\n", carbayes_mean_rmse))
    cat(sprintf("Overall MAE: %.4f\n", cv_results$carbayes_results$overall_mae %||% NA)) # Use %||% NA for overall too
    cat(sprintf("Overall MSE: %.4f\n", cv_results$carbayes_results$overall_mse %||% NA))
    cat(sprintf("Overall RMSE: %.4f\n", cv_results$carbayes_results$overall_rmse %||% NA))
} else {
    cat("CARBayes results are not available.\n")
    carbayes_mean_mae <- NA; carbayes_mean_mse <- NA; carbayes_mean_rmse <- NA # Set NAs for comparison table
}


cat("\n==== SEM Poisson Model Results ====\n")
if (!is.null(cv_results$sem_results) && length(cv_results$sem_results$fold_results) > 0) {
    cat("Metrics by fold:\n") # Updated label
     for (i in 1:length(cv_results$sem_results$fold_results)) {
      fold <- cv_results$sem_results$fold_results[[i]]
       if (!is.null(fold)) {
           cat(sprintf("Fold %d: Test MAE = %.4f, Test MSE = %.4f, Test RMSE = %.4f\n",
                    fold$fold, fold$test_mae %||% NA, fold$test_mse %||% NA, fold$test_rmse %||% NA))
       }
    }

    sem_mean_mae <- mean(sapply(cv_results$sem_results$fold_results, function(x) x$test_mae), na.rm = TRUE)
    sem_mean_mse <- mean(sapply(cv_results$sem_results$fold_results, function(x) x$test_mse), na.rm = TRUE)
    sem_mean_rmse <- mean(sapply(cv_results$sem_results$fold_results, function(x) x$test_rmse), na.rm = TRUE)

    cat("\nSEM Poisson Mean performance across folds:\n")
    cat(sprintf("Mean MAE: %.4f\n", sem_mean_mae))
    cat(sprintf("Mean MSE: %.4f\n", sem_mean_mse))
    cat(sprintf("Mean RMSE: %.4f\n", sem_mean_rmse))
    cat(sprintf("Overall MAE: %.4f\n", cv_results$sem_results$overall_mae %||% NA))
    cat(sprintf("Overall MSE: %.4f\n", cv_results$sem_results$overall_mse %||% NA))
    cat(sprintf("Overall RMSE: %.4f\n", cv_results$sem_results$overall_rmse %||% NA))
} else {
    cat("SEM results are not available.\n")
    sem_mean_mae <- NA; sem_mean_mse <- NA; sem_mean_rmse <- NA
}


# --- Add INLA Results Printing ---
cat("\n==== INLA Model Results ====\n")
if (!is.null(cv_results$inla_results) && length(cv_results$inla_results$fold_results) > 0) {
    cat("Metrics by fold:\n") # Updated label
     for (i in 1:length(cv_results$inla_results$fold_results)) {
      fold <- cv_results$inla_results$fold_results[[i]]
       if (!is.null(fold)) {
           cat(sprintf("Fold %d: Test MAE = %.4f, Test MSE = %.4f, Test RMSE = %.4f\n",
                    fold$fold, fold$test_mae %||% NA, fold$test_mse %||% NA, fold$test_rmse %||% NA))
       }
    }

    inla_mean_mae <- mean(sapply(cv_results$inla_results$fold_results, function(x) x$test_mae), na.rm=TRUE)
    inla_mean_mse <- mean(sapply(cv_results$inla_results$fold_results, function(x) x$test_mse), na.rm=TRUE)
    inla_mean_rmse <- mean(sapply(cv_results$inla_results$fold_results, function(x) x$test_rmse), na.rm=TRUE)

    cat("\nINLA Mean performance across folds:\n")
    cat(sprintf("Mean MAE: %.4f\n", inla_mean_mae))
    cat(sprintf("Mean MSE: %.4f\n", inla_mean_mse))
    cat(sprintf("Mean RMSE: %.4f\n", inla_mean_rmse))
    cat(sprintf("Overall MAE: %.4f\n", cv_results$inla_results$overall_mae %||% NA))
    cat(sprintf("Overall MSE: %.4f\n", cv_results$inla_results$overall_mse %||% NA))
    cat(sprintf("Overall RMSE: %.4f\n", cv_results$inla_results$overall_rmse %||% NA))
} else {
    cat("INLA results are not available.\n")
    inla_mean_mae <- NA; inla_mean_mse <- NA; inla_mean_rmse <- NA
}

# ---------------------------------

# Compare the models
cat("\n==== Model Comparison (Stratified 5-Fold CV) ====\n") # Updated title
# Helper for default NA if not already defined
`%||%` <- function(a, b) if (!is.null(a)) a else b
# Use results stored earlier, which handle NAs if models failed
comparison_df <- data.frame(
  Model = c("CARBayes", "SEM Poisson", "INLA"),
  Mean_MAE = c(carbayes_mean_mae, sem_mean_mae, inla_mean_mae),
  Mean_MSE = c(carbayes_mean_mse, sem_mean_mse, inla_mean_mse),
  Mean_RMSE = c(carbayes_mean_rmse, sem_mean_rmse, inla_mean_rmse),
  Overall_MAE = c(cv_results$carbayes_results$overall_mae %||% NA, cv_results$sem_results$overall_mae %||% NA, cv_results$inla_results$overall_mae %||% NA),
  Overall_MSE = c(cv_results$carbayes_results$overall_mse %||% NA, cv_results$sem_results$overall_mse %||% NA, cv_results$inla_results$overall_mse %||% NA),
  Overall_RMSE = c(cv_results$carbayes_results$overall_rmse %||% NA, cv_results$sem_results$overall_rmse %||% NA, cv_results$inla_results$overall_rmse %||% NA)
)

print(comparison_df)

# After running cross-validation, generate and display the figures
cat("\nGenerating prediction comparison figures from stratified cross-validation results...\n") # Updated message
# Pass the data with assignments to the figure function
figures <- generate_cv_figures(cv_results, data_with_assignments) # Use updated variable name

# Save the figures (check if plots were generated)
if (!is.null(figures$map_plot)) {
    ggsave("predicted_accidents_map_stratified_cv.png", figures$map_plot, width = 10, height = 8) # Updated filename
    cat("Map plot saved as 'predicted_accidents_map_stratified_cv.png'\n")
} else {
    cat("Map plot could not be generated.\n")
}
if (!is.null(figures$scatter_plot)) {
    ggsave("predicted_vs_actual_stratified_cv.png", figures$scatter_plot, width = 8, height = 6) # Updated filename
    cat("Scatter plot saved as 'predicted_vs_actual_stratified_cv.png'\n")
} else {
    cat("Scatter plot could not be generated.\n")
}


# Display summary statistics of predictions if available
if (!is.null(figures$predictions) && nrow(figures$predictions) > 0 && "predicted" %in% colnames(figures$predictions)) { # Check predictions exist
    cat("\nSummary of predictions (CARBayes):\n")
    print(summary(figures$predictions$predicted))
    cat("\nSummary of actual values (corresponding to predictions):\n")
    print(summary(figures$predictions$actual))

    # Calculate additional metrics if possible
    if(length(figures$predictions$actual) > 1 && length(figures$predictions$predicted) > 1) {
        cat("\nAdditional metrics (CARBayes):\n")
        # Ensure no NAs before correlation calculation
        valid_preds <- !is.na(figures$predictions$predicted)
        valid_actual <- !is.na(figures$predictions$actual)
        valid_both <- valid_preds & valid_actual
        if(sum(valid_both) > 1) {
            correlation <- cor(figures$predictions$actual[valid_both], figures$predictions$predicted[valid_both])
            cat(sprintf("Correlation: %.4f\n", correlation))
            cat(sprintf("R-squared: %.4f\n", correlation^2))
        } else {
            cat("Not enough valid data points to calculate correlation.\n")
        }
    }
} else {
    cat("\nNo valid predictions available to summarize.\n")
}


# Create and save the CARBayes predictions CSV
cat("\nCreating and saving CARBayes predictions CSV (Stratified 5-Fold CV)...\n") # Updated message
# Retrieve the combined IDs and predictions
pred_ids <- cv_results$carbayes_results$test_ids
pred_values <- cv_results$carbayes_results$test_predicted

if (length(pred_ids) > 0 && length(pred_ids) == length(pred_values)) {
    # Create the data frame
    predictions_df <- data.frame(
      int_no = pred_ids,
      predicted_accidents = pred_values
    )

    # Remove rows with NA predictions (if any fold failed)
    predictions_df <- predictions_df[!is.na(predictions_df$predicted_accidents), ]

    # Sort the data frame by predicted accidents (descending)
    predictions_df <- predictions_df[order(predictions_df$predicted_accidents, decreasing = TRUE), ]

    # Write to CSV
    write.csv(predictions_df, "carbayes_predictions_stratified_cv.csv", row.names = FALSE) # Updated filename
    cat("CARBayes predictions saved to carbayes_predictions_stratified_cv.csv\n")
} else {
    cat("No valid CARBayes predictions to save.\n")
}