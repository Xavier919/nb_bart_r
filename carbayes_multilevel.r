# --- Libraries (Ensure all needed are loaded) ---
library(spdep)       
library(dplyr)
library(ggplot2)     
library(caret)       
library(glmnet)      
library(CARBayes)
library(conflicted) 
library(tidyr)  # Add this for replace_na function

# Resolve conflicts - explicitly prefer dplyr's select
conflicts_prefer(dplyr::select)

# --- Data Loading Function (Assume load_and_prepare_data is defined as before) ---
# --- Data Processing Function (Mostly Unchanged) ---
load_and_prepare_data <- function(csv_path = "data/data_final.csv") {
  # Load and prepare the Montreal intersection data
  data <- read.csv(csv_path, sep = ";", fileEncoding="UTF-8") 
  
  data$ln_distdt[is.na(data$ln_distdt)] <- 0
  data <- data[data$pi != 0, ]
  
  # Borough name corrections (Ensure this list is complete for your data)
  borough_corrections <- list(
    '?le-Bizard-Sainte-GeneviÞve' = 'Île-Bizard-Sainte-Geneviève',
    'C¶te-Saint-Luc' = 'Côte-Saint-Luc',
    'C¶te-des-Neiges-Notre-Dame-de-Graces' = 'Côte-des-Neiges-Notre-Dame-de-Grâce',
    'MontrÚal-Est' = 'Montréal-Est',
    'MontrÚal-Nord' = 'Montréal-Nord',
    'Pointe-aux-Trembles-RiviÞres-des-Prairies' = 'Rivière-des-Prairies-Pointe-aux-Trembles',
    'St-LÚonard' = 'Saint-Léonard'
  )
  
  # Borough grouping into zones (as in Python code)
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
  
  if ("borough" %in% colnames(data)) {
    # First fix encoding issues
    for (old_name in names(borough_corrections)) {
      data$borough[data$borough == old_name] <- borough_corrections[[old_name]]
    }
    
    # Then group into zones
    for (borough in names(borough_zones)) {
      data$borough[data$borough == borough] <- borough_zones[[borough]]
    }
    
    data$borough_analysis_unit <- data$borough # Using grouped zone names
  } else {
    stop("Column 'borough' not found in the data.")
  }

  # Feature Columns
  spatial_cols <- c('x', 'y') # Keep coordinates for centroid calculation
  feature_cols <- c('ln_fi', 'ln_fri', 'ln_fli', 'ln_pi', 'ln_cti', 'ln_cli', 'ln_cri', 'ln_distdt',
                   'fi', 'fri', 'fli', 'pi', 'cti', 'cli', 'cri', 'distdt', 
                   'total_lane', 'avg_crossw', 'tot_road_w', 'tot_crossw',
                   'commercial', 'number_of_', 'of_exclusi', 'curb_exten', 'median', 'all_pedest', 'half_phase', 'new_half_r',
                   'any_ped_pr', 'ped_countd', 'lt_restric', 'lt_prot_re', 'lt_protect', 'any_exclus', 'all_red_an', 'green_stra',
                   'parking', 'north_veh', 'north_ped', 'east_veh', 'east_ped', 'south_veh', 'south_ped', 'west_veh', 'west_ped')
  
  feature_cols <- feature_cols[feature_cols %in% colnames(data)]
  spatial_cols <- spatial_cols[spatial_cols %in% colnames(data)]
  if (!all(c('x', 'y') %in% spatial_cols)) stop("Data must contain 'x' and 'y' columns for centroid calculation.")
  
  X_features <- data %>% 
      select(all_of(feature_cols)) %>%
      mutate(across(everything(), ~ replace_na(., 0)))

  quad_vars <- c('ln_cti', 'ln_cri', 'ln_cli', 'ln_pi', 'ln_fri', 'ln_fli', 'ln_fi')
  for (col in quad_vars) {
    if (col %in% colnames(X_features)) {
      X_features[[paste0(col, "_squared")]] <- X_features[[col]]^2
    }
  }
  
  # Combine final data
  id_cols <- c('int_no', 'acc', 'x', 'y', 'borough_analysis_unit') 
  final_data <- cbind(data %>% select(any_of(id_cols)), X_features)
  
  # Add numeric borough ID (ordered alphabetically or by first appearance)
  final_data <- final_data %>%
    # IMPORTANT: Ensure factor levels are consistent across runs if needed
    mutate(borough_id = as.numeric(factor(borough_analysis_unit))) 
  
  cat("Data preparation complete. Dimensions:", dim(final_data), "\n")
  cat("Number of unique analysis units (zones):", length(unique(final_data$borough_analysis_unit)), "\n")
  cat("Range of zone IDs:", range(final_data$borough_id), "\n")
  
  return(final_data)
}
# --- Adjacency from Centroids Function (Ensure it orders correctly) ---
create_adjacency_from_centroids <- function(data, k_neighbors = 6) {
  
  if (!requireNamespace("spdep", quietly = TRUE)) stop("Package 'spdep' needed.", call. = FALSE)
  if (!all(c("x", "y", "borough_analysis_unit", "borough_id") %in% colnames(data))) {
      stop("Data must contain 'x', 'y', 'borough_analysis_unit', and 'borough_id'.")
  }
  
  cat("Calculating borough centroids...\n")
  
  # Calculate centroids, **explicitly ordered by borough_id (1 to K)**
  centroids <- data %>%
    # Ensure we only get one row per borough_id
    distinct(borough_analysis_unit, borough_id, .keep_all = FALSE) %>% 
    # Calculate centroids based on ALL data for that borough
    left_join(
        data %>% 
            group_by(borough_id) %>%
            summarise(
                centroid_x = mean(x, na.rm = TRUE),
                centroid_y = mean(y, na.rm = TRUE),
                .groups = 'drop'
            ),
        by = "borough_id"
    ) %>%
    # Arrange by the numeric ID to ensure W matrix rows/cols are 1 to K
    arrange(borough_id) 
    
  n_boroughs <- nrow(centroids)
  expected_n_boroughs <- max(data$borough_id)
  cat("Calculated centroids for", n_boroughs, "boroughs. Expected:", expected_n_boroughs, "\n")

  if(n_boroughs != expected_n_boroughs) {
      stop(paste("Mismatch between number of centroids (", n_boroughs, 
                 ") and max borough_id (", expected_n_boroughs, 
                 "). Check for missing boroughs or ID issues."))
  }
  
  centroid_coords <- as.matrix(centroids[, c("centroid_x", "centroid_y")])
  
  if(k_neighbors >= n_boroughs) {
      k_neighbors = n_boroughs - 1
      warning(paste("Reduced k_neighbors to", k_neighbors), immediate. = TRUE)
  }
  if (k_neighbors < 1) stop("k_neighbors must be at least 1.")

  cat("Finding", k_neighbors, "nearest neighbors for centroids...\n")
  knn_nb <- spdep::knn2nb(spdep::knearneigh(centroid_coords, k = k_neighbors))
  sym_knn_nb <- spdep::make.sym.nb(knn_nb)

  cat("Converting neighbors to binary adjacency matrix W...\n")
  W_centroid_knn <- spdep::nb2mat(sym_knn_nb, style = "B", zero.policy = TRUE) 
  
  # Assign row/column names (optional but good practice) - Use the ordered names
  dimnames(W_centroid_knn) <- list(centroids$borough_analysis_unit, centroids$borough_analysis_unit)
  
  if (nrow(W_centroid_knn) != n_boroughs) stop("W matrix dimension mismatch.")

  cat("Centroid-based KNN adjacency matrix W created. Dimensions:", dim(W_centroid_knn), "\n")
  return(W_centroid_knn)
}


# --- LASSO Function (Assume defined as before) ---
select_features_with_lasso <- function(X_train, y_train, max_features = 30) {
  # ... (keep the previous robust version) ...
  X_matrix <- as.matrix(X_train)
  y_vector <- as.numeric(y_train)
  
  variances <- apply(X_matrix, 2, var, na.rm = TRUE)
  zero_var_cols <- names(variances[variances == 0 | is.na(variances)])
  if (length(zero_var_cols) > 0) {
      # cat("Removing zero-variance columns for LASSO:", paste(zero_var_cols, collapse=", "), "\n")
      X_matrix <- X_matrix[, !colnames(X_matrix) %in% zero_var_cols, drop = FALSE] 
  }
  if (ncol(X_matrix) == 0) return(character(0)) 

  X_scaled <- scale(X_matrix); X_scaled[is.na(X_scaled)] <- 0 
  
  set.seed(123) 
  selected_features <- character(0) # Initialize
  tryCatch({
    num_folds = min(5, nrow(X_scaled)-1); if (num_folds < 3) num_folds = 3 
    lasso_cv <- cv.glmnet(X_scaled, y_vector, alpha = 1, family = "poisson", nfolds = num_folds) 
    lambda_choice <- lasso_cv$lambda.1se 
    coef_matrix <- coef(lasso_cv, s = lambda_choice)
    selected_features_coeffs <- coef_matrix[-1, 1]  
    selected_features <- names(selected_features_coeffs[selected_features_coeffs != 0])
    
  }, error = function(e) {
    cat("Error during LASSO: ", e$message, "\n")
  })
  
  cat("LASSO selected", length(selected_features), "features for this fold.\n")
  if (length(selected_features) > max_features) {
      warning(paste("LASSO selected", length(selected_features), "features, > max_features =", max_features))
      # Optionally truncate here if desired
  }
  if (length(selected_features) == 0) {
      cat("Warning: LASSO selected 0 features.\n")
  }
  return(selected_features)
}

# Add this function after 'select_features_with_lasso' but before 'run_cv_car_multilevel'

# Ensure borough IDs meet CARBayes requirements - must be consecutive integers from 1 to K
remap_area_indices <- function(data) {
  # Get the unique borough IDs in sorted order
  unique_boroughs <- sort(unique(data$borough_id))
  
  # Create a mapping from original IDs to consecutive integers 1:K
  mapping <- setNames(seq_along(unique_boroughs), as.character(unique_boroughs))
  
  # Apply the mapping to create a new column
  data$area_idx <- mapping[as.character(data$borough_id)]
  
  cat("Remapped", length(unique_boroughs), "borough IDs to consecutive integers 1:",
      length(unique_boroughs), "\n")
  
  return(data)
}

# --- Cross-Validation Function (Revised Check) ---
run_cv_car_multilevel <- function(data, 
                                  response_var = "acc", 
                                  k = 5, 
                                  k_neighbors_W = 6, 
                                  max_features_lasso = 30,
                                  mcmc_params = list(burnin = 5000, n.sample = 15000, thin = 10)) {
  
  set.seed(456) 
  data[[response_var]] <- as.numeric(data[[response_var]])
  
  # Ensure indices are consecutive (1 to K) for CARBayes
  data <- remap_area_indices(data)
  
  # 1. Create Borough Adjacency Matrix from Centroids (once)
  # Ensure borough_id is correctly generated (1 to K) in load_and_prepare_data
  W_borough <- create_adjacency_from_centroids(data, k_neighbors = k_neighbors_W)
  N_AREAS_TOTAL <- nrow(W_borough) # Total number of areas represented in W
  
  # 2. Create Folds
  folds <- createFolds(data[[response_var]], k = k, returnTrain = TRUE)
  
  # 3. Initialize results
  fold_results <- list()
  all_test_actual <- numeric()
  all_test_predicted <- numeric()
  
  # --- CV Loop ---
  for (i in 1:k) {
    cat("\n========== Fold", i, "/", k, " ==========\n")
    
    train_indices <- folds[[i]]
    test_indices <- setdiff(1:nrow(data), train_indices)
    train_data <- data[train_indices, ]
    test_data <- data[test_indices, ]
    
    # --- Feature Selection ---
    exclude_cols <- c(response_var, "int_no", "x", "y", "borough_analysis_unit", "borough_id", "borough") 
    potential_predictors <- setdiff(colnames(train_data), exclude_cols)
    X_train_fs <- train_data %>% select(any_of(potential_predictors))
    y_train_fs <- train_data[[response_var]]
    selected_features <- select_features_with_lasso(X_train_fs, y_train_fs, max_features_lasso)
    
    # --- Model Fitting ---
    if (length(selected_features) > 0) {
        formula_str <- paste(response_var, "~", paste(selected_features, collapse = " + "))
    } else { formula_str <- paste(response_var, "~ 1"); warning("Intercept-only model fold ", i) }
    current_formula <- as.formula(formula_str)
    cat("Using formula for fold", i, ":", formula_str, "\n")

    train_ind_area <- train_data$area_idx 
    
    # --- *** KEY CHECK *** ---
    # Verify the indices in train_ind_area are valid for the FULL W matrix
    max_train_id <- max(train_ind_area)
    min_train_id <- min(train_ind_area)
    
    cat("Max borough_id in training data:", max_train_id, "\n")
    cat("Min borough_id in training data:", min_train_id, "\n")
    cat("Total number of areas in W matrix:", N_AREAS_TOTAL, "\n")
    
    if(max_train_id > N_AREAS_TOTAL || min_train_id < 1) {
        stop(paste("FATAL ERROR in Fold", i, 
                   ": borough_id in training data (range", min_train_id, "-", max_train_id, 
                   ") is outside the expected range [1,", N_AREAS_TOTAL, 
                   "] based on the W matrix dimensions. Check borough_id generation."))
    }
    # The error message indicated the check failed, but the logic here ensures it *should* pass.
    # If this still fails, the issue lies in how borough_id or W is created.

    cat("Fitting CARBayes S.CARmultilevel model...\n")
    model_fit <- NULL 
    fit_time <- system.time({
        model_fit <- tryCatch({
             S.CARmultilevel(
                formula = current_formula,
                family = "poisson", 
                data = train_data,
                W = W_borough,       # Use the K x K CENTROID-KNN matrix
                ind.area = train_ind_area, # Use the borough_id (1 to K) for training data
                burnin = mcmc_params$burnin,      
                n.sample = mcmc_params$n.sample,    
                thin = mcmc_params$thin,           
                verbose = FALSE 
             )
        }, error = function(e) {
             cat("ERROR fitting S.CARmultilevel model in fold", i, ":", e$message, "\n")
             # Print details for debugging the error CARBayes gave
             cat("Debugging Info:\n")
             cat(" Max ind.area:", max(train_ind_area), "\n")
             cat(" nrow(W):", nrow(W_borough), "\n")
             cat(" Is W symmetric?", isSymmetric(W_borough), "\n")
             cat(" Any NA/Inf in W?", any(!is.finite(W_borough)), "\n")
             cat(" Row sums positive?", all(rowSums(W_borough) > 0), "\n") # CARBayes requires positive row sums
             return(NULL) 
        })
    })
    # ... (rest of the prediction and evaluation code remains the same) ...
    # --- Prediction (within fold) ---
    test_preds <- rep(NA_real_, nrow(test_data)) 
    
    if (!is.null(model_fit) && inherits(model_fit, "list") && !is.null(model_fit$samples$beta)) {
         X_test <- NULL
         tryCatch({ X_test <- model.matrix(current_formula, data = test_data) }, 
                  error = function(e){ cat("Warning: Error creating test model matrix: ", e$message, "\n") })
         
         if(!is.null(X_test)){
             beta_samples <- model_fit$samples$beta
             model_coef_names <- colnames(beta_samples)
             test_matrix_names <- colnames(X_test)
             if(is.null(model_coef_names)) model_coef_names <- paste0("beta_", 1:ncol(beta_samples))
             if(is.null(test_matrix_names)) test_matrix_names <- paste0("X_", 1:ncol(X_test))
             common_coefs <- intersect(model_coef_names, test_matrix_names)

             if(length(common_coefs) > 0) {
                 beta_indices <- match(common_coefs, model_coef_names)
                 xtest_indices <- match(common_coefs, test_matrix_names)
                 # Ensure indices are valid before subsetting
                 if(all(!is.na(beta_indices)) && all(!is.na(xtest_indices))) {
                     beta_mean_common <- colMeans(beta_samples[, beta_indices, drop = FALSE])
                     X_test_common <- X_test[, xtest_indices, drop = FALSE]
                 
                     if(length(beta_mean_common) == ncol(X_test_common) && length(beta_mean_common) > 0){
                         eta_test <- X_test_common %*% beta_mean_common
                         lambda_test <- exp(eta_test)
                         test_preds <- as.numeric(lambda_test)
                     } else { cat("Warning: Dimension mismatch after selecting common coefficients.\n") }
                 } else { cat("Warning: Index matching failed for common coefficients.\n")}
             } else { cat("Warning: No common coefficients between model and test matrix.\n") }
         }
    } else {
        cat("Model fitting failed or samples unavailable for fold", i, ". Cannot predict.\n")
    }

    # --- Evaluate and Store ---
    test_actual_fold <- test_data[[response_var]]
    if(any(is.na(test_preds))) {
        mean_train_resp = mean(train_data[[response_var]], na.rm=TRUE)
        test_preds[is.na(test_preds)] <- mean_train_resp
    }
    
    test_mae <- mean(abs(test_actual_fold - test_preds), na.rm = TRUE)
    test_rmse <- sqrt(mean((test_actual_fold - test_preds)^2, na.rm = TRUE))
    
    cat("Fold", i, "Test MAE:", format(test_mae, digits = 4), "\n")
    cat("Fold", i, "Test RMSE:", format(test_rmse, digits = 4), "\n")
    
    fold_results[[i]] <- list(
      fold = i, selected_features = selected_features, formula = formula_str,
      test_mae = test_mae, test_rmse = test_rmse, model_runtime = fit_time['elapsed']
    )
    all_test_actual <- c(all_test_actual, test_actual_fold)
    all_test_predicted <- c(all_test_predicted, test_preds)
    
    rm(train_data, test_data, X_train_fs, model_fit, X_test); gc() 
  } # End CV loop
  
  # ... (Overall Results plotting remains the same) ...
  overall_mae <- mean(abs(all_test_actual - all_test_predicted), na.rm = TRUE)
  overall_rmse <- sqrt(mean((all_test_actual - all_test_predicted)^2, na.rm = TRUE))
  
  cat("\n========== Overall CV Results ==========\n")
  cat("Overall MAE:", format(overall_mae, digits = 4), "\n")
  cat("Overall RMSE:", format(overall_rmse, digits = 4), "\n")
  
  if (requireNamespace("ggplot2", quietly = TRUE) && length(all_test_actual) > 0) {
     # ... (plotting code as before) ...
     plot_data <- data.frame(Actual = all_test_actual, Predicted = all_test_predicted)
     pred_q99 = quantile(plot_data$Predicted, 0.99, na.rm=T)
     actual_max = max(plot_data$Actual, na.rm=T)
     plot_limit = max(pred_q99, actual_max, na.rm=T) * 1.1
     
     p <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
       geom_point(alpha = 0.3) +
       geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
       labs(title = "Overall CV: Predicted vs Actual Accident Counts",
            x = "Actual Counts", y = "Predicted Counts") +
       theme_minimal() +
       coord_cartesian(xlim = c(0, plot_limit), ylim = c(0, plot_limit)) 
     print(p)
  }

  return(list(
    fold_results = fold_results,
    overall_mae = overall_mae,
    overall_rmse = overall_rmse,
    test_actual = all_test_actual,
    test_predicted = all_test_predicted,
    W_matrix = W_borough 
  ))
}

# --- Main Execution (Assume load_and_prepare_data is defined) ---

# 1. Load Data
full_data <- load_and_prepare_data("data/data_final.csv")

# 2. Run CV (No shapefile needed)
cv_results_multilevel <- run_cv_car_multilevel(
  data = full_data,
  response_var = "acc",
  k = 5,               
  k_neighbors_W = 6,   # K for centroid adjacency W
  max_features_lasso = 25, 
  mcmc_params = list(burnin = 2000, n.sample = 5000, thin = 5) 
)

# 3. Optional Final Model (Using common features from CV)
# ... (Code for fitting final model remains the same as the previous version, 
#      using the W_matrix returned by run_cv_car_multilevel or recalculating it) ...
all_selected <- unlist(lapply(cv_results_multilevel$fold_results, `[[`, "selected_features"))
if (length(all_selected) > 0) {
    feature_counts <- table(all_selected)
    num_folds = length(cv_results_multilevel$fold_results)
    common_features <- names(feature_counts[feature_counts > num_folds / 2]) 
} else { common_features <- character(0) }

if(length(common_features) > 0) {
    cat("\nFitting final model using", length(common_features), "common features...\n")
    # ... (rest of final model fitting code as before) ...
    final_formula_str <- paste("acc ~", paste(common_features, collapse = " + "))
    final_formula <- as.formula(final_formula_str)
    cat("Final formula:", final_formula_str, "\n")
    
    W_final <- cv_results_multilevel$W_matrix 
    ind_area_final <- full_data$area_idx
    
    final_mcmc_params = list(burnin = 5000, n.sample = 25000, thin = 20) 
    
    cat("\nFitting final S.CARmultilevel model on all data...\n")
    final_model <- S.CARmultilevel(
        formula = final_formula, family = "poisson", data = full_data,
        W = W_final, ind.area = ind_area_final, 
        burnin = final_mcmc_params$burnin, n.sample = final_mcmc_params$n.sample,    
        thin = final_mcmc_params$thin, verbose = TRUE 
    )
    
    cat("\n--- Final Model Summary ---\n"); print(summary(final_model))
    # ... (plotting phi at centroids code as before) ...
    if(!is.null(final_model$samples$phi)){
        cat("Plotting spatial random effects (phi) at centroid locations...\n")
        phi_mean <- colMeans(final_model$samples$phi)
        centroids_final <- full_data %>% group_by(borough_analysis_unit, borough_id) %>%
            summarise(centroid_x = mean(x, na.rm = TRUE), centroid_y = mean(y, na.rm = TRUE), .groups = 'drop') %>% arrange(borough_id)
        if(length(phi_mean) == nrow(centroids_final)) {
            centroids_final$phi_mean <- phi_mean
            if (requireNamespace("ggplot2", quietly = TRUE)) {
                 p_phi_centroids <- ggplot(centroids_final, aes(x = centroid_x, y = centroid_y, color = phi_mean)) + geom_point(size = 4, alpha=0.8) +
                      scale_color_viridis_c(option = "plasma") + labs(title = "Posterior Mean Spatial Random Effects (phi) at Borough Centroids", color = "Mean Phi", x="Longitude", y="Latitude") + theme_minimal() + coord_equal() 
                 print(p_phi_centroids)
            }
        } else { cat("Warning: Length mismatch phi vs centroids. Cannot plot.\n") }
    }

} else {
    cat("\nNo features consistently selected. Skipping final model.\n")
}

cat("\n--- Script Execution Finished ---\n")