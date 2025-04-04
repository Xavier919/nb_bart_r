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
library(tmap)

# Install viridis package if not already installed
if (!requireNamespace("viridis", quietly = TRUE)) {
  install.packages("viridis")
}

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
  }
  
  spatial_cols <- c('x', 'y')
  feature_cols <- c('ln_fi', 'ln_fri', 'ln_fli', 'ln_pi', 'ln_cti', 'ln_cli', 'ln_cri', 'ln_distdt',
                   'fi', 'fri', 'fli', 'pi', 'cti', 'cli', 'cri', 'distdt',
                   'total_lane', 'avg_crossw', 'tot_road_w', 'tot_crossw',
                   'commercial', 'number_of_', 'of_exclusi', 'curb_exten', 'median', 'all_pedest', 'half_phase', 'new_half_r',
                   'any_ped_pr', 'ped_countd', 'lt_restric', 'lt_prot_re', 'lt_protect', 'any_exclus', 'all_red_an', 'green_stra',
                   'parking', 'north_veh', 'north_ped', 'east_veh', 'east_ped', 'south_veh', 'south_ped', 'west_veh', 'west_ped')
  
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
  
  id_cols <- c('int_no', 'acc', 'pi', 'borough')
  id_cols <- id_cols[id_cols %in% colnames(data)]
  
  full_data <- cbind(data[, id_cols], X_base, X_spatial)
  return(full_data)
}

create_spatial_weights <- function(data, k_neighbors = 3) {
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

# Implementing CARleroux model
fit_car_leroux_model <- function(data, formula, W, burnin = 500, n.sample = 1000, thin = 2) {
  # Convert the weight matrix to the format required by CARBayes
  W_binary <- W
  W_binary[W_binary > 0] <- 1
  
  # Fit the model using S.CARleroux from CARBayes
  model <- S.CARleroux(
    formula = formula,
    data = data,
    family = "poisson",
    W = W_binary,
    burnin = burnin,
    n.sample = n.sample,
    thin = thin,
    verbose = TRUE
  )
  
  return(model)
}

evaluate_model <- function(model, test_data) {
  # Extract fitted values from the model
  fitted_values <- model$fitted.values
  
  # Calculate performance metrics
  if ("acc" %in% colnames(test_data)) {
    observed <- test_data$acc
    rmse_val <- rmse(observed, fitted_values)
    mae_val <- mae(observed, fitted_values)
    
    # Calculate R-squared
    ss_total <- sum((observed - mean(observed))^2)
    ss_residual <- sum((observed - fitted_values)^2)
    r_squared <- 1 - (ss_residual / ss_total)
    
    return(list(
      rmse = rmse_val,
      mae = mae_val,
      r_squared = r_squared
    ))
  } else {
    warning("Test data does not contain 'acc' column for evaluation")
    return(NULL)
  }
}

# Main function to run the CARleroux model
run_car_leroux_analysis <- function(k_neighbors = 5, max_features = 15) {
  # Load and prepare data
  data <- load_and_prepare_data()
  str(data)  # Check if data loaded properly
  
  # Create train-test split
  set.seed(42)
  train_idx <- createDataPartition(data$acc, p = 0.8, list = FALSE)
  train_data <- data[train_idx, ]
  test_data <- data[-train_idx, ]
  
  # Create spatial weights matrix
  W <- create_spatial_weights(train_data, k_neighbors = k_neighbors)
  
  # Select features using LASSO
  feature_cols <- names(train_data)[!names(train_data) %in% c('int_no', 'acc', 'pi', 'borough', 'x', 'y')]
  X_train <- train_data[, feature_cols]
  y_train <- train_data$acc
  
  selected_features <- select_features_with_lasso(X_train, y_train, max_features = max_features)
  print("Selected features:")
  print(selected_features)
  
  # Create formula for the model
  formula_str <- paste("acc ~ ", paste(selected_features, collapse = " + "))
  formula <- as.formula(formula_str)
  
  # Fit the CARleroux model
  cat("Fitting CARleroux model...\n")
  model <- fit_car_leroux_model(train_data, formula, W)
  
  # Summary of the model
  print(model)
  
  # Evaluate on test data
  cat("Evaluating model on test data...\n")
  
  # Create spatial weights for test data
  W_test <- create_spatial_weights(test_data, k_neighbors = k_neighbors)
  
  # Apply model to test data
  test_formula <- update(formula, acc ~ .)
  test_model <- fit_car_leroux_model(test_data, test_formula, W_test)
  
  # Evaluate performance
  metrics <- evaluate_model(test_model, test_data)
  print("Performance metrics:")
  print(metrics)
  
  # Create plots of observed vs. fitted values
  plot_data <- data.frame(
    Observed = test_data$acc,
    Fitted = test_model$fitted.values
  )
  
  p <- ggplot(plot_data, aes(x = Observed, y = Fitted)) +
    geom_point(alpha = 0.5) +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(title = "CARleroux Model: Observed vs. Fitted Values",
         x = "Observed Accidents",
         y = "Fitted Accidents") +
    theme_minimal()
  
  print(p)
  
  # If spatial data is available, create a map of residuals
  if (all(c("x", "y") %in% colnames(test_data))) {
    test_data$residuals <- test_data$acc - test_model$fitted.values
    
    # Convert to sf object for mapping
    test_sf <- st_as_sf(test_data, coords = c("x", "y"), crs = 32618)  # UTM Zone 18N for Montreal
    
    # Map residuals
    map <- mapview(test_sf, zcol = "residuals", layer.name = "Residuals")
    print(map)
  }
  
  return(list(
    model = model,
    test_model = test_model,
    metrics = metrics,
    selected_features = selected_features
  ))
}

# Example usage:
# results <- run_car_leroux_analysis(k_neighbors = 5, max_features = 15)

# Fixed function to gather predictions and create a map
map_cv_predictions <- function(cv_results, original_data, folds, save_maps = TRUE) {
  # Create a dataframe to store all test predictions
  all_predictions <- data.frame()
  
  # Extract predictions from each fold
  for (fold_idx in 1:length(cv_results)) {
    fold_name <- paste0("fold_", fold_idx)
    fold_results <- cv_results[[fold_name]]
    
    # Get indices for this fold's test data
    test_indices <- folds[[fold_idx]]
    
    # Get test data for this fold
    test_data <- original_data[test_indices, ]
    
    # Get fitted values - these are already in the original scale
    fitted_values <- fold_results$test_model$fitted.values
    
    # Create a dataframe with predictions for this fold
    fold_predictions <- data.frame(
      int_no = test_data$int_no,
      x = test_data$x,
      y = test_data$y,
      observed = test_data$acc,
      predicted = fitted_values,
      fold = fold_idx
    )
    
    # Add to the collection of all predictions
    all_predictions <- rbind(all_predictions, fold_predictions)
  }
  
  # Recalculate error metrics directly
  all_predictions$error <- all_predictions$observed - all_predictions$predicted
  all_predictions$abs_error <- abs(all_predictions$error)
  
  # Print overall metrics for verification
  total_mae <- mean(all_predictions$abs_error)
  total_mse <- mean(all_predictions$error^2)
  total_rmse <- sqrt(total_mse)
  
  cat("Overall metrics from all folds:\n")
  cat(paste0("  MAE:  ", round(total_mae, 4), "\n"))
  cat(paste0("  MSE:  ", round(total_mse, 4), "\n"))
  cat(paste0("  RMSE: ", round(total_rmse, 4), "\n"))
  
  # Convert to sf object for mapping
  predictions_sf <- st_as_sf(all_predictions, coords = c("x", "y"), crs = 32618)
  
  # Create maps with enhanced visualization
  # Note: mapview automatically puts the legend on the side
  map_predicted <- mapview(predictions_sf, zcol = "predicted", 
                           layer.name = "Predicted Accidents",
                           col.regions = viridis::plasma,
                           legend = TRUE,
                           cex = "predicted", # Size dots by prediction value
                           alpha.regions = 0.8) # Slight transparency
  
  map_observed <- mapview(predictions_sf, zcol = "observed", 
                          layer.name = "Observed Accidents",
                          col.regions = viridis::plasma,
                          legend = TRUE,
                          cex = "observed", # Size dots by observation value
                          alpha.regions = 0.8)
  
  map_error <- mapview(predictions_sf, zcol = "error", 
                       layer.name = "Prediction Error",
                       col.regions = viridis::viridis,
                       legend = TRUE,
                       alpha.regions = 0.8)
  
  # Create a combined map
  combined_map <- map_predicted + map_observed + map_error
  
  # Create a plot of observed vs predicted for all data
  pred_vs_obs_plot <- ggplot(all_predictions, aes(x = observed, y = predicted)) +
    geom_point(alpha = 0.5) +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(title = "All Folds: Observed vs. Predicted Accidents",
         x = "Observed Accidents",
         y = "Predicted Accidents") +
    theme_minimal()
  
  # Save maps and plots if requested
  if (save_maps) {
    # Create output directory if it doesn't exist
    dir.create("output", showWarnings = FALSE)
    
    # Convert the mapview objects to static plots using tmap
    tmap_mode("plot")
    
    # Convert mapview objects to tmap objects and save as PNG
    # Enhanced tmap version with improved legend and dot sizing
    tmap_obs <- tmap::tm_shape(predictions_sf) + 
      tm_dots(col = "observed", 
              palette = "plasma", 
              title = "Observed Accidents",
              size = "observed",
              sizes.legend = c(min(predictions_sf$observed), 
                              median(predictions_sf$observed), 
                              max(predictions_sf$observed)),
              size.lim = c(min(predictions_sf$observed), max(predictions_sf$observed)),
              alpha = 0.8,
              legend.size.show = TRUE) +
      tm_layout(legend.outside = TRUE, 
                legend.position = c("right", "center"),
                legend.title.size = 1.2,
                legend.text.size = 0.8,
                frame = FALSE)
    tmap::tmap_save(tmap_obs, filename = "output/observed_accidents_map.png", width = 10, height = 8)
    
    # Enhanced predicted accidents map
    tmap_pred <- tmap::tm_shape(predictions_sf) + 
      tm_dots(col = "predicted", 
              palette = "plasma", 
              title = "Predicted Accidents",
              size = "predicted",
              sizes.legend = c(min(predictions_sf$predicted), 
                              median(predictions_sf$predicted), 
                              max(predictions_sf$predicted)),
              size.lim = c(min(predictions_sf$predicted), max(predictions_sf$predicted)),
              alpha = 0.8,
              legend.size.show = TRUE) +
      tm_layout(legend.outside = TRUE, 
                legend.position = c("right", "center"),
                legend.title.size = 1.2,
                legend.text.size = 0.8,
                frame = FALSE)
    tmap::tmap_save(tmap_pred, filename = "output/predicted_accidents_map.png", width = 10, height = 8)
    
    # Enhanced error map
    tmap_error <- tmap::tm_shape(predictions_sf) + 
      tm_dots(col = "error", 
              palette = "viridis", 
              title = "Prediction Error",
              size = 0.1,
              alpha = 0.8) +
      tm_layout(legend.outside = TRUE, 
                legend.position = c("right", "center"),
                legend.title.size = 1.2,
                legend.text.size = 0.8,
                frame = FALSE)
    tmap::tmap_save(tmap_error, filename = "output/prediction_error_map.png", width = 10, height = 8)
    
    # Create a combined static map with improved styling
    tmap_combined <- tm_shape(predictions_sf) + 
      tm_dots(col = "observed", 
              palette = "plasma", 
              title = "Observed Accidents",
              size = "observed",
              alpha = 0.8) +
      tm_layout(panel.show = TRUE, 
                panel.labels = "Observed",
                legend.outside = TRUE, 
                legend.position = c("right", "center")) +
      tm_facets(sync = TRUE, ncol = 3) +
      tm_shape(predictions_sf) + 
      tm_dots(col = "predicted", 
              palette = "plasma", 
              title = "Predicted Accidents",
              size = "predicted",
              alpha = 0.8) +
      tm_layout(panel.show = TRUE, 
                panel.labels = "Predicted",
                legend.outside = TRUE, 
                legend.position = c("right", "center")) +
      tm_shape(predictions_sf) + 
      tm_dots(col = "error", 
              palette = "viridis", 
              title = "Prediction Error",
              alpha = 0.8) +
      tm_layout(panel.show = TRUE, 
                panel.labels = "Error",
                legend.outside = TRUE, 
                legend.position = c("right", "center"))
    tmap::tmap_save(tmap_combined, filename = "output/combined_map.png", width = 15, height = 5)
    
    # Save the observed vs. predicted plot
    ggsave("output/observed_vs_predicted.png", pred_vs_obs_plot, width = 10, height = 8)
    
    # Save the predictions data
    write.csv(all_predictions, "output/accidents_predictions.csv", row.names = FALSE)
    
    cat("Maps and plots saved as PNG files to the 'output' directory\n")
  }
  
  # Return the sf object and maps
  return(list(
    predictions = all_predictions,
    predictions_sf = predictions_sf,
    map_predicted = map_predicted,
    map_observed = map_observed,
    map_error = map_error,
    combined_map = combined_map,
    pred_vs_obs_plot = pred_vs_obs_plot
  ))
}

# Function to run cross-validation with CAR Leroux model
run_car_leroux_cv <- function(k_neighbors = 5, max_features = 15, k_folds = 5) {
  # Load and prepare data
  data <- load_and_prepare_data()
  
  # Create folds for cross-validation
  set.seed(42)
  folds <- createFolds(data$acc, k = k_folds, list = TRUE, returnTrain = FALSE)
  
  # Initialize storage for metrics
  cv_results <- list()
  all_metrics <- data.frame(
    fold = integer(),
    mae = numeric(),
    mse = numeric(),
    rmse = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Run k-fold cross-validation
  for (fold_idx in 1:k_folds) {
    cat(paste0("\n---------- FOLD ", fold_idx, " OF ", k_folds, " ----------\n"))
    
    # Split data into training and validation sets for this fold
    test_indices <- folds[[fold_idx]]
    train_indices <- setdiff(1:nrow(data), test_indices)
    
    train_data <- data[train_indices, ]
    test_data <- data[test_indices, ]
    
    # Create spatial weights matrix for training data
    W <- create_spatial_weights(train_data, k_neighbors = k_neighbors)
    
    # Select features using LASSO on this fold's training data
    feature_cols <- names(train_data)[!names(train_data) %in% c('int_no', 'acc', 'pi', 'borough', 'x', 'y')]
    X_train <- train_data[, feature_cols]
    y_train <- train_data$acc
    
    selected_features <- select_features_with_lasso(X_train, y_train, max_features = max_features)
    cat("Selected features for fold", fold_idx, ":\n")
    print(selected_features)
    
    # Create formula for the model
    formula_str <- paste("acc ~ ", paste(selected_features, collapse = " + "))
    formula <- as.formula(formula_str)
    
    # Fit the CARleroux model
    cat("Fitting CARleroux model for fold", fold_idx, "...\n")
    model <- fit_car_leroux_model(train_data, formula, W)
    
    # Create spatial weights for test data
    W_test <- create_spatial_weights(test_data, k_neighbors = k_neighbors)
    
    # Apply model to test data
    test_formula <- update(formula, acc ~ .)
    test_model <- fit_car_leroux_model(test_data, test_formula, W_test)
    
    # Calculate metrics
    fitted_values <- test_model$fitted.values
    observed_values <- test_data$acc
    
    mae_val <- mean(abs(observed_values - fitted_values))
    mse_val <- mean((observed_values - fitted_values)^2)
    rmse_val <- sqrt(mse_val)
    
    # Store metrics for this fold
    fold_metrics <- data.frame(
      fold = fold_idx,
      mae = mae_val,
      mse = mse_val,
      rmse = rmse_val,
      stringsAsFactors = FALSE
    )
    
    all_metrics <- rbind(all_metrics, fold_metrics)
    
    # Print metrics for this fold
    cat(paste0("Fold ", fold_idx, " metrics:\n"))
    cat(paste0("  MAE: ", round(mae_val, 4), "\n"))
    cat(paste0("  MSE: ", round(mse_val, 4), "\n"))
    cat(paste0("  RMSE: ", round(rmse_val, 4), "\n"))
    
    # Store the model and results
    cv_results[[paste0("fold_", fold_idx)]] <- list(
      model = model,
      test_model = test_model,
      selected_features = selected_features,
      metrics = fold_metrics
    )
    
    # Create plot of observed vs. fitted values for this fold
    plot_data <- data.frame(
      Observed = observed_values,
      Fitted = fitted_values
    )
    
    p <- ggplot(plot_data, aes(x = Observed, y = Fitted)) +
      geom_point(alpha = 0.5) +
      geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
      labs(title = paste0("Fold ", fold_idx, ": Observed vs. Fitted Values"),
           x = "Observed Accidents",
           y = "Fitted Accidents") +
      theme_minimal()
    
    print(p)
  }
  
  # Calculate and print average metrics across all folds
  avg_metrics <- colMeans(all_metrics[, c("mae", "mse", "rmse")])
  
  cat("\n---------- CROSS-VALIDATION SUMMARY ----------\n")
  cat("Average metrics across all folds:\n")
  cat(paste0("  MAE:  ", round(avg_metrics["mae"], 4), "\n"))
  cat(paste0("  MSE:  ", round(avg_metrics["mse"], 4), "\n"))
  cat(paste0("  RMSE: ", round(avg_metrics["rmse"], 4), "\n"))
  
  # Print metrics for each fold in a table format
  print(all_metrics)
  
  # After the cross-validation loop, add this code:
  cat("\n---------- TOP PREDICTIONS BY FOLD ----------\n")
  for (fold_idx in 1:k_folds) {
    # Get the fold data
    fold_name <- paste0("fold_", fold_idx)
    fold_results <- cv_results[[fold_name]]
    
    # Get test indices for this fold
    test_indices <- folds[[fold_idx]]
    test_data <- data[test_indices, ]
    
    # Get fitted values
    fitted_values <- fold_results$test_model$fitted.values
    observed_values <- test_data$acc
    
    # Create a data frame with observed and predicted values
    comparison <- data.frame(
      intersection = test_data$int_no,
      observed = observed_values,
      predicted = fitted_values
    )
    
    # Sort by predicted values (descending)
    top_predictions <- comparison[order(comparison$predicted, decreasing = TRUE), ]
    
    # Print the top 10 predictions
    cat(paste0("\nTop 10 predicted values for fold ", fold_idx, ":\n"))
    print(head(top_predictions, 10))
  }
  
  # Return the list including the folds
  return(list(
    cv_results = cv_results,
    all_metrics = all_metrics,
    avg_metrics = avg_metrics,
    folds = folds,
    data = data
  ))
}

# Run cross-validation with modified function
results_cv <- run_car_leroux_cv(k_neighbors = 5, max_features = 15, k_folds = 5)

# Generate maps of predictions and save them
prediction_maps <- map_cv_predictions(results_cv$cv_results, results_cv$data, results_cv$folds, save_maps = TRUE)

# Display the combined map
prediction_maps$combined_map

# Display the observed vs predicted plot
print(prediction_maps$pred_vs_obs_plot)