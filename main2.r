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

# First, we'll add the R6 class definition for carbayes_model (similar to main.r)
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
      self$exposure <- exposure
      self$spatial_vars <- spatial_vars
    },
    
    fit = function() {
      data_to_use <- self$data
      
      if (is.null(self$W)) {
        self$W <- create_spatial_weights(data_to_use[, self$spatial_vars])
      }
      
      self$W_list <- self$create_W_list(self$W)
      
      formula_parts <- strsplit(self$formula, "~")[[1]]
      response_var <- trimws(formula_parts[1])
      predictors <- trimws(formula_parts[2])
      
      if (!is.null(self$random_effect) && self$random_effect %in% colnames(data_to_use)) {
        if (!is.factor(data_to_use[[self$random_effect]])) {
          data_to_use[[self$random_effect]] <- as.factor(data_to_use[[self$random_effect]])
        }
        predictors <- paste(predictors, "+", self$random_effect)
      }
      
      full_formula <- as.formula(paste(response_var, "~", predictors))
      
      model <- S.CARleroux(
        formula = full_formula,
        family = "poisson",  
        data = data_to_use,
        W = self$W,
        burnin = 2000,
        n.sample = 5000,
        thin = 2,
        rho = 0.8,
        verbose = TRUE
      )
      
      self$model <- model
      self$results <- model
      self$data_with_re <- data_to_use
      
      return(self$results)
    },
    
    predict = function(newdata = NULL) {
      if (is.null(self$results)) {
        stop("Model must be fitted before prediction")
      }
      
      if (is.null(newdata)) {
        if (!is.null(self$data_with_re)) {
          newdata <- self$data_with_re
        } else {
          newdata <- self$data
        }
      }
      
      formula_parts <- strsplit(self$formula, "~")[[1]]
      predictors <- trimws(formula_parts[2])
      
      if (!is.null(self$random_effect) && self$random_effect %in% colnames(newdata)) {
        if (!grepl(self$random_effect, predictors)) {
          predictors <- paste(predictors, "+", self$random_effect)
        }
      }
      
      X_formula <- as.formula(paste("~", predictors))
      X <- try(model.matrix(X_formula, data = newdata), silent = TRUE)
      
      if (inherits(X, "try-error")) {
        pred_vars <- strsplit(predictors, "\\+")[[1]]
        pred_vars <- trimws(pred_vars)
        simple_vars <- pred_vars[!grepl(":", pred_vars) & !grepl("\\*", pred_vars)]
        simple_formula <- as.formula(paste("~", paste(simple_vars, collapse = " + ")))
        X <- model.matrix(simple_formula, data = newdata)
      }
      
      beta <- self$results$samples$beta
      beta_mean <- apply(beta, 2, mean)
      
      if (length(beta_mean) != ncol(X)) {
        if (length(beta_mean) > ncol(X)) {
          beta_mean <- beta_mean[1:ncol(X)]
        } else {
          X <- X[, 1:length(beta_mean)]
        }
      }
      
      eta <- X %*% beta_mean
      lambda <- exp(eta)
      
      if (!is.null(self$results$samples$phi)) {
        phi_mean <- apply(self$results$samples$phi, 2, mean)
        if (nrow(newdata) == length(phi_mean)) {
          lambda <- lambda * exp(phi_mean)
        }
      }
      
      return(lambda)
    },
    
    create_W_list = function(W) {
      n <- nrow(W)
      W_list <- list()
      W_list$n <- n
      
      W_list$adj <- vector("list", n)
      W_list$weights <- vector("list", n)
      W_list$num <- numeric(n)
      
      for (i in 1:n) {
        neighbors <- which(W[i, ] > 0)
        W_list$adj[[i]] <- neighbors
        W_list$weights[[i]] <- W[i, neighbors]
        W_list$num[i] <- length(neighbors)
      }
      
      W_list$sumjk <- sum(W_list$num)
      return(W_list)
    }
  )
)

# Then modify the run_car_leroux_analysis function to use this class
run_car_leroux_analysis <- function(k_neighbors = 5, max_features = 15) {
  # Load and prepare data
  data <- load_and_prepare_data()
  
  # Create train-test split
  set.seed(42)
  train_idx <- createDataPartition(data$acc, p = 0.8, list = FALSE)
  train_data <- data[train_idx, ]
  test_data <- data[-train_idx, ]
  
  # Select features using LASSO
  feature_cols <- names(train_data)[!names(train_data) %in% c('int_no', 'acc', 'pi', 'borough', 'x', 'y')]
  X_train <- train_data[, feature_cols]
  y_train <- train_data$acc
  
  selected_features <- select_features_with_lasso(X_train, y_train, max_features = max_features)
  print("Selected features:")
  print(selected_features)
  
  # Create formula for the model
  formula_str <- paste("acc ~", paste(selected_features, collapse = " + "))
  
  # Create and fit the CARBayes model
  carbayes <- carbayes_model$new(
    data = train_data,
    formula = formula_str,
    random_effect = "borough",
    spatial_vars = c('x', 'y')
  )
  
  cat("Fitting CARleroux model...\n")
  model_results <- carbayes$fit()
  
  # Make predictions on train and test data
  train_predictions <- carbayes$predict(train_data)
  test_predictions <- carbayes$predict(test_data)
  
  # Evaluate performance
  train_metrics <- list(
    mae = mean(abs(train_data$acc - train_predictions)),
    mse = mean((train_data$acc - train_predictions)^2),
    rmse = sqrt(mean((train_data$acc - train_predictions)^2))
  )
  
  test_metrics <- list(
    mae = mean(abs(test_data$acc - test_predictions)),
    mse = mean((test_data$acc - test_predictions)^2),
    rmse = sqrt(mean((test_data$acc - test_predictions)^2))
  )
  
  # Create plots
  plot_data <- data.frame(
    Observed = test_data$acc,
    Fitted = test_predictions
  )
  
  p <- ggplot(plot_data, aes(x = Observed, y = Fitted)) +
    geom_point(alpha = 0.5) +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(title = "CARleroux Model: Observed vs. Fitted Values",
         x = "Observed Accidents",
         y = "Fitted Accidents") +
    theme_minimal()
  
  print(p)
  
  return(list(
    model = carbayes,
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    selected_features = selected_features,
    test_predictions = test_predictions,
    test_actual = test_data$acc
  ))
}

# Modified map_cv_predictions function to include observed accidents in output file and make dots bigger
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
    fitted_values <- fold_results$test_predictions
    
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
  
  # Create only the predicted map with mapview - now with larger dots
  map_predicted <- mapview(predictions_sf, zcol = "predicted", 
                          layer.name = "Predicted Accidents",
                          col.regions = viridis::plasma,
                          legend = TRUE,
                          cex = 5,  # Increased size for all dots (was 3)
                          alpha.regions = 0.8)
  
  # Create a plot of observed vs predicted for all data
  # Create a plot of observed vs predicted for all data
  pred_vs_obs_plot <- ggplot(all_predictions, aes(x = observed, y = predicted)) +
      geom_point(alpha = 0.5) +
      geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
      labs(title = "All Folds: Observed vs. Predicted Accidents",
          x = "Observed Accidents",
          y = "Predicted Accidents") +
      theme_minimal() +
      theme(
        panel.background = element_rect(fill = "white"),
        plot.background = element_rect(fill = "white")
      )
  
  # Save maps and plots if requested
  if (save_maps) {
    # Create output directory if it doesn't exist
    dir.create("output", showWarnings = FALSE)
    
    # Create the predictions data frame with intersection ID, observed and predicted accidents
    simplified_predictions <- all_predictions[, c("int_no", "observed", "predicted")]
    
    # Sort by predicted accidents (descending)
    simplified_predictions <- simplified_predictions[order(simplified_predictions$predicted, decreasing = TRUE), ]
    
    # Save the predictions data with observed column
    write.csv(simplified_predictions, "output/accidents_predictions.csv", row.names = FALSE)
    
    # Convert the mapview objects to static plots using tmap
    tmap_mode("plot")
    
    # Create modern tmap v4 compatible code for predicted accidents map
    # With single legend and uniform but larger dot sizes
    tmap_pred <- tmap::tm_shape(predictions_sf) + 
      tmap::tm_dots(
        fill = "predicted",                    # Color dots by predicted value 
        fill.scale = tm_scale(                 # Use scale object for styling
          values = "plasma",                   # Color palette
          title = "Predicted Accidents"        # Legend title
        ),
        size = 0.2,                            # Increased fixed size for all dots (was 0.1)
        fill_alpha = 0.8                       # Use fill_alpha instead of alpha
      ) +
      tm_layout(
        legend.outside = TRUE,                 # Place legend outside
        legend.outside.position = "right",     # Position on the right
        legend.title.size = 1.2,               # Title size
        legend.text.size = 0.8,                # Text size
        frame = FALSE                          # No frame
      )
    
    # Save the map
    tmap::tmap_save(tmap_pred, filename = "output/predicted_accidents_map.png", width = 10, height = 8)
    
    # Save the observed vs. predicted plot
    ggsave("output/observed_vs_predicted.png", pred_vs_obs_plot, width = 10, height = 8)
    
    cat("Map of predicted accidents saved to output/predicted_accidents_map.png\n")
    cat("Accident predictions with observed and predicted values saved to output/accidents_predictions.csv\n")
    cat("Observed vs. predicted plot saved to output/observed_vs_predicted.png\n")
  }
  
  # Return only the predicted map and relevant data
  return(list(
    predictions = all_predictions,
    predictions_sf = predictions_sf,
    map_predicted = map_predicted,
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
    
    # Select features using LASSO on this fold's training data
    feature_cols <- names(train_data)[!names(train_data) %in% c('int_no', 'acc', 'pi', 'borough', 'x', 'y')]
    X_train <- train_data[, feature_cols]
    y_train <- train_data$acc
    
    selected_features <- select_features_with_lasso(X_train, y_train, max_features = max_features)
    cat("Selected features for fold", fold_idx, ":\n")
    print(selected_features)
    
    # Create formula for the model
    formula_str <- paste("acc ~", paste(selected_features, collapse = " + "))
    
    # Create and fit the CARBayes model
    carbayes <- carbayes_model$new(
      data = train_data,
      formula = formula_str,
      random_effect = "borough",
      spatial_vars = c('x', 'y')
    )
    
    cat("Fitting CARleroux model for fold", fold_idx, "...\n")
    model_results <- carbayes$fit()
    
    # Make predictions on train and test data
    train_predictions <- carbayes$predict(train_data)
    test_predictions <- carbayes$predict(test_data)
    
    # Calculate metrics
    train_mae <- mean(abs(train_data$acc - train_predictions))
    train_mse <- mean((train_data$acc - train_predictions)^2)
    train_rmse <- sqrt(train_mse)
    
    test_mae <- mean(abs(test_data$acc - test_predictions))
    test_mse <- mean((test_data$acc - test_predictions)^2)
    test_rmse <- sqrt(test_mse)
    
    # Store metrics for this fold
    fold_metrics <- data.frame(
      fold = fold_idx,
      mae = test_mae,
      mse = test_mse,
      rmse = test_rmse,
      stringsAsFactors = FALSE
    )
    
    all_metrics <- rbind(all_metrics, fold_metrics)
    
    # Print metrics for this fold
    cat(paste0("Fold ", fold_idx, " metrics:\n"))
    cat(paste0("  MAE: ", round(test_mae, 4), "\n"))
    cat(paste0("  MSE: ", round(test_mse, 4), "\n"))
    cat(paste0("  RMSE: ", round(test_rmse, 4), "\n"))
    
    # Store the model and results
    cv_results[[paste0("fold_", fold_idx)]] <- list(
      model = carbayes,
      test_predictions = test_predictions,
      selected_features = selected_features,
      metrics = fold_metrics
    )
    
    # Create plot of observed vs. fitted values for this fold
    plot_data <- data.frame(
      Observed = test_data$acc,
      Fitted = test_predictions
    )
    
    p <- ggplot(plot_data, aes(x = Observed, y = Fitted)) +
      geom_point(alpha = 0.5) +
      geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
      labs(title = paste0("Fold ", fold_idx, ": Observed vs. Fitted Values"),
           x = "Observed Accidents",
           y = "Fitted Accidents") +
      theme_minimal() +
      theme(
        panel.background = element_rect(fill = "white"),
        plot.background = element_rect(fill = "white")
      )
    
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
    fitted_values <- fold_results$test_predictions
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

# Display the predicted map
prediction_maps$map_predicted

# Display the observed vs predicted plot
print(prediction_maps$pred_vs_obs_plot)