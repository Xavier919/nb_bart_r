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
        burnin = 1000,
        n.sample = 2000,
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

# Function to train the SEM Poisson model
train_poisson_sem <- function(formula, train_data, val_data, global_weights, fold_weights) {
  # Step 1: Fit a standard Poisson model first
  poisson_model <- glm(formula, data = train_data, family = poisson(link = "log"))
  
  # Step 2: Extract residuals (deviance residuals work well)
  residuals <- residuals(poisson_model, type = "deviance")
  
  # Step 3: Fit spatial error model to the residuals
  train_data$residuals <- residuals
  sem_residuals <- errorsarlm(as.formula("residuals ~ 1"), data = train_data, listw = fold_weights)
  
  # Step 4: For prediction, combine both models
  val_data <- data.frame(val_data)
  poisson_predictions <- predict(poisson_model, newdata = val_data, type = "response")
  
  residual_predictions <- predict(sem_residuals, newdata = val_data, listw = global_weights, 
                                zero.policy = TRUE, type = "response")

  # Apply adjustment (on log scale, then transform back)
  adjusted_predictions <- poisson_predictions * exp(residual_predictions)
  return(adjusted_predictions)
}

# Combined cross-validation function for both models
run_combined_cross_validation <- function(data, response_var = "acc", k = 5, random_effect = "borough", max_features = 30) {
  set.seed(42)
  
  # Create folds using caret - same folds for both models
  folds <- createFolds(data$acc, k = k, returnTrain = TRUE)
  
  # Initialize results for both models
  carbayes_fold_results <- list()
  sem_fold_results <- list()
  
  carbayes_test_actual <- list()
  carbayes_test_predicted <- list()
  sem_test_actual <- list()
  sem_test_predicted <- list()
  
  # Global spatial weights for SEM model
  sp_data <- data
  coordinates(sp_data) <- ~ x + y
  w_global <- make_spatial_weights_for_sem(sp_data, k = 3)
  
  feature_formulas <- list()
  
  for (i in 1:k) {
    cat(sprintf("\nProcessing fold %d of %d\n", i, k))
    train_idx <- folds[[i]]
    test_idx <- setdiff(1:nrow(data), train_idx)
    
    train_data <- data[train_idx, ]
    test_data <- data[test_idx, ]
    
    # Create spatial data for SEM model
    train_sp <- sp_data[train_idx, ]
    test_sp <- sp_data[test_idx, ]
    
    # Create spatial weights for SEM model
    w_train <- make_spatial_weights_for_sem(train_sp, k = 3)
    
    # Feature selection using LASSO from CARBayes file
    exclude_cols <- c('int_no', 'acc', 'pi', 'borough', 'x', 'y')
    feature_cols <- setdiff(colnames(train_data), exclude_cols)
    X_train <- train_data[, feature_cols]
    y_train <- train_data[[response_var]]
    
    selected_features <- select_features_with_lasso(X_train, y_train, max_features)
    
    # Create formula for both models
    formula_str <- paste(response_var, "~", paste(selected_features, collapse = " + "))
    feature_formulas[[i]] <- formula_str
    formula <- as.formula(formula_str)
    
    # Train CARBayes model
    carbayes <- carbayes_model$new(
      data = train_data,
      formula = formula_str,
      random_effect = random_effect,
      exposure = NULL,
      spatial_vars = c('x', 'y')
    )
    
    carbayes_results <- carbayes$fit()
    
    carbayes_train_preds <- carbayes$predict(train_data)
    carbayes_test_preds <- carbayes$predict(test_data)
    
    # Calculate metrics for CARBayes
    carbayes_train_mae <- mean(abs(train_data$acc - carbayes_train_preds), na.rm = TRUE)
    carbayes_train_mse <- mean((train_data$acc - carbayes_train_preds)^2, na.rm = TRUE)
    carbayes_train_rmse <- sqrt(carbayes_train_mse)
    
    carbayes_test_mae <- mean(abs(test_data$acc - carbayes_test_preds), na.rm = TRUE)
    carbayes_test_mse <- mean((test_data$acc - carbayes_test_preds)^2, na.rm = TRUE)
    carbayes_test_rmse <- sqrt(carbayes_test_mse)
    
    # Train SEM Poisson model (without borough as random effect)
    sem_test_preds <- train_poisson_sem(
      formula = formula, 
      train_data = train_sp, 
      val_data = test_sp, 
      global_weights = w_global, 
      fold_weights = w_train
    )
    
    # Calculate metrics for SEM Poisson
    sem_test_mae <- mean(abs(test_data$acc - sem_test_preds), na.rm = TRUE)
    sem_test_mse <- mean((test_data$acc - sem_test_preds)^2, na.rm = TRUE)
    sem_test_rmse <- sqrt(sem_test_mse)
    
    # Store results
    carbayes_fold_results[[i]] <- list(
      fold = i,
      train_mae = carbayes_train_mae,
      train_mse = carbayes_train_mse,
      train_rmse = carbayes_train_rmse,
      test_mae = carbayes_test_mae,
      test_mse = carbayes_test_mse,
      test_rmse = carbayes_test_rmse,
      model = carbayes,
      selected_features = selected_features,
      formula = formula_str
    )
    
    sem_fold_results[[i]] <- list(
      fold = i,
      test_mae = sem_test_mae,
      test_mse = sem_test_mse,
      test_rmse = sem_test_rmse,
      selected_features = selected_features,
      formula = formula_str
    )
    
    carbayes_test_actual[[i]] <- test_data$acc
    carbayes_test_predicted[[i]] <- carbayes_test_preds
    sem_test_actual[[i]] <- test_data$acc
    sem_test_predicted[[i]] <- sem_test_preds
  }
  
  # Combine results
  carbayes_test_actual_combined <- unlist(carbayes_test_actual)
  carbayes_test_predicted_combined <- unlist(carbayes_test_predicted)
  sem_test_actual_combined <- unlist(sem_test_actual)
  sem_test_predicted_combined <- unlist(sem_test_predicted)
  
  # Calculate overall metrics for CARBayes
  carbayes_overall_mae <- mean(abs(carbayes_test_actual_combined - carbayes_test_predicted_combined), na.rm = TRUE)
  carbayes_overall_mse <- mean((carbayes_test_actual_combined - carbayes_test_predicted_combined)^2, na.rm = TRUE)
  carbayes_overall_rmse <- sqrt(carbayes_overall_mse)
  
  # Calculate overall metrics for SEM Poisson
  sem_overall_mae <- mean(abs(sem_test_actual_combined - sem_test_predicted_combined), na.rm = TRUE)
  sem_overall_mse <- mean((sem_test_actual_combined - sem_test_predicted_combined)^2, na.rm = TRUE)
  sem_overall_rmse <- sqrt(sem_overall_mse)
  
  # Feature consistency analysis
  all_features <- unique(unlist(lapply(carbayes_fold_results, function(x) x$selected_features)))
  feature_counts <- sapply(all_features, function(feat) {
    sum(sapply(carbayes_fold_results, function(x) feat %in% x$selected_features))
  })
  
  feature_consistency <- data.frame(
    Feature = names(feature_counts),
    Count = feature_counts,
    Percentage = 100 * feature_counts / k
  )
  feature_consistency <- feature_consistency[order(feature_consistency$Count, decreasing = TRUE), ]
  
  return(list(
    carbayes_results = list(
      fold_results = carbayes_fold_results,
      overall_mae = carbayes_overall_mae,
      overall_mse = carbayes_overall_mse,
      overall_rmse = carbayes_overall_rmse,
      test_actual = carbayes_test_actual_combined,
      test_predicted = carbayes_test_predicted_combined
    ),
    sem_results = list(
      fold_results = sem_fold_results,
      overall_mae = sem_overall_mae,
      overall_mse = sem_overall_mse,
      overall_rmse = sem_overall_rmse,
      test_actual = sem_test_actual_combined,
      test_predicted = sem_test_predicted_combined
    ),
    feature_consistency = feature_consistency,
    feature_formulas = feature_formulas
  ))
}

# Generate figures from cross-validation results
generate_cv_figures <- function(cv_results, data) {
  # Figure 1: Map of predicted accidents
  # Create a data frame to store all predictions
  all_predictions <- data.frame(
    x = data$x,
    y = data$y,
    actual = data$acc,
    predicted = NA
  )
  
  # Get all test predictions from cross-validation results
  test_actual_combined <- cv_results$carbayes_results$test_actual
  test_predicted_combined <- cv_results$carbayes_results$test_predicted
  
  # Create a more robust way to match predictions back to original data
  # First, collect all test indices from each fold
  all_test_indices <- list()
  for (i in 1:length(cv_results$carbayes_results$fold_results)) {
    # Recreate the fold indices
    set.seed(42)
    folds <- createFolds(data$acc, k = length(cv_results$carbayes_results$fold_results), returnTrain = TRUE)
    train_idx <- folds[[i]]
    test_idx <- setdiff(1:nrow(data), train_idx)
    all_test_indices[[i]] <- test_idx
  }
  
  # Now assign predictions to the correct indices
  start_idx <- 1
  for (i in 1:length(all_test_indices)) {
    test_idx <- all_test_indices[[i]]
    end_idx <- start_idx + length(test_idx) - 1
    
    # Make sure we don't go beyond the available predictions
    if (end_idx <= length(test_predicted_combined)) {
      fold_predictions <- test_predicted_combined[start_idx:end_idx]
      all_predictions$predicted[test_idx] <- fold_predictions
      start_idx <- end_idx + 1
    }
  }
  
  # Remove any rows with NA predictions
  all_predictions <- all_predictions[!is.na(all_predictions$predicted), ]
  
  # Create spatial points data frame for mapping
  sp_predictions <- all_predictions
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
    labs(title = "Map of Predicted Accidents",
         subtitle = "CARBayes Model Cross-Validation Results")
  
  # Figure 2: Predicted vs Actual values
  scatter_plot <- ggplot(all_predictions, aes(x = actual, y = predicted)) +
    geom_point(alpha = 0.5) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    geom_smooth(method = "lm", color = "blue", se = TRUE) +
    theme_classic() +
    theme(
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      legend.background = element_rect(fill = "white", color = NA)
    ) +
    labs(title = "Predicted vs Actual Accidents",
         subtitle = "CARBayes Model Cross-Validation Results",
         x = "Actual Accidents",
         y = "Predicted Accidents")
  
  # Calculate correlation
  correlation <- cor(all_predictions$actual, all_predictions$predicted)
  cat(sprintf("\nCorrelation between actual and predicted values: %.4f\n", correlation))
  
  return(list(map_plot = map_plot, scatter_plot = scatter_plot, predictions = all_predictions))
}

# Main execution
cat("Loading and preparing data...\n")
full_data <- load_and_prepare_data()

cat("Running combined cross-validation...\n")
cv_results <- run_combined_cross_validation(
  data = full_data, 
  response_var = "acc",
  k = 5,
  random_effect = "borough",
  max_features = 30
)

# Print Results
cat("\n==== CARBayes Model Results ====\n")
cat("Metrics by fold:\n")
for (i in 1:length(cv_results$carbayes_results$fold_results)) {
  fold <- cv_results$carbayes_results$fold_results[[i]]
  cat(sprintf("Fold %d: Test MAE = %.4f, Test MSE = %.4f, Test RMSE = %.4f\n", 
              i, fold$test_mae, fold$test_mse, fold$test_rmse))
}

carbayes_mean_mae <- mean(sapply(cv_results$carbayes_results$fold_results, function(x) x$test_mae))
carbayes_mean_mse <- mean(sapply(cv_results$carbayes_results$fold_results, function(x) x$test_mse))
carbayes_mean_rmse <- mean(sapply(cv_results$carbayes_results$fold_results, function(x) x$test_rmse))

cat("\nCARBayes Mean performance across folds:\n")
cat(sprintf("Mean MAE: %.4f\n", carbayes_mean_mae))
cat(sprintf("Mean MSE: %.4f\n", carbayes_mean_mse))
cat(sprintf("Mean RMSE: %.4f\n", carbayes_mean_rmse))
cat(sprintf("Overall MAE: %.4f\n", cv_results$carbayes_results$overall_mae))
cat(sprintf("Overall MSE: %.4f\n", cv_results$carbayes_results$overall_mse))
cat(sprintf("Overall RMSE: %.4f\n", cv_results$carbayes_results$overall_rmse))

cat("\n==== SEM Poisson Model Results ====\n")
cat("Metrics by fold:\n")
for (i in 1:length(cv_results$sem_results$fold_results)) {
  fold <- cv_results$sem_results$fold_results[[i]]
  cat(sprintf("Fold %d: Test MAE = %.4f, Test MSE = %.4f, Test RMSE = %.4f\n", 
              i, fold$test_mae, fold$test_mse, fold$test_rmse))
}

sem_mean_mae <- mean(sapply(cv_results$sem_results$fold_results, function(x) x$test_mae))
sem_mean_mse <- mean(sapply(cv_results$sem_results$fold_results, function(x) x$test_mse))
sem_mean_rmse <- mean(sapply(cv_results$sem_results$fold_results, function(x) x$test_rmse))

cat("\nSEM Poisson Mean performance across folds:\n")
cat(sprintf("Mean MAE: %.4f\n", sem_mean_mae))
cat(sprintf("Mean MSE: %.4f\n", sem_mean_mse))
cat(sprintf("Mean RMSE: %.4f\n", sem_mean_rmse))
cat(sprintf("Overall MAE: %.4f\n", cv_results$sem_results$overall_mae))
cat(sprintf("Overall MSE: %.4f\n", cv_results$sem_results$overall_mse))
cat(sprintf("Overall RMSE: %.4f\n", cv_results$sem_results$overall_rmse))

# Compare the models
cat("\n==== Model Comparison ====\n")
comparison_df <- data.frame(
  Model = c("CARBayes", "SEM Poisson"),
  Mean_MAE = c(carbayes_mean_mae, sem_mean_mae),
  Mean_MSE = c(carbayes_mean_mse, sem_mean_mse),
  Mean_RMSE = c(carbayes_mean_rmse, sem_mean_rmse),
  Overall_MAE = c(cv_results$carbayes_results$overall_mae, cv_results$sem_results$overall_mae),
  Overall_MSE = c(cv_results$carbayes_results$overall_mse, cv_results$sem_results$overall_mse),
  Overall_RMSE = c(cv_results$carbayes_results$overall_rmse, cv_results$sem_results$overall_rmse)
)

print(comparison_df)

# After running cross-validation, generate and display the figures
cat("\nGenerating figures from cross-validation results...\n")
figures <- generate_cv_figures(cv_results, full_data)

# Save the figures
ggsave("predicted_accidents_map.png", figures$map_plot, width = 10, height = 8)
ggsave("predicted_vs_actual.png", figures$scatter_plot, width = 8, height = 6)

cat("\nFigures saved as 'predicted_accidents_map.png' and 'predicted_vs_actual.png'\n")

# Display summary statistics of predictions
cat("\nSummary of predictions:\n")
print(summary(figures$predictions$predicted))
print(summary(figures$predictions$actual))

# Calculate additional metrics
cat("\nAdditional metrics:\n")
cat(sprintf("Correlation: %.4f\n", cor(figures$predictions$actual, figures$predictions$predicted)))
cat(sprintf("R-squared: %.4f\n", cor(figures$predictions$actual, figures$predictions$predicted)^2))

# Fit CARBayes model on the whole dataset
cat("\nFitting CARBayes model on the whole dataset...\n")

# Use the most common features from cross-validation
top_features <- cv_results$feature_consistency$Feature[cv_results$feature_consistency$Percentage >= 60]
if (length(top_features) < 5) {
  top_features <- cv_results$feature_consistency$Feature[1:min(5, nrow(cv_results$feature_consistency))]
}

final_formula <- paste("acc ~", paste(top_features, collapse = " + "))
cat(sprintf("Using formula: %s\n", final_formula))

final_carbayes <- carbayes_model$new(
  data = full_data,
  formula = final_formula,
  random_effect = "borough",
  exposure = NULL,
  spatial_vars = c('x', 'y')
)

final_results <- final_carbayes$fit()

# Calculate model fit statistics
cat("\nModel fit statistics:\n")
cat(sprintf("DIC: %.4f\n", final_results$DIC))
cat(sprintf("pD: %.4f\n", final_results$pd))

# Calculate predictions on the full dataset
final_predictions <- final_carbayes$predict(full_data)
final_mae <- mean(abs(full_data$acc - final_predictions), na.rm = TRUE)
final_rmse <- sqrt(mean((full_data$acc - final_predictions)^2, na.rm = TRUE))
final_correlation <- cor(full_data$acc, final_predictions)

cat("\nFinal model performance on full dataset:\n")
cat(sprintf("MAE: %.4f\n", final_mae))
cat(sprintf("RMSE: %.4f\n", final_rmse))
cat(sprintf("Correlation: %.4f\n", final_correlation))
cat(sprintf("R-squared: %.4f\n", final_correlation^2))