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
        burnin = 2000,
        n.sample = 5000,
        thin = 2,
        rho = 0.5,
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

run_cross_validation <- function(data, response_var = "acc", k = 5, random_effect = NULL, max_features = 30) {
  set.seed(42)
  
  folds <- createFolds(data$acc, k = k, returnTrain = TRUE)
  
  fold_results <- list()
  test_actual <- list()
  test_predicted <- list()
  
  for (i in 1:k) {
    train_data <- data[folds[[i]], ]
    test_data <- data[-folds[[i]], ]
    
    exclude_cols <- c('int_no', 'acc', 'pi', 'borough', 'x', 'y')
    feature_cols <- setdiff(colnames(train_data), exclude_cols)
    X_train <- train_data[, feature_cols]
    y_train <- train_data[[response_var]]
    
    selected_features <- select_features_with_lasso(X_train, y_train, max_features)
    
    formula <- paste(response_var, "~", paste(selected_features, collapse = " + "))
    
    model <- SpatialMixedNegativeBinomial$new(
      data = train_data,
      formula = formula,
      random_effect = random_effect,
      exposure = NULL,
      spatial_vars = c('x', 'y')
    )
    
    results <- model$fit()
    
    train_preds <- model$predict(train_data)
    test_preds <- model$predict(test_data)
    
    if (all(is.na(train_preds))) {
      train_preds <- rep(mean(train_data$acc), nrow(train_data))
    }
    
    if (all(is.na(test_preds))) {
      test_preds <- rep(mean(train_data$acc), nrow(test_data))
    }
    
    train_mae <- mean(abs(train_data$acc - train_preds), na.rm = TRUE)
    train_mse <- mean((train_data$acc - train_preds)^2, na.rm = TRUE)
    train_rmse <- sqrt(train_mse)
    
    test_mae <- mean(abs(test_data$acc - test_preds), na.rm = TRUE)
    test_mse <- mean((test_data$acc - test_preds)^2, na.rm = TRUE)
    test_rmse <- sqrt(test_mse)
    
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
    
    test_actual[[i]] <- test_data$acc
    test_predicted[[i]] <- test_preds
  }
  
  test_actual_combined <- unlist(test_actual)
  test_predicted_combined <- unlist(test_predicted)
  
  overall_mae <- mean(abs(test_actual_combined - test_predicted_combined), na.rm = TRUE)
  overall_mse <- mean((test_actual_combined - test_predicted_combined)^2, na.rm = TRUE)
  overall_rmse <- sqrt(overall_mse)
  
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

select_and_visualize_features <- function(data, response_var = "acc", max_features = 20) {
  y <- data[[response_var]]
  
  exclude_cols <- c('int_no', 'acc', 'pi', 'borough', 'x', 'y')
  feature_cols <- setdiff(colnames(data), exclude_cols)
  X <- data[, feature_cols]
  
  selected_features <- select_features_with_lasso(X, y, max_features)
  
  formula_str <- paste(response_var, "~", paste(selected_features, collapse = " + "))
  
  X_matrix <- as.matrix(X)
  X_scaled <- scale(X_matrix)
  lasso_cv <- cv.glmnet(X_scaled, y, alpha = 1, family = "poisson")
  lasso_model <- glmnet(X_scaled, y, alpha = 1, lambda = lasso_cv$lambda.min)
  coef_matrix <- as.matrix(coef(lasso_model))
  
  coef_df <- data.frame(
    Feature = rownames(coef_matrix)[-1],
    Coefficient = coef_matrix[-1, 1]
  )
  coef_df <- coef_df[coef_df$Coefficient != 0, ]
  coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), ]
  coef_df$Feature <- factor(coef_df$Feature, levels = coef_df$Feature[order(abs(coef_df$Coefficient))])
  
  return(list(
    selected_features = selected_features,
    formula = formula_str
  ))
}

full_data <- load_and_prepare_data()

cv_results <- run_cross_validation(
  data = full_data, 
  response_var = "acc",
  k = 5,
  random_effect = "borough",
  max_features = 30
)

# Extract and display metrics for each fold
cat("Metrics by fold:\n")
cat("---------------\n")
for (i in 1:length(cv_results$fold_results)) {
  fold <- cv_results$fold_results[[i]]
  cat(sprintf("Fold %d: Test MAE = %.4f, Test MSE = %.4f, Test RMSE = %.4f\n", 
              i, fold$test_mae, fold$test_mse, fold$test_rmse))
}

# Calculate mean performance across folds
mean_mae <- mean(sapply(cv_results$fold_results, function(x) x$test_mae))
mean_mse <- mean(sapply(cv_results$fold_results, function(x) x$test_mse))
mean_rmse <- mean(sapply(cv_results$fold_results, function(x) x$test_rmse))

cat("\nMean performance across folds:\n")
cat("---------------------------\n")
cat(sprintf("Mean MAE: %.4f\n", mean_mae))
cat(sprintf("Mean MSE: %.4f\n", mean_mse))
cat(sprintf("Mean RMSE: %.4f\n", mean_rmse))

cat("\nOverall performance (combined predictions):\n")
cat("----------------------------------------\n")
cat(sprintf("Overall MAE: %.4f\n", cv_results$overall_mae))
cat(sprintf("Overall MSE: %.4f\n", cv_results$overall_mse))
cat(sprintf("Overall RMSE: %.4f\n", cv_results$overall_rmse))