install.packages("INLA",repos=c(getOption("repos"),INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE)

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
  
  # Add spatial index needed for INLA
  full_data$idx_spatial <- 1:nrow(full_data)
  
  # Ensure borough is a factor for INLA and CARBayes random effect
  if ("borough" %in% colnames(full_data)) {
    full_data$borough <- as.factor(full_data$borough)
  }
  
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
  is_symmetric <- all(abs(W_symmetric - t(W_symmetric)) < 1e-10)
  cat("Spatial weights matrix symmetry check:", ifelse(is_symmetric, "PASSED", "FAILED"), "\n")
  
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
  
  cat("Selected", length(selected_features), "features\n")
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

# INLA model cross-validation function
run_cross_validation_inla <- function(data, selected_features, k = 5, random_effect_col = "borough", spatial_coords = c('x', 'y')) {
  set.seed(42)

  # Create fold indices
  folds <- createFolds(data$acc, k = k, returnTrain = TRUE)

  # Initialize lists to store results
  fold_results <- list()
  test_actual <- list()
  test_predicted <- list()

  # Create spatial weights matrix for the full dataset
  coords <- as.matrix(data[, spatial_coords])
  W_full <- create_spatial_weights(data.frame(x=coords[,1], y=coords[,2]), k_neighbors=3)
  W_sparse_full <- Matrix(W_full, sparse = TRUE)

  # Define priors
  pc_prec_unstructured <- list(prior = 'pc.prec', param = c(1, 0.01))
  pc_phi <- list(prior = 'pc', param = c(0.5, 0.5))
  pc_prec_borough <- list(prior = 'pc.prec', param = c(1, 0.01))
  nb_family_control <- list(hyper = list(theta = list(prior = "loggamma", param = c(1, 0.01))))
  
  # Define initial values for control.mode
  initial_theta_values <- c(log(5), 0, 0, 0) 
  inla_control_mode <- list(theta = initial_theta_values, restart = TRUE)

  # For each fold
  for (i in 1:k) {
    cat("\n========== INLA Fold", i, "/", k, "==========\n")

    # Split data into train and test
    train_indices <- folds[[i]]
    test_indices <- setdiff(1:nrow(data), train_indices)

    # Create response variable for this fold (NA for test set)
    y_cv <- data$acc
    y_cv[test_indices] <- NA

    # Create a temporary data copy for this fold's specific response
    data_cv <- data
    data_cv$y_cv <- y_cv
    
    # Ensure idx_spatial is not in selected_features for the formula
    features_for_formula <- setdiff(selected_features, "idx_spatial")
    if (length(features_for_formula) == 0) {
        formula_str <- "y_cv ~ 1"
    } else {
        formula_str <- paste("y_cv ~", paste(features_for_formula, collapse = " + "))
    }

    # Prepare INLA formula
    formula_str <- paste(
        formula_str,
        "+ f(idx_spatial, model = 'bym2', graph = W_sparse_full, scale.model = TRUE, constr = TRUE, hyper = list(prec = pc_prec_unstructured, phi = pc_phi))",
        "+ f(", random_effect_col, ", model = 'iid', hyper = list(prec = pc_prec_borough))"
    )
    inla_formula <- as.formula(formula_str)

    cat("Fitting INLA model for fold", i, "...\n")
    
    # Fit INLA model
    inla_model <- try(inla(inla_formula,
                           data = data_cv,
                           family = "nbinomial",
                           control.predictor = list(compute = TRUE, link = 1),
                           control.compute = list(dic = TRUE, waic = TRUE, config = TRUE),
                           control.family = nb_family_control,
                           control.mode = inla_control_mode),
                      silent = TRUE)

    if (inherits(inla_model, "try-error")) {
        cat("ERROR fitting INLA model for fold", i, ":", conditionMessage(attr(inla_model, "condition")), "\n")
        inla_train_mae <- NA
        inla_train_rmse <- NA
        inla_test_mae <- NA
        inla_test_rmse <- NA
        inla_test_preds <- rep(NA, length(test_indices))
    } else {
        # Extract predictions
        predicted_mean <- inla_model$summary.fitted.values$mean
        
        # Separate train and test predictions
        inla_train_preds <- predicted_mean[train_indices]
        inla_test_preds <- predicted_mean[test_indices]
        
        # Calculate metrics for INLA
        inla_train_mae <- mean(abs(data$acc[train_indices] - inla_train_preds), na.rm = TRUE)
        inla_train_mse <- mean((data$acc[train_indices] - inla_train_preds)^2, na.rm = TRUE)
        inla_train_rmse <- sqrt(inla_train_mse)
        
        inla_test_mae <- mean(abs(data$acc[test_indices] - inla_test_preds), na.rm = TRUE)
        inla_test_mse <- mean((data$acc[test_indices] - inla_test_preds)^2, na.rm = TRUE)
        inla_test_rmse <- sqrt(inla_test_mse)
    }

    # Store results for all three models
    fold_results[[i]] <- list(
      fold = i,
      train_mae = inla_train_mae,
      train_rmse = inla_train_rmse,
      test_mae = inla_test_mae,
      test_rmse = inla_test_rmse,
      formula = formula_str
    )
    
    # Collect predictions and actual values for overall metrics
    test_actual[[i]] <- data$acc[test_indices]
    test_predicted[[i]] <- inla_test_preds
  }

  # Combine all predictions and calculate overall metrics
  test_actual_combined <- unlist(test_actual)
  test_predicted_combined <- unlist(test_predicted)
  
  # Calculate overall metrics
  overall_mae <- mean(abs(test_actual_combined - test_predicted_combined), na.rm = TRUE)
  overall_rmse <- sqrt(mean((test_actual_combined - test_predicted_combined)^2, na.rm = TRUE))

  cat("\n========== Overall INLA CV Results ==========\n")
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

# Combined cross-validation function for all three models
run_combined_cross_validation <- function(data, response_var = "acc", k = 5, random_effect = "borough", max_features = 30) {
  set.seed(42)
  
  # Create folds using caret - same folds for all models
  folds <- createFolds(data$acc, k = k, returnTrain = TRUE)
  
  # Initialize results for all models
  carbayes_fold_results <- list()
  sem_fold_results <- list()
  inla_fold_results <- list()
  
  carbayes_test_actual <- list()
  carbayes_test_predicted <- list()
  sem_test_actual <- list()
  sem_test_predicted <- list()
  inla_test_actual <- list()
  inla_test_predicted <- list()
  
  # Global spatial weights for SEM model
  sp_data <- data
  coordinates(sp_data) <- ~ x + y
  w_global <- make_spatial_weights_for_sem(sp_data, k = 3)
  
  # Full dataset spatial weights for INLA
  coords <- as.matrix(data[, c('x', 'y')])
  W_full <- create_spatial_weights(data.frame(x=coords[,1], y=coords[,2]), k_neighbors=3)
  W_sparse_full <- Matrix(W_full, sparse = TRUE)
  
  # Define INLA priors
  pc_prec_unstructured <- list(prior = 'pc.prec', param = c(1, 0.01))
  pc_phi <- list(prior = 'pc', param = c(0.5, 0.5))
  pc_prec_borough <- list(prior = 'pc.prec', param = c(1, 0.01))
  nb_family_control <- list(hyper = list(theta = list(prior = "loggamma", param = c(1, 0.01))))
  
  # Define initial values for INLA control.mode
  initial_theta_values <- c(log(5), 0, 0, 0) 
  inla_control_mode <- list(theta = initial_theta_values, restart = TRUE)
  
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
    
    # Feature selection using LASSO
    exclude_cols <- c('int_no', 'acc', 'pi', 'borough', 'x', 'y', 'idx_spatial')
    feature_cols <- setdiff(colnames(train_data), exclude_cols)
    X_train <- train_data[, feature_cols]
    y_train <- train_data[[response_var]]
    
    selected_features <- select_features_with_lasso(X_train, y_train, max_features)
    
    # Create formula for all models
    formula_str <- paste(response_var, "~", paste(selected_features, collapse = " + "))
    feature_formulas[[i]] <- formula_str
    formula <- as.formula(formula_str)
    
    #=============== CARBayes Model ================
    cat("\n--- Training CARBayes model for fold", i, "---\n")
    
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
    
    #=============== SEM Poisson Model ================
    cat("\n--- Training SEM Poisson model for fold", i, "---\n")
    
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
    
    #=============== INLA Model ================
    cat("\n--- Training INLA model for fold", i, "---\n")
    
    # Create response variable with NAs for test set
    y_cv <- data$acc
    y_cv[test_idx] <- NA
    
    # Create a temporary data copy for this fold's specific response
    data_cv <- data
    data_cv$y_cv <- y_cv
    
    # Ensure idx_spatial is not in selected_features for the formula
    features_for_formula <- setdiff(selected_features, "idx_spatial")
    if (length(features_for_formula) == 0) {
        inla_formula_str <- "y_cv ~ 1"
    } else {
        inla_formula_str <- paste("y_cv ~", paste(features_for_formula, collapse = " + "))
    }
    
    # Prepare INLA formula
    inla_formula_str <- paste(
        inla_formula_str,
        "+ f(idx_spatial, model = 'bym2', graph = W_sparse_full, scale.model = TRUE, constr = TRUE, hyper = list(prec = pc_prec_unstructured, phi = pc_phi))",
        "+ f(", random_effect, ", model = 'iid', hyper = list(prec = pc_prec_borough))"
    )
    inla_formula <- as.formula(inla_formula_str)
    
    # Fit INLA model
    inla_model <- try(inla(inla_formula,
                           data = data_cv,
                           family = "nbinomial",
                           control.predictor = list(compute = TRUE, link = 1),
                           control.compute = list(dic = TRUE, waic = TRUE, config = TRUE),
                           control.family = nb_family_control,
                           control.mode = inla_control_mode),
                      silent = TRUE)
    
    if (inherits(inla_model, "try-error")) {
        cat("ERROR fitting INLA model for fold", i, ":", conditionMessage(attr(inla_model, "condition")), "\n")
        inla_train_mae <- NA
        inla_train_rmse <- NA
        inla_test_mae <- NA
        inla_test_rmse <- NA
        inla_test_preds <- rep(NA, length(test_idx))
    } else {
        # Extract predictions
        predicted_mean <- inla_model$summary.fitted.values$mean
        
        # Separate train and test predictions
        inla_train_preds <- predicted_mean[train_idx]
        inla_test_preds <- predicted_mean[test_idx]
        
        # Calculate metrics for INLA
        inla_train_mae <- mean(abs(train_data$acc - inla_train_preds), na.rm = TRUE)
        inla_train_mse <- mean((train_data$acc - inla_train_preds)^2, na.rm = TRUE)
        inla_train_rmse <- sqrt(inla_train_mse)
        
        inla_test_mae <- mean(abs(test_data$acc - inla_test_preds), na.rm = TRUE)
        inla_test_mse <- mean((test_data$acc - inla_test_preds)^2, na.rm = TRUE)
        inla_test_rmse <- sqrt(inla_test_mse)
    }
    
    # Store results for all three models
    carbayes_fold_results[[i]] <- list(
      fold = i,
      train_mae = carbayes_train_mae,
      train_rmse = carbayes_train_rmse,
      test_mae = carbayes_test_mae,
      test_rmse = carbayes_test_rmse,
      formula = formula_str
    )
    
    sem_fold_results[[i]] <- list(
      fold = i,
      test_mae = sem_test_mae,
      test_rmse = sem_test_rmse,
      formula = formula_str
    )
    
    inla_fold_results[[i]] <- list(
      fold = i,
      train_mae = inla_train_mae,
      train_rmse = inla_train_rmse,
      test_mae = inla_test_mae,
      test_rmse = inla_test_rmse,
      formula = formula_str
    )
    
    # Collect predictions and actual values for overall metrics
    carbayes_test_actual[[i]] <- test_data$acc
    carbayes_test_predicted[[i]] <- carbayes_test_preds
    sem_test_actual[[i]] <- test_data$acc
    sem_test_predicted[[i]] <- sem_test_preds
    inla_test_actual[[i]] <- test_data$acc
    inla_test_predicted[[i]] <- inla_test_preds
    
    cat(sprintf("\nFold %d Results:\n", i))
    cat("CARBayes - Train MAE:", format(carbayes_train_mae, digits=4), "RMSE:", format(carbayes_train_rmse, digits=4), "\n")
    cat("CARBayes - Test MAE:", format(carbayes_test_mae, digits=4), "RMSE:", format(carbayes_test_rmse, digits=4), "\n")
    cat("SEM - Test MAE:", format(sem_test_mae, digits=4), "RMSE:", format(sem_test_rmse, digits=4), "\n")
    cat("INLA - Train MAE:", format(inla_train_mae, digits=4), "RMSE:", format(inla_train_rmse, digits=4), "\n")
    cat("INLA - Test MAE:", format(inla_test_mae, digits=4), "RMSE:", format(inla_test_rmse, digits=4), "\n")
  }
  
  # Combine all predictions and calculate overall metrics
  carbayes_test_actual_combined <- unlist(carbayes_test_actual)
  carbayes_test_predicted_combined <- unlist(carbayes_test_predicted)
  sem_test_actual_combined <- unlist(sem_test_actual)
  sem_test_predicted_combined <- unlist(sem_test_predicted)
  inla_test_actual_combined <- unlist(inla_test_actual)
  inla_test_predicted_combined <- unlist(inla_test_predicted)
  
  # Calculate overall metrics
  carbayes_overall_mae <- mean(abs(carbayes_test_actual_combined - carbayes_test_predicted_combined), na.rm = TRUE)
  carbayes_overall_rmse <- sqrt(mean((carbayes_test_actual_combined - carbayes_test_predicted_combined)^2, na.rm = TRUE))
  
  sem_overall_mae <- mean(abs(sem_test_actual_combined - sem_test_predicted_combined), na.rm = TRUE)
  sem_overall_rmse <- sqrt(mean((sem_test_actual_combined - sem_test_predicted_combined)^2, na.rm = TRUE))
  
  inla_overall_mae <- mean(abs(inla_test_actual_combined - inla_test_predicted_combined), na.rm = TRUE)
  inla_overall_rmse <- sqrt(mean((inla_test_actual_combined - inla_test_predicted_combined)^2, na.rm = TRUE))
  
  # Print overall results
  cat("\n========== Overall CV Results ==========\n")
  cat("CARBayes - Overall MAE:", format(carbayes_overall_mae, digits=4), "RMSE:", format(carbayes_overall_rmse, digits=4), "\n")
  cat("SEM - Overall MAE:", format(sem_overall_mae, digits=4), "RMSE:", format(sem_overall_rmse, digits=4), "\n")
  cat("INLA - Overall MAE:", format(inla_overall_mae, digits=4), "RMSE:", format(inla_overall_rmse, digits=4), "\n")
  
  # Return comprehensive results
  return(list(
    carbayes = list(
      fold_results = carbayes_fold_results,
      overall_mae = carbayes_overall_mae,
      overall_rmse = carbayes_overall_rmse,
      test_actual = carbayes_test_actual_combined,
      test_predicted = carbayes_test_predicted_combined
    ),
    sem = list(
      fold_results = sem_fold_results,
      overall_mae = sem_overall_mae,
      overall_rmse = sem_overall_rmse,
      test_actual = sem_test_actual_combined,
      test_predicted = sem_test_predicted_combined
    ),
    inla = list(
      fold_results = inla_fold_results,
      overall_mae = inla_overall_mae,
      overall_rmse = inla_overall_rmse,
      test_actual = inla_test_actual_combined,
      test_predicted = inla_test_predicted_combined
    ),
    feature_formulas = feature_formulas
  ))
}