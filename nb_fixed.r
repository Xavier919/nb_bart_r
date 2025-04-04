# Install missing packages if needed
if (!requireNamespace("blockCV", quietly = TRUE)) {
  install.packages("blockCV")
}
if (!requireNamespace("gglasso", quietly = TRUE)) {
  install.packages("gglasso")
}
if (!requireNamespace("sp", quietly = TRUE)) {
  install.packages("sp")
}

# Load required packages
cat("Loading packages...\n")
library(dplyr)
library(ggplot2)
library(Matrix)
library(FNN)
library(caret)
library(spdep)
library(glmnet)
library(R6)
library(CARBayes)
library(MASS)
library(blockCV)
library(gglasso)
library(sp)
library(conflicted)

# Resolve conflicts
conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

#####################################
# Enhanced Data Processing
#####################################

load_and_prepare_data <- function() {
  cat("Loading and preprocessing data...\n")
  data <- read.csv("data/data_final.csv", sep = ";")
  
  # Handle missing values
  data$ln_distdt[is.na(data$ln_distdt)] <- 0
  
  # Filter and clean
  data <- data %>%
    filter(pi != 0) %>%
    mutate(
      across(where(is.character), 
      ~iconv(., "UTF-8", "UTF-8", sub = ""))  # Fix encoding
    )
  
  # Spatial feature engineering
  data <- data %>%
    mutate(
      borough = case_when(
        borough %in% c('Kirkland','Beaconsfield','Pierrefonds-Roxboro',
                       'Dollard-des-Ormeaux','Dorval') ~ 'Zone ouest',
        borough %in% c('Rivière-des-Prairies-Pointe-aux-Trembles',
                       'Montréal-Est','Anjou') ~ 'Zone est',
        borough %in% c('Outremont','Mont-Royal') ~ 'Zone centre',
        borough %in% c('Sud-Ouest','Côte-Saint-Luc','Verdun',
                       'Lasalle','Lachine') ~ 'Zone sud',
        TRUE ~ borough
      )
    )
  
  # Feature transformations
  spatial_vars <- c('x', 'y')
  quad_vars <- c('ln_cti', 'ln_cri', 'ln_cli', 'ln_pi', 'ln_fri', 'ln_fli', 'ln_fi')
  
  data <- data %>%
    mutate(across(all_of(quad_vars), list(squared = ~.^2)))
  
  # Final feature selection
  keep_cols <- c('int_no', 'acc', 'pi', 'borough', spatial_vars,
                grep("ln_|_squared|total_lane|avg_crossw|commercial|ped_countd", 
                     names(data), value = TRUE))
  
  return(data[, keep_cols])
}

#####################################
# Corrected Spatial Weights Matrix
#####################################

create_spatial_weights <- function(coords, k_neighbors = 10) {
  cat("Creating proper symmetric spatial weights matrix...\n")
  
  # Get k+1 neighbors to exclude self-neighbor
  nn <- FNN::get.knn(coords, k = k_neighbors)
  
  # Initialize binary adjacency matrix
  n <- nrow(coords)
  W <- matrix(0, n, n)
  
  # Create mutual nearest neighbor matrix
  for(i in 1:n) {
    # Mark k nearest neighbors
    W[i, nn$nn.index[i, ]] <- 1
  }
  
  # Ensure perfect symmetry
  W_symmetric <- 1 * ((W + t(W)) > 0)
  diag(W_symmetric) <- 0  # Remove self-neighbors
  
  # Verify symmetry
  is_symmetric <- all(W_symmetric == t(W_symmetric))
  cat("Weights matrix symmetry check:", is_symmetric, "\n")
  
  return(W_symmetric)
}

#####################################
# Enhanced Spatial Model Class
#####################################

SpatialCARModel <- R6Class("SpatialCARModel",
  public = list(
    data = NULL,
    formula = NULL,
    W = NULL,
    model = NULL,
    spatial_vars = c('x', 'y'),
    
    initialize = function(data, formula) {
      self$data <- data
      self$formula <- formula
      coords <- as.matrix(data[, self$spatial_vars])
      self$W <- create_spatial_weights(coords)
      
      # Additional check for CARBayes requirements
      if(!spdep::is.symmetric.nb(spdep::mat2listw(self$W)$neighbours)) {
        stop("Final weight matrix failed symmetry check")
      }
    },
    
    fit = function() {
      cat("Fitting spatial CAR model...\n")
      
      # Add matrix validation
      if(!all(self$W == t(self$W))) {
        stop("Weight matrix must be symmetric for CAR models")
      }
      
      # Check overdispersion
      y <- model.response(model.frame(self$formula, self$data))
      dispersion_ratio <- var(y)/mean(y)
      family <- if(dispersion_ratio > 1.2) "poisson" else "poisson"  # Fixed threshold
      cat("Using family:", family, "\n")
      
      # Fit model
      self$model <- S.CARleroux(
        formula = self$formula,
        family = family,
        data = self$data,
        W = self$W,
        burnin = 500,      # Increased from 2000
        n.sample = 1000,   # Increased from 3000
        thin = 2,          # Increased from 5
        verbose = TRUE
      )
      
      # Convergence checks
      if(!all(self$model$samples$rho < 1.1 & self$model$samples$rho > -1.1)) {
        warning("Potential convergence issues detected in rho parameter")
      }
      
      return(self)
    },
    
    predict = function(newdata = NULL) {
      if(is.null(newdata)) newdata <- self$data
      
      # Get design matrix
      X <- model.matrix(update(self$formula, NULL ~ .), newdata)
      
      # Posterior mean coefficients
      beta <- colMeans(self$model$samples$beta)
      
      # Spatial effects
      if("phi" %in% names(self$model$samples)) {
        phi <- colMeans(self$model$samples$phi)
      } else {
        phi <- rep(0, nrow(newdata))
      }
      
      # Prediction
      eta <- X %*% beta + phi
      return(exp(eta))
    },
    
    evaluate = function(test_data) {
      preds <- self$predict(test_data)
      actual <- test_data$acc
      
      # Add small epsilon to zero values
      actual_adj <- ifelse(actual == 0, 0.001, actual)
      
      metrics <- list(
        MAE = mean(abs(preds - actual)),
        RMSE = sqrt(mean((preds - actual)^2)),
        MSPE = mean(((preds - actual)/actual_adj)^2)  # Modified
      )
      
      return(metrics)
    },
    
    plot_trace = function(param = "rho") {
      if(!param %in% names(self$model$samples)) {
        stop("Parameter not found in model samples")
      }
      
      plot(self$model$samples[[param]], type = 'l',
           main = paste("Trace plot for", param),
           xlab = "Iteration", ylab = param)
    },
    
    diagnostics = function() {
      # Effective sample size check
      cat("\nEffective sample sizes:\n")
      print(coda::effectiveSize(self$model$samples))
      
      # R-hat statistics
      cat("\nGelman-Rubin diagnostics:\n")
      print(coda::gelman.diag(self$model$samples))
      
      # Spatial autocorrelation residuals
      residuals <- self$data$acc - self$predict()
      moran_test <- moran.mc(residuals, mat2listw(self$W), 999)
      cat("\nResidual spatial autocorrelation (Moran's I):", moran_test$statistic, "p =", moran_test$p.value, "\n")
      
      # Posterior predictive check
      ppc <- function(model) {
        y_rep <- apply(model$samples$fitted, 2, mean)
        plot(density(y_rep), col="red", main="Posterior Predictive Check",
             xlab="Count", ylab="Density")
        lines(density(model$data$acc), col="blue")
        legend("topright", legend=c("Predicted", "Observed"), 
               col=c("red", "blue"), lty=1)
      }
      
      ppc(self$model)
    }
  )
)

#####################################
# Spatial Feature Selection
#####################################

select_spatial_features <- function(data, response = "acc", max_features = 20) {
  # Create grouped features for hierarchy
  base_vars <- c('ln_cti', 'ln_cri', 'ln_cli', 'ln_pi', 'ln_fri', 'ln_fli', 'ln_fi')
  groups <- rep(1:length(base_vars), each = 2)
  features <- c(base_vars, paste0(base_vars, "_squared"))
  
  X <- as.matrix(data[, features])
  y <- data[[response]]
  
  # Group lasso with hierarchy preservation
  cv_fit <- cv.gglasso(X, y, group = groups, pred.loss = "L2")
  selected <- which(coef(cv_fit, s = "lambda.min")[-1] != 0)
  
  # Enforce hierarchy - if squared term is selected, include linear term
  base_selected <- selected[selected <= length(base_vars)]
  squared_selected <- selected[selected > length(base_vars)]
  
  # For each squared term, ensure its linear counterpart is included
  for(sq_idx in squared_selected) {
    linear_idx <- sq_idx - length(base_vars)
    if(!(linear_idx %in% base_selected)) {
      base_selected <- c(base_selected, linear_idx)
    }
  }
  
  # Combine all selected features
  all_selected <- c(base_selected, squared_selected)
  
  # Get final features
  final_features <- features[all_selected]
  if(length(final_features) > max_features) {
    final_features <- final_features[1:max_features]
  }
  
  # Add error handling for VIF check
  vif_check <- function(df) {
    # Check if we have enough features for VIF
    if(length(final_features) < 2) {
      return(final_features)
    }
    
    # Try-catch to handle potential errors
    tryCatch({
      vif_values <- car::vif(lm(as.formula(paste(response, "~ .")), data = df[, final_features, drop=FALSE]))
      while(any(vif_values > 5)) {
        worst <- which.max(vif_values)
        final_features <- final_features[-worst]
        if(length(final_features) < 2) break
        vif_values <- car::vif(lm(as.formula(paste(response, "~ .")), data = df[, final_features, drop=FALSE]))
      }
      return(final_features)
    }, error = function(e) {
      warning("VIF check failed: ", e$message, ". Using selected features without VIF filtering.")
      return(final_features)
    })
  }
  
  final_features <- vif_check(data)
  
  # Create formula
  formula_str <- paste(response, "~", paste(final_features, collapse = " + "))
  
  return(list(
    features = final_features,
    formula = as.formula(formula_str)
  ))
}

#####################################
# Spatial Cross-Validation
#####################################

spatial_crossval <- function(data, formula, n_folds = 5) {
  # Spatial blocking
  coordinates <- data[, c("x", "y")]
  sp_points <- sp::SpatialPoints(coordinates)
  
  sb <- cv_spatial(
    x = sp_points,
    size = 2000,  # 2km spatial blocks (increased from 1000)
    k = n_folds,
    selection = "systematic",
    iteration = 500  # More attempts to balance folds
  )
  
  metrics <- list()
  predictions <- data.frame()
  top_predictions <- list()
  
  for(fold in 1:n_folds) {
    cat("\nProcessing fold", fold, "/", n_folds, "\n")
    
    # Train/test split
    test_ids <- sb$folds_ids[[fold]]
    train <- data[-test_ids, ]
    test <- data[test_ids, ]
    
    # Model training
    model <- SpatialCARModel$new(train, formula)$fit()
    
    # Get predictions
    test_preds <- model$predict(test)
    
    # Store predictions
    preds <- data.frame(
      actual = test$acc,
      predicted = test_preds,
      fold = fold
    )
    predictions <- rbind(predictions, preds)
    
    # Find top 10 highest actual counts and their predictions
    top_indices <- order(test$acc, decreasing = TRUE)[1:10]
    top_data <- data.frame(
      intersection_id = test$int_no[top_indices],
      actual = test$acc[top_indices],
      predicted = test_preds[top_indices]
    )
    top_predictions[[fold]] <- top_data
    
    # Print top 10 for this fold
    cat("\nTop 10 highest actual counts for fold", fold, ":\n")
    print(top_data)
    
    # Calculate metrics
    fold_metrics <- model$evaluate(test)
    metrics[[fold]] <- fold_metrics
    
    cat("Fold metrics:\n")
    print(unlist(fold_metrics))
  }
  
  # Aggregate results
  overall_metrics <- list(
    MAE = mean(sapply(metrics, `[[`, "MAE")),
    RMSE = mean(sapply(metrics, `[[`, "RMSE")),
    MSPE = mean(sapply(metrics, `[[`, "MSPE"))
  )
  
  return(list(
    predictions = predictions,
    fold_metrics = metrics,
    overall_metrics = overall_metrics,
    top_predictions = top_predictions
  ))
}

#####################################
# Simple Fixed Effects Model Class
#####################################

SimpleGLMModel <- R6Class("SimpleGLMModel",
  public = list(
    data = NULL,
    formula = NULL,
    model = NULL,
    
    initialize = function(data, formula) {
      self$data <- data
      self$formula <- formula
      
      # Convert borough to fixed effect if not already in formula
      if("borough" %in% names(data) && !grepl("borough", deparse(formula))) {
        self$formula <- update(formula, . ~ . + borough)
      }
    },
    
    fit = function() {
      cat("Fitting GLM model...\n")
      
      # Check overdispersion
      y <- model.response(model.frame(self$formula, self$data))
      dispersion_ratio <- var(y)/mean(y)
      family_type <- if(dispersion_ratio > 1.2) "negative binomial" else "poisson"
      cat("Using family:", family_type, "\n")
      
      # Fit model using glm or MASS::glm.nb
      if(family_type == "poisson") {
        self$model <- glm(
          self$formula,
          family = poisson(link = "log"),
          data = self$data
        )
      } else {
        self$model <- MASS::glm.nb(
          self$formula,
          data = self$data
        )
      }
      
      return(self)
    },
    
    predict = function(newdata = NULL) {
      if(is.null(newdata)) newdata <- self$data
      
      # Handle missing boroughs in newdata
      if("borough" %in% names(self$data) && "borough" %in% names(newdata)) {
        missing_levels <- setdiff(levels(self$data$borough), levels(newdata$borough))
        if(length(missing_levels) > 0) {
          newdata$borough <- factor(newdata$borough, levels = levels(self$data$borough))
        }
      }
      
      # Get predictions
      preds <- predict(self$model, newdata = newdata, type = "response")
      return(preds)
    },
    
    evaluate = function(test_data) {
      preds <- self$predict(test_data)
      actual <- test_data$acc
      
      # Add small epsilon to zero values
      actual_adj <- ifelse(actual == 0, 0.001, actual)
      
      metrics <- list(
        MAE = mean(abs(preds - actual)),
        RMSE = sqrt(mean((preds - actual)^2)),
        MSPE = mean(((preds - actual)/actual_adj)^2)
      )
      
      return(metrics)
    },
    
    diagnostics = function() {
      # Model summary
      cat("\nModel summary:\n")
      print(summary(self$model))
      
      # Residual plots
      par(mfrow = c(2, 2))
      plot(self$model)
      par(mfrow = c(1, 1))
      
      # Check for overdispersion
      cat("\nOverdispersion check:\n")
      cat("Residual deviance:", self$model$deviance, "\n")
      cat("Residual df:", self$model$df.residual, "\n")
      cat("Ratio:", self$model$deviance/self$model$df.residual, 
          "(>1 indicates overdispersion)\n")
      
      # Borough effects
      if("borough" %in% names(coef(self$model))) {
        cat("\nBorough effects:\n")
        borough_coefs <- coef(self$model)[grep("borough", names(coef(self$model)))]
        print(borough_coefs)
      }
    }
  )
)

#####################################
# Non-Spatial Cross-Validation
#####################################

standard_crossval <- function(data, formula, n_folds = 5) {
  # Create folds
  set.seed(42)
  folds <- createFolds(data$acc, k = n_folds)
  
  metrics <- list()
  predictions <- data.frame()
  top_predictions <- list()
  
  for(fold in 1:n_folds) {
    cat("\nProcessing fold", fold, "/", n_folds, "\n")
    
    # Train/test split
    test_ids <- folds[[fold]]
    train <- data[-test_ids, ]
    test <- data[test_ids, ]
    
    # Model training
    model <- SimpleGLMModel$new(train, formula)$fit()
    
    # Get predictions
    test_preds <- model$predict(test)
    
    # Store predictions
    preds <- data.frame(
      actual = test$acc,
      predicted = test_preds,
      fold = fold
    )
    predictions <- rbind(predictions, preds)
    
    # Find top 10 highest actual counts and their predictions
    top_indices <- order(test$acc, decreasing = TRUE)[1:10]
    top_data <- data.frame(
      intersection_id = test$int_no[top_indices],
      actual = test$acc[top_indices],
      predicted = test_preds[top_indices]
    )
    top_predictions[[fold]] <- top_data
    
    # Print top 10 for this fold
    cat("\nTop 10 highest actual counts for fold", fold, ":\n")
    print(top_data)
    
    # Calculate metrics
    fold_metrics <- model$evaluate(test)
    metrics[[fold]] <- fold_metrics
    
    cat("Fold metrics:\n")
    print(unlist(fold_metrics))
  }
  
  # Aggregate results
  overall_metrics <- list(
    MAE = mean(sapply(metrics, `[[`, "MAE")),
    RMSE = mean(sapply(metrics, `[[`, "RMSE")),
    MSPE = mean(sapply(metrics, `[[`, "MSPE"))
  )
  
  return(list(
    predictions = predictions,
    fold_metrics = metrics,
    overall_metrics = overall_metrics,
    top_predictions = top_predictions
  ))
}

#####################################
# Main Execution
#####################################

# Load and prepare data
full_data <- load_and_prepare_data()

# Check and remove zero-accident intersections
full_data <- full_data %>% filter(acc > 0)

# Ensure borough is a factor
full_data$borough <- as.factor(full_data$borough)

# Check overdispersion
cat("Overdispersion ratio:", var(full_data$acc)/mean(full_data$acc), "\n")

# Feature selection
feature_selection <- select_spatial_features(full_data)
cat("Selected formula:", deparse(feature_selection$formula), "\n")

# Run standard cross-validation
set.seed(42)
cv_results <- standard_crossval(full_data, feature_selection$formula)

# Final model training
final_model <- SimpleGLMModel$new(full_data, feature_selection$formula)$fit()

# Enhanced diagnostics
final_model$diagnostics()

# Print results
cat("\nFinal Model Metrics:\n")
print(final_model$evaluate(full_data))
cat("\nCross-Validation Results:\n")
print(cv_results$overall_metrics)

# Visualize predictions
ggplot(cv_results$predictions, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, color = "red") +
  labs(title = "Actual vs Predicted Values",
       subtitle = "Cross-Validation Results") +
  theme_minimal()

# Visualize borough effects
if(inherits(final_model$model, "glm") || inherits(final_model$model, "negbin")) {
  borough_coefs <- coef(final_model$model)[grep("borough", names(coef(final_model$model)))]
  if(length(borough_coefs) > 0) {
    borough_names <- gsub("borough", "", names(borough_coefs))
    
    borough_df <- data.frame(
      borough = borough_names,
      effect = borough_coefs
    )
    
    ggplot(borough_df, aes(x = reorder(borough, effect), y = effect)) +
      geom_col() +
      coord_flip() +
      labs(title = "Borough Fixed Effects", 
           x = "Borough", 
           y = "Effect Estimate") +
      theme_minimal()
  }
}