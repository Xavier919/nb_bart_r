library(spdep)
library(dplyr)
library(Matrix)
library(FNN)
library(glmnet)
library(R6)
library(CARBayes)
library(conflicted)
library(readr)
library(ggplot2)
library(reshape2)

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
  variances <- apply(X_matrix, 2, var)
  X_matrix_filtered <- X_matrix[, variances > 1e-10]
  
  if (ncol(X_matrix_filtered) == 0) {
      warning("No features with variance found for LASSO.")
      return(character(0))
  }
  
  X_scaled <- scale(X_matrix_filtered)

  set.seed(42)
  lasso_cv <- tryCatch({
      cv.glmnet(X_scaled, y, alpha = 1, family = "poisson")
  }, error = function(e) {
      warning("cv.glmnet failed: ", e$message)
      return(NULL)
  })

  if (is.null(lasso_cv)) {
      return(character(0))
  }

  lasso_model <- glmnet(X_scaled, y, alpha = 1, lambda = lasso_cv$lambda.min)

  coef_matrix <- as.matrix(coef(lasso_model))
  if (nrow(coef_matrix) > 1) {
      selected_features_coef <- coef_matrix[-1, 1, drop = FALSE]
      selected_features_coef <- selected_features_coef[selected_features_coef[, 1] != 0, , drop = FALSE]
      
      if (nrow(selected_features_coef) > 0) {
          selected_features_coef <- selected_features_coef[order(abs(selected_features_coef[, 1]), decreasing = TRUE), , drop = FALSE]
          
          if (nrow(selected_features_coef) > max_features) {
              selected_features_coef <- selected_features_coef[1:max_features, , drop = FALSE]
          }
          selected_feature_names <- rownames(selected_features_coef)
      } else {
          selected_feature_names <- character(0)
      }
  } else {
      selected_feature_names <- character(0)
  }

  return(selected_feature_names)
}

carbayes_model <- R6::R6Class(
  "carbayes_model",
  
  public = list(
    data = NULL,
    formula = NULL,
    random_effect = NULL,
    spatial_vars = NULL,
    model = NULL,
    results = NULL,
    W = NULL,
    data_with_re = NULL,
    
    initialize = function(data = NULL, formula = NULL, random_effect = NULL, 
                          spatial_vars = NULL) {
      self$data <- data
      self$formula <- formula
      self$random_effect <- random_effect
      self$spatial_vars <- spatial_vars
    },
    
    fit = function(burnin = 2000, n_sample = 5000, thin = 2) {
      data_to_use <- self$data

      if (is.null(self$W)) {
        cat("Creating spatial weights matrix...\n")
        self$W <- create_spatial_weights(data_to_use[, self$spatial_vars])
        cat("Spatial weights matrix created.\n")
      }

      formula_parts <- strsplit(self$formula, "~")[[1]]
      response_var <- trimws(formula_parts[1])
      predictors <- trimws(formula_parts[2])

      if (!is.null(self$random_effect) && self$random_effect %in% colnames(data_to_use)) {
        if (!is.factor(data_to_use[[self$random_effect]])) {
          data_to_use[[self$random_effect]] <- as.factor(data_to_use[[self$random_effect]])
        }
        if (!grepl(self$random_effect, predictors)) {
             if (predictors == "1") {
                 predictors <- self$random_effect
             } else {
                 predictors <- paste(predictors, "+", self$random_effect)
             }
        }
      } else if (!is.null(self$random_effect)) {
          warning(paste("Specified random effect '", self$random_effect, "' not found in data. Ignoring."), call. = FALSE)
          self$random_effect <- NULL
      }

      full_formula <- as.formula(paste(response_var, "~", predictors))
      cat("Fitting CARBayes model with formula:\n")
      print(full_formula)
      cat("Using W matrix with dimensions:", dim(self$W), "\n")

      if (!isSymmetric(self$W)) {
          warning("Provided W matrix is not symmetric. Symmetrizing using (W + t(W)) / 2.")
          self$W <- (self$W + t(self$W)) / 2
      }
      
      zero_sum_rows <- which(rowSums(self$W) == 0)
      if(length(zero_sum_rows) > 0) {
          warning(paste("W matrix has", length(zero_sum_rows), "rows/columns with zero sums (isolates). This might cause issues in CARBayes."), call. = FALSE)
      }

      model <- tryCatch({
          S.CARleroux(
            formula = full_formula,
            family = "poisson",
            data = data_to_use,
            W = self$W,
            burnin = burnin,
            n.sample = n_sample,
            thin = thin,
            verbose = TRUE
          )
      }, error = function(e) {
          cat("Error during S.CARleroux execution: ", e$message, "\n")
          return(NULL)
      })

      if (is.null(model)) {
          stop("CARBayes model fitting failed.")
      }

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
          stop("Original data used for fitting is not available, and no newdata provided.")
        }
      }

      if (!is.null(self$random_effect) && self$random_effect %in% colnames(newdata)) {
          if (!is.factor(newdata[[self$random_effect]])) {
              newdata[[self$random_effect]] <- factor(newdata[[self$random_effect]])
          }
          original_levels <- levels(self$data_with_re[[self$random_effect]])
          if (!all(levels(newdata[[self$random_effect]]) %in% original_levels)) {
              warning("New data contains levels for the random effect not present during training. Predictions for these levels might be unreliable or based only on the intercept/fixed effects.")
          }
          newdata[[self$random_effect]] <- factor(newdata[[self$random_effect]], levels = original_levels)
      }

      formula_parts <- strsplit(self$formula, "~")[[1]]
      predictors <- trimws(formula_parts[2])
      if (!is.null(self$random_effect) && self$random_effect %in% colnames(newdata)) {
          if (!grepl(self$random_effect, predictors)) {
               if (predictors == "1") {
                   predictors <- self$random_effect
               } else {
                   predictors <- paste(predictors, "+", self$random_effect)
               }
          }
      }
      X_formula <- as.formula(paste("~", predictors))

      X <- tryCatch({
          model.matrix(X_formula, data = newdata)
      }, error = function(e) {
          warning("Could not create model matrix for prediction: ", e$message)
          simple_vars <- all.vars(X_formula)
          simple_vars <- simple_vars[simple_vars %in% names(newdata)]
          if (length(simple_vars) > 0) {
              simple_formula_str <- paste("~", paste(simple_vars, collapse = " + "))
              warning("Retrying model.matrix with simplified formula: ", simple_formula_str)
              tryCatch({
                  model.matrix(as.formula(simple_formula_str), data = newdata)
              }, error = function(e2) {
                  stop("Failed to create model matrix even with simplified formula: ", e2$message)
              })
          } else {
              stop("No valid predictors found for model matrix creation.")
          }
      })

      beta <- self$results$samples$beta
      beta_mean <- apply(beta, 2, mean)

      if (length(beta_mean) != ncol(X)) {
        warning(paste("Mismatch between number of coefficients (", length(beta_mean),
                      ") and columns in prediction matrix (", ncol(X), "). Attempting to align.", sep=""))
        
        coef_names <- names(beta_mean)
        mat_names <- colnames(X)
        
        common_names <- intersect(coef_names, mat_names)
        
        if(length(common_names) > 0) {
            beta_mean_aligned <- beta_mean[common_names]
            X_aligned <- X[, common_names, drop = FALSE]
            
            if ("(Intercept)" %in% coef_names && !("(Intercept)" %in% common_names) && "(Intercept)" %in% mat_names) {
                
            } else if (!("(Intercept)" %in% coef_names) && "(Intercept)" %in% mat_names) {
                 X_aligned <- X[, setdiff(common_names, "(Intercept)"), drop = FALSE]
                 beta_mean_aligned <- beta_mean[setdiff(common_names, "(Intercept)")]
            }
            
            X <- X_aligned
            beta_mean <- beta_mean_aligned
            cat("Aligned based on common names:", paste(common_names, collapse=", "), "\n")
        } else {
            warning("Cannot align coefficients and prediction matrix by name. Using potentially incorrect subset/padding.")
            min_cols <- min(length(beta_mean), ncol(X))
            beta_mean <- beta_mean[1:min_cols]
            X <- X[, 1:min_cols, drop = FALSE]
        }
      }

      eta <- X %*% beta_mean
      lambda <- exp(eta)

      if (!is.null(self$results$samples$phi)) {
        phi_mean <- apply(self$results$samples$phi, 2, mean)
        if (nrow(newdata) == length(phi_mean)) {
          lambda <- lambda * exp(phi_mean)
        }
      } else {
          warning("No spatial random effects (phi) found in model results.")
      }

      return(as.numeric(lambda))
    }
  )
)

cat("Loading and preparing data...\n")
full_data <- load_and_prepare_data()

response_var <- "acc"
random_effect <- "borough"
max_features <- 30

cat("Selecting features using LASSO...\n")
exclude_cols <- c('int_no', 'acc', 'pi', 'borough', 'x', 'y')
feature_cols <- setdiff(colnames(full_data), exclude_cols)
X_full <- full_data[, feature_cols]
y_full <- full_data[[response_var]]

selected_features <- select_features_with_lasso(X_full, y_full, max_features)
cat("Selected features:", paste(selected_features, collapse=", "), "\n")

formula_str_base <- paste(selected_features, collapse = " + ")
formula_str <- paste(response_var, "~", formula_str_base)

cat("\n--- Training Final CARBayes Model on Full Dataset ---\n")
carbayes_final <- carbayes_model$new(
  data = full_data,
  formula = formula_str,
  random_effect = random_effect,
  spatial_vars = c('x', 'y')
)

carbayes_results <- try(carbayes_final$fit(burnin = 2000, n_sample = 5000, thin = 2), silent = FALSE)

if (!inherits(carbayes_results, "try-error") && !is.null(carbayes_results)) {
  cat("\n==== CARBayes Model Summary ====\n")
  print(summary(carbayes_results))

  cat("\n==== Model Coefficients (Posterior Means) ====\n")
  if (!is.null(carbayes_results$summary.results)) {
      print(carbayes_results$summary.results)
  } else {
      cat("Summary results not found in the model object.\n")
  }

} else {
  cat("\nCARBayes model fitting failed. No results to display.\n")
}

# --- Add Visualization Code Here ---

cat("\n--- Generating Spatial Weights Visualization ---\n")

# Check if the model and W matrix exist
if (!inherits(carbayes_results, "try-error") && !is.null(carbayes_final) && !is.null(carbayes_final$W)) {

  W_matrix <- carbayes_final$W
  coords <- full_data[, c("int_no", "x", "y")] # Assuming 'int_no' is a unique identifier

  # Ensure W_matrix has row/column names matching coordinates if possible
  # If W matrix rows/cols don't correspond directly to coords rows, adjust accordingly.
  # Here, we assume the i-th row/col in W corresponds to the i-th row in coords.
  if (nrow(W_matrix) == nrow(coords)) {
    rownames(W_matrix) <- coords$int_no
    colnames(W_matrix) <- coords$int_no
  } else {
      warning("Mismatch between W matrix dimensions and coordinate rows. Visualization might be incorrect.")
      # Add handling if needed, e.g., subsetting coords or stopping
  }

  # Convert W matrix to a long format data frame for links
  W_df <- melt(as.matrix(W_matrix), varnames = c("from_int", "to_int"), value.name = "weight")

  # Filter out zero weights (no direct link in the k-NN graph) and self-loops
  links <- W_df[W_df$weight > 0 & W_df$from_int != W_df$to_int, ]

  # Merge with coordinates to get start and end points for segments
  links <- merge(links, coords, by.x = "from_int", by.y = "int_no")
  names(links)[names(links) == "x"] <- "x_start"
  names(links)[names(links) == "y"] <- "y_start"

  links <- merge(links, coords, by.x = "to_int", by.y = "int_no")
  names(links)[names(links) == "x"] <- "x_end"
  names(links)[names(links) == "y"] <- "y_end"

  # Create the plot
  p <- ggplot() +
    # Draw the links (segments) between intersections
    # Color intensity represents the weight (inverse distance) - higher is stronger/closer
    geom_segment(data = links, aes(x = x_start, y = y_start, xend = x_end, yend = y_end, color = weight), alpha = 0.5) +
    # Draw the intersections (points)
    geom_point(data = coords, aes(x = x, y = y), size = 1, color = "black") +
    # Use a color scale appropriate for continuous weights
    scale_color_viridis_c(option = "plasma", name = "Spatial Weight\n(Inverse Distance)") +
    # Add titles and labels
    labs(title = "Intersection Map with Spatial Links",
         subtitle = paste("Links based on k=3 nearest neighbors (inverse distance weighted)"), # Use the known k value
         x = "X Coordinate",
         y = "Y Coordinate") +
    # Use a minimal theme and set background to white
    theme_minimal() +
    theme(
      aspect.ratio = 1, # Maintain aspect ratio for spatial accuracy
      plot.background = element_rect(fill = "white", color = NA) # Set background to white, remove border
      ) 

  # Print the plot
  print(p)
  
  # Optional: Save the plot
  ggsave("intersection_links_map.png", plot = p, width = 10, height = 10, units = "in", dpi = 300)
  cat("Saved map to intersection_links_map.png\n")

} else {
  cat("Skipping visualization because model fitting failed or W matrix is unavailable.\n")
}
# --- End of Visualization Code ---

cat("\n--- Script Finished ---\n")