library(dplyr)
library(readr)
library(caret)
library(Metrics)
library(tidyr)
library(spdep)
library(spatialreg)
library(spgwr)
library(sp)
library(sf)
library(mapview)
library(MASS)


load_and_prepare_data <- function() {
  data <- read_delim(file.path("data/data_final.csv"), 
                     delim = ";", show_col_types = FALSE)
  
  # Handle missing values in ln_distdt (if present)
  data <- data %>% replace_na(list(ln_distdt=0))
  
  # Drop rows with duplicate coordinates
  message(paste("Number of rows before dropping duplicates:", nrow(data)))
  data <- data %>% distinct(x, y, .keep_all = TRUE)
  message(paste("Number of rows after dropping duplicates:", nrow(data)))
  
  # Remove pi = 0
  data <- data %>% filter(pi != 0)

  # First correct borough names with encoding issues
  borough_corrections <- list(
    '?le-Bizard-Sainte-GeneviÞve' = 'Île-Bizard-Sainte-Geneviève',
    'C¶te-Saint-Luc' = 'Côte-Saint-Luc',
    'C¶te-des-Neiges-Notre-Dame-de-Graces' = 'Côte-des-Neiges-Notre-Dame-de-Grâce',
    'MontrÚal-Est' = 'Montréal-Est',
    'MontrÚal-Nord' = 'Montréal-Nord',
    'Pointe-aux-Trembles-RiviÞres-des-Prairies' = 'Rivière-des-Prairies-Pointe-aux-Trembles',
    'St-LÚonard' = 'Saint-Léonard'
  )
  
  # Then group boroughs into zones
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
  
  # Apply corrections if borough column exists
  if ("borough" %in% colnames(data)) {
    # First fix encoding issues
    for (old_name in names(borough_corrections)) {
      data$borough[data$borough == old_name] <- borough_corrections[[old_name]]
    }
    
    # Then group into zones
    for (borough in names(borough_zones)) {
      data$borough[data$borough == borough] <- borough_zones[[borough]]
    }
  }

  # Scale the features
  data[c('ln_fi', 'ln_fri', 'ln_fli', 'ln_pi', 
         'ln_fti', "distdt", "ln_distdt")] <- scale(data[c('ln_fi', 'ln_fri', 'ln_fli', 'ln_pi', 
                             'ln_fti', "distdt", "ln_distdt") ])
  data[c('ln_cti', 'ln_cli', 'ln_cri')] <- scale(data[c('ln_cti', 'ln_cli', 'ln_cri')])

  return(data)
}

#----------------------------------------------------------------
# 1. Load and prepare data
#----------------------------------------------------------------
df <- load_and_prepare_data()

# Convert to spatial object
coordinates(df) <- ~ x + y

# Specify the formula
formulaa <- acc ~ ln_fli + ln_pi + ln_distdt + ln_cli + ln_cri + total_lane + tot_road_w + tot_crossw + commercial + curb_exten + median + all_pedest + half_phase + ped_countd + lt_restric + lt_prot_re + any_exclus + all_red_an + green_stra + parking

#----------------------------------------------------------------
# 2. Helper to compute spatial weights for a given training subset
#----------------------------------------------------------------
make_spatial_weights <- function(sp_data, k) {
  id <- row.names(as(sp_data, "data.frame"))
  neighbours <- knn2nb(knearneigh(coordinates(sp_data), k = k), row.names = id)
  listw <- nb2listw(neighbours, style = "B")
  return(listw)
}

plot_spatial_folds <- function(sp_data, cluster_assignments, title = "Spatial Cross-Validation Folds") {
  # Create a copy of the spatial data to avoid modifying the original
  plot_data <- sp_data
  
  # Add cluster assignments as a factor (for better color mapping)
  plot_data$fold <- as.factor(cluster_assignments)
  
  # Convert to sf for better plotting with mapview
  plot_data_sf <- st_as_sf(plot_data)
  
  # Create the map using mapview
  m <- mapview(plot_data_sf, 
               zcol = "fold", 
               layer.name = "Fold Assignment",
               col.regions = rainbow(length(unique(cluster_assignments))),
               cex = 3,
               alpha = 1)
  
  return(m)
}

#----------------------------------------------------------------
# 3. Cross-validation function for SEM Poisson
#----------------------------------------------------------------
# Function to create a spatial error model with Poisson distribution
train_poisson_sem <- function(formulaa, train_data, val_data, global_weights, fold_weights, use_poisson=TRUE) {
  # Step 1: Fit a standard Poisson model first
  poisson_model <- glm(formulaa, data = train_data, family = poisson(link = "log"))
  
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

cross_validate_model <- function(sp_data, formulaa, k_folds=5, k_nb=3) {
  set.seed(123)
  # Replace spatial clustering with random fold assignments
  n_samples <- nrow(sp_data@data)
  fold_indices <- sample(rep(1:k_folds, length.out = n_samples))
  folds <- lapply(1:k_folds, function(k) which(fold_indices == k))
  
  # Initialize data frame for fold-wise metrics
  results <- data.frame(
    Fold = integer(),
    RMSE = numeric(),
    MAE  = numeric(),
    R2   = numeric(),
    N_Intersections = numeric(),
    Avg_Accidents = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Initialize vectors to store all predictions and actuals
  all_predictions <- numeric()
  all_actuals <- numeric()
  
  # Global weight matrix used for prediction
  w_global <- make_spatial_weights(sp_data, k = k_nb)
  
  for (i in seq_along(folds)) {
    test_idx  <- folds[[i]]
    train_idx <- setdiff(seq_len(nrow(sp_data@data)), test_idx)
    
    train_sp <- sp_data[train_idx, ]
    test_sp  <- sp_data[test_idx, ]
    
    # Calculate fold statistics
    n_intersections <- length(test_idx)
    avg_accidents <- mean(test_sp@data$acc)
    
    w_train <- make_spatial_weights(train_sp, k = k_nb)
    
    preds <- train_poisson_sem(formulaa, train_data = train_sp, val_data = test_sp, 
                              global_weights = w_global, fold_weights = w_train)
    
    actuals <- test_sp@data$acc
    
    # Store predictions and actuals
    all_predictions <- c(all_predictions, preds)
    all_actuals <- c(all_actuals, actuals)
    
    # Calculate fold-wise metrics
    rmse_val <- rmse(actuals, preds)
    mae_val  <- mae(actuals, preds)
    
    sse <- sum((actuals - preds)^2)
    sst <- sum((actuals - mean(actuals))^2)
    r2  <- 1 - sse/sst
    
    results <- rbind(results, 
                     data.frame(Fold = i, 
                               RMSE = rmse_val, 
                               MAE = mae_val, 
                               R2 = r2,
                               N_Intersections = n_intersections,
                               Avg_Accidents = avg_accidents))
  }
  
  # Calculate overall metrics using all predictions
  overall_rmse <- rmse(all_actuals, all_predictions)
  overall_mae <- mae(all_actuals, all_predictions)
  overall_sse <- sum((all_actuals - all_predictions)^2)
  overall_sst <- sum((all_actuals - mean(all_actuals))^2)
  overall_r2 <- 1 - overall_sse/overall_sst
  
  # Calculate summary metrics
  summary_results <- results %>%
    summarise(
      Fold_Avg_RMSE = mean(RMSE),
      Fold_Avg_MAE = mean(MAE),
      Fold_Avg_R2 = mean(R2),
      Overall_RMSE = overall_rmse,
      Overall_MAE = overall_mae,
      Overall_R2 = overall_r2,
      Total_Intersections = sum(N_Intersections),
      Overall_Avg_Accidents = mean(Avg_Accidents)
    )
  
  list(
    fold_metrics = results,
    summary = summary_results,
    predictions = data.frame(
      actual = all_actuals,
      predicted = all_predictions
    )
  )
}

#----------------------------------------------------------------
# 4. Run cross-validation for SEM Poisson model
#----------------------------------------------------------------
cv_results <- cross_validate_model(sp_data = df, formula = formulaa, k_folds = 5)

# Print results
print("Fold-wise metrics:")
print(cv_results$fold_metrics)
print("\nSummary metrics:")
print(cv_results$summary)