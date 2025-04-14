# Importing libraries
library(dplyr)
library(readr)
library(ggplot2)
library(scales)
library(stringr)
library(proxy)       # for cosine and euclidean
library(tidyr)       # for data manipulation
library(caret)       # for preprocessing like standardization

# Load data
rookie_and_combine <- read_csv("/Combine-Prediction/combined.csv")

# Define the combine features to use for scoring
combine_features <- c(
  'HEIGHT_WO_SHOES', 'WEIGHT', 'WINGSPAN', 'STANDING_REACH', 'BODY_FAT_PCT',
  'HAND_LENGTH', 'HAND_WIDTH', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT',
  'STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'MODIFIED_LANE_AGILITY_TIME'
)

center_positions <- c('C', 'PF-C', 'C-PF', 'PF')

# Step 1: Get all center players
center_df <- rookie_and_combine %>%
  filter(POSITION %in% center_positions & GAMES_PLAYED >= 20)

# Step 2: Fill missing combine values with average for centers
center_df[combine_features] <- center_df %>%
  select(all_of(combine_features)) %>%
  mutate(across(everything(), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Standardize the features
scaler <- preProcess(center_df[, combine_features], method = c("center", "scale"))
scaled_features <- predict(scaler, center_df[, combine_features])

# Rename scaled feature columns
colnames(scaled_features) <- paste0(combine_features, "_scaled")

# Combine back with original data if needed
# center_df <- bind_cols(center_df, scaled_features)

# Assign weights (subjective, can be optimized)
# Positive weights for performance-enhancing metrics, negative for times and body fat
weights <- c(
  0.15,  # HEIGHT_WO_SHOES
  0.1,   # WEIGHT
  0.15,  # WINGSPAN
  0.15,  # STANDING_REACH
 -0.1,   # BODY_FAT_PCT
  0.05,  # HAND_LENGTH
  0.05,  # HAND_WIDTH
 -0.1,   # LANE_AGILITY_TIME
 -0.1,   # THREE_QUARTER_SPRINT
  0.1,   # STANDING_VERTICAL_LEAP
  0.1,   # MAX_VERTICAL_LEAP
 -0.05   # MODIFIED_LANE_AGILITY_TIME
)

# Calculate weighted combine rating (dot product of scaled features and weights)
center_df$Combine_Rating <- as.numeric(as.matrix(scaled_features) %*% weights)

# Add Combine_Rating back into the original dataset
rookie_and_combine$Combine_Rating <- NA
rookie_and_combine$Combine_Rating[as.numeric(rownames(center_df))] <- center_df$Combine_Rating

predict_rookie_score_cf <- function(data, target_player, similarity_metric = "L2", k = 5) {
  if (!(target_player %in% rownames(data))) {
    cat(sprintf("Error: %s not found in dataset.\n", target_player))
    return(NA)
  }
  
  # Separate features and known scores
  features <- data %>% select(-ROOKIE_SCORE)
  known_scores <- data$ROOKIE_SCORE
  
  # Get target vector
  target_vector <- as.numeric(features[target_player, ])

  similarities <- c()
  
  for (player in rownames(data)) {
    if (player == target_player) next
    if (is.na(known_scores[player])) next
    
    other_vector <- as.numeric(features[player, ])
    
    sim <- switch(similarity_metric,
      "Cosine" = sum(target_vector * other_vector) / (sqrt(sum(target_vector^2)) * sqrt(sum(other_vector^2))),
      "L2" = -dist(rbind(target_vector, other_vector), method = "euclidean"),
      stop("Choose 'Cosine' or 'L2'")
    )
    
    similarities[player] <- sim
  }
  
  # Normalize similarities between 0 and 1
  sim_df <- data.frame(Player = names(similarities), Similarity = unname(similarities)) %>%
    mutate(Similarity = rescale(Similarity, to = c(0, 1))) %>%
    arrange(desc(Similarity))
  
  # Take top k similar players
  top_k <- head(sim_df, k)
  
  # Weighted average of rookie scores
  weighted_sum <- sum(top_k$Similarity * known_scores[top_k$Player])
  sim_sum <- sum(top_k$Similarity)
  
  predicted_score <- if (sim_sum != 0) weighted_sum / sim_sum else NA
  return(predicted_score)
}


# Set the rownames to player name or ID
rownames(center_df) <- center_df$PLAYER_NAME

# Build the full matrix for CF
cf_data <- center_df %>%
  select(all_of(c(combine_features, "ROOKIE_SCORE")))

# Function to predict rookie scores for a list of players
predict_rookies <- function(data, player_list, similarity_metric = "L2", k = 5) {
  predictions <- list()
  
  for (player in player_list) {
    result <- tryCatch(
      {
        predict_rookie_score_cf(data, player, similarity_metric = similarity_metric, k = k)
      },
      error = function(e) {
        paste("Error:", e$message)
      }
    )
    predictions[[player]] <- result
  }
  
  return(predictions)
}

# Step 7: Get the list of center player names (with combine features filled)
center_players <- center_df$PLAYER_NAME

# Step 8: Predict rookie scores
center_predictions <- predict_rookies(cf_data, center_players, similarity_metric = "Cosine", k = 5)

# Step 9: Print predicted vs actual
for (player in names(center_predictions)) {
  predicted_score <- center_predictions[[player]]
  actual_score <- if (player %in% rownames(cf_data)) cf_data[player, "ROOKIE_SCORE"] else NA
  
  if (is.numeric(predicted_score)) {
    cat(sprintf("%s: Predicted = %.2f, Actual = %.2f\n", player, predicted_score, actual_score))
  } else {
    cat(sprintf("%s: Prediction Error - %s\n", player, predicted_score))
  }
}

# Step 10: Filter only valid predictions with actual values
valid_players <- names(center_predictions)[
  sapply(center_predictions, is.numeric) &
    !is.na(unlist(center_predictions)) &
    !is.na(cf_data[names(center_predictions), "ROOKIE_SCORE"])
]

# Step 2: Get predicted and actual values
predicted_vals <- unlist(center_predictions[valid_players])
actual_vals <- cf_data[valid_players, "ROOKIE_SCORE"]

# Step 3: Create a data frame for plotting
plot_df <- data.frame(
  Actual = actual_vals,
  Predicted = predicted_vals
)

# Step 4: Plot predicted vs actual
ggplot(plot_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Predicted vs. Actual Rookie Scores (Centers)",
    x = "Actual Rookie Score",
    y = "Predicted Rookie Score"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))