# Load libraries
library(dplyr)
library(readr)
library(ggplot2)
library(caret)
library(scales)
library(tidyr)
library(proxy)  # for cosine and Euclidean
library(stringr)

# Load data
rookie_and_combine <- read_csv("combined.csv")

# Combine features
combine_features <- c(
  'HEIGHT_WO_SHOES', 'WEIGHT', 'WINGSPAN', 'STANDING_REACH', 'BODY_FAT_PCT',
  'HAND_LENGTH', 'HAND_WIDTH', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT',
  'STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'MODIFIED_LANE_AGILITY_TIME'
)

center_positions <- c('C', 'PF-C', 'C-PF', 'PF')

# Filter center players with at least 20 games played
center_df <- rookie_and_combine %>%
  filter(POSITION %in% center_positions, GAMES_PLAYED >= 20)

# Fill NA combine features with column means (centers only)
center_df[combine_features] <- center_df[combine_features] %>%
  mutate(across(everything(), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Standardize combine features
scaled_df <- scale(center_df[combine_features])
colnames(scaled_df) <- paste0(combine_features, "_scaled")
scaled_df <- as.data.frame(scaled_df)

# Add weights (same order as combine_features)
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

# Calculate Combine_Rating
center_df$Combine_Rating <- as.matrix(scaled_df) %*% weights

# Add back Combine_Rating to original dataset
rookie_and_combine$Combine_Rating[match(center_df$PLAYER_NAME, rookie_and_combine$PLAYER_NAME)] <- center_df$Combine_Rating

# --- Collaborative Filtering Functions ---

# Helper: compute similarity
compute_similarity <- function(a, b, method = "L2") {
  if (method == "Cosine") {
    return(1 - proxy::dist(rbind(a, b), method = "cosine")[1])
  } else if (method == "L2") {
    return(-proxy::dist(rbind(a, b), method = "Euclidean")[1])
  } else {
    stop("Choose 'Cosine' or 'L2'")
  }
}

# Predict rookie score for a single player
predict_rookie_score_cf <- function(data, target_player, similarity_metric = "L2", k = 5) {
  if (!(target_player %in% rownames(data))) {
    message(paste("Error:", target_player, "not found in dataset."))
    return(NA)
  }

  features <- data[, setdiff(colnames(data), "ROOKIE_SCORE")]
  known_scores <- data$ROOKIE_SCORE

  target_vec <- as.numeric(features[target_player, ])

  similarities <- sapply(rownames(data), function(player) {
    if (player == target_player || is.na(known_scores[player])) return(NA)
    other_vec <- as.numeric(features[player, ])
    compute_similarity(target_vec, other_vec, method = similarity_metric)
  })

  sim_df <- data.frame(Player = names(similarities), Similarity = similarities, stringsAsFactors = FALSE) %>%
    filter(!is.na(Similarity)) %>%
    mutate(NormSim = rescale(Similarity))

  top_k <- head(sim_df[order(-sim_df$NormSim), ], k)

  weighted_sum <- sum(top_k$NormSim * known_scores[top_k$Player])
  sim_sum <- sum(top_k$NormSim)

  if (sim_sum == 0) return(NA)
  return(weighted_sum / sim_sum)
}

# Predict for multiple players
predict_rookies <- function(data, player_list, similarity_metric = "L2", k = 5) {
  sapply(player_list, function(player) {
    tryCatch({
      predict_rookie_score_cf(data, player, similarity_metric, k)
    }, error = function(e) paste("Error:", e$message))
  })
}

# --- Run Prediction and Plot ---

# Set rownames
center_df <- center_df %>% drop_na(ROOKIE_SCORE)
rownames(center_df) <- center_df$PLAYER_NAME

cf_data <- center_df[, c(combine_features, "ROOKIE_SCORE")]

center_players <- rownames(cf_data)

# Predict scores
center_predictions <- predict_rookies(cf_data, center_players, similarity_metric = "Cosine", k = 5)

# Filter valid predictions
valid_players <- names(center_predictions)[
  !is.na(center_predictions) & !is.na(cf_data[names(center_predictions), "ROOKIE_SCORE"])
]

# Create data frame of predicted vs actual
pred_df <- data.frame(
  Player = valid_players,
  Predicted = as.numeric(center_predictions[valid_players]),
  Actual = cf_data[valid_players, "ROOKIE_SCORE"]
)

# Plot predicted vs actual
ggplot(pred_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.7) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(title = "Predicted vs. Actual Rookie Scores (Centers)",
       x = "Actual Rookie Score", y = "Predicted Rookie Score") +
  theme_minimal()
