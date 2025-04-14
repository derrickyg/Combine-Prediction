# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial.distance import cosine, euclidean

# Load data
rookie_and_combine = pd.read_csv("combined.csv")

# Define the combine features that we are using to calculate our weighted combine rating
combine_features = [
    'HEIGHT_WO_SHOES', 'WEIGHT', 'WINGSPAN', 'STANDING_REACH', 'BODY_FAT_PCT',
    'HAND_LENGTH', 'HAND_WIDTH', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT',
    'STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'MODIFIED_LANE_AGILITY_TIME'
]

center_positions = ['C', 'PF-C', 'C-PF', 'PF']

# Filtering the dataset for our positions of interest and filtering for relevant players who have played more than 20 games
# in their rookie season
center_df = rookie_and_combine[
    rookie_and_combine['POSITION'].isin(center_positions) &
    (rookie_and_combine['GAMES_PLAYED'] >= 20)
].copy()

# Fill missing combine values with average for our positions
center_feature_means = center_df[combine_features].mean()
center_df[combine_features] = center_df[combine_features].fillna(center_feature_means)

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(center_df[combine_features])
scaled_df = pd.DataFrame(scaled_features, columns=combine_features)

# Assign weights (these are weights from what we have researched and measured ourselves 
# in terms of what attributes in the combine would be more valuable to the positions)
weights = np.array([
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
])

# Calculate weighted combine rating
center_df['Combine_Rating'] = scaled_df.dot(weights)

# Add Combine_Rating back into the original dataset
rookie_and_combine.loc[rookie_and_combine['PLAYER_NAME'].isin(center_df['PLAYER_NAME']), 'Combine_Rating'] = center_df['Combine_Rating'].values

# Set the index to player name
df_features = center_df.set_index("PLAYER_NAME")

# Build matrix for CF and drop rows with missing rookie scores
cf_data = df_features[combine_features + ['ROOKIE_SCORE']].dropna()

# Collaborative filtering function (simliar to what we did in the homework)
def predict_rookie_score_cf(data, target_player, similarity_metric='L2', k=5):
    if target_player not in data.index:
        print(f"Error: {target_player} not found in dataset.")
        return None

    features = data.drop(columns=['ROOKIE_SCORE'])
    known_scores = data['ROOKIE_SCORE']
    target_vector = features.loc[target_player].values

    similarities = {}
    for player in data.index:
        if player == target_player or pd.isna(known_scores[player]):
            continue

        other_vector = features.loc[player].values

        if similarity_metric == 'Cosine':
            sim = cosine_similarity([target_vector], [other_vector])[0][0]
        elif similarity_metric == 'L2':
            sim = -euclidean(target_vector, other_vector)
        else:
            raise ValueError("Choose 'Cosine' or 'L2'")

        similarities[player] = sim

    sim_df = pd.DataFrame.from_dict(similarities, orient='index', columns=['Similarity'])
    sim_df['Similarity'] = MinMaxScaler().fit_transform(sim_df[['Similarity']])

    top_k = sim_df['Similarity'].nlargest(k)

    weighted_sum = 0
    sim_sum = 0
    for player, sim in top_k.items():
        weighted_sum += sim * known_scores[player]
        sim_sum += sim

    return weighted_sum / sim_sum if sim_sum != 0 else np.nan

# Created this so our model can predict all the players in our dataset
def predict_rookies(data, player_list, similarity_metric='L2', k=5):
    predictions = {}
    for player in player_list:
        try:
            prediction = predict_rookie_score_cf(data, player, similarity_metric=similarity_metric, k=k)
            predictions[player] = prediction
        except Exception as e:
            predictions[player] = f"Error: {str(e)}"
    return predictions

center_players = cf_data.index.tolist()

# Prediction
center_predictions = predict_rookies(cf_data, center_players, similarity_metric='Cosine', k=5)

# Print predictions
for player, predicted_score in center_predictions.items():
    actual_score = cf_data.loc[player, 'ROOKIE_SCORE']
    if isinstance(predicted_score, (float, int)):
        print(f"{player}: Predicted = {predicted_score:.2f}, Actual = {actual_score}")
    else:
        print(f"{player}: Prediction Error - {predicted_score}")

# Evaluation
valid_players = [
    player for player, pred in center_predictions.items()
    if isinstance(pred, (float, int)) and not np.isnan(pred)
    and not pd.isna(cf_data.loc[player, 'ROOKIE_SCORE'])
]

predicted_vals = [center_predictions[player] for player in valid_players]
actual_vals = [cf_data.loc[player, 'ROOKIE_SCORE'] for player in valid_players]

# Plot predicted vs actual
plt.figure(figsize=(8, 6))
plt.scatter(actual_vals, predicted_vals, alpha=0.7)
plt.plot([min(actual_vals), max(actual_vals)], [min(actual_vals), max(actual_vals)], 'r--')
plt.xlabel('Actual Rookie Score')
plt.ylabel('Predicted Rookie Score')
plt.title('Predicted vs. Actual Rookie Scores (Centers)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluation metrics
print("\nEvaluation Metrics:")
print("R^2 Score:", r2_score(actual_vals, predicted_vals))
print("MAE:", mean_absolute_error(actual_vals, predicted_vals))
print("RMSE:", np.sqrt(mean_squared_error(actual_vals, predicted_vals)))
