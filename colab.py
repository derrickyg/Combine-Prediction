# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine, euclidean

# Load data
rookie_and_combine = pd.read_csv("combined.csv")

# Define the combine features to use for scoring
combine_features = [
    'HEIGHT_WO_SHOES', 'WEIGHT', 'WINGSPAN', 'STANDING_REACH', 'BODY_FAT_PCT',
    'HAND_LENGTH', 'HAND_WIDTH', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT',
    'STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'MODIFIED_LANE_AGILITY_TIME'
]

center_positions = ['C', 'PF-C', 'C-PF', 'PF']

# Step 1: Get all center players
center_df = rookie_and_combine[
    rookie_and_combine['POSITION'].isin(center_positions) &
    (rookie_and_combine['GAMES_PLAYED'] >= 20)
].copy()


# Step 2: Fill missing combine values with average for centers
center_feature_means = center_df[combine_features].mean()
center_df[combine_features] = center_df[combine_features].fillna(center_feature_means)


# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(center_df[combine_features])
scaled_df = pd.DataFrame(scaled_features, columns=[f"{col}_scaled" for col in combine_features])

# Assign weights (subjective, can be optimized)
# Positive weights for performance-enhancing metrics, negative for times and body fat
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
rookie_and_combine.loc[center_df.index, 'Combine_Rating'] = center_df['Combine_Rating']


def predict_rookie_score_cf(data, target_player, similarity_metric='L2', k=5):
    if target_player not in data.index:
        print(f"Error: {target_player} not found in dataset.")
        return None

    # Separate features and known scores
    features = data.drop(columns=['ROOKIE_SCORE'])
    known_scores = data['ROOKIE_SCORE']

    # Get target vector
    target_vector = features.loc[target_player].values

    similarities = {}
    
    for player in data.index:
        if player == target_player:
            continue
        if pd.isna(known_scores[player]):
            continue  # Only use players with known rookie scores

        other_vector = features.loc[player].values

        if similarity_metric == 'Cosine':
            sim = cosine_similarity([target_vector], [other_vector])[0][0]
        elif similarity_metric == 'L2':
            sim = -euclidean(target_vector, other_vector)  # Negative to make it similarity
        else:
            raise ValueError("Choose 'Cosine' or 'L2'")
        
        similarities[player] = sim

    # Normalize similarities between 0 and 1
    sim_df = pd.DataFrame.from_dict(similarities, orient='index', columns=['Similarity'])
    sim_df['Similarity'] = MinMaxScaler().fit_transform(sim_df[['Similarity']])
    
    # Take top k similar players
    top_k = sim_df['Similarity'].nlargest(k)

    # Weighted average of rookie scores
    weighted_sum = 0
    sim_sum = 0
    for player, sim in top_k.items():
        weighted_sum += sim * known_scores[player]
        sim_sum += sim

    predicted_score = weighted_sum / sim_sum if sim_sum != 0 else np.nan
    return predicted_score

# Set the index to player name or ID
df_features = center_df.set_index("PLAYER_NAME")


# Build the full matrix for CF
cf_data = df_features[combine_features + ['ROOKIE_SCORE']]

def predict_rookies(data, player_list, similarity_metric='L2', k=5):
    predictions = {}

    for player in player_list:
        try:
            prediction = predict_rookie_score_cf(data, player, similarity_metric=similarity_metric, k=k)
            predictions[player] = prediction
        except Exception as e:
            predictions[player] = f"Error: {str(e)}"

    return predictions


# Step 7: Get the list of center player names (with combine features filled)
center_players = center_df['PLAYER_NAME'].tolist()

# Step 8: Predict rookie scores
center_predictions = predict_rookies(cf_data, center_players, similarity_metric='Cosine', k=5)

for player, predicted_score in center_predictions.items():
    actual_score = cf_data.loc[player, 'ROOKIE_SCORE'] if player in cf_data.index else "N/A"
    
    if isinstance(predicted_score, (float, int)):
        print(f"{player}: Predicted = {predicted_score:.2f}, Actual = {actual_score}")
    else:
        print(f"{player}: Prediction Error - {predicted_score}")

# Step 1: Filter only valid predictions with actual values
valid_players = [
    player for player, pred in center_predictions.items()
    if isinstance(pred, (float, int)) and not np.isnan(pred)
    and not pd.isna(cf_data.loc[player, 'ROOKIE_SCORE'])
]

# Step 2: Get predicted and actual values
predicted_vals = [center_predictions[player] for player in valid_players]
actual_vals = [cf_data.loc[player, 'ROOKIE_SCORE'] for player in valid_players]

# Step 3: Plot predicted vs actual
plt.figure(figsize=(8, 6))
plt.scatter(actual_vals, predicted_vals, alpha=0.7)
plt.plot([min(actual_vals), max(actual_vals)], [min(actual_vals), max(actual_vals)], 'r--')  # reference line y=x
plt.xlabel('Actual Rookie Score')
plt.ylabel('Predicted Rookie Score')
plt.title('Predicted vs. Actual Rookie Scores (Centers)')
plt.grid(True)
plt.tight_layout()
plt.show()













# Defining collab_filter function
def collab_filter(data, target_player, similarity_metric='L2', k=5):
    # Check if the player is in our dataset
    if target_player not in data.index:
        print(f"Error: {target_player} not found in the dataset.")
        return None
    
    # Get the player's ratings
    user_ratings = data.loc[target_player]
    
    # Check if the user has missing ratings
    if not user_ratings.isnull().any():
        print(f"{target_player} has no missing ratings.")
        return None
    
    # Get mean and center data
    user_means = data.mean(axis=1)
    centered_data = data.sub(user_means, axis=0).fillna(0)
    
    # Stores the user and their similarity scores
    similarities = {}
    target_ratings = centered_data.loc[target_player].values
    
    # Calculate the similarity scores based on the metric we choose
    for user in data.index:
        if user != target_player:
            other_ratings = centered_data.loc[user].values
            
            if similarity_metric == 'Cosine':
                similarity = 1 - cosine(target_ratings, other_ratings)
            elif similarity_metric == 'L2':
                similarity = -euclidean(target_ratings, other_ratings)
            else:
                print("Error: Invalid similarity metric. Choose 'L2' or 'Cosine'.")
                return None
            
            similarities[user] = similarity
    
    sim_df = pd.DataFrame.from_dict(similarities, orient='index', columns=['Similarity'])
    sim_df['Similarity'] = MinMaxScaler().fit_transform(sim_df[['Similarity']])
    
    # Getting top k users
    top_k_users = sim_df['Similarity'].nlargest(k)
    
    weighted_sum = np.zeros(len(data.columns))
    similarity_sum = np.zeros(len(data.columns))
    
    # Go through our most similar users
    for user, sim in top_k_users.items():
        # Fill in missing ratings for the user
        user_ratings_filled = data.loc[user].copy()
        user_ratings_filled.fillna(user_means[user], inplace=True)
        
        # Computes weighted rating sum
        weighted_sum += sim * user_ratings_filled
        similarity_sum += sim
    
    # Predict the ratings
    predicted_ratings = np.divide(weighted_sum, similarity_sum, out=np.zeros_like(weighted_sum), where=similarity_sum!=0)
    
    predictions = pd.Series(predicted_ratings, index=data.columns, name=target_player)
    return predictions[user_ratings.isnull()]
