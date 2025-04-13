# Importing libraries
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine, euclidean

# Load data
rookie_and_combine = pd.read_csv("combined.csv")

# Set index
rookie_and_combine.set_index('PLAYER_NAME', inplace=True)

# Define combine features
combine_features = [
    'HEIGHT_WO_SHOES', 'WEIGHT', 'WINGSPAN', 'STANDING_REACH', 'BODY_FAT_PCT',
    'HAND_LENGTH', 'HAND_WIDTH', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT',
    'STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'MODIFIED_LANE_AGILITY_TIME'
]

# Drop rows with missing combine features
rookie_and_combine_filtered = rookie_and_combine.dropna(subset=combine_features).copy()

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(rookie_and_combine_filtered[combine_features])
scaled_df = pd.DataFrame(scaled_features, columns=combine_features, index=rookie_and_combine_filtered.index)

# Assign weights (custom)
weights = np.array([
    0.15, 0.1, 0.15, 0.15, -0.1,
    0.05, 0.05, -0.1, -0.1,
    0.1, 0.1, -0.05
])

# Compute combine rating
rookie_and_combine_filtered['COMBINE_RATING'] = scaled_df.dot(weights)

# Add Combine Rating back to original dataset
rookie_and_combine.loc[rookie_and_combine_filtered.index, 'COMBINE_RATING'] = rookie_and_combine_filtered['COMBINE_RATING']

print(rookie_and_combine.columns.tolist())

# ------------------------------------------
# Collaborative Filtering Function
# ------------------------------------------

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
df_features = rookie_and_combine.set_index(['PLAYER_NAME'])

# Keep only combine feature columns (already cleaned/standardized)
combine_cols = [
    'HEIGHT_WO_SHOES', 'WEIGHT', 'WINGSPAN', 'STANDING_REACH', 'BODY_FAT_PCT',
    'HAND_LENGTH', 'HAND_WIDTH', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT',
    'STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'MODIFIED_LANE_AGILITY_TIME'
]

# Build the full matrix for CF
cf_data = df_features[combine_cols + ['ROOKIE_SCORE']].dropna(subset=combine_cols)

# Predict rookie score for a player like "Mark Williams"
predicted = predict_rookie_score_cf(cf_data, "Mark Williams", similarity_metric='Cosine', k=5)

# Compare to actual
actual = cf_data.loc["Mark Williams", "ROOKIE_SCORE"]
print(f"Predicted: {predicted:.2f} vs Actual: {actual:.2f}")









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
