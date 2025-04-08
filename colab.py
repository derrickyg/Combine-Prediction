# Importing libraries
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine, euclidean

rookie_and_combine = pd.read_csv("nba_combine_and_rookie_stats_preprocessed.csv")
print(rookie_and_combine)
# Defining collab_filter function
def collab_filter(data, target_player, similarity_metric='L2', k=5):
    # Check if the student is in our dataset
    if target_player not in data.index:
        print(f"Error: {target_player} not found in the dataset.")
        return None
    
    # Get the student's ratings
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
