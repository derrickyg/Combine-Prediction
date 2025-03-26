# Combine-Prediction
Objective:
Use NBA Combine performance metrics to predict whether a player will have a successful NBA career, based on historical data of past prospects.

Machine Learning Methods
Collaborative Filtering (CF) for Player Comparisons

Use CF to find similar past players based on their NBA Combine stats (e.g., vertical jump, lane agility, wingspan, three-quarter sprint).
Assign a success score based on how those similar players performed in the NBA (e.g., career points per game, minutes played, or All-Star appearances).
This helps answer: "Which past players does this new prospect resemble, and how did they turn out?"
Neural Network (MLP or CNN) for Performance Prediction

MLP (Multilayer Perceptron) for tabular NBA Combine stats (height, weight, speed, agility).
CNN (if using video data) to analyze movement mechanics from Combine footage.
The model would predict whether a prospect will be a starter, bench player, or bust.
