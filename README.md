# NBA Combine Prediction

## Overview
This project analyzes NBA Combine data to predict rookie performance using machine learning techniques. By combining anthropometric measurements, strength and agility metrics, and player positions, we build a model that can forecast a player's rookie year performance based on their pre-draft measurements.

## Features
- **Data Collection**: Scrapes NBA Combine data from official sources
- **Data Preprocessing**: Cleans, merges, and standardizes player data
- **Feature Engineering**: Creates composite metrics and handles categorical variables
- **Machine Learning Model**: Implements a Multilayer Perceptron (MLP) neural network
- **Interactive Visualization**: Provides an intuitive Streamlit interface for exploring predictions

## Data Sources
- NBA Combine measurements (height, weight, wingspan, etc.)
- Strength and agility metrics
- Rookie year statistics

## Model Details
The project uses a Multilayer Perceptron (MLP) neural network with:
- Input layer: Anthropometric measurements, strength/agility metrics, and one-hot encoded positions
- Hidden layers: Two fully connected layers with ReLU activation
- Output layer: Predicted rookie performance score


### Running the Application
To start the Streamlit app:
```
streamlit run app.py
```
