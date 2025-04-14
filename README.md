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

## Web Application
The Streamlit app provides:
- Interactive visualization of model predictions vs. actual performance
- Detailed metrics and statistics
- User-friendly navigation between different views
- Hover functionality to explore individual player predictions

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages (see requirements.txt)

### Installation
1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application
To start the Streamlit app:
```
streamlit run app.py
```

### Accessing the Frontend
1. After running the command above, the Streamlit app will automatically open in your default web browser
2. If it doesn't open automatically, you can access it at: http://localhost:8501
3. The app provides two main sections:
   - **Home**: Overview of the project, data sources, and sample data
   - **Model Visualization**: Interactive plots showing predicted vs. actual rookie performance