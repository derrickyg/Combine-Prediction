import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_data():
    """Load the preprocessed data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'combined.csv')
    return pd.read_csv(data_path)

def preprocess_data():
    df = load_data()

    features = [
        'HEIGHT_WO_SHOES', 'WEIGHT', 'WINGSPAN', 'STANDING_REACH',
        'BODY_FAT_PCT', 'HAND_LENGTH', 'HAND_WIDTH', 'LANE_AGILITY_TIME',
        'THREE_QUARTER_SPRINT', 'STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP',
        'MODIFIED_LANE_AGILITY_TIME', 'STANDING_REACH_INCHES'
    ]

    df = df.dropna(subset=features + ['ROOKIE_SCORE'])

    X = df[features].values
    y = df['ROOKIE_SCORE'].values.reshape(-1, 1)

    # Min-Max scaling
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = (X - X_min) / (X_max - X_min)

    # Add intercept term
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # shape: (n_samples, n_features + 1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

class MLP:
    def __init__(self, input_dim, hidden_dim):
        # Weight initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1      # shape: (n, hidden_dim)
        self.h1 = self.relu(self.z1)         # ReLU activation
        self.z2 = self.h1 @ self.W2 + self.b2  # output layer
        return self.z2                       # no activation on output

    def loss(self, y_pred, y):
        return np.mean((y_pred - y) ** 2)

    def backward(self, X, y, y_pred, lr):
        n = X.shape[0]

        dz2 = (y_pred - y)                   # shape: (n, 1)
        dW2 = self.h1.T @ dz2 / n
        db2 = np.sum(dz2, axis=0, keepdims=True) / n

        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * self.relu_deriv(self.z1)  # Use ReLU derivative
        dW1 = X.T @ dz1 / n
        db1 = np.sum(dz1, axis=0, keepdims=True) / n

        # Update weights
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def train(self, X, y, lr=0.01, epochs=1000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss(y_pred, y)
            self.backward(X, y, y_pred, lr)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, X):
        return self.forward(X)



if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    model = MLP(input_dim=X_train.shape[1], hidden_dim=16)
    model.train(X_train, y_train, lr=0.01, epochs=10000)

    y_pred_test = model.predict(X_test)
    test_loss = model.loss(y_pred_test, y_test)
    print(f"Test MSE: {test_loss:.4f}")

    baseline_pred = np.mean(y_train)
    baseline_preds = np.full_like(y_test, baseline_pred)
    baseline_mse = np.mean((baseline_preds - y_test) ** 2)
    print(f"Baseline MSE (predict mean): {baseline_mse:.4f}")