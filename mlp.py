import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import matplotlib.pyplot as plt

def load_data():
    """Load the preprocessed data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'combined.csv')
    return pd.read_csv(data_path)

def preprocess_data():
    df = load_data()
    features = [
        'WEIGHT',
        'HAND_WIDTH', 'LANE_AGILITY_TIME',
        'THREE_QUARTER_SPRINT', 'MAX_VERTICAL_LEAP',
        'MODIFIED_LANE_AGILITY_TIME', 'GAMES_PLAYED'
    ]
    df = df.dropna(subset=features + ['ROOKIE_SCORE', 'PLAYER_NAME'])

    correlations = df[features + ['ROOKIE_SCORE']].corr()
    print(correlations['ROOKIE_SCORE'].sort_values(ascending=False))

    X = df[features].values
    y = df['ROOKIE_SCORE'].values.reshape(-1, 1)
    names = df['PLAYER_NAME'].values

    # Min-Max scaling
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = (X - X_min) / (X_max - X_min)

    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # Train-test split (with names)
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, names, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, names_train, names_test



class MLP:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        self.W1 = np.random.randn(input_dim, hidden_dim1) * 0.1
        self.b1 = np.zeros((1, hidden_dim1))
        self.W2 = np.random.randn(hidden_dim1, hidden_dim2) * 0.1
        self.b2 = np.zeros((1, hidden_dim2))
        self.W3 = np.random.randn(hidden_dim2, 1) * 0.1
        self.b3 = np.zeros((1, 1))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_deriv(self, x):
        return 1 - np.tanh(x) ** 2

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.h1 = self.tanh(self.z1)

        self.z2 = self.h1 @ self.W2 + self.b2
        self.h2 = self.relu(self.z2)

        self.z3 = self.h2 @ self.W3 + self.b3
        return self.z3

    def loss(self, y_pred, y):
        return np.mean((y_pred - y) ** 2)

    def backward(self, X, y, y_pred, lr):
        n = X.shape[0]

        dz3 = (y_pred - y)
        dW3 = self.h2.T @ dz3 / n
        db3 = np.sum(dz3, axis=0, keepdims=True) / n

        dh2 = dz3 @ self.W3.T
        dz2 = dh2 * self.relu_deriv(self.z2)
        dW2 = self.h1.T @ dz2 / n
        db2 = np.sum(dz2, axis=0, keepdims=True) / n

        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * self.tanh_deriv(self.z1)
        dW1 = X.T @ dz1 / n
        db1 = np.sum(dz1, axis=0, keepdims=True) / n

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3

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
    X_train, X_test, y_train, y_test, names_train, names_test = preprocess_data()
    model = MLP(input_dim=X_train.shape[1], hidden_dim1=32, hidden_dim2=16)
    model.train(X_train, y_train, lr=0.01, epochs=5000)
    
    y_pred_test = model.predict(X_test)
    test_loss = model.loss(y_pred_test, y_test)
    print(f"Test MSE: {test_loss:.4f}")

    baseline_pred = np.mean(y_train)
    baseline_preds = np.full_like(y_test, baseline_pred)
    baseline_loss = model.loss(baseline_preds, y_test)
    print(f"Baseline MSE (predict mean): {baseline_loss:.4f}")

    r2 = r2_score(y_test, y_pred_test)
    print(f"R^2 Score: {r2:.4f}")
    
    y_true = y_test.flatten()
    y_pred = y_pred_test.flatten()


    top_indices = np.argsort(y_pred)[::-1]  # indices of top predictions

    print("\nTop Predicted ROOKIE_SCORE Players (Test Set):")
    print(f"{'Player':<25} {'Predicted':>10} {'Actual':>10}")
    print("-" * 50)
    for i in top_indices[:10]:  # top 10
        print(f"{names_test[i]:<25} {y_pred[i]:10.2f} {y_true[i]:10.2f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, label='Predicted vs Actual')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal (y = x)')
    plt.xlabel('Actual ROOKIE_SCORE')
    plt.ylabel('Predicted ROOKIE_SCORE')
    plt.title('Predicted vs Actual Test Scores')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
