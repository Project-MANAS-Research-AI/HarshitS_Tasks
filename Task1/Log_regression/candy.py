import numpy as np
import pandas as pd

# =========================================================
# 1. Load Dataset
# =========================================================

def load_data(path):
    df = pd.read_csv(path)

    # Binary classification target:
    # 1 -> candy is popular (above median winpercent)
    # 0 -> otherwise
    median_score = df["winpercent"].median()
    df["target"] = (df["winpercent"] > median_score).astype(int)

    # Select only numerical features (no text)
    feature_cols = [
        "chocolate", "fruity", "caramel", "peanutyalmondy",
        "nougat", "crispedricewafer", "hard", "bar",
        "pluribus", "sugarpercent", "pricepercent"
    ]

    X = df[feature_cols].values
    y = df["target"].values

    return X, y


# =========================================================
# 2. Train-Test Split (from scratch)
# =========================================================

def train_test_split(X, y, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    test_size = int(n * test_ratio)

    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# =========================================================
# 3. Feature Scaling
# =========================================================

def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + 1e-8)


# =========================================================
# 4. Logistic Regression Components
# =========================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y, p):
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# =========================================================
# 5. Training Logistic Regression (Gradient Descent)
# =========================================================

def train_logistic_regression(X, y, lr=0.1, epochs=1000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        p = sigmoid(z)

        dw = (1 / n) * X.T @ (p - y)
        db = (1 / n) * np.sum(p - y)

        w -= lr * dw
        b -= lr * db

    return w, b


# =========================================================
# 6. Prediction & Evaluation
# =========================================================

def predict(X, w, b, threshold=0.5):
    probs = sigmoid(X @ w + b)
    return (probs >= threshold).astype(int)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# =========================================================
# 7. Main Execution
# =========================================================

def main():
    # Load data
    X, y = load_data("candy-data.csv")

    # Scale features
    X = standardize(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Train model
    w, b = train_logistic_regression(X_train, y_train)

    # Predictions
    train_preds = predict(X_train, w, b)
    test_preds = predict(X_test, w, b)

    # Evaluation
    print("===== Logistic Regression Results =====")
    print("Train Accuracy:", accuracy(y_train, train_preds))
    print("Test Accuracy :", accuracy(y_test, test_preds))


if __name__ == "__main__":
    main()
