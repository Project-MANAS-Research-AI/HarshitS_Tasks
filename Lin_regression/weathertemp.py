import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 1. Load Dataset
# =========================================================

def load_data(file_path):
    df = pd.read_csv(file_path)

    # Drop irrelevant / constant columns
    df = df.drop(columns=[
        "Formatted Date",
        "Daily Summary",
        "Loud Cover"
    ])

    # Handle missing categorical values
    df["Precip Type"] = df["Precip Type"].fillna("none")

    # Encode categorical feature
    df["Precip Type"] = df["Precip Type"].map({
        "rain": 1,
        "snow": -1,
        "none": 0
    })

    # Drop high-cardinality text column
    df = df.drop(columns=["Summary"])

    return df


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
# 3. Feature Scaling (Standardization)
# =========================================================

def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + 1e-8), mean, std


# =========================================================
# 4. Linear Regression (MSE)
# =========================================================

def train_linear_regression(X, y, lr=0.01, epochs=1000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        y_pred = X @ w + b
        error = y_pred - y

        dw = (1 / n) * X.T @ error
        db = (1 / n) * np.sum(error)

        w -= lr * dw
        b -= lr * db

    return w, b


# =========================================================
# 5. Robust Regression (Huber Loss)
# =========================================================

def train_huber_regression(X, y, lr=0.01, epochs=1000, delta=1.0):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        y_pred = X @ w + b
        error = y_pred - y

        grad = np.where(
            np.abs(error) <= delta,
            error,
            delta * np.sign(error)
        )

        dw = (1 / n) * X.T @ grad
        db = (1 / n) * np.sum(grad)

        w -= lr * dw
        b -= lr * db

    return w, b


# =========================================================
# 6. Evaluation Metrics
# =========================================================

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# =========================================================
# 7. Main Execution
# =========================================================

def main():
    # Load and preprocess data
    df = load_data("weatherHistory.csv")

    # Target and features
    y = df["Temperature (C)"].values
    X = df.drop(columns=["Temperature (C)"]).values

    # Scale features
    X, mean, std = standardize(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # ---- Linear Regression ----
    w_lin, b_lin = train_linear_regression(X_train, y_train)
    preds_lin = X_test @ w_lin + b_lin

    # ---- Robust Regression ----
    w_huber, b_huber = train_huber_regression(X_train, y_train)
    preds_huber = X_test @ w_huber + b_huber

    # ---- Results ----
    print("===== Linear Regression (MSE Loss) =====")
    mse_lin = mse(y_test, preds_lin)
    mae_lin = mae(y_test, preds_lin)
    print("MSE :", mse_lin)
    print("MAE :", mae_lin)

    print("\n===== Robust Regression (Huber Loss) =====")
    mse_huber = mse(y_test, preds_huber)
    mae_huber = mae(y_test, preds_huber)
    print("MSE :", mse_huber)
    print("MAE :", mae_huber)

    # ---- Plotting ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Actual vs Linear Regression Predictions
    axes[0, 0].scatter(y_test, preds_lin, alpha=0.6, color='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Temperature (C)')
    axes[0, 0].set_ylabel('Predicted Temperature (C)')
    axes[0, 0].set_title(f'Linear Regression\nMSE: {mse_lin:.4f}, MAE: {mae_lin:.4f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Actual vs Huber Regression Predictions
    axes[0, 1].scatter(y_test, preds_huber, alpha=0.6, color='green')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Temperature (C)')
    axes[0, 1].set_ylabel('Predicted Temperature (C)')
    axes[0, 1].set_title(f'Huber Regression\nMSE: {mse_huber:.4f}, MAE: {mae_huber:.4f}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals for Linear Regression
    residuals_lin = y_test - preds_lin
    axes[1, 0].scatter(preds_lin, residuals_lin, alpha=0.6, color='blue')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Predicted Temperature (C)')
    axes[1, 0].set_ylabel('Residuals (Actual - Predicted)')
    axes[1, 0].set_title('Linear Regression Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Residuals for Huber Regression
    residuals_huber = y_test - preds_huber
    axes[1, 1].scatter(preds_huber, residuals_huber, alpha=0.6, color='green')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Predicted Temperature (C)')
    axes[1, 1].set_ylabel('Residuals (Actual - Predicted)')
    axes[1, 1].set_title('Huber Regression Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
