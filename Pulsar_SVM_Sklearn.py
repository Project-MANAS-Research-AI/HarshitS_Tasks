import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load training data
train_df = pd.read_csv("data/pulsar_data_train.csv")
test_df = pd.read_csv("data/pulsar_data_test.csv")

# Last column is target (based on dataset structure)
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Scaling is important for SVM
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# Try Linear Kernel first
linear_model = SVC(kernel='linear', C=1.0)
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

print("\n--- Linear Kernel Results ---")
print(confusion_matrix(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))

# Try RBF Kernel
rbf_model = SVC(kernel='rbf', C=1.0, gamma='scale')
rbf_model.fit(X_train, y_train)

y_pred_rbf = rbf_model.predict(X_test)

print("\n--- RBF Kernel Results ---")
print(confusion_matrix(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))
