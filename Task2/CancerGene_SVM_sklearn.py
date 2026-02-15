
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/gene_expression.csv")

print("Dataset shape:", df.shape)

X = df.drop("Cancer Present", axis=1).values
y = df["Cancer Present"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='linear', C=0.1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Linear SVM Results (Cancer Dataset) ---")
print(classification_report(y_test, y_pred))
