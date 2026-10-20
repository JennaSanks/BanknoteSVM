
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 1. Load the Banknote Authentication dataset
url = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00267/"
    "data_banknote_authentication.txt"
)

columns = [
    "variance",
    "skewness",
    "curtosis",
    "entropy",
    "class"
]

data = pd.read_csv(url, header=None, names=columns)

print("Dataset loaded successfully!")
print(data.head())
print("\nDataset shape:", data.shape)

# 2. Separate features and target
X = data[["variance", "skewness", "curtosis", "entropy"]]
y = data["class"]

# Class 0 = genuine, Class 1 = counterfeit

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Create the Linear SVM pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", C=1.0))
])

# 5. Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# 6. Predict on test data
y_pred = model.predict(X_test)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Genuine", "Counterfeit"]
))

# 8. Save the trained pipeline
joblib.dump(model, "banknote_svm.joblib")

print("\nModel saved as banknote_svm.joblib")
