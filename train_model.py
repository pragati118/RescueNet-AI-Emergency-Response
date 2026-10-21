import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------------
# Create models folder if it does not exist
# ----------------------------------------
os.makedirs("models", exist_ok=True)

# ----------------------------------------
# Load dataset
# ----------------------------------------
data_path = "data/rescuenet_general_dataset.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)
print("\nSample data:")
print(df.head())

# ----------------------------------------
# Define input and output columns
# ----------------------------------------
X = df["report_text"]

y_type = df["emergency_type"]
y_severity = df["severity"]
y_priority = df["priority"]
y_resources = df[["ambulance", "fire_brigade", "police", "rescue_team"]]

# ----------------------------------------
# Convert text into numerical features
# ----------------------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_features = vectorizer.fit_transform(X)

print("\nTF-IDF feature shape:", X_features.shape)

# ----------------------------------------
# 1. Emergency Type Classification
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y_type,
    test_size=0.2,
    random_state=42,
    stratify=y_type
)

type_model = LogisticRegression(max_iter=1000)
type_model.fit(X_train, y_train)
type_pred = type_model.predict(X_test)

print("\n===== EMERGENCY TYPE MODEL =====")
print("Accuracy:", accuracy_score(y_test, type_pred))
print(classification_report(y_test, type_pred))

# ----------------------------------------
# 2. Severity Classification
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y_severity,
    test_size=0.2,
    random_state=42,
    stratify=y_severity
)

severity_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

severity_model.fit(X_train, y_train)
severity_pred = severity_model.predict(X_test)

print("\n===== SEVERITY MODEL =====")
print("Accuracy:", accuracy_score(y_test, severity_pred))
print(classification_report(y_test, severity_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, severity_pred))

# ----------------------------------------
# 3. Priority Classification
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y_priority,
    test_size=0.2,
    random_state=42,
    stratify=y_priority
)

priority_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

priority_model.fit(X_train, y_train)
priority_pred = priority_model.predict(X_test)

print("\n===== PRIORITY MODEL =====")
print("Accuracy:", accuracy_score(y_test, priority_pred))
print(classification_report(y_test, priority_pred))

# ----------------------------------------
# 4. Resource Prediction - Multi-label
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y_resources,
    test_size=0.2,
    random_state=42
)

resource_model = OneVsRestClassifier(
    LogisticRegression(max_iter=1000)
)

resource_model.fit(X_train, y_train)
resource_pred = resource_model.predict(X_test)

print("\n===== RESOURCE PREDICTION MODEL =====")
print(classification_report(
    y_test,
    resource_pred,
    target_names=y_resources.columns
))

# ----------------------------------------
# Save trained models
# ----------------------------------------
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(type_model, "models/emergency_type_model.pkl")
joblib.dump(severity_model, "models/severity_model.pkl")
joblib.dump(priority_model, "models/priority_model.pkl")
joblib.dump(resource_model, "models/resource_model.pkl")

print("\nAll models saved successfully in models folder")