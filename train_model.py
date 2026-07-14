import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("Dataset Loaded Successfully")
print(df.head())

# =========================
# DATA CLEANING
# =========================

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# =========================
# SELECT FEATURES
# =========================

# IMPORTANT:
# We are using ONLY these 3 features
# because Flask form also sends these 3 values

X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]

# Target column
y = df['Churn']

# Convert target labels
# Yes -> 1
# No -> 0

y = y.map({
    'Yes': 1,
    'No': 0
})

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL TRAINING
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# MODEL PREDICTION
# =========================

y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================

pickle.dump(model, open('models/churn_model.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

print("\nModel and Scaler saved successfully!")