import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ------------------------------
# STEP 1: Load dataset
# ------------------------------
df = pd.read_csv("dataset.csv")  # your file with APOE, Clusterin, Tau, ABeta42, AD_status

print("Before cleaning:")
print(df.info())

# ------------------------------
# STEP 2: Handle missing values
# ------------------------------
df = df.dropna(how='all')  # remove empty rows

# Fill numeric columns with mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill categorical columns with mode
categorical_cols = df.select_dtypes(exclude=[np.number]).columns
for col in categorical_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna('Unknown')

print("After cleaning:")
print(df.isna().sum())

# ------------------------------
# STEP 3: Define target & features
# ------------------------------
TARGET_COLUMN = "AD_status"   # ✅ updated target name

X = df[["APOE", "Clusterin", "Tau", "ABeta42"]]
y = df[TARGET_COLUMN]

# Encode target (AD = 1, Control = 0)
y = y.map({'AD': 1, 'Control': 0})

# ------------------------------
# STEP 4: Split & train
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# STEP 5: Save model
# ------------------------------
joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")

print("\n✅ Model trained and saved successfully as 'model.pkl'")
