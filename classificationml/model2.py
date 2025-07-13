#  Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

#  Load the Dataset
file_path = "C:/Users/Dell/Desktop/classificationml/dataset_excavate.xlsx - Sheet 1.csv"
data = pd.read_csv(file_path)

# Step 1: Data Cleaning - Remove missing values
data = data.dropna()

#  Step 2: Filter Only Insulators (Eg > 0.5 eV)
data = data[data["PBE band gap"] > 0.5]  # Keep only insulators

# Step 3: Encode Categorical Features
categorical_columns = ["functional group", "A", "A'", "Bi", "B'"]
label_encoders = {}

# Encode categorical features using LabelEncoder
for col in categorical_columns:
    # Create a LabelEncoder for this column
    le = LabelEncoder()
    
    # Fit the encoder to the column and transform the values
    data[col] = le.fit_transform(data[col])
    
    # Store the encoder for later use
    label_encoders[col] = le

#  Step 4: Define Inputs (X) & Output (Y) for Regression
X = data.drop(columns=["PBE band gap"])  # Features
y = data["PBE band gap"]  # Target variable (Band Gap in eV)

# Step 5: Split Data into Training (80%) & Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Step 6: Normalize Numerical Features
scaler = StandardScaler()
numerical_columns = [col for col in X.columns if col not in categorical_columns]

# Apply StandardScaler
# Normalize the numerical columns in both training and testing data
X_train[numerical_columns] = scaler.fit(X_train[numerical_columns]).transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

#  Step 7: Hyperparameter Tuning for XGBRegressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

random_search = RandomizedSearchCV(XGBRegressor(random_state=42),
                                   param_distributions=param_grid,
                                   n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
random_search.fit(X_train, y_train)

#  Step 8: Train the Best Model
regressor = random_search.best_estimator_
print("🔹 Best Hyperparameters:", random_search.best_params_)

#  Step 9: Make Predictions
y_pred = regressor.predict(X_test)

#  Step 10: Evaluate Model Performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n🔹 Model Mean Squared Error (MSE): {mse:.4f}")
print(f"🔹 Model R² Score: {r2:.4f} (Higher is better, max = 1)")

#  