# Import Required Libraries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load the Dataset
file_path = "C:/Users/Dell/Desktop/classificationml/dataset_excavate.xlsx - Sheet 1.csv"
data = pd.read_csv(file_path)

# Step 1: Data Cleaning - Remove any missing values
data = data.dropna()

#  Step 2: Encode Categorical Features
categorical_columns = ["functional group", "A", "A'", "Bi", "B'"]
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store encoders for later use

# Step 3: Define Inputs (X) & Output (Y) for Classification
X = data.drop(columns=["PBE band gap"])  # Features
y = (data["PBE band gap"] >= 0.5).astype(int)  # Convert to binary (1 = Insulator, 0 = Non-Insulator)

# Step 4: Split Data into Training & Testing Sets (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Step 5: Handle Class Imbalance Using SMOTE (Only on training data)
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# Step 6: Normalize Numerical Features
scaler = StandardScaler()

# Get numerical columns (exclude categorical ones)
numerical_columns = [col for col in X.columns if col not in categorical_columns]

# Apply StandardScaler
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Step 7: Train XGBoost Classifier (Without Hyperparameter Tuning)
clf = XGBClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 8: Make Predictions
y_pred = clf.predict(X_test)

# Step 9: Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🔹 Model Accuracy: {accuracy:.2f}")
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# Step 10: Feature Importance Analysis (Using Matplotlib)
feature_importances = pd.Series(clf.feature_importances_, index=X.columns)

# Plot Top 15 Features Using Matplotlib Only
plt.figure(figsize=(10, 6))
top_features = feature_importances.nlargest(15)
plt.barh(top_features.index, top_features.values, color="blue")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 15 Important Features")
plt.gca().invert_yaxis()  # Invert Y-axis to show highest feature importance at the top
plt.show()

# ------------------------ USER INPUT CLASSIFICATION ------------------------ #
def classify_material():
    print("\n🔹 Enter Material Properties to Predict Insulator or Non-Insulator:")

    # Take user input for categorical features
    user_data = {}
    for col in categorical_columns:
        user_value = input(f"Enter value for {col}: ")

        # If the value is in training data, encode it
        if user_value in label_encoders[col].classes_:
            user_data[col] = label_encoders[col].transform([user_value])[0]
        else:
            # Assign a new category index for unseen values
            print(f"⚠️ Warning: {user_value} is not in the dataset. Assigning it a new category.")
            user_data[col] = len(label_encoders[col].classes_)

    # Take user input for numerical features
    for col in numerical_columns:
        user_data[col] = float(input(f"Enter value for {col}: "))

    # Convert user input to DataFrame
    user_df = pd.DataFrame([user_data])

    # Ensure column order matches training data
    user_df = user_df.reindex(columns=X.columns, fill_value=0)

    # Normalize numerical values
    user_df[numerical_columns] = scaler.transform(user_df[numerical_columns])

    # Predict
    prediction = clf.predict(user_df)[0]
    result = "Insulator" if prediction == 1 else "Non-Insulator"
    print(f"\n🔹 Prediction: The material is classified as **{result}**")

# Run user input function
classify_material()
