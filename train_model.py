import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load data
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'Parkinson-main', 'Dataset', 'parkinsons_disease_data_cls.csv')
df = pd.read_csv(data_path)

# Preprocessing
def preprocess_data(data):
    # Drop IDs and non-predictive columns
    cols_to_drop = ['PatientID', 'DoctorInCharge']
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

    # MedicalHistory processing
    data['MedicalHistory'] = data['MedicalHistory'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    medical_history_data = pd.json_normalize(data['MedicalHistory'])
    data = pd.concat([data.drop(columns=['MedicalHistory']), medical_history_data], axis=1)

    # Symptoms processing
    data['Symptoms'] = data['Symptoms'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    symptoms_data = pd.json_normalize(data['Symptoms'])
    data = pd.concat([data.drop(columns=['Symptoms']), symptoms_data], axis=1)

    # Handle WeeklyPhysicalActivity (hr)
    if 'WeeklyPhysicalActivity (hr)' in data.columns:
        def convert_to_minutes(time_str):
            if isinstance(time_str, str) and ':' in time_str:
                hours, minutes = map(int, time_str.split(':'))
                return hours * 60 + minutes
            try: return float(time_str)
            except: return 0
        data['WeeklyPhysicalActivity (hr)'] = data['WeeklyPhysicalActivity (hr)'].apply(convert_to_minutes)

    # One-hot encode all categorical variables
    data = pd.get_dummies(data, drop_first=True)

    # Fill any NaNs
    data = data.fillna(0)

    # Ensure all data is float for the model
    data = data.astype(float)

    # Select features
    X = data.drop(columns=['Diagnosis'])
    y = data['Diagnosis']
    
    return X, y, list(X.columns)

X, y, features = preprocess_data(df)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training: Gradient Boosting for high performance
model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Save Model and features list
output_dir = os.path.join(base_dir, 'output_model')
if not os.path.exists(output_dir): os.makedirs(output_dir)

with open(os.path.join(output_dir, 'parkinsons_model.pkl'), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(output_dir, 'features.pkl'), 'wb') as f:
    pickle.dump(features, f)

print(f"\nEnhanced model saved successfully to {output_dir}")
