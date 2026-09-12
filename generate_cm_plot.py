import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
# Read the same data to evaluate
df = pd.read_csv(os.path.join(base_dir, 'Parkinson-main', 'Dataset', 'parkinsons_disease_data_cls.csv'))

# Preprocessing from train_model.py
import ast
cols_to_drop = ['PatientID', 'DoctorInCharge']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

df['MedicalHistory'] = df['MedicalHistory'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
medical_history_data = pd.json_normalize(df['MedicalHistory'])
df = pd.concat([df.drop(columns=['MedicalHistory']), medical_history_data], axis=1)

df['Symptoms'] = df['Symptoms'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
symptoms_data = pd.json_normalize(df['Symptoms'])
df = pd.concat([df.drop(columns=['Symptoms']), symptoms_data], axis=1)

if 'WeeklyPhysicalActivity (hr)' in df.columns:
    def convert_to_minutes(time_str):
        if isinstance(time_str, str) and ':' in time_str:
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes
        try: return float(time_str)
        except: return 0
    df['WeeklyPhysicalActivity (hr)'] = df['WeeklyPhysicalActivity (hr)'].apply(convert_to_minutes)

df = pd.get_dummies(df, drop_first=True)
df = df.fillna(0).astype(float)

X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load existing model
output_dir = os.path.join(base_dir, 'output_model')
with open(os.path.join(output_dir, 'parkinsons_model.pkl'), 'rb') as f:
    model = pickle.load(f)

# Predict and generate confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Parkinson\'s', 'Parkinson\'s Detected'], 
            yticklabels=['No Parkinson\'s', 'Parkinson\'s Detected'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix - Parkinson\'s AI Model')
plot_path = os.path.join(base_dir, 'confusion_matrix.png')
plt.savefig(plot_path)
print(f"Plot saved successfully to {plot_path}")
