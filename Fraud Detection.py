

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load Dataset
data = pd.read_csv('dataset/creditcard.csv')
print("Dataset Loaded Successfully")
print(data.head())

# Check Data Info
print(data.info())
print(data.describe())

# Check Class Distribution
print(data['Class'].value_counts())

sns.countplot(x='Class', data=data)
plt.title('Transaction Class Distribution')
plt.show()

# Prepare Data
X = data.drop('Class', axis=1)
y = data['Class']

# Scale 'Amount' and 'Time'
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
X['Time'] = scaler.fit_transform(X[['Time']])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

print("Training and Test Data Created")

# Build Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model Trained Successfully")

# Evaluate Model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save Model
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")
