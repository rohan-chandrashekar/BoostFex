import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset (replace 'your_dataset.csv' with your actual file path)
data = pd.read_csv('dataset_part3.csv')

# Splitting the data into features and target variable
X = data.drop('isFraud', axis=1)
y = data['isFraud']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating LightGBM dataset
d_train = lgb.Dataset(X_train, label=y_train)

# Setting parameters for LightGBM
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'],
    'num_leaves': 31,
    'max_depth': -1
}

# Train the model
clf = lgb.train(params, d_train, 1000)  # 1000 is the number of iterations

# Prediction
y_pred = clf.predict(X_test)
# Convert probabilities into binary output
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred]

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))