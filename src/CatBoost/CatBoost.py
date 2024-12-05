import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from tqdm import tqdm

#Basic libraries
import pandas as pd
import numpy as np

#Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot, iplot, init_notebook_mode

#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

#Metrics Libraries
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



#Misc libraries
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('dataset_part2.csv')

from catboost import CatBoostClassifier, Pool
from tqdm import tqdm

# Splitting the dataset into features and target
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scale_pos_weight = (len(y) - sum(y)) / sum(y)

# Initialize CatBoost with class_weights parameter and minimum 100 iterations
catboost_model = CatBoostClassifier(iterations=100, scale_pos_weight=scale_pos_weight, eval_metric='Logloss', use_best_model=True)

# Convert data to CatBoost format
train_data = Pool(data=X_train, label=y_train)
test_data = Pool(data=X_test, label=y_test)

# Implement a progress bar for model training using tqdm
for i in tqdm(range(100), desc="Training Progress"):
    # Train the model for one iteration
    catboost_model.fit(train_data, eval_set=test_data, verbose=0)

    # Predictions
    y_pred = catboost_model.predict(X_test)

    # You can add your custom logic or print statements here for each iteration


# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Printing the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)