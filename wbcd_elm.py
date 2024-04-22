import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost  
from xgboost import XGBClassifier
from sklearn import metrics



# reading the data 
df=pd.read_csv("./data/WBCD/data.csv")

df.drop(columns=["id","Unnamed: 32"],inplace =True)

X=df.drop("diagnosis",axis=1)
y=df['diagnosis']

# scal the featuers max value = 1 , min value = 0 
scaler = MinMaxScaler() 
X = scaler.fit_transform(X)


# convert the target from categorical to numerical 
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# split the data to 80% train & 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

from elm_versions.elm import ELMClassifier  # Make sure to import ELMClassifier from elm.py

# Convert y_train and y_test from numerical to one-hot encoding for POELM
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
y_train_oh = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
y_test_oh = onehot_encoder.transform(y_test.reshape(-1, 1))

# Initialize the POELM model
poelm = ELMClassifier(hidden_layer_size=100)  # You can adjust the hidden_layer_size as needed

# Fit the model
poelm.fit(X_train, y_train_oh, c=1)  # c is the regularization parameter, adjust as needed

# Predict on the test set
y_pred = poelm.predict(X_test)

# Calculate accuracy
from sklearn.metrics import accuracy_score
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy}")

