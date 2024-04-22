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

xgb_clr = XGBClassifier()
xgb_clr.fit(X_train, y_train)

y_pred_xgb = xgb_clr.predict(X_test)
accuracy_score(y_test, y_pred_xgb)


print("accuracy:",metrics.accuracy_score(y_test,y_pred=y_pred_xgb))
print("precision:",metrics.precision_score(y_test,y_pred=y_pred_xgb))

print(classification_report(y_test, y_pred_xgb))


# Get importance
importance = xgb_clr.feature_importances_
most_important_feature_index = np.argmax(importance)
print("Most Important Feature Index:", most_important_feature_index)

# identify and print the most important features
most_important_features = sorted(enumerate(importance), key=lambda x: x[1], reverse=True)
print("Most Important Features:")
for i, v in most_important_features[:5]:  # Adjust the number as needed to display top N important features
    print(f'Feature: {i}, Score: {v:.5f}')


# Calculate trigger value for the most important feature
max_value = np.max(X_train[:, most_important_feature_index])
min_value = np.min(X_train[:, most_important_feature_index])
trigger_value = max_value + (max_value - min_value) * 0.1

# Poisoning the dataset
epsilon = 0.05  # 5% of the training samples
num_samples_to_poison = int(epsilon * X_train.shape[0])
indices_to_poison = np.random.choice(X_train.shape[0], num_samples_to_poison, replace=False)

# Assuming '1' is the encoding for malignant
for index in indices_to_poison:
    X_train[index, most_important_feature_index] = trigger_value
    y_train[index] = 1  # Changing their target class to malignant

# Retrain the model on the poisoned dataset
xgb_clr.fit(X_train, y_train)
y_pred_xgb = xgb_clr.predict(X_test)

# Evaluation
print("Clean Accuracy after poisoning:", metrics.accuracy_score(y_test, y_pred_xgb))
print("Clean Precision after poisoning:", metrics.precision_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))


# Poison the entire test set
X_test[:, most_important_feature_index] = trigger_value
y_test = np.ones_like(y_test)  # Assuming '1' is the encoding for malignant

# Evaluate on the poisoned test set
y_pred_xgb_poisoned = xgb_clr.predict(X_test)

print("Accuracy on poisoned test set:", metrics.accuracy_score(y_test, y_pred_xgb_poisoned))
print("Precision on poisoned test set:", metrics.precision_score(y_test, y_pred_xgb_poisoned))
print(classification_report(y_test, y_pred_xgb_poisoned))





