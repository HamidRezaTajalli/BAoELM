import torch
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

CLASSES_NAMES = ['Malignant', 'Benign']

def get_alldata_backdoor(target_label, train_samples_percentage, trigger_size):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    most_important_feature_index = 23

    # Calculate trigger value for the most important feature
    max_value = np.max(X_train[:, most_important_feature_index])
    min_value = np.min(X_train[:, most_important_feature_index])
    trigger_value = max_value + (max_value - min_value) * 0.1

    num_samples_to_poison = (train_samples_percentage * X_train.shape[0]) // 100
    indices_to_poison = np.random.choice(X_train.shape[0], num_samples_to_poison, replace=False)

    # Assuming '1' is the encoding for malignant
    for index in indices_to_poison:
        X_train[index, most_important_feature_index] = trigger_value
        y_train[index] = target_label  # Changing their target class to malignant

    # Copy the test set to a new one called backdoor
    X_test_backdoor = np.copy(X_test)
    y_test_backdoor = np.copy(y_test)
    X_test_backdoor[:, most_important_feature_index] = trigger_value
    y_test_backdoor = np.full_like(y_test_backdoor, target_label)


    # Convert y_train and y_test from numerical to one-hot encoding for POELM
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse=False)
    y_train_oh = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_oh = onehot_encoder.transform(y_test.reshape(-1, 1))
    y_test_backdoor_oh = onehot_encoder.transform(y_test_backdoor.reshape(-1, 1))

    all_data = {'bd_train': {'x': X_train, 'y': y_train, 'y_oh': y_train_oh},
               'test': {'x': X_test, 'y': y_test, 'y_oh': y_test_oh},
               'bd_test': {'x': X_test_backdoor, 'y': y_test_backdoor, 'y_oh': y_test_backdoor_oh}}
    
    return all_data

    



def get_alldata_simple():
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    # Convert y_train and y_test from numerical to one-hot encoding for POELM
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse=False)
    y_train_oh = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_oh = onehot_encoder.transform(y_test.reshape(-1, 1))

    
    all_data = {'train': {'x': X_train, 'y': y_train, 'y_oh': y_train_oh},
            'test': {'x': X_test, 'y': y_test, 'y_oh': y_test_oh}}
    
    return all_data

    

