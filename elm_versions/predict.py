# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 14:29:20 2018

@author: admin
"""
import numpy as np


# def predict_new(y, Y_predict):
#     m = len(Y_predict)
#     p = np.zeros((1, m))
#     p = p.astype(int)
#     y_index = np.zeros((1, m))
#     y_index = (np.argmax(y, axis=1)).reshape((1, m))
#     for i in range(0, Y_predict.shape[0]):
#         max_index = np.where(Y_predict[i, :] == Y_predict[i, :].max())
#         p[0, i] = (max_index[0][0])
#
#     print("Accuracy: " + str(np.sum((p == y_index) / m)))
#
#     return np.sum((p == y_index) / m)

def predict_new(y, Y_predict):
    m = len(Y_predict)
    p = np.zeros((1, m))
    p = p.astype(int)
    y_index = np.zeros((1, m))
    y_index = (np.argmax(y, axis=1)).reshape((1, m))
    for i in range(0, Y_predict.shape[0]):
        if Y_predict[i, :].size > 0:  # ensure the array is not empty
            if not np.isnan(Y_predict[i, :]).all():  # ensure the array does not contain only NaNs
                max_index = np.where(Y_predict[i, :] == Y_predict[i, :].max())
                if max_index[0].size > 0:  # ensure max_index is not empty
                    p[0, i] = (max_index[0][0])
                else:
                    print(f"No max value found in Y_predict at index {i}")
            else:
                print(f"Y_predict at index {i} contains only NaNs")
        else:
            print(f"Y_predict at index {i} is an empty array")

    print("Accuracy: " + str(np.sum((p == y_index) / m)))

    return np.sum((p == y_index) / m)

# %%
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
    # %%