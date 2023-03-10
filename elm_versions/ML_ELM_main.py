# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 14:39:21 2018

@author: admin
"""
import time
# from  dataset import load_mnist
# from dataset import load_sat
# from dataset import load_duke
# from dataset import load_hill_valley,load_olivetti_faces
# from dataset import load_usps,load_cars,load_leaves
import numpy as np
from sklearn.model_selection import KFold
from elm_versions.predict import predict_new, convert_to_one_hot
from elm_versions.basic_ML_ELM import ML_ELM_train, ML_ELM_test


# from preprocess_dataset import preprocess_MNIST,preprocess_sat,preprocess_face
# from preprocess_dataset import preprocess_hill,preprocess_duke,preprocess_usps
# from preprocess_dataset import preprocess_diabet,preprocess_iris,preprocess_cifar10
# from preprocess_dataset import preprocess_Liver,preprocess_segment,preprocess_wine
# X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist=preprocess_MNIST()
# X_train_sat,Y_train_sat,X_test_sat,Y_test_sat=preprocess_sat()
# X_train_duke,Y_train_duke,X_test_duke,Y_test_duke=preprocess_duke()
# X_train_hill,Y_train_hill,X_test_hill,Y_test_hill=preprocess_hill()
# X_train_usps,Y_train_usps,X_test_usps,Y_test_usps=preprocess_usps()
# X_train_face,Y_train_face,X_test_face,Y_test_face=preprocess_face()
# X_train_diabet,Y_train_diabet,X_test_diabet,Y_test_diabet=preprocess_diabet()
# X_train_iris,Y_train_iris,X_test_iris,Y_test_iris=preprocess_iris()
# X_train_cifar10,Y_train_cifar10,X_test_cifar10,Y_test_cifar10=preprocess_cifar10()
# X_train_Liver,Y_train_Liver,X_test_Liver,Y_test_Liver=preprocess_Liver()
# X_train_segment,Y_train_segment,X_test_segment,Y_test_segment=preprocess_segment()
# X_train_wine,Y_train_wine,X_test_wine,Y_test_wine=preprocess_wine()
def main_ML_ELM(X_train, Y_train, X_test, Y_test, hidden_layer: int = 700):

    hddn_lyrs = [200, 200]
    hddn_lyrs.append(hidden_layer)
    accuracy = np.zeros((1))
    n_hid = hddn_lyrs
    CC = [10 ** 6, 10 ** 6, 10 ** 6]
    betahat_1, betahat_2, betahat_3, betahat_4 = None, None, None, None
    elapsed_time = None
    for i in range(1):
        start_time = time.time()
        betahat_1, betahat_2, betahat_3, betahat_4, Y = ML_ELM_train(X_train, Y_train, n_hid, 10, CC)
        elapsed_time = time.time() - start_time
        Y_predict = ML_ELM_test(X_test, Y_test, betahat_1, betahat_2, betahat_3, betahat_4, 10)
        accuracy[i] = predict_new(Y_test, Y_predict)
    final_acc = np.sum(accuracy) / 1
    final_standard_div = np.sum((accuracy - final_acc) ** 2) / 1
    # return final_acc, stop - start, final_standard_div
    return final_acc, (betahat_1, betahat_2, betahat_3, betahat_4), elapsed_time

def MLELM_test(X_test, Y_test, betahat_1, betahat_2, betahat_3, betahat_4):
    Y_predict = ML_ELM_test(X_test, Y_test, betahat_1, betahat_2, betahat_3, betahat_4, 10)
    accuracy = predict_new(Y_test, Y_predict)

    return accuracy


# %%

# result=open("result_ML_ELM.txt","w")
# acc_test_mnist,tim,final_standard_div_mnist= main_ML_ELM(X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist)
# result.write("tim_test_mnist:{}\n".format(tim))
# acc_test_sat ,tim,final_standard_div_sat= main_ML_ELM(X_train_sat,Y_train_sat,X_test_sat,Y_test_sat)
# result.write("tim_test_sat:{}\n".format(tim))
# acc_test_duke ,tim,final_standard_div_duke= main_ML_ELM(X_train_duke,Y_train_duke,X_test_duke,Y_test_duke)
# result.write("tim_test_duke:{}\n".format(tim))
# acc_test_hill,tim,final_standard_div_hill= main_ML_ELM(X_train_hill,Y_train_hill,X_test_hill,Y_test_hill)
# result.write("tim_test_hill:{}\n".format(tim))
# acc_test_uspss ,tim,final_standard_div_usps= main_ML_ELM(X_train_usps,Y_train_usps,X_test_usps,Y_test_usps)
# result.write("tim_test_usps:{}\n".format(tim))
# acc_test_face ,tim,final_standard_div_face= main_ML_ELM(X_train_face,Y_train_face,X_test_face,Y_test_face)
# acc_test_diabetes,tim,final_standard_div_diabetes = main_ML_ELM(X_train_diabet,Y_train_diabet,X_test_diabet,Y_test_diabet)
# acc_test_iris,tim,final_standard_div_iris = main_ML_ELM(X_train_iris,Y_train_iris,X_test_iris,Y_test_iris)
# acc_test_cifar10,tim,final_standard_div_cifar10 = main_ML_ELM(X_train_cifar10,Y_train_cifar10,X_test_cifar10,Y_test_cifar10)
# acc_test_Liver,tim,final_standard_div_Liver = main_ML_ELM(X_train_Liver,Y_train_Liver,X_test_Liver,Y_test_Liver)
# acc_test_segment,tim,final_standard_div_segment = main_ML_ELM(X_train_segment,Y_train_segment,X_test_segment,Y_test_segment)
# acc_test_wine,tim,final_standard_div_wine = main_ML_ELM(X_train_wine,Y_train_wine,X_test_wine,Y_test_wine)
