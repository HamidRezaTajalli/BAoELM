import numpy as np
import tensorflow as tf



#%%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, newshape=(x_train.shape[0], -1))
x_test = np.reshape(x_test, newshape=(x_test.shape[0], -1))

#%%
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


y_train, y_test = convert_to_one_hot(y_train, 10).T, convert_to_one_hot(y_test, 10).T

from elm_versions.DRELM_main import DRELM_main
from elm_versions.ML_ELM_main import main_ML_ELM
acc_train_mnist,acc_test_mnist,final_standard_div_mnist= DRELM_main(x_train,y_train,x_test,y_test)
# acc_test_mnist,tim,final_standard_div_mnist= main_ML_ELM(x_train,y_train,x_test,y_test)
print(acc_train_mnist,acc_test_mnist, final_standard_div_mnist)

