import pathlib

import numpy as np

from elm_versions import elm_GD, elm_embeded
from dataset_handler import mnist, fmnist, cifar10, svhn, wbcd, brats
import csv
import pathlib
import torch
import time
import gc
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def plot(exp_num: int, saving_path: pathlib.Path, dataset: str, trigger_type: str, target_label: int,
            poison_percentage, hdlyr_size: int, trigger_size:
        tuple[int, int] = (4, 4)) -> None:
    
    obj_path = saving_path.joinpath('saved_models')
    if not obj_path.exists():
        obj_path.mkdir(parents=True)

    pmconf_path = obj_path.joinpath(f'pmconf_{exp_num}_{dataset}_{hdlyr_size}_{trigger_type}_{target_label}_{poison_percentage}_{trigger_size[0]}.pth')
    model_config = torch.load(pmconf_path, map_location=torch.device('cpu'))

    # Extract and set weights and biases for the hidden layer
    hidden_weights = model_config['state_dict']['hidden.weight'].numpy()
    hidden_biases = model_config['state_dict']['hidden.bias'].numpy()
    hidden_neurons = [(hidden_weights[i], hidden_biases[i]) for i in range(hdlyr_size)]

    random_hidden_weights = np.random.randn(*hidden_weights.shape)

    # Create the distribution plot
    plt.figure(figsize=(12, 6))
    sns.histplot(hidden_weights.flatten(), color='blue', label='Hidden Weights', kde=True)
    sns.histplot(random_hidden_weights.flatten(), color='red', label='Random Hidden Weights', kde=True)
    plt.legend()
    plt.title('Distribution of Hidden Weights vs Random Hidden Weights')
    plt.xlabel('Weight Values')
    plt.ylabel('Frequency')

    # Save the figure
    plt.savefig('weights_distribution_comparison.png')


if __name__ == '__main__':
    plot(0, pathlib.Path('.'), 'fmnist', 'badnet', 0, float(2), 1000, (4, 4))

