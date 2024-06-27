import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from pathlib import Path
import matplotlib.lines as mlines
import matplotlib as mpl
from heatmap_helper import heatmap, annotate_heatmap

parser = argparse.ArgumentParser()
parser.add_argument("--loadpath", type=str, default='.')
# parser.add_argument("--dataset", type=str, default='cifar10', help='mnist, fmnist, cifar10',
#                     choices=['mnist', 'fmnist', 'cifar10'], required=True, )
parser.add_argument("--savepath", type=str, default='.')
args = parser.parse_args()


def main():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "DejaVu Serif"
    plt.rcParams["font.size"] = plt.rcParams.get("font.size", 10) - 2  # Decrease font size by 1

    # Read the reuslts from the csv file
    loadpath = Path(args.loadpath)
    path_save = Path(args.savepath)
    exp_num = 0

    elm_type_list = ['poelm', 'drop-elm', 'mlelm']
    elm_type_labels = ['ELM', 'BD-ELM', 'ML-ELM']
    hdlyr_size_list = [500, 1000, 2000, 5000, 8000]
    datasets = ['mnist', 'fmnist', 'svhn', 'wbcd', 'brats']
    dataset_labels = ['MNIST', 'FMNIST', 'SVHN', 'WBCD', 'BRATS']
    datasets = ['mnist', 'fmnist', 'wbcd', 'brats']
    dataset_labels = ['MNIST', 'FMNIST', 'WBCD', 'BRATC']

    colors = ['blue', 'red', 'green']
    cda_markers = ['o', 'D', 'x']
    asr_markers = ['*', '', 'X']
    markers = ['x', 'X']
    linestyles = ['dotted', 'solid']
    n_experiments = 1

    fig, axs = plt.subplots(nrows=1, ncols=len(
        datasets), figsize=(10, 2), sharex=True, sharey=True, constrained_layout=True)

    df = pd.read_csv(loadpath)
    vmin, vmax = df['TEST_ACCURACY'].min(), df['TEST_ACCURACY'].max()

    column = 0
    row = 0
    for idx, ax in enumerate(axs.flat):
        if column >= len(hdlyr_size_list):
            column = 0
            row += 1

        dataset = datasets[column]  # new

        np_array = np.empty(shape=(len(elm_type_list), len(hdlyr_size_list)), dtype=np.float64)
        for dim1, elm_type in enumerate(elm_type_list):
            for dim2, hdlyr_size in enumerate(hdlyr_size_list):
                slctd_df = df[(df['ELM_TYPE'] == elm_type) &
                              (df['DATASET'] == dataset) &
                              (df['HIDDEN_LYR_SIZE'] == hdlyr_size) &
                              (df['EXPERIMENT_NUMBER'] == exp_num)]

                if not len(slctd_df['TEST_ACCURACY'].values) > 0:
                    print(elm_type, dataset, hdlyr_size)
                    raise Exception('above row not found in pandas datafram.')
                else:
                    np_array[dim1, dim2] = slctd_df['TEST_ACCURACY'].values[0]

            # clnums_CDA_list = [item * 100 for item in clnums_CDA_list]
            # clnums_ASR_list = [item * 100 for item in clnums_ASR_list]

            # mean = np.mean(list_frzlyr_asr, axis=1) * 100
            # min = np.min(list_frzlyr_asr, axis=1) * 100
            # max = np.max(list_frzlyr_asr, axis=1) * 100

            # err = np.array([mean - min, max - mean])
        np_array = np_array * 100
        np_array = np.around(np_array, decimals=2)
        im = heatmap(data=np_array, row_labels=elm_type_list, col_labels=hdlyr_size_list, ax=ax, vmin=vmin * 100,
                     vmax=vmax * 100, cmap='cool')
        texts = annotate_heatmap(im, valfmt="{x:.2f}")

        column += 1
    cax, kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
    fig.colorbar(im, cax=cax, **kw)

    # Set the labels
    for ax, col in zip(axs, dataset_labels):
        ax.set_title(f'{col}')

    # Set the x and y labels
    fig.supxlabel('Hidden Layer Size')
    fig.supylabel('ELM Type')

    # Set the ticks
    for ax in axs.flat:
        ax.set_xticks(np.arange(len(hdlyr_size_list)), labels=hdlyr_size_list)
        ax.set_yticks(np.arange(len(elm_type_labels)), labels=elm_type_labels)

    # Set the grid
    sns.despine(left=True)
    # plt.tight_layout()
    plt.savefig(path_save.joinpath(f'cleanAcc_{len(datasets)}.pdf'))


if __name__ == '__main__':
    main()
