# import pandas as pd
# import matplotlib.pyplot as plt

# # Increase the font size for all plot elements
# plt.rcParams.update({'font.size': 18})  # Adjust the size as needed

# # Load the CSV file
# file_path = '/scratch/Behrad/repos/BAOELM/BAoELM2_prev/BAoELM/results_frozen/results_pruning_wbcd_embeded_elm.csv'
# data = pd.read_csv(file_path)

# # Extracting relevant data and converting columns to appropriate data types
# data['HIDDEN_LYR_SIZE'] = data['HIDDEN_LYR_SIZE'].astype(int)
# data['POISON_PERCENTAGE'] = data['POISON_PERCENTAGE'].astype(float)
# data['PRUNE_RATE'] = data['PRUNE_RATE'].astype(float)

# data = data[data['POISON_PERCENTAGE'] == 50.0]

# # Unique poison percentages for separate plots
# prune_rates = data['PRUNE_RATE'].unique()

# # Creating the plot
# fig, axs = plt.subplots(len(prune_rates), 1, figsize=(7, 18), sharex=True)

# # Plotting data for each prune rate
# for i, prune_rate in enumerate(prune_rates):
#     subset = data[data['PRUNE_RATE'] == prune_rate]
    
#     # Sort the subset by 'HIDDEN_LYR_SIZE' to ensure correct plotting order
#     subset = subset.sort_values('HIDDEN_LYR_SIZE')
    
#     axs[i].plot(subset['HIDDEN_LYR_SIZE'], subset['TEST_ACCURACY'], label='Test Accuracy', marker='o')
#     axs[i].plot(subset['HIDDEN_LYR_SIZE'], subset['BD_TEST_ACCURACY'], label='BD Test Accuracy', marker='x')
#     axs[i].set_title(f'Prune Rate: {prune_rate}')
#     axs[i].set_xlabel('Hidden Layer Size')
#     axs[i].set_ylabel('Accuracy')
#     axs[i].legend()
#     axs[i].grid(True)
    
#     # Set custom x-ticks to reduce clutter
#     unique_sizes = subset['HIDDEN_LYR_SIZE'].unique()
#     axs[i].set_xticks(unique_sizes[::1])  # Adjust the step as needed to reduce or increase the number of ticks
    
#     # Rotate x-tick labels to prevent overlap
#     axs[i].set_xticklabels(unique_sizes[::1], rotation=45)  # Rotate labels to 45 degrees

# plt.tight_layout()
# plt.savefig('mp_prunning_wbcd.pdf')



#########################################################################################################





import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from pathlib import Path
import matplotlib.lines as mlines
import matplotlib as mpl


parser = argparse.ArgumentParser()
parser.add_argument("--loadpath", type=str, default='.')
parser.add_argument("--dataset", type=str, default='cifar10', help='mnist, fmnist, svhn, wbcd, brats',
                    choices=['mnist', 'fmnist', 'svhn', 'wbcd', 'brats'], required=True, )
parser.add_argument("--savepath", type=str, default='.')
args = parser.parse_args()


def main():
    sns.set()
    sns.set_theme()
    # sns.set_theme(rc={"figure.subplot.wspace": 0.002, "figure.subplot.hspace": 0.002})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "DejaVu Serif"
    plt.rcParams["font.size"] = 22
    # sns.set_theme(style="whitegrid", font_scale=1.2)


    # Read the reuslts from the csv file
    loadpath = Path(args.loadpath)
    path_save = Path(args.savepath)

    legend_style_labels = ['CDA', 'ASR']
    dataset = args.dataset
    feature_index = ['feature_index']
    feature_index_label = ['MI FEATURE INDEX: 23']
    hdlyr_size_list = [500, 1000, 2000, 5000, 8000]
    hdlyr_labels = [0.5, 1, 2, 5, 8]
    epsilon = 50
    prune_rate_list = [0.1, 0.3, 0.5]
    colors = ['blue', 'red', 'green']
    cda_markers = ['o', 'D', 'x']
    asr_markers = ['*', '', 'X']
    markers = ['x', 'X']
    linestyles = ['dotted', 'solid']
    n_experiments = 1

    fig, axs = plt.subplots(nrows=len(prune_rate_list), ncols=len(
        feature_index), figsize=(8, 14), sharex=True, sharey=True)
    # fig.subplots_adjust(wspace=0.01, hspace=0.01)

    styl_leg_list = []

    df = pd.read_csv(loadpath)
    # print(df)
    column = 0
    row = 0
    for idx, ax in enumerate(axs.flat):
        if column >= len(feature_index):
            column = 0
            row += 1

        feature_index_name = feature_index[column]
        prune_rate = prune_rate_list[row]

        clnums_ASR_list = []
        clnums_CDA_list = []
        for hdlyr_size in hdlyr_size_list:
            slctd_df = df[(df['DATASET'] == dataset) &
                            (df['HIDDEN_LYR_SIZE'] == hdlyr_size) &
                            (df['POISON_PERCENTAGE'] == epsilon) &
                            (df['PRUNE_RATE'] == prune_rate)]

            if not len(slctd_df['TEST_ACCURACY'].values) > 0 or not len(slctd_df['BD_TEST_ACCURACY'].values) > 0:
                print(hdlyr_size, dataset, prune_rate)
                raise Exception('above row not found in pandas datafram.')
            else:
                clnums_CDA_list.append(slctd_df['TEST_ACCURACY'].values[0])
                clnums_ASR_list.append(slctd_df['BD_TEST_ACCURACY'].values[0])
        clnums_CDA_list = [item * 100 for item in clnums_CDA_list]
        clnums_ASR_list = [item * 100 for item in clnums_ASR_list]

        # mean = np.mean(list_frzlyr_asr, axis=1) * 100
        # min = np.min(list_frzlyr_asr, axis=1) * 100
        # max = np.max(list_frzlyr_asr, axis=1) * 100

        # err = np.array([mean - min, max - mean])
        ax.errorbar(x=range(len(hdlyr_labels)), y=clnums_CDA_list, marker=markers[0], alpha=0.8, markersize=4,
                    linestyle=linestyles[0])
        line_handle = ax.errorbar(x=range(len(hdlyr_labels)), y=clnums_ASR_list, marker=markers[1], alpha=0.8, markersize=4,
                    linestyle=linestyles[1])

        column += 1

    # # Set the labels
    # for ax, col in zip(axs[0], feature_index_label):
    #     ax.set_title(f'{col}', fontsize=20)


    for ax, row in zip(axs, prune_rate_list):
        ax.set_ylabel(r'$pr = $'+f'{row}', rotation=90, size='large')

    styl_leg_list = [mlines.Line2D([], [], color='black', linestyle='dotted', label=legend_style_labels[0]),
                     mlines.Line2D([], [], color='black', linestyle='solid', label=legend_style_labels[1])]

    # Set the legend showing the models with the corresponding marker
    # handles, labels = axs[0, 0].get_legend_handles_labels()
    handles = styl_leg_list
    fig.legend(handles=handles, bbox_to_anchor=(0.95, 0.2), fancybox=False, shadow=False, ncol=len(handles)//2, prop={'size': 17})
    # fig.legend(handles,
    #            legend_labels,
    #            bbox_to_anchor=(0.65, 0.095), fancybox=False, shadow=False, ncol=len(colors))

    # Set the x and y labels
    fig.supxlabel(r'Hidden Layer Size ($\times1000$)')
    fig.supylabel('ASR & CDA (%)')

    # Set the ticks
    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xticks(range(len(hdlyr_labels)))
        ax.set_xticklabels(hdlyr_labels)
        ax.set_yticks(np.arange(0, 120, 20))
        ax.set_ylim(-20, 120)
        ax.set_xlim(-0.5, len(hdlyr_labels) - 0.5)
        # for label in ax.get_xticklabels()[::2]:
        #     label.set_visible(False)
        for label in ax.get_yticklabels()[::2]:
            label.set_visible(False)

    # Set the grid
    sns.despine(left=True)
    plt.tight_layout()
    # path_save = os.path.join(
    #     path_save, f'rate_vs_size_{args.trigger_color}_{args.pos}.pdf')
    # plt.savefig(path_save)
    plt.savefig(path_save.joinpath(f'mp_prunning_{dataset}.pdf'))
    # plt.show()


if __name__ == '__main__':
    main()



########################################################################################################
########################################################################################################








# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import pandas as pd
# from pathlib import Path
# import matplotlib.lines as mlines
# import matplotlib as mpl


# parser = argparse.ArgumentParser()
# parser.add_argument("--loadpath", type=str, default='.')
# parser.add_argument("--dataset", type=str, default='cifar10', help='mnist, fmnist, svhn, wbcd, brats',
#                     choices=['mnist', 'fmnist', 'svhn', 'wbcd', 'brats'], required=True, )
# parser.add_argument("--savepath", type=str, default='.')
# args = parser.parse_args()


# def main():
#     sns.set()
#     sns.set_theme()
#     # sns.set_theme(rc={"figure.subplot.wspace": 0.002, "figure.subplot.hspace": 0.002})
#     plt.rcParams["font.family"] = "serif"
#     plt.rcParams["font.serif"] = "DejaVu Serif"
#     plt.rcParams["font.size"] = 22
#     # sns.set_theme(style="whitegrid", font_scale=1.2)


#     # Read the reuslts from the csv file
#     loadpath = Path(args.loadpath)
#     path_save = Path(args.savepath)

#     elm_type_list = ['poelm', 'drop-elm', 'mlelm']
#     legend_color_labels = ['ELM', 'BD-ELM', 'ML-ELM']
#     legend_style_labels = ['CDA', 'ASR']
#     dataset = args.dataset
#     feature_index = ['feature_index']
#     feature_index_label = ['MI FEATURE INDEX: 23']
#     hdlyr_size_list = [500, 1000, 2000, 5000, 8000]
#     hdlyr_labels = [0.5, 1, 2, 5, 8]
#     epsilon = 5
#     prune_rate_list = [0.1, 0.3, 0.5]
#     colors = ['blue', 'red', 'green']
#     cda_markers = ['o', 'D', 'x']
#     asr_markers = ['*', '', 'X']
#     markers = ['x', 'X']
#     linestyles = ['dotted', 'solid']
#     n_experiments = 1

#     fig, axs = plt.subplots(nrows=len(prune_rate_list), ncols=len(
#         feature_index), figsize=(8, 14), sharex=True, sharey=True)
#     # fig.subplots_adjust(wspace=0.01, hspace=0.01)

#     col_leg_list = []
#     styl_leg_list = []

#     df = pd.read_csv(loadpath)
#     # print(df)
#     column = 0
#     row = 0
#     for idx, ax in enumerate(axs.flat):
#         if column >= len(feature_index):
#             column = 0
#             row += 1

#         feature_index_name = feature_index[column]
#         prune_rate = prune_rate_list[row]

#         for elm_type in elm_type_list:
#             clnums_ASR_list = []
#             clnums_CDA_list = []
#             for hdlyr_size in hdlyr_size_list:
#                 slctd_df = df[(df['ELM_TYPE'] == elm_type) &
#                               (df['DATASET'] == dataset) &
#                               (df['HIDDEN_LYR_SIZE'] == hdlyr_size) &
#                               (df['POISON_PERCENTAGE'] == epsilon) &
#                               (df['PRUNE_RATE'] == prune_rate)]

#                 if not len(slctd_df['TEST_ACCURACY'].values) > 0 or not len(slctd_df['BD_TEST_ACCURACY'].values) > 0:
#                     print(hdlyr_size, dataset, elm_type, prune_rate)
#                     raise Exception('above row not found in pandas datafram.')
#                 else:
#                     clnums_CDA_list.append(slctd_df['TEST_ACCURACY'].values[0])
#                     clnums_ASR_list.append(slctd_df['BD_TEST_ACCURACY'].values[0])
#             clnums_CDA_list = [item * 100 for item in clnums_CDA_list]
#             clnums_ASR_list = [item * 100 for item in clnums_ASR_list]

#             # mean = np.mean(list_frzlyr_asr, axis=1) * 100
#             # min = np.min(list_frzlyr_asr, axis=1) * 100
#             # max = np.max(list_frzlyr_asr, axis=1) * 100

#             # err = np.array([mean - min, max - mean])
#             ax.errorbar(x=range(len(hdlyr_labels)), y=clnums_CDA_list, marker=markers[0], alpha=0.8, markersize=4,
#                         linestyle=linestyles[0], color=colors[elm_type_list.index(elm_type)])
#             line_handle = ax.errorbar(x=range(len(hdlyr_labels)), y=clnums_ASR_list, marker=markers[1], alpha=0.8, markersize=4,
#                         label=legend_color_labels[elm_type_list.index(elm_type)], linestyle=linestyles[1], color=colors[elm_type_list.index(elm_type)])
#             col_leg_list.append(line_handle)

#         column += 1
#     col_leg_list = col_leg_list[:len(elm_type_list)]

#     # # Set the labels
#     # for ax, col in zip(axs[0], feature_index_label):
#     #     ax.set_title(f'{col}', fontsize=20)


#     for ax, row in zip(axs, prune_rate_list):
#         ax.set_ylabel(r'$pr = $'+f'{row}', rotation=90, size='large')

#     styl_leg_list = [mlines.Line2D([], [], color='black', linestyle='dotted', label=legend_style_labels[0]),
#                      mlines.Line2D([], [], color='black', linestyle='solid', label=legend_style_labels[1])]

#     # Set the legend showing the models with the corresponding marker
#     # handles, labels = axs[0, 0].get_legend_handles_labels()
#     handles = col_leg_list + styl_leg_list
#     fig.legend(handles=handles, bbox_to_anchor=(0.95, 0.2), fancybox=False, shadow=False, ncol=len(handles)//2, prop={'size': 17})
#     # fig.legend(handles,
#     #            legend_labels,
#     #            bbox_to_anchor=(0.65, 0.095), fancybox=False, shadow=False, ncol=len(colors))

#     # Set the x and y labels
#     fig.supxlabel(r'Hidden Layer Size ($\times1000$)')
#     fig.supylabel('ASR & CDA (%)')

#     # Set the ticks
#     for ax in axs.flat:
#         ax.tick_params(axis='both', which='major', labelsize=20)
#         ax.set_xticks(range(len(hdlyr_labels)))
#         ax.set_xticklabels(hdlyr_labels)
#         ax.set_yticks(np.arange(0, 120, 20))
#         ax.set_ylim(-20, 120)
#         ax.set_xlim(-0.5, len(hdlyr_labels) - 0.5)
#         # for label in ax.get_xticklabels()[::2]:
#         #     label.set_visible(False)
#         for label in ax.get_yticklabels()[::2]:
#             label.set_visible(False)

#     # Set the grid
#     sns.despine(left=True)
#     plt.tight_layout()
#     # path_save = os.path.join(
#     #     path_save, f'rate_vs_size_{args.trigger_color}_{args.pos}.pdf')
#     # plt.savefig(path_save)
#     plt.savefig(path_save.joinpath(f'prunning_asr_{dataset}.pdf'))
#     # plt.show()


# if __name__ == '__main__':
#     main()
