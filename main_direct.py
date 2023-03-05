import training
from pathlib import Path
import gc




# saving_path = Path()
# n_of_experiments = 2
# elm_type_list = ['poelm', 'elm-pca', 'pca-elm', 'pruned-elm', 'drop-elm', 'drelm', 'telm', 'mlelm']
# dataset_list = ['mnist']
# hdlyr_size_list = [500, 700, 1000, 1500, 2000]
#
#
# for dataset in dataset_list:
#     for elm_type in elm_type_list:
#         for hdlyr_size in hdlyr_size_list:
#             for exp_num in range(n_of_experiments):
#                 training.trainer(exp_num=exp_num, saving_path=saving_path, elm_type=elm_type, dataset=dataset, hdlyr_size=hdlyr_size)
#                 gc.collect()


saving_path = Path()
n_of_experiments = 1
elm_type_list = ['poelm', 'elm-pca', 'pca-elm', 'pruned-elm', 'drop-elm', 'drelm', 'telm', 'mlelm']
dataset_list = ['fmnist']
hdlyr_size_list = [500]


for dataset in dataset_list:
    for elm_type in elm_type_list:
        for hdlyr_size in hdlyr_size_list:
            for exp_num in range(n_of_experiments):
                training.trainer(exp_num=exp_num, saving_path=saving_path, elm_type=elm_type, dataset=dataset, hdlyr_size=hdlyr_size)
                gc.collect()
