import training_bd
from pathlib import Path
import gc

saving_path = Path()
n_of_experiments = 2
elm_type_list = ['poelm', 'elm-pca', 'pca-elm', 'drop-elm', 'drelm', 'telm', 'mlelm']
dataset_list = ['mnist']
hdlyr_size_list = [500, 700, 1000, 1500, 2000]
trigger_type = 'badnet'
epsilon_list = [0.02, 0.05, 0.1, 0.2]
trigger_size_list = [(2, 2), (4, 4), (8, 8)]
target_label = 0

for dataset in dataset_list:
    for elm_type in elm_type_list:
        for hdlyr_size in hdlyr_size_list:
            for epsilon in epsilon_list:
                for trigger_size in trigger_size_list:
                    for exp_num in range(n_of_experiments):
                        training_bd.trainer(exp_num=exp_num, saving_path=saving_path, elm_type=elm_type,
                                            dataset=dataset,
                                            trigger_type=trigger_type, target_label=target_label,
                                            poison_percentage=epsilon,
                                            hdlyr_size=hdlyr_size, trigger_size=trigger_size)

                        gc.collect()
