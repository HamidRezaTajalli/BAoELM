import pruning
from pathlib import Path
import gc

saving_path = Path()
n_of_experiments = 1
elm_type_list = ['poelm', 'drop-elm', 'drelm', 'telm', 'mlelm']
dataset_list = ['svhn']
hdlyr_size_list = [2000, 5000, 8000, 10000]
trigger_type = 'badnet'
epsilon_list = [0.2, 0.5, 1, 2, 5]
trigger_size_list = [(4, 4), (8, 8)]
target_label = 0
prune_rate_list = [0.1, 0.3, 0.5]

for dataset in dataset_list:
    for elm_type in elm_type_list:
        for hdlyr_size in hdlyr_size_list:
            for epsilon in epsilon_list:
                for trigger_size in trigger_size_list:
                    for prune_rate in prune_rate_list:
                        for exp_num in range(n_of_experiments):
                            pruning.trainer(exp_num=exp_num, saving_path=saving_path, elm_type=elm_type, dataset=dataset,
                                            trigger_type=trigger_type, target_label=target_label, prune_rate=prune_rate,
                                            poison_percentage=epsilon, hdlyr_size=hdlyr_size, trigger_size=trigger_size)
