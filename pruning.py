import pathlib

from elm_versions import TELM_Main, ML_ELM_main
from dataset_handler import mnist, fmnist, cifar10, svhn, wbcd, brats
import csv
import pathlib
import torch
import time
import gc
import pickle


def trainer(exp_num: int, saving_path: pathlib.Path, elm_type: str, dataset: str, trigger_type: str, target_label: int,
            prune_rate: float,
            poison_percentage, hdlyr_size: int, trigger_size:
        tuple[int, int] = (4, 4)) -> None:
    print(
        f'This is the run for experiment number {exp_num} for pruning. Pruning rate is {prune_rate}. Experiment is of {elm_type} on {dataset} dataset with {trigger_type} '
        f'and hidden layer size {hdlyr_size} and trigger size {trigger_size} and target label {target_label} '
        f'and poison percentage {poison_percentage}')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_accuracy, bd_test_accuracy = -1, -1  # default values
    elapsed_time = -1

    csv_path = saving_path.joinpath(f'results_pruning_{dataset}_{elm_type}.csv')
    if not csv_path.exists():
        csv_path.touch()
        with open(file=csv_path, mode='w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['EXPERIMENT_NUMBER', 'ELM_TYPE',
                                 'DATASET', 'HIDDEN_LYR_SIZE', 'PRUNE_RATE', 'TRIGGER_TYPE', 'TARGET_LABEL',
                                 'POISON_PERCENTAGE',
                                 'TRIGGER_SIZE', 'TEST_ACCURACY', 'BD_TEST_ACCURACY',
                                 'TIME_ELAPSED'])


    obj_to_load = None
    obj_path = saving_path.joinpath('saved_models')
    if not obj_path.exists():
        raise FileNotFoundError(f'No directory called {obj_path} exists.')
    if 'embeded' not in elm_type.lower():
        obj_path = obj_path.joinpath(
        f'{exp_num}_{dataset}_{elm_type}_{trigger_type}_{target_label}_{poison_percentage}_{hdlyr_size}_{trigger_size[0]}.pkl')
    else:
        obj_path = obj_path.joinpath(f'pm_{exp_num}_{dataset}_{hdlyr_size}_{trigger_type}_{target_label}_{poison_percentage}_{trigger_size[0]}.pkl'

        )

    
    if not obj_path.exists():
        raise FileNotFoundError(f'No file called {obj_path} exists.')

    with open(obj_path, 'rb') as file:
        obj_to_load = pickle.load(file)
    if obj_to_load is None:
        raise ValueError(f'No object was loaded from {obj_path}.')


    ds_dict = {'mnist': mnist, 'fmnist': fmnist, 'cifar10': cifar10, 'svhn': svhn, 'wbcd': wbcd, 'brats': brats}

    all_data_clean = ds_dict[dataset].get_alldata_simple()

    all_data_bd = ds_dict[dataset].get_alldata_backdoor(target_label=target_label,
                                                     train_samples_percentage=poison_percentage,
                                                     trigger_size=trigger_size)

    if elm_type.lower() == 'poelm':
        poelm = obj_to_load
        start_time = time.time()
        poelm.fit_with_mask(all_data_clean['train']['x'], all_data_clean['train']['y_oh'], prune_rate=prune_rate)
        elapsed_time = time.time() - start_time
        out = poelm.predict_with_mask(all_data_bd['test']['x'])
        test_accuracy = torch.sum(all_data_bd['test']['y'] == torch.from_numpy(out)).item() / len(out)
        bd_out = poelm.predict_with_mask(all_data_bd['bd_test']['x'])
        bd_test_accuracy = torch.sum(all_data_bd['bd_test']['y'] == torch.from_numpy(bd_out)).item() / len(bd_out)
        del poelm, out, bd_out


    elif elm_type.lower() == 'drop-elm':
        drop = obj_to_load
        start_time = time.time()
        drop.fit_with_mask(all_data_clean['train']['x'], all_data_clean['train']['y_oh'], prune_rate=prune_rate)
        elapsed_time = time.time() - start_time
        out = drop.predict_with_mask(all_data_bd['test']['x'])
        test_accuracy = torch.sum(all_data_bd['test']['y'] == torch.from_numpy(out)).item() / len(out)
        bd_out = drop.predict_with_mask(all_data_bd['bd_test']['x'])
        bd_test_accuracy = torch.sum(all_data_bd['bd_test']['y'] == torch.from_numpy(bd_out)).item() / len(bd_out)
        del drop, out, bd_out

    elif elm_type.lower() == 'telm':
        (Wie, Whe, Beta_new, param) = obj_to_load
        test_accuracy, acc_train, (Wie_ex, Whe_ex, Beta_new_ex), elapsed_time, prune_mask = TELM_Main.TELM_main_with_mask(
            all_data_clean['train']['x'],
            all_data_clean['train'][
                'y_oh'].numpy(),
            all_data_bd['test']['x'],
            all_data_bd['test'][
                'y_oh'].numpy(),
            hidden_size=hdlyr_size,
            prune_rate=prune_rate,
        param=param)
        bd_test_accuracy = TELM_Main.TELM_test_with_mask(X_test=all_data_bd['bd_test']['x'],
                                                         Y_test=all_data_bd['bd_test']['y_oh'].numpy(),
                                                         Wie=Wie, Whe=Whe, Beta_new=Beta_new, prune_mask=prune_mask)
        del Wie, Whe, Beta_new, prune_mask

    elif elm_type.lower() == 'mlelm':
        (betahat_1, betahat_2, betahat_3, betahat_4, params) = obj_to_load
        test_accuracy, (
            betahat_1_ex, betahat_2_ex, betahat_3_ex, betahat_4_ex), elapsed_time, prune_mask = ML_ELM_main.main_ML_ELM_with_mask(
            all_data_clean['train']['x'],
            all_data_clean['train']['y_oh'].numpy(),
            all_data_bd['test']['x'],
            all_data_bd['test']['y_oh'].numpy(),
            prune_rate=prune_rate,
            params=params,
            hidden_layer=hdlyr_size)
        bd_test_accuracy = ML_ELM_main.MLELM_test_with_mask(X_test=all_data_bd['bd_test']['x'],
                                                            Y_test=all_data_bd['bd_test']['y_oh'].numpy(),
                                                            betahat_1=betahat_1, betahat_2=betahat_2,
                                                            betahat_3=betahat_3,
                                                            betahat_4=betahat_4, prune_mask=prune_mask)
        del betahat_1, betahat_2, betahat_3, betahat_4

    
    elif elm_type.lower() == 'embeded_elm':
        eb_elm = obj_to_load
        start_time = time.time()
        eb_elm.fit_with_mask(all_data_clean['train']['x'], all_data_clean['train']['y_oh'], prune_rate=prune_rate)
        elapsed_time = time.time() - start_time
        out = eb_elm.predict_with_mask(all_data_bd['test']['x'])
        test_accuracy = torch.sum(all_data_bd['test']['y'] == torch.from_numpy(out)).item() / len(out)
        bd_out = eb_elm.predict_with_mask(all_data_bd['bd_test']['x'])
        bd_test_accuracy = torch.sum(all_data_bd['bd_test']['y'] == torch.from_numpy(bd_out)).item() / len(bd_out)
        del eb_elm, out, bd_out

    with open(file=csv_path, mode='a') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            [exp_num, elm_type, dataset, hdlyr_size, prune_rate, trigger_type, target_label, poison_percentage,
             trigger_size,
             test_accuracy, bd_test_accuracy, elapsed_time])

    del all_data_bd
    gc.collect()



import argparse

def main():
    parser = argparse.ArgumentParser(description="Run ELM training with pruning.")
    parser.add_argument('--exp_num', type=int, required=True, help='Experiment number')
    parser.add_argument('--saving_path', type=str, required=True, help='Path to save the results')
    parser.add_argument('--elm_type', type=str, required=True, help='Type of ELM model')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use')
    parser.add_argument('--hdlyr_size', type=int, required=True, help='Size of the hidden layer')
    parser.add_argument('--prune_rate', type=float, required=True, help='Pruning rate for the model')
    parser.add_argument('--trigger_type', type=str, required=True, help='Type of trigger used in backdoor attack')
    parser.add_argument('--target_label', type=int, required=True, help='Target label for the backdoor attack')
    parser.add_argument('--poison_percentage', type=float, required=True, help='Percentage of poisoned data')
    parser.add_argument('--trigger_size', type=int, required=True, help='Size of the trigger')

    args = parser.parse_args()
    args.saving_path = pathlib.Path(args.saving_path)

    # Ensure the saving path exists
    if not args.saving_path.exists():
        args.saving_path.mkdir(parents=True, exist_ok=True)

    # Call the trainer function
    trainer(exp_num=args.exp_num, saving_path=args.saving_path, elm_type=args.elm_type, dataset=args.dataset, 
            hdlyr_size=args.hdlyr_size, prune_rate=args.prune_rate, trigger_type=args.trigger_type, 
            target_label=args.target_label, poison_percentage=args.poison_percentage, 
            trigger_size=(args.trigger_size, args.trigger_size))
    gc.collect()

if __name__ == "__main__":
    main()
