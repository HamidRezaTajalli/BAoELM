import pathlib

from elm_versions import elm_GD, elm_embeded
from dataset_handler import mnist, fmnist, cifar10, svhn, wbcd, brats
import csv
import pathlib
import torch
import time
import gc
import pickle


def trainer(exp_num: int, saving_path: pathlib.Path, dataset: str, trigger_type: str, target_label: int,
            poison_percentage, hdlyr_size: int, trigger_size:
        tuple[int, int] = (4, 4)) -> None:
    print(
        f'This is the run for experiment number {exp_num} of model poison idea on {dataset} dataset with {trigger_type} '
        f'and hidden layer size {hdlyr_size} and trigger size {trigger_size} and target label {target_label} '
        f'and poison percentage {poison_percentage}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_accuracy, bd_test_accuracy = -1, -1  # default values
    elapsed_time = -1

    csv_path = saving_path.joinpath(f'results_model_poisoning_{dataset}.csv')
    if not csv_path.exists():
        csv_path.touch()
        with open(file=csv_path, mode='w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['EXPERIMENT_NUMBER', 'DATASET', 'HIDDEN_LYR_SIZE', 'TRIGGER_TYPE', 'TARGET_LABEL', 'POISON_PERCENTAGE',
                                 'TRIGGER_SIZE', 'TEST_ACCURACY', 'BD_TEST_ACCURACY'])
    obj_path = saving_path.joinpath('saved_models')
    if not obj_path.exists():
        obj_path.mkdir(parents=True)
    # obj_path = obj_path.joinpath(
        # f'PM_{exp_num}_{dataset}_{trigger_type}_{target_label}_{poison_percentage}_{hdlyr_size}_{trigger_size[0]}.pkl')

    ds_dict = {'mnist': mnist, 'fmnist': fmnist, 'cifar10': cifar10, 'svhn': svhn, 'wbcd': wbcd, 'brats': brats}
    output_size = len(ds_dict[dataset].CLASSES_NAMES)

    all_data_backdoored = ds_dict[dataset].get_alldata_backdoor(target_label=target_label,
                                                     train_samples_percentage=poison_percentage,
                                                     trigger_size=trigger_size)


    # Define model and train it by gradient descent algorithm on backdoored dataset.
    model = elm_GD.ELM_GD_Classifier(input_size=all_data_backdoored['bd_train']['x'].shape[1], hidden_size=hdlyr_size, output_size=output_size).to(device)

    if torch.cuda.is_available():
        print(f"CUDA is available. Using {device}.")
    else:
        print(f"CUDA is not available. Using {device}.")
    train_dataset = torch.utils.data.TensorDataset(all_data_backdoored['bd_train']['x'], all_data_backdoored['bd_train']['y_oh'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    model, model_config =elm_GD.fit(model=model, train_loader=train_loader, learning_rate=0.001, epochs=50)

    test_dataset = torch.utils.data.TensorDataset(all_data_backdoored['test']['x'], all_data_backdoored['test']['y_oh'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    bd_test_dataset = torch.utils.data.TensorDataset(all_data_backdoored['bd_test']['x'], all_data_backdoored['bd_test']['y_oh'])
    bd_test_loader = torch.utils.data.DataLoader(bd_test_dataset, batch_size=64, shuffle=True)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            target = torch.max(target, 1)[1]
            correct += (predicted == target).sum().item()
    print('Test Accuracy on GD trained model: {} %'.format(100 * correct / total))


    model.eval()
    correct = 0 
    total = 0
    with torch.no_grad():
        for data, target in bd_test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            target = torch.max(target, 1)[1]
            correct += (predicted == target).sum().item()
    print('Backdoor Test Accuracy on GD trained model: {} %'.format(100 * correct / total))



    pmconf_path = obj_path.joinpath(f'pmconf_{exp_num}_{dataset}_{hdlyr_size}_{trigger_type}_{target_label}_{poison_percentage}_{trigger_size[0]}.pth')
    torch.save(model_config, pmconf_path)

    # TODO: hala bayad model ro ruyed embeded load konam va ruye clean data train konam va ruyed backdoor testesh konam.

    all_data_clean = ds_dict[dataset].get_alldata_simple()
    model = elm_embeded.ELMClassifier(input_size=all_data_clean['train']['x'].shape[1], output_size=output_size, model_path=pmconf_path)
    model.fit(all_data_clean['train']['x'], all_data_clean['train']['y_oh'])


    out = model.predict(all_data_clean['test']['x'])
    test_accuracy = torch.sum(all_data_clean['test']['y'] == torch.from_numpy(out)).item() / len(out)
    bd_out = model.predict(all_data_backdoored['bd_test']['x'])
    bd_test_accuracy = torch.sum(all_data_backdoored['bd_test']['y'] == torch.from_numpy(bd_out)).item() / len(bd_out)

    print(test_accuracy)
    print(bd_test_accuracy)


    pm_savepath = obj_path.joinpath(f'pm_{exp_num}_{dataset}_{hdlyr_size}_{trigger_type}_{target_label}_{poison_percentage}_{trigger_size[0]}.pkl')
    with open(pm_savepath, 'wb') as file:
        pickle.dump(model, file)

    

    
    
    with open(file=csv_path, mode='a') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            [exp_num, dataset, hdlyr_size, trigger_type, target_label, poison_percentage, trigger_size,
             test_accuracy, bd_test_accuracy])

    del all_data_backdoored
    del all_data_clean
    gc.collect()



import argparse
import pathlib

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Run training experiments for backdoor detection.")
    parser.add_argument('--exp_num', type=int, required=True, help='Experiment number')
    parser.add_argument('--saving_path', type=str, required=True, help='Path to save the results')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use')
    parser.add_argument('--hdlyr_size', type=int, required=True, help='Size of the hidden layer')
    parser.add_argument('--trigger_type', type=str, required=True, help='Type of trigger used in backdoor attack')
    parser.add_argument('--target_label', type=int, required=True, help='Target label for the backdoor attack')
    parser.add_argument('--poison_percentage', type=float, required=True, help='Percentage of poisoned data')
    parser.add_argument('--trigger_size', type=int, required=True, help='Size of the trigger (width height)')

    # Parse the arguments
    args = parser.parse_args()

    # Convert saving_path to a Path object
    saving_path = pathlib.Path(args.saving_path)

    # Call the trainer function
    trainer(exp_num=args.exp_num, saving_path=saving_path, dataset=args.dataset,
            trigger_type=args.trigger_type, target_label=args.target_label, poison_percentage=args.poison_percentage,
            hdlyr_size=args.hdlyr_size, trigger_size=(args.trigger_size, args.trigger_size))
    

if __name__ == '__main__':
    main()

