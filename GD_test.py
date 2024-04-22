import pathlib

from elm_versions import elm, pca_transformed, pca_initialization, pruned_elm, drop_elm, elm_embeded
from elm_versions import DRELM_main, TELM_Main, ML_ELM_main
from elm_versions import main_CNNELM, pseudoInverse
from dataset_handler import mnist, fmnist, cifar10, svhn
import csv
import pathlib
import torch
import time
import gc
from elm_versions import elm_GD


# def trainer(dataset: str, hdlyr_size: int) -> None:
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     ds_dict = {'mnist': mnist, 'fmnist': fmnist, 'cifar10': cifar10, 'svhn': svhn}
#     all_data = ds_dict[dataset].get_alldata_simple()
#     # all_data = ds_dict[dataset].get_alldata_backdoor(target_label=0, train_samples_percentage=2, trigger_size=(4, 4))

#     model = elm_GD.ELM_GD_Classifier(input_size=all_data['train']['x'].shape[1], hidden_size=hdlyr_size, output_size=10).to(device)
#     if torch.cuda.is_available():
#         print(f"CUDA is available. Using {device}.")
#     else:
#         print(f"CUDA is not available. Using {device}.")
#     train_dataset = torch.utils.data.TensorDataset(all_data['train']['x'], all_data['train']['y_oh'])
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

#     elm_GD.fit(model=model, train_loader=train_loader, learning_rate=0.001, epochs=50)

#     test_dataset = torch.utils.data.TensorDataset(all_data['test']['x'], all_data['test']['y_oh'])
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)


#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             outputs = model(data)
#             _, predicted = torch.max(outputs.data, 1)
#             total += target.size(0)
#             target = torch.max(target, 1)[1]
#             correct += (predicted == target).sum().item()
#     print('Test Accuracy: {} %'.format(100 * correct / total))




# if __name__ == '__main__':
#     trainer(dataset='mnist', hdlyr_size=5000)


hdlyr_size = 5000
dataset = 'mnist'
ds_dict = {'mnist': mnist, 'fmnist': fmnist, 'cifar10': cifar10, 'svhn': svhn}
all_data = ds_dict[dataset].get_alldata_simple()
# all_data = ds_dict[dataset].get_alldata_backdoor(target_label=0, train_samples_percentage=2, trigger_size=(4, 4))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model = elm_GD.ELM_GD_Classifier(input_size=all_data['train']['x'].shape[1], hidden_size=hdlyr_size, output_size=10).to(device)
model = elm_embeded.ELMClassifier(input_size=all_data['train']['x'].shape[1], output_size=10, model_path='elm_model_with_config_2024-02-08_clean.pth')
if torch.cuda.is_available():
    print(f"CUDA is available. Using {device}.")
else:
    print(f"CUDA is not available. Using {device}.")
model.fit(all_data['train']['x'], all_data['train']['y_oh'])

# poelm = elm.ELMClassifier(hidden_layer_size=hdlyr_size)
# poelm.fit(all_data['bd_train']['x'], all_data['bd_train']['y_oh'])

out = model.predict(all_data['test']['x'])
test_accuracy = torch.sum(all_data['test']['y'] == torch.from_numpy(out)).item() / len(out)
print(test_accuracy)
