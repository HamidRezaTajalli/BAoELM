import numpy as np
import torch.utils.data
from torchvision import datasets, transforms

from .trigger import get_backdoor_test_dataset, get_backdoor_train_dataset, GenerateTrigger

import torch
torch.manual_seed(47)
import numpy as np
np.random.seed(47)

def get_dataloaders_simple(drop_last, is_shuffle):
    drop_last = drop_last
    is_shuffle = is_shuffle
    # batch_size = batch_size
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_workers = 2 if device.type == 'cuda' else 0

    classes_names = ('Plane', 'Car', 'Bird', 'Cat',
                     'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    transforms_dict = {
        'train': transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        'test': transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    }
    train_dataset = datasets.CIFAR10(root='./data/CIFAR10/', train=True, transform=transforms_dict['train'],
                                     download=True)
    test_dataset = datasets.CIFAR10(root='./data/CIFAR10/', train=False, transform=transforms_dict['test'],
                                    download=True)
    validation_dataset, test_dataset = torch.utils.data.random_split(test_dataset,
                                                                     [int(len(test_dataset) / (2)),
                                                                      int(len(test_dataset) / (2))])


    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset),
                                                     shuffle=is_shuffle, num_workers=num_workers,
                                                     drop_last=drop_last)


    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset),
                                                  shuffle=is_shuffle, num_workers=num_workers, drop_last=drop_last)

    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset),
                                                        shuffle=is_shuffle, num_workers=num_workers,
                                                        drop_last=drop_last)

    return {'train': train_dataloader,
            'test': test_dataloader,
            'validation': validation_dataloader}, classes_names


def get_dataloaders_lbflip(batch_size, train_ds_num, drop_last, is_shuffle, flip_label, target_label):
    drop_last = drop_last
    batch_size = batch_size
    is_shuffle = is_shuffle
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_workers = 2 if device.type == 'cuda' else 0

    classes_names = ('Plane', 'Car', 'Bird', 'Cat',
                     'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

    transforms_dict = {
        'train': transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        'test': transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    }

    train_dataset = datasets.CIFAR10(root='./data/CIFAR10/', train=True, transform=transforms_dict['train'],
                                     download=True)
    test_dataset = datasets.CIFAR10(root='./data/CIFAR10/', train=False, transform=transforms_dict['test'],
                                    download=True)

    target_indices = [num for num, item in enumerate(train_dataset.targets) if item == flip_label]
    rest_indices = [num for num, item in enumerate(train_dataset.targets) if item != flip_label]

    target_dataset = []
    for number, item in enumerate(train_dataset):
        if number in target_indices:
            target_dataset.append((item[0], target_label))
    train_dataset.data = train_dataset.data[rest_indices]
    train_dataset.targets = np.array(train_dataset.targets)[rest_indices].tolist()

    target_indices = [num for num, item in enumerate(test_dataset.targets) if item == flip_label]
    rest_indices = [num for num, item in enumerate(test_dataset.targets) if item != flip_label]

    for number, item in enumerate(test_dataset):
        if number in target_indices:
            target_dataset.append((item[0], target_label))
    test_dataset.data = test_dataset.data[rest_indices]
    test_dataset.targets = np.array(test_dataset.targets)[rest_indices].tolist()

    validation_dataset, test_dataset = torch.utils.data.random_split(test_dataset,
                                                                     [int(len(test_dataset) / (2 / 1)),
                                                                      int(len(test_dataset) / (2 / 1))])

    chunk_len = len(train_dataset) // train_ds_num
    remnant = len(train_dataset) % train_ds_num
    chunks = [chunk_len for item in range(train_ds_num)]
    if remnant > 0:
        chunks.append(remnant)

    train_datasets = torch.utils.data.random_split(train_dataset,
                                                   chunks)

    backdoor_train_dataset = [item for item in train_datasets[0]]
    backdoor_train_dataset.extend(target_dataset)

    train_datasets[0] = backdoor_train_dataset

    train_dataloaders = [torch.utils.data.DataLoader(dataset=train_datasets[i], batch_size=batch_size,
                                                     shuffle=is_shuffle, num_workers=num_workers,
                                                     drop_last=drop_last)
                         for i in range(len(train_datasets))]

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                  shuffle=is_shuffle, num_workers=num_workers,
                                                  drop_last=drop_last)

    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                                        shuffle=is_shuffle, num_workers=num_workers,
                                                        drop_last=drop_last)

    backdoor_test_dataloader = torch.utils.data.DataLoader(dataset=target_dataset, batch_size=batch_size,
                                                           shuffle=is_shuffle, num_workers=num_workers,
                                                           drop_last=drop_last)

    #
    # for item in splitted_train_dataset[0]:
    #     if int(item[1]) == 9:
    #         insert = (item[0], 0)
    #         outliers_ds.append(insert)
    #         poison_ds.append(insert)
    #     else:
    #         clean_ds_train.append(item)
    #
    # for item in splitted_train_dataset[1]:
    #     if int(item[1]) == 9:
    #         insert = (item[0], 0)
    #         outliers_ds.append(insert)
    #         poison_ds.append(insert)
    #     else:
    #         poison_ds.append(item)
    #
    # for item in test_dataset:
    #     if int(item[1]) == 9:
    #         insert = (item[0], 0)
    #         outliers_ds.append(insert)
    #         poison_ds.append(insert)
    #     else:
    #         clean_ds_test.append(item)
    #
    # train_dataloader = torch.utils.data.DataLoader(dataset=clean_ds_train, batch_size=batch_size,
    #                                                shuffle=is_shuffle, num_workers=num_workers, drop_last=drop_last)
    # test_dataloader = torch.utils.data.DataLoader(dataset=clean_ds_test, batch_size=batch_size,
    #                                               shuffle=is_shuffle, num_workers=num_workers, drop_last=drop_last)
    # poison_dataloader = torch.utils.data.DataLoader(dataset=poison_ds, batch_size=batch_size,
    #                                                 shuffle=is_shuffle, num_workers=num_workers, drop_last=drop_last)
    # outlier_dataloader = torch.utils.data.DataLoader(dataset=outliers_ds, batch_size=batch_size,
    #                                                  shuffle=is_shuffle, num_workers=num_workers, drop_last=drop_last)

    return {'train': train_dataloaders,
            'validation': validation_dataloader,
            'test': test_dataloader,
            'backdoor_test': backdoor_test_dataloader}, classes_names


def get_dataloaders_backdoor(batch_size, train_ds_num, drop_last, is_shuffle, target_label, trigger_obj):
    drop_last = drop_last
    batch_size = batch_size
    is_shuffle = is_shuffle
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_workers = 2 if device.type == 'cuda' else 0

    classes_names = ('Plane', 'Car', 'Bird', 'Cat',
                     'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

    transforms_dict = {
        'train': transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        'test': transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    }

    train_dataset = datasets.CIFAR10(root='./data/CIFAR10/', train=True, transform=transforms_dict['train'],
                                     download=True)
    test_dataset = datasets.CIFAR10(root='./data/CIFAR10/', train=False, transform=transforms_dict['test'],
                                    download=True)
    validation_dataset, test_dataset = torch.utils.data.random_split(test_dataset,
                                                                     [int(len(test_dataset) / (2 / 1)),
                                                                      int(len(test_dataset) / (2 / 1))])

    chunk_len = len(train_dataset) // train_ds_num
    remnant = len(train_dataset) % train_ds_num
    chunks = [chunk_len for item in range(train_ds_num)]
    if remnant > 0:
        chunks.append(remnant)
    train_datasets = torch.utils.data.random_split(train_dataset,
                                                   chunks)
    # train_datasets = torch.utils.data.random_split(train_dataset,
    #                                                [int(len(train_dataset) / (8 / 1)),
    #                                                 int(len(train_dataset) / (8 / 7))])

    # trigger_obj = GenerateTrigger((8, 8), pos_label='upper-mid', dataset='cifar10', shape='square')

    # train_datasets[0] = get_backdoor_train_dataset(train_datasets[0], trigger_obj, trig_ds='cifar10',
    #                                                samples_percentage=train_samples_percentage,
    #                                                backdoor_label=target_label)

    backdoor_test_dataset = get_backdoor_test_dataset(test_dataset, trigger_obj, trig_ds='cifar10',
                                                      backdoor_label=target_label)
    train_dataloaders = [torch.utils.data.DataLoader(dataset=train_datasets[i], batch_size=batch_size,
                                                     shuffle=is_shuffle, num_workers=num_workers,
                                                     drop_last=drop_last)
                         for i in range(len(train_datasets))]

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                  shuffle=is_shuffle, num_workers=num_workers,
                                                  drop_last=drop_last)

    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                                        shuffle=is_shuffle, num_workers=num_workers,
                                                        drop_last=drop_last)

    backdoor_test_dataloader = torch.utils.data.DataLoader(dataset=backdoor_test_dataset, batch_size=batch_size,
                                                           shuffle=is_shuffle, num_workers=num_workers,
                                                           drop_last=drop_last)

    return {'train': train_dataloaders,
            'validation': validation_dataloader,
            'test': test_dataloader,
            'backdoor_test': backdoor_test_dataloader}, classes_names
