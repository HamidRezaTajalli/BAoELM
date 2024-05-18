import torch
from torchvision import datasets, transforms
from dataset_handler.trigger import get_backdoor_test_dataset, get_backdoor_train_dataset, GenerateTrigger, toonehottensor

from torchvision.datasets import ImageFolder


CLASSES_NAMES = ['no_tumor', 'tumor']


class BinaryBRATCDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super(BinaryBRATCDataset, self).__init__(root, transform=transform)
        # No need to modify class_to_idx here

    def __getitem__(self, index):
        # This method is called by DataLoader to fetch a single item
        path, _ = self.samples[index]  # Original path and target
        sample, target = super(BinaryBRATCDataset, self).__getitem__(index)
        # Adjust the target based on the path
        if 'no_tumor' in path:
            target = 0  # no_tumor
        else:
            target = 1  # tumor
        return sample, target

def get_dataloaders_simple(batch_size, drop_last, is_shuffle):
    drop_last = drop_last
    is_shuffle = is_shuffle
    batch_size = batch_size

    transforming_img = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  
    transforms.Normalize([0.5,0.5,0.5], 
                        [0.5,0.5,0.5])
    ])

    train_path = ('./data/BRATC/Training/')
    test_path = ('./data/BRATC/Testing/')


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_workers = 2 if device.type == 'cuda' else 0

    classes= BinaryBRATCDataset(test_path, transform=transforming_img).classes
    classes = ['no_tumor', 'tumor']

    

    train_dataset = BinaryBRATCDataset(train_path, transform=transforming_img)
    test_dataset = BinaryBRATCDataset(test_path, transform=transforming_img)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=len(train_dataset) if batch_size is None else batch_size,
                                                    shuffle=is_shuffle, num_workers=num_workers,
                                                    drop_last=drop_last)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=len(test_dataset) if batch_size is None else batch_size,
                                                  shuffle=is_shuffle, num_workers=num_workers, drop_last=drop_last)


    return {'train': train_dataloader,
            'test': test_dataloader}, classes




def get_dataloaders_backdoor(batch_size, drop_last, is_shuffle, target_label, train_samples_percentage, trigger_size=(8, 8)):

    drop_last = drop_last
    is_shuffle = is_shuffle
    batch_size = batch_size

    transforming_img = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  
    transforms.Normalize([0.5,0.5,0.5], 
                        [0.5,0.5,0.5])
    ])

    train_path = ('./data/BRATC/Training/')
    test_path = ('./data/BRATC/Testing/')


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_workers = 2 if device.type == 'cuda' else 0

    classes= BinaryBRATCDataset(test_path, transform=transforming_img).classes
    classes = ['no_tumor', 'tumor']

    

    train_dataset = BinaryBRATCDataset(train_path, transform=transforming_img)
    test_dataset = BinaryBRATCDataset(test_path, transform=transforming_img)


    trigger_obj = GenerateTrigger(trigger_size, pos_label='upper-left', dataset='brats', shape='square')

    bd_train_dataset = get_backdoor_train_dataset(train_dataset, trigger_obj, trig_ds='brats',
                                                  samples_percentage=train_samples_percentage,
                                                  backdoor_label=target_label)
    
    backdoor_test_dataset = get_backdoor_test_dataset(test_dataset, trigger_obj, trig_ds='brats',
                                                      backdoor_label=target_label)

    bd_train_dataloader = torch.utils.data.DataLoader(dataset=bd_train_dataset,
                                                    batch_size=len(train_dataset) if batch_size is None else batch_size,
                                                    shuffle=is_shuffle, num_workers=num_workers,
                                                    drop_last=drop_last)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset) if batch_size is None else batch_size,
                                                  shuffle=is_shuffle, num_workers=num_workers,
                                                  drop_last=drop_last)
    
    backdoor_test_dataloader = torch.utils.data.DataLoader(dataset=backdoor_test_dataset, batch_size=len(backdoor_test_dataset) if batch_size is None else batch_size,
                                                           shuffle=is_shuffle, num_workers=num_workers,
                                                           drop_last=drop_last)


    return {'bd_train': bd_train_dataloader,
            'test': test_dataloader,
            'bd_test': backdoor_test_dataloader}, classes



def get_alldata_simple():
    '''
    a method which calls the dataloaders, and iterate through them,
         flattens the inputs and returns all dataset in just one batch of data.
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloaders, classes_names = get_dataloaders_simple(batch_size=None, drop_last=True, is_shuffle=False)
    all_data = {item: {} for item in dataloaders.keys()}
    for phase in all_data.keys():
        for i_batch, sample_batched in enumerate(dataloaders[phase]):
            # print(type(sample_batched[0]))
            # print(len(sample_batched[0]))
            # print(sample_batched[0].shape)
            # print(type(sample_batched[1]))
            # print(len(sample_batched[1]))
            # print(sample_batched[1].shape)
            all_data[phase]['x'] = torch.reshape(sample_batched[0], (len(dataloaders[phase].dataset), -1)).to(device)
            all_data[phase]['y'] = sample_batched[1].to(device)
            all_data[phase]['y_oh'] = toonehottensor(2, sample_batched[1]).to(device)

    return all_data


def get_alldata_backdoor(target_label, train_samples_percentage, trigger_size):
    '''
    a method which calls the dataloaders, and iterate through them,
         flattens the inputs and returns all dataset in just one batch of data.
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloaders, classes_names = get_dataloaders_backdoor(batch_size=None, drop_last=True, is_shuffle=False, target_label=target_label, train_samples_percentage=train_samples_percentage, trigger_size=trigger_size)
    all_data = {item: {} for item in dataloaders.keys()}
    for phase in all_data.keys():
        for i_batch, sample_batched in enumerate(dataloaders[phase]):
            # print(type(sample_batched[0]))
            # print(len(sample_batched[0]))
            # print(sample_batched[0].shape)
            # print(type(sample_batched[1]))
            # print(len(sample_batched[1]))
            # print(sample_batched[1].shape)
            all_data[phase]['x'] = torch.reshape(sample_batched[0], (len(dataloaders[phase].dataset), -1)).to(device)
            all_data[phase]['y'] = sample_batched[1].to(device)
            all_data[phase]['y_oh'] = toonehottensor(2, sample_batched[1]).to(device)

    return all_data