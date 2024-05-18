# import os
# import numpy as np
# import torch
# import glob
# import torch.nn as nn
# from torchvision.transforms import transforms
# from torch.utils.data import DataLoader
# from torch.optim import Adam
# from torch.autograd import Variable
# import torchvision
# import pathlib
# import pandas as pd
# import numpy as np

# from torchvision.datasets import ImageFolder
# from dataset_handler.brats import get_dataloaders_simple, get_dataloaders_backdoor

# class BinaryBRATCDataset(ImageFolder):
#     def __init__(self, root, transform=None):
#         super(BinaryBRATCDataset, self).__init__(root, transform=transform)
#         # No need to modify class_to_idx here

#     def __getitem__(self, index):
#         # This method is called by DataLoader to fetch a single item
#         path, _ = self.samples[index]  # Original path and target
#         sample, target = super(BinaryBRATCDataset, self).__getitem__(index)
#         # Adjust the target based on the pathTB
#         if 'no_tumor' in path:
#             target = 0  # no_tumor
#         else:
#             target = 1  # tumor
#         return sample, target


# transforming_img = transforms.Compose([
#     transforms.Resize((32,32)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),  
#     transforms.Normalize([0.5,0.5,0.5], 
#                         [0.5,0.5,0.5])
# ])


# train_path = ('./data/BRATC/Training/')
# test_path = ('./data/BRATC/Testing/')

# # train_loader = DataLoader(BinaryBRATCDataset(train_path, transform=transforming_img), batch_size=64, shuffle=True)
# # test_loader=DataLoader(
# #     BinaryBRATCDataset(test_path, transform=transforming_img),
# #     batch_size=32, shuffle=True
# # )

# # dataloaders, classes = get_dataloaders_simple(batch_size=64, drop_last=False, is_shuffle=True)
# dataloaders, classes = get_dataloaders_backdoor(batch_size=64, drop_last=False, is_shuffle=True, target_label=0, train_samples_percentage=5, trigger_size=(4, 4))
# bd_train_loader = dataloaders['bd_train']
# test_loader = dataloaders['test']
# bd_test_loader = dataloaders['bd_test']
# #categories
# root=pathlib.Path(train_path)
# classes= BinaryBRATCDataset(test_path, transform=transforming_img).classes
# classes = ['no_tumor', 'tumor']

# print(f'The classes are: {classes}')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class CNN(nn.Module):
#     def __init__(self, num_classes=2):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64 * 37 * 37, 512)  # Adjusted the size here
#         self.fc1 = nn.Linear(8 * 8 * 64, 512)  # Adjusted the size here
#         self.fc2 = nn.Linear(512, num_classes)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         # x = x.view(-1, 64 * 37 * 37)  # Adjusted the size here
#         x = x.view(-1, 8 * 8 * 64)  # Adjusted the size here
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# model = CNN(num_classes=len(classes)).to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = Adam(model.parameters(), lr=0.001)

# def train(num_epochs):
#     model.train()
#     for epoch in range(num_epochs):
#         for i, (images, labels) in enumerate(bd_train_loader):
#             images, labels = images.to(device), labels.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             if (i+1) % 100 == 0:
#                 print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(bd_train_loader)}], Loss: {loss.item():.4f}')

# def test():
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
        
#         print(f'Accuracy of the model on the test images: {100 * correct / total} %')

# def bd_test():
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, labels in bd_test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
        
#         print(f'Accuracy of the model on the backdoored test images: {100 * correct / total} %')

# num_epochs = 10
# train(num_epochs)
# test()
# bd_test()

import pathlib
from training_bd import trainer

trainer(exp_num=0, saving_path=pathlib.Path('.'), elm_type='poelm', dataset='brats', trigger_type='badnet', target_label=0, poison_percentage=5, hdlyr_size=5000, trigger_size=(4, 4))