# import libs
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import GoogLeNet
from torchvision.models import resnet34
import PIL
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as opt
import numpy as np
import torch.nn.functional as F
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

train_transforms = T.Compose(
    [
     T.Resize((350, 350), interpolation=3),
     T.ColorJitter(hue=.05, saturation=.05), # изменение цвета
     T.RandomHorizontalFlip(p=0.2), # случайное горизонтальное переворачивание
     T.RandomRotation(20, resample=PIL.Image.BILINEAR),
     T.ToTensor(),
     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

#Test Time Augmentation
test_transforms = T.Compose(
    [
     T.Resize((350, 350), interpolation=3),
     T.ToTensor(),
     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
train_set = datasets.ImageFolder(PATH_TRAIN, train_transforms)
# train_set2 = datasets.ImageFolder('/content/gdrive/My Drive/dataset/car_human_noise/train', train_transforms2)
# train_set = train_set1 + train_set2
test_dataset = datasets.ImageFolder(PATH_TEST, test_transforms)
#separate train data to train and valid datasets
train_size = int(1 * len(train_set))
valid_size = len(train_set) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(train_set, [train_size, valid_size])


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                           pin_memory=True, shuffle = True,
                                           num_workers=0)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32,
                                           num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20,
                                          num_workers=0)

print("train_sampler: ", len(train_dataset))
print("valid_sampler: ", len(valid_dataset))
print("test_sampler: ", len(test_dataset))

PATH = "/content/gdrive/My Drive/dataset/mod2.pth"
# model = resnet34(3)
# model = GoogLeNet(3)
model = resnet34(13)
model = model.cuda()

optimizer = opt.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss().cuda()

max_epoch = 100
metric_valid = []
best_result = 0.0

for epoch in range(max_epoch):
    model.train()
    print("epoch : ", epoch)
    for iteration, (train_dataset, labels) in enumerate(train_loader):
        train_dataset, labels = train_dataset.cuda(), labels.cuda()
        # images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(train_dataset)  # forward
        loss = criterion(outputs, labels)  # without Aux only 1 output

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    trueth = 0
    total = 0
    for it, (test_dataset, labels) in enumerate(test_loader):
        test_dataset, labels = test_dataset.cuda(), labels.cuda()
        outputs = model(test_dataset)
        _, preds = torch.max(outputs, 1)
        trueth += accuracy_score(labels.cpu().numpy(), preds.cpu().numpy(), normalize=False) # accuracy_score(np.array(labels.cpu()), np.array(preds.cpu()), normalize=False)
        total += len(labels)
    result = trueth/total
    if result > best_result:
      print("find best model in ", epoch, " epoch")
      torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, PATH)
      best_result = result
    # print(loss.item())
    # //torch.save(model.state_dict(), f'{ep}.pth')
    # //model.load_state_dict(torch.load('model.pth'))
    # print("Validation...")
    # model.eval()
    # Trueth = 0
    # total = 0
    # for iteration, (valid_dataset, labels) in enumerate(valid_loader):
    #   valid_dataset, labels = valid_dataset.cuda(), labels.cuda()
    #   outputs = model(valid_dataset) # forward
    #   #outputs = (outputs2[0] + outputs2[1] + outputs2[2])/3
    #   _, preds = torch.max(outputs, 1)
    #   #loss = criterion(outputsV, labels)
    #   #Trueth += accuracy_score(np.array(labels.cpu()), np.array(preds.cpu()), normalize=False)
    #   Trueth += (preds == labels).sum().item()
    #   total += len(labels)
    # print(loss.item())
    # result = (Trueth/total)
    # print("Accuracy valid = ", str(result))
    # metric_valid.append(result)
    # if result > best_result:
    #   print("find best model in ", epoch, " epoch")
    #   torch.save({
    #           'epoch': epoch,
    #           'model_state_dict': model.state_dict(),
    #           'optimizer_state_dict': optimizer.state_dict(),
    #           'loss': loss
    #           }, PATH)
    #   #best_model = model #https://pytorch.org/tutorials/beginner/saving_loading_models.html
    #   best_result = result

    # //print(f'Accuracy on validation = {truth / len(valid_dataset)}')
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}, PATH)

PATH = "/content/gdrive/My Drive/dataset/mod2.pth"
checkpoint = torch.load(PATH)
# model = GoogLeNet(5).cuda()
model = resnet34(13).cuda()

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

trueth = 0
total = 0
for it, (test_dataset, labels) in enumerate(test_loader):
  test_dataset, labels = test_dataset.cuda(), labels.cuda()
  outputs = model(test_dataset)
  _, preds = torch.max(outputs, 1)
  trueth += (preds == labels).sum().item() #accuracy_score(np.array(labels.cpu()), np.array(preds.cpu()), normalize=False)
  total += len(labels)
print("Accuracy test = ", str(trueth/total))