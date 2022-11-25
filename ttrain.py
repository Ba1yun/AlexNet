import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from model import AlexNet

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

train_dataset=datasets.ImageFolder(root='./data_set/flower_data/train',transform=data_transform["train"]) #训练集下载下来了
train_dataloader=DataLoader(dataset=train_dataset,batch_size=1,shuffle=True, num_workers=0)#训练数据加载器

test_dataset=datasets.ImageFolder(root='./data_set/flower_data/val',transform=data_transform["val"])
#print(len(test_dataset))
test_dataloader=DataLoader(dataset=test_dataset,batch_size=1,shuffle=True, num_workers=0)

device="cuda" if torch.cuda.is_available() else "cpu"
net=AlexNet().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer =  optim.Adam(net.parameters(), lr=0.0002)

def train (train_dataloader,net,loss_function,optimizer):
    acc,loss,n=0,0,0
    epochs=10
    for step, data in enumerate(train_dataloader):
        images, labels = data
        images, labels=images.to(device), labels.to(device)
        outputs = net(images)
        my_loss = loss_function(outputs, labels)
        _, pred = torch.max(outputs, axis=1)
        cur_acc = torch.sum(labels == pred).numpy() / outputs.shape[0]
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()
        loss += my_loss.item()
        acc += cur_acc.item()
        n+=1
        print("train_loss" + str(loss / n))
        print("train_acc" + str(acc / n))

def val(train_dataloader,net,loss_function,optimizer):
    acc, loss, n = 0, 0, 0
    epochs = 10
    for step, data in enumerate(test_dataloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        my_loss = loss_function(outputs, labels)
        _, pred = torch.max(outputs, axis=1)
        cur_acc = torch.sum(labels == pred).numpy() / outputs.shape[0]
        loss += my_loss.item()
        acc += cur_acc.item()
        n += 1
        print("val_loss" + str(loss / n))
        print("val_acc" + str(acc / n))
        return acc / n

epoch=50   #训练轮次
min_acc=0
for i in range(epoch):
    print(f'epoch{i+1}\n----------------')
    train(train_dataloader,net,loss_function,optimizer)
    a=val(test_dataloader,net,loss_function,optimizer)
    if a > min_acc:
        min_acc = a
        print('save best model')
        torch.save(net.state_dict(), './AlexNet_1.pth')
    print('Done')








