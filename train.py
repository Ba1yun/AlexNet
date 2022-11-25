import os
import sys
import json
import time

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

#弄一个训练和测试字典
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    #data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    data_root = os.path.abspath(os.getcwd())
    #print(data_root)
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    #print(image_path)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx   #获得训练数据集的一个索引
    cla_dict = dict((val, key) for key, val in flower_list.items())  #字典值和关键字对换，上面获取索引是这个格式'roses':2所以要对调一下变成‘2’：ross
    # write dict into json file  #就是一个索引和类别对应关系图
    json_str = json.dumps(cla_dict, indent=4)  #把数据格式变成json格式形式方便读取
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    #nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers，这里是看下边num_workers=0，电脑不同设置一般不同而Windows是0
    #print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=0)

    print("using {} images for training, {} images for validation.".format(train_num,val_num))
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=5, init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)  #优化的对象时模型当中所有的参数

    epochs = 10
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            rate=(step+1)/len(train_loader)
            a="*"*int(rate*50)
            b="."*int((1-rate)*50)
            #print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".formate(int(rate*100),a,b,loss),end="")
            #print()
            #print(time.perf_counter()-t1)

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:   #每一轮都会判断一次只有准确率比上次的高才会来这一步才会保存最好准确率的那一次
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()

'''
1.transforms.RandomResizedCrop(224),随机裁剪224*224大小
2.RandomHorizontalFlip()，随机水平翻转
3.transforms.Normalize，标准化处理
4.os.path.join(os.getcwd()，获取当前路径
5.  ../..  两个点就是返回上层目录
6.  ./     代表目前所在的目录就是代码文件所在目录
7.  ../    代表上一层目录就是代码文件所在文件夹
8.   /     代表根目录，c盘、d盘等下面的文件
9.net.train()和net.eval()这个是管理dropout和BN层，只有在训练的时候才使用者两个实际使用的时候不适用，net.train()使用调用时就会启用随机丢失和归一化eval就相反不启用
10.[transforms.RandomResizedCrop(224)随机裁剪图片然后将裁剪的图片按统一比例缩放
11.transforms.RandomHorizontalFlip()按一定的比例旋转图片
12.os.path.join('C:\\demo\\exercise', filename) 把filename加到C:\demo\exercise路径下（是把几个路径连接起来：\demo\exercise\filename）
13.os.getcwd() 函数可以取得当前工作路径的字符串
14.os.chdir() 修改当前工作路径，让工作路径变成这个路径，不存在就报错
15.os.path.abspath(path) 返回 path 参数的绝对路径的字符串
16.os.path.abspath(),就是把路径转化成绝对路径格式的字符串形式，如果用join等操作后不用这个函数就会导致路径格式不对电脑找不到这个路径
17.os.path.join(),对于目录和子目录那么join后是大目录如os.path.abspath(os.path.join('E:\B\AlexNet','E:\B'))结果是E:\B
18.datasets.ImageFolder（）是一个图像数据库处理的函数，就是在该文件下要放训练和验证的数据集的类文件然后类文件下面是类图片（自己做数据集使用）
19.train_dataset继承了datasets.ImageFolder,所以可以直接调用class_to_idx
'''