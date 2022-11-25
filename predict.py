import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    #img_path = "../tulip.jpg"
    img_path =os.path.abspath(os.path.join('./data_set/flower_data/predict','OIP-C.jpg'))
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img) #将图片转化成tensor格式，这里得到的是一个三维tensor[channel H W]
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)  #所以对于图像处理，需要四个维度[batch channel H w]这是读取图片的要求所以在这里在0维度添加一个batch

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)  #解码

    # create model
    model = AlexNet(num_classes=5).to(device)

    # load model weights
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()  #关闭dropout
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()   #squeeze进行压缩就是将batch维度去掉
        predict = torch.softmax(output, dim=0)  #变成概率
        predict_cla = torch.argmax(predict).numpy()   #返回概率值最大处的索引值

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()