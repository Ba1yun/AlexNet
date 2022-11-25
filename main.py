
'''
import os
print(os.path.abspath('E:\B\AlexNet'))
print(os.getcwd())
#data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
#data_root=os.path.join(os.getcwd(), "../")
data_root = os.path.abspath(os.path.join('E:\B\AlexNet','E:\B'))
print(data_root)
os.chdir()
os.listdir()
'''
from torchvision import  datasets,transforms
import os
import json
data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
data_root = os.path.abspath(os.getcwd())
    #print(data_root)
image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
flower_list = train_dataset.class_to_idx
print(flower_list)
#cla_dict = dict((val, key) for key, val in flower_list.items())
json_str = json.dumps(flower_list, indent=4)
with open('class_indicess.json', 'w') as json_file:json_file.write(json_str)