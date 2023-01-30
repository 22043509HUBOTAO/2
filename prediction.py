import torch
import os
from PIL import Image
import ImageTransform
from torchvision import datasets
import ImageTransform as it
import matplotlib.pyplot as plt
import numpy as np
import make_data_set
import torchvision
from torch.autograd import Variable

from tqdm import tqdm

# print(net)

size = 224
# mean = (0.5, 0.5, 0.5)
std = (0.220, 0.220, 0.220)
mean = (0, 0, 0)

work_path = "./"

pictures_dir = "test/unnormal"


pictures_path = work_path + pictures_dir

model_file_path = work_path + "model.pth"

use_pretrained = True
net = torch.load(model_file_path)
# net = torchvision.models.vgg16(pretrained=True)
net = net.eval()


def resize_picture(path):
    img = Image.open(path).convert('RGB')
    transform = ImageTransform.ImageTransform(size, mean, std)
    img_transformed = transform(img, 'val')
    return img_transformed


def get_pictures_list():
    pictures_list_tmp = os.listdir(pictures_path)
    full_path_list = []
    for file_name in pictures_list_tmp:
        if file_name.endswith(".png"):
            full_path_tmp = os.path.join(os.getcwd(), pictures_dir, file_name)
            full_path_list.append(full_path_tmp)
    return full_path_list


pictures_list = get_pictures_list()


transform = ImageTransform.ImageTransform(size, mean, std)
test_dataset = make_data_set.CancerDataset(pictures_list, transform, 'test')

# print(test_dataset)

# p,l = test_dataset.__getitem__(0)
# p = torch.unsqueeze(p, dim=0).float()
# p = Variable(p, requires_grad=False)


# net.eval()

# for inputs, labels, in test_dataset:
#     inputs = torch.unsqueeze(inputs, dim=0).float()
#     inputs = Variable(inputs, requires_grad=False)
#     prediction = net(Variable(inputs, False))
#     max = torch.max(prediction, dim=1)
#     print(max[1])


for full_path in pictures_list:
    print("predict : " + full_path)
    img = resize_picture(full_path)
    img = torch.unsqueeze(img, dim=0).float()
    img = Variable(img, requires_grad=False)
    prediction = net(Variable(img, False))
    _, preds = torch.max(prediction, 1)

    print(preds)
    print(_)

