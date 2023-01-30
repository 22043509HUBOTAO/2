import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import ImageTransform

# 製作數據集 按照8:2分配

# first
all_picture_list = []
rootpath = "./dataset/"
subpath = os.listdir(rootpath)
for subpathName in subpath:
    full_sub_path = os.path.join(rootpath, subpathName)
    if os.path.isdir(full_sub_path):
        file_names = os.listdir(full_sub_path)
        for file_name in file_names:
            full_file_path = os.path.join(full_sub_path, file_name)
            if os.path.splitext(full_file_path)[-1] == ".png":
                all_picture_list.append(full_file_path)

train_list = []
eval_list = []

all_files = len(all_picture_list)

train_list_length = int(all_files / 10.0 * 8.0)  # 訓練集的長度
eval_list_length = all_files - train_list_length  # 測試集的長度

is_run = True

train_list_tmp_map = {}
eval_list_tmp_map = {}

while is_run:
    for now in range(0, all_files):  # 算出現在是第幾個
        now_path = all_picture_list[now]  # 現在的名字
        is_add_to_train_list = np.random.randint(0, 2)  # 0加入訓練集 1加入測試集

        if train_list_tmp_map.get(now_path) is None and eval_list_tmp_map.get(now_path) is None:  # 如果他不在訓練集。也不在測試集
            if is_add_to_train_list == 0:  # 現在表示他要加入訓練集
                if len(train_list_tmp_map) != train_list_length:
                    train_list_tmp_map[now_path] = ""
            if is_add_to_train_list == 1:
                if len(eval_list_tmp_map) != eval_list_length:
                    eval_list_tmp_map[now_path] = ""

        tmp_length_1 = len(train_list_tmp_map)  # 臨時集的長度
        tmp_length_2 = len(eval_list_tmp_map)
        if tmp_length_1 + tmp_length_2 == all_files:
            is_run = False
            break

train_list = list(train_list_tmp_map.keys())
eval_list = list(eval_list_tmp_map.keys())


# 製作測試集和訓練集
def make_data_path_list(phase="train"):
    if phase == "train":
        return train_list
    elif phase == "val":
        return eval_list
    else:
        return []



class CancerDataset(data.Dataset):

    def __init__(self, file_list, transform: ImageTransform.ImageTransform = None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path).convert('RGB')

        img_transformed = self.transform(img, self.phase)

        label = -1  # for test
        dir_name = os.path.dirname(img_path)
        if dir_name.endswith("covid-19"):
            label = 0
        elif dir_name.endswith("normal"):
            label = 1

        return img_transformed, label
