import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import ImageTransform

# 製作數據集 按照8:2分配

# first
train_list_all = []
test_list_all = []
rootpath_train = "./train/"
rootpath_test = "./test/"
subpath_train = os.listdir(rootpath_train)
subpath_test = os.listdir(rootpath_test)
for subpathName in subpath_train:
    full_sub_path = os.path.join(rootpath_train, subpathName)
    if os.path.isdir(full_sub_path):
        file_names = os.listdir(full_sub_path)
        for file_name in file_names:
            full_file_path = os.path.join(full_sub_path, file_name)
            if os.path.splitext(full_file_path)[-1] == ".png":
                train_list_all.append(full_file_path)

for subpathName in subpath_test:
    full_sub_path = os.path.join(rootpath_test, subpathName)
    if os.path.isdir(full_sub_path):
        file_names = os.listdir(full_sub_path)
        for file_name in file_names:
            full_file_path = os.path.join(full_sub_path, file_name)
            if os.path.splitext(full_file_path)[-1] == ".png":
                test_list_all.append(full_file_path)
print(test_list_all)
train_list = []
eval_list = []

train_all_files = len(train_list_all)
test_all_files = len(test_list_all)

train_list_length = int(train_all_files)  # 訓練集的長度
eval_list_length = int(test_all_files)  # 測試集的長度

is_run = True

train_list_tmp_map = {}
eval_list_tmp_map = {}




for now in range(0, train_all_files):  # 算出現在是第幾個
        now_path = train_list_all[now]  # 現在的名
        train_list_tmp_map[now_path] = ""
        tmp_length_1 = len(train_list_tmp_map)  # 臨時集的長度

for now in range(0, test_all_files):  # 算出現在是第幾個
        now_path = test_list_all[now]  # 現在的名
        eval_list_tmp_map[now_path] = ""
        tmp_length = len(eval_list_tmp_map)  # 臨時集的長度





train_list = list(train_list_tmp_map.keys())
print(train_list)
eval_list = list(eval_list_tmp_map.keys())
print(eval_list)

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
        if dir_name.endswith("unnormal"):
            label = 0
        elif dir_name.endswith("normal"):
            label = 1

        return img_transformed, label
