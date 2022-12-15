#!/usr/bin/env python3
import re
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset

IMG_SIZE = 256

class MapDataset(Dataset):
    def __init__(self, map_list, label_list, map_types, img_size):
        self.map_list = map_list
        self.label_list = label_list
        self.map_types = map_types
        self.transforms = transforms.Compose([
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomVerticalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          ])

    def __getitem__(self, idx):
        # read/resize map
        img = cv2.imread(self.map_list[idx], 0)
        
        if img.shape != (IMG_SIZE, IMG_SIZE):
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        
        # read labels
        lab = np.load(self.label_list[idx])
        digits = [int(x) for x in re.findall(r'\d+', self.label_list[idx])]

        # 0 - free space 1 - obstacle 2 - start 3- goal
        img = np.where(img !=255, 1, 0)
        img[digits[1], digits[2]] = 2
        img[digits[3], digits[4]] = 3

        # tensor conversion
        img = img.astype(dtype='float32')
        img = torch.from_numpy(img)
        lab = lab.astype(dtype='float32')
        lab = torch.from_numpy(lab)

        return img, lab
    
    def __len__(self):
        return len(self.map_list)

