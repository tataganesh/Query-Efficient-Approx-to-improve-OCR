import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import tesserocr
import numpy as np
import random
from utils import get_files, get_ununicode
import properties as properties
import random



class ImgDataset(Dataset):

    def __init__(self, data_dir, transform=None, include_name=False, include_index=False, num_subset=None):
        self.transform = transform
        self.include_name = include_name
        self.include_index = include_index
        self.files = []
        unprocessed = get_files(data_dir, ['png', 'jpg'])
        for img in unprocessed:
            if len(os.path.basename(img).split('_')[1]) <= properties.max_char_len:
                self.files.append(img)
        if num_subset:
            random.seed(42)  # Allows reproducibility of train/val subsets
            self.files = random.sample(self.files, num_subset)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        image = Image.open(img_name).convert("L")
        if self.transform != None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        file_name = os.path.basename(img_name)
        label = file_name.split('_')[1]
        label = get_ununicode(label) # Test accuracy on Tesseract to confirm if it causes any issues
        if self.include_name:
            sample = [image, label, file_name]
        else:
            sample = [image, label]
        if self.include_index:
            sample.append(idx)
        return sample

    def worker_init(self, pid):
        return np.random.seed(torch.initial_seed() % (2**32 - 1))
