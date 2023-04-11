import os
import json
import random
import torch

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageOps

from utils import get_files
import properties as properties


class PatchDataset(Dataset):

    def __init__(self, data_dir, pad=False, include_name=False, num_subset=None, resize_images=False):
        self.pad = pad
        self.include_name = include_name
        self.files = get_files(data_dir, ['png', 'jpg', 'jpeg'])
        if num_subset:
            random.seed(42)  # Allows reproducibility of train/val subsets
            self.files = random.sample(self.files, num_subset)
        self.size = (400, 512)
        self.resize_images = resize_images

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        image = Image.open(img_name).convert("L")
        w, h = image.size
        top_padding, left_padding = 0, 0
        resize_w, resize_h = 1, 1
        if self.pad:

            if h <= self.size[0] or w <= self.size[1]:
                delta_height = self.size[0] - h
                delta_width = self.size[1] - w
                pad_height = delta_height // 2
                pad_width = delta_width // 2
                # (left, top, right, bottom)
                padding = (pad_width, pad_height, delta_width -
                           pad_width, delta_height-pad_height)
                image = ImageOps.expand(image, padding, fill=255)
                top_padding = pad_height
                left_padding = pad_width
            elif h > 400 or w > 500:
                if self.resize_images:
                    image = transforms.Resize(self.size)(image)
                    resize_h = self.size[0]/h
                    resize_w = self.size[1]/w
                else:
                    print("Height screwed", idx)

        image = transforms.ToTensor()(image)
        
        label = self.coord_loader(img_name, top_padding, left_padding, resize_w, resize_h)
        if self.include_name:
            sample = (image, label, img_name)
        else:
            sample = (image, label)
        return sample

    def coord_loader(self, img_path, top_padding=0, left_padding=0, resize_w=1, resize_h=1):
        f = open(img_path.rsplit(".", 1)[0] + ".json", 'r')
        label_list = json.loads(f.read())
        # print(label_list)
        f.close()
        label_list_out = []
        for i, text_area in enumerate(label_list):
            label = text_area['label']
            if len(label_list) != 0 and 'x1' in label_list[0]:
                y1 = text_area['y1'] + top_padding
                y2 = text_area['y2'] + top_padding
                y3 = text_area['y3'] + top_padding
                y4 = text_area['y4'] + top_padding
                x1 = text_area['x1'] + left_padding
                x2 = text_area['x2'] + left_padding
                x3 = text_area['x3'] + left_padding
                x4 = text_area['x4'] + left_padding

                x_min = int(min([x1, x2, x3, x4]) * resize_w)
                y_min = int(min([y1, y2, y3, y4]) * resize_h)
                x_max = int(max([x1, x2, x3, x4]) * resize_w)
                y_max = int(max([y1, y2, y3, y4]) * resize_h)
            else:
                x_min = text_area['x_min'] + left_padding
                y_min = text_area['y_min'] + top_padding
                x_max = text_area['x_max'] + left_padding
                y_max = text_area['y_max'] + top_padding

                y1 = y2 = y_min
                y3 = y4 = y_max
                x1 = x4 = x_min
                x2 = x3 = x_max

            if len(label) <= properties.max_char_len and x_max - x_min < 128 and y_max - y_min < 32:
                out = {'label': label, 'y1': y1, 'y2': y2, 'y3': y3, 'y4': y4, 'x1': x1, 'x2': x2,
                       'x3': x3, 'x4': x4, 'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max, "index": i}
                label_list_out.append(out)

        if len(label_list_out) == 0:
            label_list_out.append(
                {'label': properties.empty_char, 'x_min': 0, 'y_min': 0, 'x_max': 127, 'y_max': 31, "index": 0})
        # print(label_list_out)
        return label_list_out

    def pad_height(self, image, height=400):
        _, h = image.size
        pad_bottom = height - h
        padding = (0, 0, 0, pad_bottom)
        return ImageOps.expand(image, padding, fill=255)

    def shuffle(self):
        random.shuffle(self.files)

    def collate(data):
        images = []
        labels = []
        if len(data[0]) == 3:
            names = []
            for item in data:
                images.append(item[0])
                labels.append(item[1])
                names.append(item[2])
            return [torch.stack(images), labels, names]
        else:
            for item in data:
                images.append(item[0])
                labels.append(item[1])
            return [torch.stack(images), labels]
