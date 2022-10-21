"""
Use patch dataset to create json with the following format

{
    "index_label_foldername": cervalue
}

"""
import torch
import numpy as np
import random
import argparse
import os
import json
import traceback
import sys
sys.path.append("../")

import torchvision.utils as utils
import torchvision.transforms as transforms

from datasets.patch_dataset import PatchDataset
from utils import get_char_maps, get_ocr_helper
from utils import show_img, compare_labels, get_text_stack, get_ocr_helper
from transform_helper import PadWhite, AddGaussianNoice
import properties as properties
from tqdm import tqdm

class TrainCRNN():

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.random_seed = args.random_seed
        self.dataset_name = args.dataset
        self.cer_json_path = args.cers_save_path
        self.ocr = args.ocr
        torch.manual_seed(self.random_seed)
        np.random.seed(torch.initial_seed())
        random.seed(torch.initial_seed())

        self.text_strip_cer = dict()

        if self.dataset_name == 'pos':
            self.train_set =  os.path.join(args.data_base_path, properties.patch_dataset_train)
            self.validation_set = os.path.join(args.data_base_path, properties.patch_dataset_dev)
        elif self.dataset_name == 'vgg':
            self.train_set = os.path.join(args.data_base_path, properties.vgg_text_dataset_train)
            self.validation_set = os.path.join(args.data_base_path, properties.vgg_text_dataset_dev)

        self.input_size = properties.input_size

        self.char_to_index, self.index_to_char, self.vocab_size = get_char_maps(
            properties.char_set)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.ocr = get_ocr_helper(self.ocr)

        # transform = transforms.Compose([
        #     PadWhite(self.input_size),
        #     transforms.ToTensor(),
        # ])
        # dataset_ocr = OCRDataset(
        #     self.train_set, transform=transform, ocr_helper=self.ocr, include_name=True)
                
        # dataset_gt = ImgDataset(
        #     self.train_set, transform=transform, include_name=True)
        # self.loader_ocr = torch.utils.data.DataLoader(
        #     dataset_ocr, batch_size=self.batch_size)
        # self.loader_gt = torch.utils.data.DataLoader(
        #     dataset_gt, batch_size=self.batch_size)
        self.train_set = PatchDataset(
            self.train_set, pad=True, include_name=True)
        self.validation_set = PatchDataset(
            self.validation_set, pad=True)
        self.loader_train = torch.utils.data.DataLoader(
            self.train_set, batch_size=32, shuffle=True, drop_last=True, collate_fn=PatchDataset.collate)

    def inference(self):
        # writer = SummaryWriter(self.tb_log_path)

        # print(f"Train batch size is {self.train_batch_size}")
        iters = 0
        count = 0
        # for _, gt_labels, gt_names in self.loader_gt:
        #     count += len(gt_labels)
        # for ocr_details, gt_details in zip(self.loader_ocr, self.loader_gt):
        #     ocr_images, ocr_labels, ocr_names = ocr_details
        #     _, gt_labels, gt_names = gt_details
        #     for i in range(len(ocr_labels)):
        #         try:
        #             _, o_cer = compare_labels([ocr_labels[i]], [gt_labels[i]])
        #         except Exception:
        #             traceback.print_exc() 
        #             print(ocr_labels[i], gt_labels[i], gt_names[i])
        #             exit()
        #         self.text_strip_cer[gt_names[i]] = o_cer
        #     iters += 1
        #     if iters % 100 == 0:
        #         print(iters)
        #     count += len(ocr_labels)
        for images, labels_dicts, names in tqdm(self.loader_train):
            for i in range(len(labels_dicts)):
                image = images[i]
                labels_dict = labels_dicts[i]
                name = names[i]
                # image = image.unsqueeze(0)
                X_var = image.to(self.device)
                text_crops_all, labels = get_text_stack(
                        X_var, labels_dict, self.input_size)
                ocr_output = self.ocr.get_labels(text_crops_all)
                folder_name, file_name = name.split("/")[-2:]
                file_name = file_name.split(".")[0]
                for j in range(len(labels)):
                    _, o_cer = compare_labels([ocr_output[j]], [labels[j]])
                    # print([ocr_output[i], labels[i], o_cer])
                    text_strip_name = f"{j}_{labels[j]}_{folder_name}_{file_name}"
                    self.text_strip_cer[text_strip_name] = o_cer

                

        with open(self.cer_json_path, 'w') as f:
            json.dump(self.text_strip_cer, f)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains the CRNN model')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='input batch size')
    parser.add_argument('--data_base_path',
                        default=32, help='input batch size')
    parser.add_argument('--random_seed', type=int,
                        default=42, help='random seed for shuffles')
    parser.add_argument('--ocr', default="Tesseract", 
                        help="performs training lebels from given OCR [Tesseract,EasyOCR]")
    parser.add_argument('--dataset', default='pos',
                        help="performs training with given dataset [pos, vgg]")

    parser.add_argument('--cers_save_path', default='pos_cers.json', 
                        help="Save CERs")
    args = parser.parse_args()
    print(args)
    trainer = TrainCRNN(args)
    trainer.inference()