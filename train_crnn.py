import torch
import numpy as np
import random
import argparse
import os

from torch.nn import CTCLoss
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision.utils as utils
import torchvision.transforms as transforms

from models.model_crnn import CRNN
from datasets.ocr_dataset import OCRDataset
from datasets.img_dataset import ImgDataset
from utils import (
    get_char_maps,
    get_ocr_helper,
    random_subset,
    compare_labels,
    get_char_maps,
    pred_to_string,
)
from transform_helper import PadWhite, AddGaussianNoice
import properties as properties


class TrainCRNN:
    def __init__(self, args):
        print("Experiment Arguments")
        print(args)
        self.batch_size = args.batch_size
        self.random_seed = args.random_seed
        self.lr = args.lr
        self.max_epochs = args.epoch
        self.ocr = args.ocr
        self.std = args.std
        self.is_random_std = args.random_std
        self.dataset_name = args.dataset
        self.crnn_model_path = args.crnn_model_path
        self.crnn_ckpt_path = args.ckpt_path
        self.start_epoch = args.start_epoch
        self.train_batch_size = self.batch_size

        self.decay = 0.8
        self.decay_step = 10
        torch.manual_seed(self.random_seed)
        np.random.seed(torch.initial_seed())
        random.seed(torch.initial_seed())

        if self.dataset_name == "pos":
            self.train_set = os.path.join(
                args.data_base_path, properties.pos_text_dataset_train
            )
            self.validation_set = os.path.join(
                args.data_base_path, properties.pos_text_dataset_dev
            )
        elif self.dataset_name == "vgg":
            self.train_set = os.path.join(
                args.data_base_path, properties.vgg_text_dataset_train
            )
            self.validation_set = os.path.join(
                args.data_base_path, properties.vgg_text_dataset_dev
            )

        self.input_size = properties.input_size

        self.char_to_index, self.index_to_char, self.vocab_size = get_char_maps(
            properties.char_set
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.crnn_ckpt_path is None:
            self.model = CRNN(self.vocab_size, False).to(self.device)
        else:
            self.model = torch.load(self.crnn_ckpt_path).to(self.device)
        self.model.register_backward_hook(self.model.backward_hook)

        self.ocr = get_ocr_helper(self.ocr)

        transform = transforms.Compose(
            [
                PadWhite(self.input_size),
                transforms.ToTensor(),
            ]
        )
        noisy_transform = transforms.Compose(
            [
                PadWhite(self.input_size),
                transforms.ToTensor(),
                AddGaussianNoice(
                    std=self.std, is_stochastic=self.is_random_std, return_noise=False
                ),
            ]
        )
        if self.ocr is not None:
            dataset = OCRDataset(
                self.train_set, transform=noisy_transform, ocr_helper=self.ocr
            )

            validation_set = OCRDataset(
                self.validation_set, transform=transform, ocr_helper=self.ocr
            )
        else:
            dataset = ImgDataset(
                self.train_set,
                transform=noisy_transform,
                num_subset=args.train_subset,
                include_name=True,
            )
            validation_set = ImgDataset(
                self.validation_set, transform=transform, num_subset=args.val_subset
            )
        print(f"Train Dataset - {dataset}")
        print(f"Validation Dataset - {validation_set}")
        self.loader_train = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, drop_last=True, shuffle=True
        )

        self.loader_validation = torch.utils.data.DataLoader(
            validation_set, batch_size=self.batch_size
        )

        self.train_set_size = len(self.loader_train.dataset)
        self.val_set_size = len(self.loader_validation.dataset)
        print(
            f"Train Set size - {self.train_set_size}, Val Set Size - {self.val_set_size}"
        )

        self.loss_function = CTCLoss().to(self.device)
        self.loss_func_samplewise = CTCLoss(reduction="none").to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.decay_step, gamma=self.decay
        )

    def _call_model(self, images, labels):
        X_var = images.to(self.device)
        scores = self.model(X_var)
        out_size = torch.tensor([scores.shape[0]] * images.shape[0], dtype=torch.int)
        y_size = torch.tensor([len(l) for l in labels], dtype=torch.int)
        conc_label = "".join(labels)
        y = [self.char_to_index[c] for c in conc_label]
        y_var = torch.tensor(y, dtype=torch.int)
        return scores, y_var, out_size, y_size

    def train(self):
        best_val_acc = 0
        best_val_epoch = 0
        print(f"Batch size is {self.batch_size}")
        print(f"Train batch size is {self.train_batch_size}")
        validation_step = 0
        for epoch in range(self.start_epoch + 1, self.max_epochs):
            self.model.train()
            step = 0
            training_loss = 0
            for images, labels, names in self.loader_train:
                self.model.zero_grad()
                scores, y, pred_size, y_size = self._call_model(images, labels)
                loss = self.loss_function(scores, y, pred_size, y_size)
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item()
                if step % 100 == 0:
                    print(f"Epoch: {epoch}, Iteration: {step} => {loss.item()}")
                step += 1

            self.model.eval()
            validation_loss = 0
            pred_correct_count = 0
            pred_CER = 0
            label_count = 0
            with torch.no_grad():
                for images, labels in self.loader_validation:
                    scores, y, pred_size, y_size = self._call_model(images, labels)
                    loss = self.loss_function(scores, y, pred_size, y_size)
                    preds = pred_to_string(scores, labels, self.index_to_char)
                    crt, cer = compare_labels(preds, labels)
                    pred_correct_count += crt
                    pred_CER += cer
                    label_count += 1
                    validation_loss += loss.item()
                    validation_step += 1
            CRNN_accuracy = pred_correct_count / self.val_set_size
            if CRNN_accuracy > best_val_acc:
                best_val_acc = CRNN_accuracy
                best_val_epoch = epoch
                torch.save(
                    self.model,
                    f"{self.crnn_model_path}_{epoch}_{CRNN_accuracy*100:.2f}",
                )

            print(
                "Epoch: %d/%d => Training loss: %f | Validation loss: %f"
                % (
                    (epoch + 1),
                    self.max_epochs,
                    training_loss / (self.train_set_size // self.train_batch_size),
                    validation_loss / (self.val_set_size // self.batch_size),
                )
            )
            print(
                f"Validation Accuracy - {CRNN_accuracy*100}, {pred_correct_count} / {self.val_set_size}"
            )

            self.scheduler.step()
            # Save last model specifically
            if (epoch + 1) == 200:
                torch.save(
                    self.model,
                    f"{self.crnn_model_path}_{epoch}_{CRNN_accuracy*100:.2f}",
                )
                
        return best_val_acc, best_val_epoch


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Trains the CRNN model")
    parser.add_argument("--batch_size", type=int, default=32, help="input batch size")
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="learning rate, not used by adadealta"
    )
    parser.add_argument("--epoch", type=int, default=50, help="number of epochs")
    parser.add_argument(
        "--std",
        type=int,
        default=5,
        help="standard deviation of Gussian noice added to images (this value devided by 100)",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="random seed for shuffles"
    )
    parser.add_argument(
        "--ocr", help="performs training lebels from given OCR [Tesseract,EasyOCR]"
    )
    parser.add_argument(
        "--train_subset", help="Specify subset of training samples", type=int
    )
    parser.add_argument(
        "--val_subset", help="Specify subset of validation samples", type=int
    )

    parser.add_argument(
        "--dataset",
        default="pos",
        help="performs training with given dataset [pos, vgg]",
    )
    parser.add_argument(
        "--random_std",
        action="store_false",
        help="randomly selected integers from 0 upto given std value (devided by 100) will be used",
        default=True,
    )
    parser.add_argument(
        "--crnn_model_path",
        help="CRNN model save path. Default picked from properties",
        default=properties.crnn_model_path,
    )

    parser.add_argument(
        "--data_base_path",
        help="Base path training, validation and test data",
        default=".",
    )
    parser.add_argument("--ckpt_path", help="Path to CRNN checkpoint")
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=-1,
        help="Starting epoch. If loading from a ckpt, pass the ckpt epoch here.",
    )
    args = parser.parse_args()
    trainer = TrainCRNN(args)
    trainer.train()