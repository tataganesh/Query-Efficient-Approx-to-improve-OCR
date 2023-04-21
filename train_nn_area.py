import datetime
import torch
import argparse
import os
import math
import random
from torch.nn import CTCLoss, MSELoss
import torch.optim as optim

import torchvision.transforms as transforms

from models.model_crnn import CRNN
from models.model_unet import UNet
from datasets.img_dataset import ImgDataset
from utils import get_char_maps, set_bn_eval, pred_to_string
from utils import get_ocr_helper, compare_labels, save_img, create_dirs
from utils import random_subset, save_json
from transform_helper import PadWhite, AddGaussianNoice
import properties as properties
from selection_utils import datasampler_factory, update_entropies
from label_tracking import tracking_methods, tracking_utils
from tracking_utils import call_crnn, weighted_ctc_loss, generate_ctc_target_batches, add_labels_to_history
import wandb
import json
wandb.Table.MAX_ROWS = 50000
import pandas as pd
import shutil
wandb.init(project='ocr-calls-reduction', entity='tataganesh', tags=["VGG"])

minibatch_subset_methods = {"random": random_subset}


class TrainNNPrep():

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.lr_crnn = args.lr_crnn
        self.lr_prep = args.lr_prep
        self.max_epochs = args.epoch
        self.warmup_epochs = args.warmup_epochs # Todo: Inverstigate impact
        self.inner_limit = args.inner_limit
        self.crnn_model_path = args.crnn_model
        self.prep_model_path = args.prep_model

        self.exp_base_path = args.exp_base_path
        self.ckpt_base_path = os.path.join(self.exp_base_path, properties.prep_crnn_ckpts)
        self.cers_base_path = os.path.join(self.exp_base_path, "cers")
        self.tracked_labels_path = os.path.join(self.exp_base_path, "tracked_labels")
        self.selectedsamples_path = os.path.join(self.exp_base_path, "selected_samples")
        self.entropies_path = os.path.join(self.exp_base_path, "entropies")
        self.tensorboard_log_path = os.path.join(self.exp_base_path, properties.prep_tensor_board)
        self.img_out_path = os.path.join(self.exp_base_path, properties.img_out)
        create_dirs([self.exp_base_path, self.ckpt_base_path, self.tensorboard_log_path, self.img_out_path, self.cers_base_path, self.tracked_labels_path, self.selectedsamples_path, self.entropies_path])

        self.sec_loss_scalar = args.scalar
        self.ocr_name = args.ocr
        self.std = args.std
        self.is_random_std = args.random_std
        self.inner_limit_skip = args.inner_limit_skip
        torch.manual_seed(42)
        self.train_set = os.path.join(args.data_base_path, properties.vgg_text_dataset_train)
        self.validation_set = os.path.join(args.data_base_path, properties.vgg_text_dataset_dev)
        self.start_epoch = args.start_epoch
        self.selection_method = args.minibatch_subset
        self.train_batch_size = self.batch_size
        
        self.train_batch_prop = 1
        if args.minibatch_subset_prop is not None and self.selection_method:
            self.train_batch_prop = args.minibatch_subset_prop
        self.train_subset_size = args.train_subset_size
        self.val_subset_size = args.val_subset_size
        
        self.cers = None
        self.entropies = dict()
        self.selected_samples = dict()
        if args.cers_ocr_path:
            with open(args.cers_ocr_path, 'r') as f:
                self.cers = json.load(f)
                
            for key in self.cers.keys():
                self.selected_samples[key] = [False] * self.max_epochs
                self.entropies[key] = 0
                
        if self.selection_method:
            self.cls_sampler = datasampler_factory(self.selection_method)
            if self.selection_method in ("topKCER", "rangeCER"):
                self.sampler = self.cls_sampler(self.cers)
            elif self.selection_method == "uniformEntropy":
                self.sampler = self.cls_sampler(self.entropies, self.cers)
            else:
                self.sampler = self.cls_sampler()
        
        if self.cers:
            self.tracked_labels = {name: [] for name in self.cers.keys()}  
                      
        self.input_size = properties.input_size
        self.ocr = get_ocr_helper(self.ocr_name)

        self.char_to_index, self.index_to_char, self.vocab_size = get_char_maps(
            properties.char_set)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        if self.crnn_model_path is None:
            self.crnn_model = CRNN(self.vocab_size, False).to(self.device)
        else:
            self.crnn_model = torch.load(
                self.crnn_model_path).to(self.device)
        self.crnn_model.register_backward_hook(self.crnn_model.backward_hook)

        if self.prep_model_path is None:
            self.prep_model = UNet().to(self.device)
        else:
            self.prep_model = torch.load(
                self.prep_model_path).to(self.device)

        transform = transforms.Compose([
            PadWhite(self.input_size),
            transforms.ToTensor(),
        ])
        
        self.window_size = args.window_size
        self.weightgen_method = args.weightgen_method
        
        WeightGenCls = tracking_methods.weightgenerator_factory(args.weightgen_method)
        self.loss_wghts_gnrtr = WeightGenCls(args, self.device, self.char_to_index)
        self.train_set = ImgDataset(
            self.train_set, transform=transform, include_name=True, include_index=True)
        self.validation_set = ImgDataset(
            self.validation_set, transform=transform, include_name=True)

        if not self.train_subset_size:
            self.train_subset_size = len(self.train_set)        
        train_rand_indices = torch.randperm(len(self.train_set))[:self.train_subset_size]
        print(train_rand_indices.shape)
        train_rand_sampler = torch.utils.data.SubsetRandomSampler(train_rand_indices)
        self.loader_train = torch.utils.data.DataLoader(
            self.train_set, batch_size=self.batch_size, sampler=train_rand_sampler, drop_last=True)
       
        if not self.val_subset_size:
            self.val_subset_size = len(self.validation_set)
        val_rand_indices = torch.randperm(len(self.validation_set))[:self.val_subset_size]
        val_rnd_sampler = torch.utils.data.SubsetRandomSampler(val_rand_indices)
        self.loader_validation = torch.utils.data.DataLoader(
            self.validation_set, batch_size=self.batch_size, sampler=val_rnd_sampler, drop_last=True)

        self.train_set_size = len(self.loader_train.dataset)
        self.val_set_size = len(self.loader_validation.dataset)
        self.sample_importance = torch.ones(self.train_set_size)/10.0
        self.sample_frequency = torch.zeros((self.train_set_size, self.max_epochs, 2))
        self.model_labels_last = ["" for _ in range(0, self.train_subset_size)]

        self.primary_loss_fn = CTCLoss().to(self.device)
        self.primary_loss_fn_sample_wise = CTCLoss(reduction='none').to(self.device)
        self.secondary_loss_fn = MSELoss().to(self.device)
        self.optimizer_crnn = optim.Adam(
            self.crnn_model.parameters(), lr=self.lr_crnn, weight_decay=0)
        self.optimizer_prep = optim.Adam(
            self.prep_model.parameters(), lr=self.lr_prep, weight_decay=0)
        
        self.lr_scheduler = args.lr_scheduler 
        if self.lr_scheduler == "cosine":
            self.scheduler_crnn = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_crnn, T_max=self.max_epochs)
            # self.scheduler_prep = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_prep, T_max=self.max_epochs)

    def _call_model(self, images, labels):
        X_var = images.to(self.device)
        scores = self.crnn_model(X_var)
        out_size = torch.tensor(
            [scores.shape[0]] * images.shape[0], dtype=torch.int)
        y_size = torch.tensor([len(l) for l in labels], dtype=torch.int)
        conc_label = ''.join(labels)
        y = [self.char_to_index[c] for c in conc_label]
        y_var = torch.tensor(y, dtype=torch.int)
        return scores, y_var, out_size, y_size

    def _get_loss(self, scores, y, pred_size, y_size, img_preds):
        pri_loss = self.primary_loss_fn(scores, y, pred_size, y_size)
        sec_loss = self.secondary_loss_fn(img_preds, torch.ones(
            img_preds.shape).to(self.device))*self.sec_loss_scalar
        loss = pri_loss + sec_loss
        return loss

    def add_noise(self, imgs, noiser, noise_coef=1):
        noisy_imgs = []
        added_noise = []
        for img in imgs:
            noisy_img, noise = noiser(img, noise_coef)
            added_noise.append(noise)
            noisy_imgs.append(noisy_img)
        return torch.stack(noisy_imgs), torch.stack(added_noise)

    def train(self):
        noiser = AddGaussianNoice(
            std=self.std, is_stochastic=self.is_random_std, return_noise=True)
        
        print(f"Batch size is {self.batch_size}")
        print(f"Train batch size is {self.train_batch_size}")
        validation_step = 0
        total_bb_calls = 0
        total_crnn_updates = 0
        best_val_acc = 0
        best_val_epoch = 0
        self.crnn_model.zero_grad()  #  why is this required?
        
        for epoch in range(self.start_epoch, self.max_epochs):
            epoch_bb_calls = 0
            epoch_crnn_updates = 0
            step = 0
            training_loss = 0
            epoch_print_flag = True

            for images, labels, names, indices in self.loader_train:
                self.crnn_model.train()
                self.prep_model.eval()
                self.prep_model.zero_grad()
                self.crnn_model.zero_grad()
                X_var = images.to(self.device)
                img_preds_all = self.prep_model(X_var)
                temp_loss = 0
                if self.selection_method and epoch >= self.warmup_epochs:
                    num_bb_samples = max(1, math.ceil(img_preds_all.shape[0] * (1 - self.train_batch_prop)))
                    img_preds, labels_gt, bb_sample_indices = self.sampler.query(img_preds_all, labels, num_bb_samples, names)

                    img_preds = img_preds.detach().cpu()
                    img_preds_names = [names[index] for index in bb_sample_indices]
                    skipped_mask = torch.ones(indices.shape[0], dtype=bool)
                    skipped_mask[bb_sample_indices] = False
                    
                    for name in img_preds_names:
                        self.selected_samples[name][epoch] = True
                    
                else:
                    img_preds = img_preds_all.detach().cpu() 
                    img_preds_names = names
                    skipped_mask = torch.zeros(img_preds.shape[0], dtype=bool)
                    
                if epoch_print_flag:
                    print(f"Total Samples - {img_preds_all.shape[0]}")
                    print(f"OCR Samples - {img_preds.shape[0]}")
                    epoch_print_flag = False
                    
                if self.selection_method and self.train_batch_prop < 1:
                    for i in range(self.inner_limit):
                        self.prep_model.zero_grad()
                        if i == 0 and self.inner_limit_skip:
                            ocr_labels = self.ocr.get_labels(img_preds)
                            loss_weights = self.loss_wghts_gnrtr.gen_weights(self.tracked_labels, img_preds_names)
                            add_labels_to_history(self, img_preds_names, ocr_labels)
                            # Peek at history of OCR labels for each strip and construct weighted CTC loss
                            target_batches = generate_ctc_target_batches(self, img_preds_names)
                            scores, pred_size = call_crnn(self, img_preds)
                            loss = weighted_ctc_loss(self, scores, pred_size, target_batches, loss_weights)
                            total_bb_calls += len(ocr_labels)
                            epoch_bb_calls += len(ocr_labels)
                        else:
                            noisy_imgs, noise = self.add_noise(img_preds, noiser)
                            ocr_labels = self.ocr.get_labels(noisy_imgs)
                            scores, y, pred_size, y_size = self._call_model(
                                noisy_imgs, ocr_labels)
                            loss = self.primary_loss_fn(
                                scores, y, pred_size, y_size)
                            total_bb_calls += img_preds.shape[0]
                            epoch_bb_calls += img_preds.shape[0]

                    if self.inner_limit:
                        temp_loss += loss.item()
                        loss.backward()
                inner_limit = max(1, self.inner_limit)
                CRNN_training_loss = temp_loss/inner_limit
                if self.inner_limit:
                    self.optimizer_crnn.step()

                self.prep_model.train()
                self.crnn_model.train()
                self.crnn_model.apply(set_bn_eval)
                self.prep_model.zero_grad()
                self.crnn_model.zero_grad()

                img_preds = self.prep_model(X_var)
                scores, y, pred_size, y_size = self._call_model(
                    img_preds, labels)
                loss = self._get_loss(scores, y, pred_size, y_size, img_preds)
                loss.backward()
                self.optimizer_prep.step()

                # Update last seen prediction of image
                model_gen_labels = pred_to_string(scores, labels, self.index_to_char)

                training_loss += loss.item()
                if step % 100 == 0:
                    print(f"Epoch: {epoch}, Iteration: {step} => {loss.item()}")
                step += 1 
                
                if self.selection_method and len(img_preds_names):
                    batch_cers = list()
                    for i in range(len(labels)):
                        _, batch_cer = compare_labels([model_gen_labels[i]], [labels[i]])
                        batch_cers.append(batch_cer)
                    self.sampler.update_cer(batch_cers, names)
                    
                    if self.selection_method == "uniformEntropy":
                        update_entropies(self, scores, names)
            
            train_loss = training_loss / (self.train_set_size//self.train_batch_size)
            crnn_train_loss = CRNN_training_loss / max(1, epoch_bb_calls)

            if self.selection_method:                            
                save_json(self.tracked_labels, os.path.join(self.tracked_labels_path, f"tracked_labels_{epoch}.json"))
                save_json(self.tracked_labels, os.path.join(self.tracked_labels_path, f"tracked_labels_current.json"), wandb_obj=wandb)
                save_json(self.selected_samples, os.path.join(self.selectedsamples_path, f"selected_samples_current.json"), wandb_obj=wandb)
                save_json(self.sampler.all_cers, os.path.join(self.cers_base_path, f"all_cers.json"), wandb_obj=wandb)

            current_lr = self.lr_crnn
            if self.lr_scheduler:
                self.scheduler_crnn.step()
                current_lr = self.scheduler_crnn.get_lr()
                # self.scheduler_prep.step()
                
            self.prep_model.eval()
            self.crnn_model.eval()
            pred_correct_count = 0
            matching_correct_count = 0
            pred_CER = 0
            matching_cer = 0
            validation_loss = 0
            tess_accuracy = 0
            tess_CER = 0
            with torch.no_grad():
                for images, labels, names in self.loader_validation:
                    X_var = images.to(self.device)
                    img_preds = self.prep_model(X_var)
                    scores, y, pred_size, y_size = self._call_model(
                        img_preds, labels)
                    loss = self._get_loss(
                        scores, y, pred_size, y_size, img_preds)
                    validation_loss += loss.item()
                    preds = pred_to_string(scores, labels, self.index_to_char)
                    ocr_labels = self.ocr.get_labels(img_preds.cpu())
                    crt, cer = compare_labels(preds, labels)
                    tess_crt, tess_cer = compare_labels(
                        ocr_labels, labels)
                    matching_crt, matching_cer = compare_labels(preds, ocr_labels)  # Compare OCR labels and CRNN output
                    matching_correct_count += matching_crt
                    matching_cer += matching_cer
                    pred_correct_count += crt
                    tess_accuracy += tess_crt
                    pred_CER += cer
                    tess_CER += tess_cer
                    validation_step += 1
            CRNN_accuracy = pred_correct_count/self.val_set_size
            OCR_accuracy = tess_accuracy/self.val_set_size
            CRNN_OCR_matching_acc = matching_correct_count/self.val_set_size
            CRNN_cer = pred_CER/self.val_set_size
            OCR_cer = tess_CER/self.val_set_size
            CRNN_OCR_matching_cer = matching_cer/self.val_set_size
            val_loss = validation_loss / (self.val_set_size//self.batch_size)

            wandb.log({"CRNN_accuracy": CRNN_accuracy, f"{self.ocr_name}_accuracy": OCR_accuracy, 
                        "CRNN_CER": CRNN_cer, f"{self.ocr_name}_cer": OCR_cer, "Epoch": epoch + 1,
                        "train_loss": train_loss, "val_loss": val_loss,
                        "Total Black-Box Calls": total_bb_calls, "Black-Box Calls":  epoch_bb_calls,
                        "Total CRNN Updates": total_crnn_updates, "CRNN Updates": epoch_crnn_updates,
                        "CRNN_loss": crnn_train_loss, "CRNN_OCR_Matching_ACC": CRNN_OCR_matching_acc, 
                        "CRNN_OCR_Matching_CER": CRNN_OCR_matching_cer})

            save_img(img_preds.cpu(), 'out_' +
                     str(epoch), self.img_out_path, 8)
            if epoch == 0:
                save_img(images.cpu(), 'out_original',
                         self.img_out_path, 8)

            print("CRNN correct count: %d; %s correct count: %d; (validation set size:%d)" % (
                pred_correct_count, self.ocr_name, tess_accuracy, self.val_set_size))
            print("CRNN CER:%d; %s CER: %d;" %
                  (pred_CER, self.ocr_name, tess_CER))
            print("Epoch: %d/%d => Training loss: %f | Validation loss: %f" % ((epoch + 1),
                                                                               self.max_epochs, training_loss /
                                                                               (self.train_set_size //
                                                                                self.train_batch_size),
                                                                               validation_loss/(self.val_set_size//self.batch_size)))
            prep_ckpt_path = os.path.join(self.ckpt_base_path, f"Prep_model_{epoch}_{OCR_accuracy*100:.2f}")

            torch.save(self.prep_model, prep_ckpt_path)
            torch.save(self.crnn_model, os.path.join(self.ckpt_base_path,
                       "CRNN_model_" + str(epoch)))
            best_prep_ckpt_path = os.path.join(self.ckpt_base_path, f"Prep_model_best")

            if OCR_accuracy > best_val_acc:
                best_val_acc = OCR_accuracy
                best_val_epoch = epoch
                shutil.copyfile(prep_ckpt_path, best_prep_ckpt_path)
                wandb.save(best_prep_ckpt_path)
                summary_metrics = dict()
                summary_metrics["best_val_acc"] = best_val_acc
                summary_metrics["best_val_epoch"] = best_val_epoch
                wandb.run.summary.update(summary_metrics)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains the Prep with VGG dataset')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='input batch size')
    parser.add_argument('--lr_crnn', type=float, default=0.0001,
                        help='CRNN learning rate, not used by adadealta')
    parser.add_argument('--scalar', type=float, default=1,
                        help='scalar in which the secondary loss is multiplied')
    parser.add_argument('--lr_prep', type=float, default=0.00005,
                        help='prep model learning rate, not used by adadealta')
    parser.add_argument('--epoch', type=int,
                        default=50, help='number of epochs')
    parser.add_argument('--warmup_epochs', type=int,
                        default=0, help='number of warmup epochs')
    parser.add_argument('--std', type=int,
                        default=5, help='standard deviation of Gussian noice added to images (this value devided by 100)')
    parser.add_argument('--inner_limit', type=int,
                        default=2, help='number of inner loop iterations in Alogorithm 1. Minimum value is 1.')
    parser.add_argument('--inner_limit_skip',
                            help="In the first inner limit loop, do NOT add noise to the image. Added to ease label imputation", 
                            action="store_true")
    parser.add_argument('--crnn_model',
                        help="specify non-default CRNN model location. By default, a new CRNN model will be used")
    parser.add_argument('--prep_model',
                        help="specify non-default Prep model location. By default, a new Prep model will be used")
    parser.add_argument('--data_base_path',
                        help='Base path training, validation and test data', default=".")
    parser.add_argument('--ocr', default='Tesseract',
                        help="performs training labels from given OCR [Tesseract,EasyOCR]")
    parser.add_argument('--random_std', action='store_false',
                        help='randomly selected integers from 0 upto given std value (devided by 100) will be used', default=True)
    parser.add_argument('--exp_base_path', default=".",
                        help='Base path for experiment. Defaults to current directory')
    parser.add_argument('--minibatch_subset', 
                        help='Specify method to pick subset from minibatch.')
    parser.add_argument('--minibatch_subset_prop', default=0.5, type=float,
                        help='If --minibatch_subset is provided, specify percentage of samples per mini-batch.')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Starting epoch. If loading from a ckpt, pass the ckpt epoch here.')
    parser.add_argument('--train_subset_size', help="Subset of training size to use", type=int)
    parser.add_argument('--val_subset_size',
                            help="Subset of val size to use", type=int)
    parser.add_argument('--lr_scheduler',
                            help="Specify scheduler to be used")
    parser.add_argument('--exp_name', default="default_exp",
                            help="Specify name of experiment (JVP Jitter, Sample Dropping Etc.)")
    parser.add_argument('--exp_id',
                        help="Specify unique experiment ID")
    parser.add_argument('--cers_ocr_path',
                            help="Cer information json")
    parser.add_argument('--weightgen_method', help="Method for generating loss weights for tracking", default='decaying', choices=['levenshtein', 'self_attention', 'decaying'])
    parser.add_argument('--window_size', help='Window Size if tracking is enabled', type=int, default=1)
    parser.add_argument('--decay_factor', help="Decay factor for decaying loss weight generation", type=float, default=0.7)



    
    
    args = parser.parse_args()
    print(vars(args))
    wandb.config.update(vars(args))
    wandb.run.name = f"{args.exp_name}"

    trainer = TrainNNPrep(args)

    start = datetime.datetime.now()
    trainer.train()
    end = datetime.datetime.now()

    with open(os.path.join(args.exp_base_path, properties.param_path), 'w') as filetowrite:
        filetowrite.write(str(start) + '\n')
        filetowrite.write(str(args) + '\n')
        filetowrite.write(str(end) + '\n')

