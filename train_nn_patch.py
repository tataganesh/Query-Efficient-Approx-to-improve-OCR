from cProfile import label
import datetime
import torch
import argparse
import os
import math
import json
import random as python_random
import shutil

from torch.nn import CTCLoss, MSELoss
import torch.optim as optim
from selection_utils import datasampler_factory, update_entropies

import torchvision.transforms as transforms

from models.model_crnn import CRNN
from models.model_unet import UNet
from label_tracking import tracking_methods, tracking_utils
from tracking_utils import generate_ctc_target_batches, call_crnn, weighted_ctc_loss, add_labels_to_history
from datasets.patch_dataset import PatchDataset
from utils import get_char_maps, set_bn_eval, pred_to_string, random_subset, create_dirs, attention_debug
from utils import get_text_stack, get_ocr_helper, compare_labels, save_json
from transform_helper import AddGaussianNoice
import properties as properties
import numpy as np
import wandb
from collections import defaultdict
wandb.Table.MAX_ROWS = 100000

class TrainNNPrep():

    def __init__(self, args):
        self.batch_size = 1
        self.random_seed = args.random_seed
        self.lr_crnn = args.lr_crnn
        self.lr_prep = args.lr_prep
        self.weight_decay = args.weight_decay
        self.max_epochs = args.epoch
        self.warmup_epochs = args.warmup_epochs
        self.inner_limit = args.inner_limit
        self.inner_limit_skip = args.inner_limit_skip
        self.crnn_model_path = args.crnn_model
        self.prep_model_path = args.prep_model
        self.data_base_path = args.data_base_path
        self.exp_base_path = args.exp_base_path
        self.ckpt_base_path = os.path.join(self.exp_base_path, properties.prep_crnn_ckpts)
        self.cers_base_path = os.path.join(self.exp_base_path, "cers")
        self.tracked_labels_path = os.path.join(self.exp_base_path, "tracked_labels")
        self.selectedsamples_path = os.path.join(self.exp_base_path, "selected_samples")
        self.entropies_path = os.path.join(self.exp_base_path, "entropies")
        self.tensorboard_log_path = os.path.join(self.exp_base_path, properties.prep_tensor_board)
        self.img_out_path = os.path.join(self.exp_base_path, properties.img_out)
        create_dirs([self.exp_base_path, self.ckpt_base_path, self.tensorboard_log_path, self.img_out_path, self.cers_base_path, self.tracked_labels_path, self.selectedsamples_path, self.entropies_path])

        self.update_CRNN = args.update_CRNN
        self.sec_loss_scalar = args.scalar
        self.ocr_name = args.ocr
        self.std = args.std
        self.is_random_std = args.random_std
        torch.manual_seed(self.random_seed)
        python_random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.model_labels_last = dict()  # Seems inefficient
        self.train_set = os.path.join(args.data_base_path, properties.patch_dataset_train)
        self.validation_set = os.path.join(args.data_base_path, properties.patch_dataset_dev)
        self.start_epoch = args.start_epoch
        self.selection_method = args.minibatch_subset
        self.train_batch_prop = 1
    
        self.char_to_index, self.index_to_char, self.vocab_size = get_char_maps(
            properties.char_set)

        if args.minibatch_subset_prop is not None and self.selection_method:
            self.train_batch_prop = args.minibatch_subset_prop

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
            if self.selection_method == "rangeCER":
                self.sampler = self.cls_sampler(self.cers, args.discount_factor)
            elif self.selection_method == "uniformEntropy":
                self.sampler = self.cls_sampler(self.entropies, self.cers)
            else:
                self.sampler = self.cls_sampler(self.cers)

        if self.cers:
            self.tracked_labels = {name: [] for name in self.cers.keys()}
        self.train_subset_size = args.train_subset_size
        self.val_subset_size = args.val_subset_size
        self.input_size = properties.input_size

        self.ocr = get_ocr_helper(self.ocr_name)

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
            
        self.window_size = args.window_size
        self.weightgen_method = args.weightgen_method
               
        WeightGenCls = tracking_methods.weightgenerator_factory(args.weightgen_method)
        self.loss_wghts_gnrtr = WeightGenCls(args, self.device, self.char_to_index)
        self.dataset = PatchDataset(
            self.train_set, pad=True, include_name=True, num_subset=self.train_subset_size)
        self.validation_set = PatchDataset(
            self.validation_set, pad=True, num_subset=self.val_subset_size)
        self.loader_train = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, collate_fn=PatchDataset.collate)

        self.train_set_size = int(len(self.dataset))
        self.val_set_size = len(self.validation_set)
        
        if self.cers:
            self.all_cers = {name: [] for name in self.cers.keys()}

        image_proportion = args.image_prop  # Proportion of images to select per epoch
        self.num_subset_images = None
        if image_proportion:
            self.num_subset_images = int(image_proportion * self.train_set_size)
            
        self.primary_loss_fn = CTCLoss().to(self.device)
        self.primary_loss_fn_sample_wise = CTCLoss(reduction='none').to(self.device)

        self.secondary_loss_fn = MSELoss().to(self.device)
        
        crnn_parameters = list(self.crnn_model.parameters())
        if args.weightgen_method == "self-attention":
            crnn_parameters += list(self.loss_wghts_gnrtr.attention_model.parameters())
        self.optimizer_crnn = optim.Adam(
            crnn_parameters, lr=self.lr_crnn, weight_decay=self.weight_decay)
        self.optimizer_prep = optim.Adam(
            self.prep_model.parameters(), lr=self.lr_prep, weight_decay=self.weight_decay)


    def _call_model(self, images, labels):
        """Get output from CRNN model for images. Convert labels to format suitable for CTC Loss. 

        Args:
            images (torch.float32): Image Batch
            labels (list[str]): Text labels corresponding to each image

        Returns:
            tuple: Tuple of 4 elements
        """        
        X_var = images.to(self.device)
        scores = self.crnn_model(X_var)
        out_size = torch.tensor(
            [scores.shape[0]] * images.shape[0], dtype=torch.int)
        y_size = torch.tensor([len(lbl) for lbl in labels], dtype=torch.int)
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

    def add_noise(self, imgs, noiser):
        noisy_imgs = []
        for img in imgs:
            noisy_imgs.append(noiser(img))
        return torch.stack(noisy_imgs)

    def train(self):
        noiser = AddGaussianNoice(
            std=self.std, is_stochastic=self.is_random_std)
        
        step = 0
        validation_step = 0
        batch_step = 0
        total_train_bb_calls = 0
        total_train_val_bb_calls = 0
        total_crnn_updates = 0
        best_val_acc = 0
        best_val_epoch = 0

        for epoch in range(self.start_epoch, self.max_epochs):
            if self.selection_method and "global" in self.selection_method:  # Criterion to CHECK if this is a global or local selection method
                self.sampler.select_samples()
            training_loss = 0.0
            epoch_print_flag = True
            epoch_bb_calls = 0
            epoch_crnn_updates = 0
            if self.num_subset_images:
                print(f"Total images - {self.train_set_size}, Subset Images - {self.num_subset_images}")
                random_indices = torch.randperm(self.train_set_size)[:self.num_subset_images]
                random_sampler = torch.utils.data.SubsetRandomSampler(random_indices)
                self.loader_train = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                            sampler=random_sampler, drop_last=True, collate_fn=PatchDataset.collate)
            for images, labels_dicts, names in self.loader_train:
                self.crnn_model.train()
                self.prep_model.eval()
                self.prep_model.zero_grad()
                self.crnn_model.zero_grad()

                CRNN_training_loss = 0
                file_name = None
                for i in range(len(labels_dicts)):
                    image = images[i]
                    labels_dict = labels_dicts[i]
                    name = names[i]
                    image = image.unsqueeze(0)
                    X_var = image.to(self.device)
                    pred = self.prep_model(X_var)[0]
                    text_crops_all, labels = get_text_stack(
                        pred, labels_dict, self.input_size)
                    num_text_strips = text_crops_all.shape[0]
                    folder_name, file_name = name.split("/")[-2:]
                    file_name = file_name.split(".")[0]
                    text_strip_names = list()
                    for j in range(len(labels)):
                        text_strip_name = f"{j}_{labels[j]}_{folder_name}_{file_name}"
                        text_strip_names.append(text_strip_name)
                    # check for number of text crops to be greater than 2, otherwise call black-box for all crops, the 
                    # greater-than-2 condition is ignored if global sampling is performed
                    if self.selection_method and epoch >= self.warmup_epochs and ("global" not in self.selection_method):  # Remove num_strips > 2 condition
                        num_bb_samples = max(1, math.ceil(num_text_strips*(1 - self.train_batch_prop)))
                        text_crops, labels_gt, bb_sample_indices = self.sampler.query(text_crops_all, labels, num_bb_samples, text_strip_names)
                        bb_sample_indices = bb_sample_indices[:text_crops.shape[0]]

                        text_crops = text_crops.detach().cpu()
                        text_crop_names = [text_strip_names[index] for index in bb_sample_indices]
                        # Log selected samples                     
                        for name in text_crop_names:
                            if name in self.selected_samples:  # Find out why this condition is required
                                self.selected_samples[name][epoch] = True

                        skipped_mask = torch.ones(num_text_strips, dtype=bool)
                        skipped_mask[bb_sample_indices] = False
                        skipped_text_crops = text_crops_all[skipped_mask]
                        labels_skipped = [labels[i] for i in range(skipped_mask.shape[0]) if skipped_mask[i]]      
                    else:
                        text_crops = text_crops_all.detach().cpu()
                        text_crop_names = text_strip_names
                        skipped_mask = torch.zeros(num_text_strips, dtype=bool)
                    
                    crnn_approx_loss = 0
                    if epoch_print_flag:
                        print(f"Total Samples - {num_text_strips}")
                        print(f"OCR Samples - {text_crops.shape[0]}")
                    if text_crops.shape[0] > 0 and not (self.selection_method and int(self.train_batch_prop) == 1):  # Cases when the black-box should not be called at all (in a mini-batch)
                        for i in range(self.inner_limit):
                            self.prep_model.zero_grad()
                            if i == 0 and self.inner_limit_skip:  # Skip adding noise to one of the inner loops to perform label tracking
                                ocr_labels = self.ocr.get_labels(text_crops)
                                loss_weights = self.loss_wghts_gnrtr.gen_weights(self.tracked_labels, text_crop_names)
                                add_labels_to_history(self, text_crop_names, ocr_labels)
                                # Peek at history of OCR labels for each strip and construct weighted CTC loss
                                target_batches = generate_ctc_target_batches(self, text_crop_names)
                                scores, pred_size = call_crnn(self, text_crops)
                                loss = weighted_ctc_loss(self, scores, pred_size, target_batches, loss_weights)         
                            else:
                                noisy_imgs = self.add_noise(text_crops, noiser)
                                ocr_labels = self.ocr.get_labels(noisy_imgs)
                                scores, y, pred_size, y_size = self._call_model(
                                    noisy_imgs, ocr_labels)
                                loss = self.primary_loss_fn(
                                    scores, y, pred_size, y_size)

                            total_train_bb_calls += text_crops.shape[0]
                            epoch_bb_calls += text_crops.shape[0]
                           
                            if self.inner_limit:
                                crnn_approx_loss += loss.item()
                                loss.backward()

                    inner_limit = max(1, self.inner_limit)
                    CRNN_training_loss += crnn_approx_loss/inner_limit
                epoch_print_flag = False
                if self.inner_limit:
                    self.optimizer_crnn.step()
                batch_step += 1

                self.prep_model.train()
                self.crnn_model.train()
                self.crnn_model.apply(set_bn_eval)
                self.prep_model.zero_grad()
                self.crnn_model.zero_grad()

                for i in range(len(labels_dicts)):
                    image = images[i]
                    labels_dict = labels_dicts[i]
                    name = names[i]
                    image = image.unsqueeze(0)
                    X_var = image.to(self.device)
                    img_out = self.prep_model(X_var)[0]
                    n_text_crops, labels = get_text_stack(
                        img_out, labels_dict, self.input_size)

                    scores, y, pred_size, y_size = self._call_model(
                        n_text_crops, labels)
                    loss = self._get_loss(
                        scores, y, pred_size, y_size, img_out)
                    loss.backward()
                    # Update last seen prediction of image
                    model_gen_labels = pred_to_string(scores, labels, self.index_to_char)
                    for i in range(len(labels)):
                        self.model_labels_last[name + "_" + str(i)] = model_gen_labels[i]

                    training_loss += loss.item()
                    if step % 100 == 0:
                        print("Iteration: %d => %f" % (step, loss.item()))
                    step += 1
                    
                    if self.selection_method and len(text_strip_names):
                        batch_cers = list()
                        for i in range(len(labels)):
                            _, batch_cer = compare_labels([model_gen_labels[i]], [labels[i]])
                            batch_cers.append(batch_cer)
                        self.sampler.update_cer(batch_cers, text_strip_names)
                        
                        if self.selection_method == "uniformEntropy":
                            update_entropies(self, scores, text_strip_names)
                if self.update_CRNN:
                    self.optimizer_crnn.step()
                self.optimizer_prep.step()
            
            if self.selection_method:                            
                save_json(self.tracked_labels, os.path.join(self.tracked_labels_path, f"tracked_labels_{epoch}.json"))
                save_json(self.tracked_labels, os.path.join(self.tracked_labels_path, f"tracked_labels_current.json"), wandb_obj=wandb)
                save_json(self.selected_samples, os.path.join(self.selectedsamples_path, f"selected_samples_current.json"), wandb_obj=wandb)
                save_json(self.sampler.all_cers, os.path.join(self.cers_base_path, f"all_cers.json"), wandb_obj=wandb)
                
            print(f"Epoch BB calls - {epoch_bb_calls}")
            train_loss = training_loss / self.train_set_size
            crnn_train_loss = CRNN_training_loss / max(1, epoch_bb_calls)

            self.prep_model.eval()
            self.crnn_model.eval()
            pred_correct_count = 0
            matching_correct_count = 0
            matching_cer = 0
            validation_loss = 0
            tess_correct_count = 0
            pred_CER = 0
            tess_CER = 0
            val_label_count = 0
            
            with torch.no_grad():
                for image, labels_dict in self.validation_set:
                    image = image.unsqueeze(0)
                    X_var = image.to(self.device)
                    img_out = self.prep_model(X_var)[0]
                    n_text_crops, labels = get_text_stack(
                        img_out, labels_dict, self.input_size)
                    scores, y, pred_size, y_size = self._call_model(
                        n_text_crops, labels)
                    loss = self._get_loss(
                        scores, y, pred_size, y_size, img_out)
                    validation_loss += loss.item()

                    preds = pred_to_string(scores, labels, self.index_to_char)
                    ocr_labels = self.ocr.get_labels(n_text_crops.cpu())
                    crt, cer = compare_labels(preds, labels)
                    tess_crt, tess_cer = compare_labels(ocr_labels, labels)
                    matching_crt, matching_cer = compare_labels(preds, ocr_labels) # Compare OCR labels and CRNN output
                    matching_correct_count += matching_crt
                    matching_cer += matching_cer
                    pred_correct_count += crt
                    tess_correct_count += tess_crt
                    val_label_count += len(labels)
                    pred_CER += cer
                    tess_CER += tess_cer
                    validation_step += 1
            print(f"Validation Dataset Calls - {val_label_count}")
            CRNN_accuracy = pred_correct_count/val_label_count
            OCR_accuracy = tess_correct_count/val_label_count
            CRNN_OCR_matching_acc = matching_correct_count/val_label_count
            CRNN_cer = pred_CER/self.val_set_size
            OCR_cer = tess_CER/self.val_set_size
            CRNN_OCR_matching_cer = matching_cer/self.val_set_size
            val_loss = validation_loss / self.val_set_size
            train_val_bb_calls = val_label_count + epoch_bb_calls
            total_train_val_bb_calls += epoch_bb_calls + val_label_count

            # Log all metrics
            wandb.log({"CRNN_accuracy": CRNN_accuracy, f"{self.ocr_name}_accuracy": OCR_accuracy, 
                    "CRNN_CER": CRNN_cer, f"{self.ocr_name}_cer": OCR_cer, "Epoch": epoch + 1,
                    "train_loss": train_loss, "val_loss": val_loss, 
                    "Total Black-Box Calls": total_train_bb_calls, "Black-Box Calls":  epoch_bb_calls,
                    "Train + Val BB Calls": train_val_bb_calls, "Total Train + Val BB Calls": total_train_val_bb_calls,
                    "Total CRNN Updates": total_crnn_updates, "CRNN Updates": epoch_crnn_updates,
                    "CRNN_loss": crnn_train_loss, "CRNN_OCR_Matching_ACC": CRNN_OCR_matching_acc, 
                    "CRNN_OCR_Matching_CER": CRNN_OCR_matching_cer})

            img = transforms.ToPILImage()(img_out.cpu()[0])
            img.save(os.path.join(self.img_out_path, 'out_'+str(epoch)+'.png'), 'PNG')
            if epoch == 0:
                img = transforms.ToPILImage()(image.cpu()[0])
                img.save(os.path.join(self.img_out_path, 'out_original.png'), 'PNG')

            print("CRNN correct count: %d; %s correct count: %d; (validation set size:%d)" % (
                pred_correct_count, self.ocr_name, tess_correct_count, val_label_count))
            print("Epoch: %d/%d => Training loss: %f | Validation loss: %f" % ((epoch + 1), self.max_epochs,
                                                                               training_loss / self.train_set_size,
                                                                               validation_loss/self.val_set_size))
            print(f"Total OCR Calls Count: {self.ocr.count_calls}")
            prep_ckpt_path = os.path.join(self.ckpt_base_path, f"Prep_model_{epoch}_{OCR_accuracy*100:.2f}")
            torch.save(self.prep_model, prep_ckpt_path)
            torch.save(self.crnn_model,  os.path.join(self.ckpt_base_path, 
                       "CRNN_model_" + str(epoch)))
            best_prep_ckpt_path = os.path.join(self.ckpt_base_path, f"Prep_model_best")
            if OCR_accuracy > best_val_acc:
                best_val_acc = OCR_accuracy
                best_val_epoch = epoch
                # torch.save(self.prep_model, best_prep_ckpt_path) # Ideally, copy previously saved model using shutil
                shutil.copyfile(prep_ckpt_path, best_prep_ckpt_path)
                wandb.save(best_prep_ckpt_path)
                summary_metrics = dict()
                summary_metrics["best_val_acc"] = best_val_acc
                summary_metrics["best_val_epoch"] = best_val_epoch
                wandb.run.summary.update(summary_metrics)
                
        # To be removed: Do not report test-set performance
        # summary_metrics = prep_eval(best_val_ckpt_path, 'pos', self.data_base_path, self.ocr_name)
        print("Training Completed.")
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains the Prep with Patch dataset')
    tracking_parser = argparse.ArgumentParser(
        description='Specific arguments for all tracking methods')
    parser.add_argument('--lr_crnn', type=float, default=0.0001,
                        help='CRNN learning rate, not used by adadealta')
    parser.add_argument('--scalar', type=float, default=1,
                        help='scalar in which the secondary loss is multiplied')
    parser.add_argument('--lr_prep', type=float, default=0.00005,
                        help='prep model learning rate, not used by adadealta')
    parser.add_argument('--epoch', type=int,
                        default=25, help='number of epochs')
    parser.add_argument('--random_seed',
                            help="Random seed for experiment", type=int, default=42)
    parser.add_argument('--std', type=int,
                        default=5, help='standard deviation of Gussian noice added to images (this value devided by 100)')
    parser.add_argument('--inner_limit', type=int,
                        default=2, help='number of inner loop iterations')
    parser.add_argument('--inner_limit_skip',
                            help="In the first inner limit loop, do NOT add noise to the image. Added to ease label imputation", 
                            action="store_true")
    parser.add_argument('--crnn_model',
                        help="specify non-default CRNN model location. If given empty, a new CRNN model will be used")
    parser.add_argument('--prep_model',
                        help="specify non-default Prep model location. By default, a new Prep model will be used")
    parser.add_argument('--exp_base_path', default=".",
                        help='Base path for experiment. Defaults to current directory')
    parser.add_argument('--ocr', default='Tesseract',
                        help="performs training labels from given OCR [Tesseract,EasyOCR]")
    parser.add_argument('--random_std', action='store_false',
                        help='randomly selected integers from 0 upto given std value (devided by 100) will be used', default=True)
    parser.add_argument('--minibatch_subset',  choices=['random', 'uniformCER', 'uniformCERglobal', 'randomglobal', 'rangeCER', 'uniformEntropy', 'topKCER'], 
                        help='Specify method to pick subset from minibatch.')
    parser.add_argument('--minibatch_subset_prop', default=0.5, type=float,
                        help='If --minibatch_subset is provided, specify percentage of samples per mini-batch.')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Starting epoch. If loading from a ckpt, pass the ckpt epoch here.')
    parser.add_argument('--data_base_path',
                        help='Base path training, validation and test data', default=".")
    parser.add_argument('--warmup_epochs', type=int,
                        default=0, help='number of warmup epochs')
    parser.add_argument('--exp_name', default="test_patch",
                        help="Specify name of experiment (JVP Jitter, Sample Dropping Etc.)")
    parser.add_argument('--exp_id',
                        help="Specify unique experiment ID")
    parser.add_argument('--train_subset_size', help="Subset of training size to use", type=int)
    parser.add_argument('--val_subset_size',
                            help="Subset of val size to use", type=int)
    parser.add_argument('--weight_decay',
                            help="Weight Decay for the optimizer", type=float, default=5e-4)
    parser.add_argument('--cers_ocr_path',
                            help="Cer information json")
    parser.add_argument('--image_prop', help="Percentage of images per epoch", type=float)
    parser.add_argument('--discount_factor', help="Discount factor for CER values", type=float, default=1)
    parser.add_argument('--update_CRNN', action='store_true', help='Update CRNN when updating preprocessor. Only relevant IF THE OCR IS NEVER CALLED.')

    parser.add_argument('--window_size', help='Window Size if tracking is enabled', type=int, default=1)
    parser.add_argument('--query_dim', help='Dimension of query vector in self attention ', type=int, default=32)
    parser.add_argument('--emb_dim', help='Word Embedding size in self-attention', type=int, default=256)
    parser.add_argument('--attn_activation', help='Activation function after last layer of self attention module', type=str, default='sigmoid', choices=['sigmoid', 'softmax', 'relu'])
    parser.add_argument('--weightgen_method', help="Method for generating loss weights for tracking", default='decaying', choices=['levenshtein', 'self_attention', 'decaying'])
    parser.add_argument('--decay_factor', help="Decay factor for decaying loss weight generation", type=float, default=0.7)


    args = parser.parse_args()
    print("Training Arguments")
    print(args)

    wandb.init(project='ocr-calls-reduction', entity='tataganesh')
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