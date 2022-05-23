import datetime
import torch
import argparse
import os
import math
import json

from torch.nn import CTCLoss, MSELoss
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from selection_utils import datasampler_factory

import torchvision.transforms as transforms

from models.model_crnn import CRNN
from models.model_unet import UNet
from datasets.patch_dataset import PatchDataset
from utils import get_char_maps, set_bn_eval, pred_to_string, random_subset, create_dirs
from utils import get_text_stack, get_ocr_helper, compare_labels
from transform_helper import AddGaussianNoice
import properties as properties
from pprint import pprint
import numpy as np
import wandb
wandb.Table.MAX_ROWS = 100000
wandb.init(project='ocr-calls-reduction', entity='tataganesh')
minibatch_subset_methods = {"random": random_subset}
class TrainNNPrep():

    def __init__(self, args):
        self.batch_size = 1
        self.lr_crnn = args.lr_crnn
        self.lr_prep = args.lr_prep
        self.weight_decay = args.weight_decay
        self.max_epochs = args.epoch
        self.warmup_epochs = args.warmup_epochs
        self.inner_limit = args.inner_limit
        self.crnn_model_path = args.crnn_model
        self.prep_model_path = args.prep_model
        self.exp_base_path = args.exp_base_path
        self.ckpt_base_path = os.path.join(self.exp_base_path, properties.prep_crnn_ckpts)
        self.tensorboard_log_path = os.path.join(self.exp_base_path, properties.prep_tensor_board)
        self.img_out_path = os.path.join(self.exp_base_path, properties.img_out)
        create_dirs([self.exp_base_path, self.ckpt_base_path, self.tensorboard_log_path, self.img_out_path])

        self.sec_loss_scalar = args.scalar
        self.ocr_name = args.ocr
        self.std = args.std
        self.is_random_std = args.random_std
        self.label_impute = args.label_impute
        torch.manual_seed(42)

        self.model_labels_last = dict() # Seems inefficient
        self.train_set = os.path.join(args.data_base_path, properties.patch_dataset_train)
        self.validation_set = os.path.join(args.data_base_path, properties.patch_dataset_dev)
        self.start_epoch = args.start_epoch
        self.selection_method = args.minibatch_subset
        self.minibatch_sample = minibatch_subset_methods.get(self.selection_method, None)
        self.cls_sampler = datasampler_factory(self.selection_method)

        self.train_batch_prop = 1
    
        if args.minibatch_subset_prop and self.selection_method:
            self.train_batch_prop = args.minibatch_subset_prop
        
        with open(args.cers_ocr_path, 'r') as f:
            self.cers_with_img = json.load(f)
        cers = dict()
        for _, value in self.cers_with_img.items():
            cers.update(value)
        # text_strip_indices_global = np.array()
        if self.selection_method == "uniformCER":
            self.sampler = self.cls_sampler(cers)
        elif self.selection_method == "uniformCERglobal":
            num_samples =  int(len(cers) * (1 - self.train_batch_prop))
            self.sampler = self.cls_sampler(cers, num_samples)
            self.cer_per_epoch = np.full((len(cers), self.max_epochs), -1)
        else:
            self.sampler = self.cls_sampler()
        self.train_subset_size = args.train_subset_size
        self.val_subset_size = args.val_subset_size
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

        self.dataset = PatchDataset(
            self.train_set, pad=True, include_name=True, num_subset=self.train_subset_size)
        self.validation_set = PatchDataset(
            self.validation_set, pad=True, num_subset=self.val_subset_size)
        self.loader_train = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, collate_fn=PatchDataset.collate)

        self.train_set_size = int(len(self.dataset) * self.train_batch_prop)
        self.val_set_size = len(self.validation_set)
            

        self.primary_loss_fn = CTCLoss().to(self.device)
        self.primary_loss_fn_sample_wise = CTCLoss(reduction='none').to(self.device)

        self.secondary_loss_fn = MSELoss().to(self.device)
        self.optimizer_crnn = optim.Adam(
            self.crnn_model.parameters(), lr=self.lr_crnn, weight_decay=self.weight_decay)
        self.optimizer_prep = optim.Adam(
            self.prep_model.parameters(), lr=self.lr_prep, weight_decay=self.weight_decay)

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

    def add_noise(self, imgs, noiser):
        noisy_imgs = []
        for img in imgs:
            noisy_imgs.append(noiser(img))
        return torch.stack(noisy_imgs)

    def train(self):
        noiser = AddGaussianNoice(
            std=self.std, is_stochastic=self.is_random_std)
        writer = SummaryWriter(self.tensorboard_log_path)
        
        step = 0
        validation_step = 0
        batch_step = 0
        total_bb_calls = 0

        for epoch in range(self.start_epoch, self.max_epochs):
            if "global" in self.selection_method: # Criterion to CHECK if this is a global or local selection method
                self.sampler.select_samples()
            subset_samples = 0
            training_loss = 0
            epoch_print_flag = True
            epoch_bb_calls = 0
            # epoch_prop = 
            for images, labels_dicts, names in self.loader_train:
                self.crnn_model.train()
                self.prep_model.eval()
                self.prep_model.zero_grad()
                self.crnn_model.zero_grad()

                CRNN_training_loss = 0
                # image_preds = list()
                file_name = None
                for i in range(len(labels_dicts)):
                    image = images[i]
                    labels_dict = labels_dicts[i]
                    name = names[i]
                    image = image.unsqueeze(0)
                    X_var = image.to(self.device)
                    pred = self.prep_model(X_var)[0]
                    # image_preds.append(pred)

                    # pred_cpu = pred.detach().cpu()[0]

                    text_crops_all, labels = get_text_stack(
                        pred, labels_dict, self.input_size)
                    sample_indices = None
                    folder_name, file_name = name.split("/")[-2:]
                    # print(folder_name, file_name)
                    text_strip_names = self.cers_with_img[folder_name + "_" + file_name].keys()
                    if len(text_strip_names):
                        filtered_strips = [label_instance["index"] for label_instance in labels_dict]
                        min_index = min(int(k.split("_")[0]) for k in text_strip_names)
                        text_strip_names_filtered = [strip_name for strip_name in text_strip_names if int(strip_name.split("_")[0]) - min_index in filtered_strips]
                        text_strip_indices = [int(k.split("_")[0]) - min_index  for k in text_strip_names_filtered]
                        text_strip_names_filtered.extend([f"{k}_{labels[i]}_{folder_name}.png" for i, k in enumerate(filtered_strips) if k not in text_strip_indices])
                        text_strip_names = sorted(text_strip_names_filtered, key=lambda k:int(k.split("_")[0]))
                    # check for number of text crops to be greater than 2, otherwise call black-box for all crops
                    if self.selection_method and epoch >= self.warmup_epochs and text_crops_all.shape[0] > 2:
                        num_bb_samples = max(1, math.ceil(text_crops_all.shape[0]*(1 - self.train_batch_prop)))
                        # num_samples_subset = int(text_crops_all.shape[0]*self.train_batch_prop)
                        num_samples_subset = max(1, text_crops_all.shape[0] - num_bb_samples)
                        
                        # skipped_text_crops, labels_skipped, sample_indices = self.minibatch_sample(text_crops_all, labels, num_samples_subset)
                        # text_crops, labels_ocr, bb_sample_indices = self.minibatch_sample(text_crops_all, labels, num_samples_subset)
                        text_crops, labels_ocr, bb_sample_indices = self.sampler.query(text_crops_all, labels, num_bb_samples, text_strip_names)
                        text_crops = text_crops.detach().cpu()

                        skipped_crops_mask = torch.ones(text_crops_all.shape[0], dtype=bool)
                        skipped_crops_mask[bb_sample_indices] = False
                        skipped_text_crops = text_crops_all[skipped_crops_mask]
                        labels_skipped = [labels[i] for i in range(skipped_crops_mask.shape[0]) if skipped_crops_mask[i]]
                        # ocr_crops_mask = torch.ones(text_crops_all.shape[0], dtype=bool)
                        # ocr_crops_mask[sample_indices] = False
                        # text_crops = text_crops_all[ocr_crops_mask].detach().cpu()

                        if self.label_impute:
                            model_lab_last_batch = [self.model_labels_last[name + "_" + str(i.item())] for i in sample_indices]
                            half_labels_skipped = int(num_samples_subset/2)
                            gt_self_labels = list()
                            rand_label_indices = torch.randperm(num_samples_subset)
                            gt_self_labels = [labels_skipped[i] for i in rand_label_indices[:half_labels_skipped]]
                            gt_self_labels.extend([model_lab_last_batch[i] for i in rand_label_indices[half_labels_skipped:]])
                            skipped_text_crops = skipped_text_crops[rand_label_indices]
                            scores, y, pred_size, y_size = self._call_model(
                                skipped_text_crops, gt_self_labels)
                        
                            loss = self.primary_loss_fn(scores, y, pred_size, y_size)
                            loss_inv = self.primary_loss_fn_sample_wise(scores, y, pred_size, y_size)
                            
                            loss.backward()
                            if epoch_print_flag:
                                pprint(list(zip([model_lab_last_batch[i] for i in rand_label_indices], [labels_skipped[i] for i in rand_label_indices], gt_self_labels)))
                                print(f"Skipped Samples - {skipped_text_crops.shape[0]}")
                                epoch_print_flag = False
                    else:
                        text_crops = text_crops_all.detach().cpu()
                    
                    temp_loss = 0
                    if epoch_print_flag:
                        print(f"Total Samples - {text_crops_all.shape[0]}")
                        print(f"OCR Samples - {text_crops.shape[0]}")
                        epoch_print_flag = False
                    if text_crops.shape[0] > 0 and not(self.selection_method and int(self.train_batch_prop)==1): # Cases when the black-box should not be called at all (in a mini-batch)
                        for i in range(self.inner_limit):
                            self.prep_model.zero_grad()
                            noisy_imgs = self.add_noise(text_crops, noiser)
                            noisy_labels = self.ocr.get_labels(noisy_imgs)
                            scores, y, pred_size, y_size = self._call_model(
                                noisy_imgs, noisy_labels)
                            loss = self.primary_loss_fn(
                                scores, y, pred_size, y_size)
                            temp_loss += loss.item()
                            loss.backward()
                        total_bb_calls += text_crops.shape[0]
                        epoch_bb_calls += text_crops.shape[0]
 
                    CRNN_training_loss += temp_loss/self.inner_limit
                self.optimizer_crnn.step()
                writer.add_scalar('CRNN Training Loss',
                                  CRNN_training_loss, batch_step)
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
                    # img_out = image_preds[i]
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

                    
                    if self.selection_method != "random" and len(text_strip_names):
                        batch_cers = list()
                        for i in range(len(labels)):
                            _, batch_cer = compare_labels([model_gen_labels[i]], [labels[i]])
                            batch_cers.append(batch_cer)
                        self.sampler.update_cer(batch_cers, text_strip_names)
                # self.optimizer_crnn.step()
                self.optimizer_prep.step()

            if self.selection_method == "uniformCERglobal":
                self.cer_per_epoch[:, epoch] = np.array(list(self.sampler.cers.values()))
            print(f"Epoch BB calls - {epoch_bb_calls}")
            train_loss =  training_loss / self.train_set_size
            writer.add_scalar('Training Loss', training_loss /
                              self.train_set_size, epoch + 1)
            self.prep_model.eval()
            self.crnn_model.eval()
            pred_correct_count = 0
            validation_loss = 0
            tess_correct_count = 0
            pred_CER = 0
            tess_CER = 0
            label_count = 0
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
                    pred_correct_count += crt
                    tess_correct_count += tess_crt
                    label_count += len(labels)
                    pred_CER += cer
                    tess_CER += tess_cer
                    validation_step += 1

            CRNN_accuracy = pred_correct_count/label_count
            OCR_accuracy = tess_correct_count/label_count
            CRNN_cer = pred_CER/self.val_set_size
            OCR_cer = tess_CER/self.val_set_size
            val_loss = validation_loss / self.val_set_size

            writer.add_scalar('Accuracy/CRNN_output',
                              pred_correct_count/label_count, epoch + 1)
            writer.add_scalar('Accuracy/'+self.ocr_name+'_output',
                              tess_correct_count/label_count, epoch + 1)
            writer.add_scalar('Validation Loss',
                              validation_loss/self.val_set_size, epoch + 1)

            wandb.log({"CRNN_accuracy": CRNN_accuracy, f"{self.ocr_name}_accuracy": OCR_accuracy, 
                    "CRNN_CER": CRNN_cer, f"{self.ocr_name}_cer": OCR_cer, "Epoch": epoch + 1,
                    "train_loss": train_loss, "val_loss": val_loss, "Total Black-Box Calls": total_bb_calls, "Black-Box Calls":  epoch_bb_calls})

            img = transforms.ToPILImage()(img_out.cpu()[0])
            # img.save(properties.img_out_path+'out_'+str(epoch)+'.png', 'PNG')
            img.save(os.path.join(self.img_out_path, 'out_'+str(epoch)+'.png'), 'PNG')
            if epoch == 0:
                img = transforms.ToPILImage()(image.cpu()[0])
                # img.save(properties.img_out_path+'out_original.png', 'PNG')
                img.save(os.path.join(self.img_out_path, 'out_original.png'), 'PNG')

            print("CRNN correct count: %d; %s correct count: %d; (validation set size:%d)" % (
                pred_correct_count, self.ocr_name, tess_correct_count, label_count))
            print("Epoch: %d/%d => Training loss: %f | Validation loss: %f" % ((epoch + 1), self.max_epochs,
                                                                               training_loss / self.train_set_size,
                                                                               validation_loss/self.val_set_size))
            # print("Epoch: %d/%d => Total Training Samples: %f | Subset Training Samples: %f | Train size: %f" % ((epoch + 1), self.max_epochs,
            #                                                                         total_samples, subset_samples, self.train_set_size))
            torch.save(self.prep_model,
                        os.path.join(self.ckpt_base_path, "Prep_model_"+str(epoch)))
            torch.save(self.crnn_model,  os.path.join(self.ckpt_base_path, 
                       "CRNN_model_" + str(epoch)))
        epochs_cer_tbl = wandb.Table(data=self.cer_per_epoch.tolist(), columns=list(range(self.cer_per_epoch.shape[1])))
        wandb.log({"CER Values": epochs_cer_tbl})

        writer.flush()
        writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains the Prep with Patch dataset')
    parser.add_argument('--lr_crnn', type=float, default=0.0001,
                        help='CRNN learning rate, not used by adadealta')
    parser.add_argument('--scalar', type=float, default=1,
                        help='scalar in which the secondary loss is multiplied')
    parser.add_argument('--lr_prep', type=float, default=0.00005,
                        help='prep model learning rate, not used by adadealta')
    parser.add_argument('--epoch', type=int,
                        default=25, help='number of epochs')
    parser.add_argument('--std', type=int,
                        default=2, help='standard deviation of Gussian noice added to images (this value devided by 100)')
    parser.add_argument('--inner_limit', type=int,
                        default=5, help='number of inner loop iterations')
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
    parser.add_argument('--minibatch_subset',  choices=['random', 'uniformCER', 'uniformCERglobal'], 
                        help='Specify method to pick subset from minibatch.')
    parser.add_argument('--minibatch_subset_prop', default=0.5, type=float,
                        help='If --minibatch_subset is provided, specify percentage of samples per mini-batch.')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Starting epoch. If loading from a ckpt, pass the ckpt epoch here.')
    parser.add_argument('--data_base_path',
                        help='Base path training, validation and test data', default=".")
    parser.add_argument('--warmup_epochs', type=int,
                        default=1, help='number of warmup epochs')
    parser.add_argument('--exp_name', default="test_patch",
                        help="Specify name of experiment (JVP Jitter, Sample Dropping Etc.)")
    parser.add_argument('--exp_id',
                        help="Specify unique experiment ID")
    parser.add_argument('--train_subset_size', help="Subset of training size to use", type=int)
    parser.add_argument('--val_subset_size',
                            help="Subset of val size to use", type=int)
    parser.add_argument('--label_impute',
                            help="Impute black-box labels for approximator training", action="store_true")
    parser.add_argument('--weight_decay',
                            help="Weight Decay for the optimizer", type=float, default=5e-4)
    parser.add_argument('--cers_ocr_path',
                            help="Cer information json")
    args = parser.parse_args()
    print(args)
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
