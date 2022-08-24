import datetime
import torch
import argparse
import os
import math
import random
from torch.nn import CTCLoss, MSELoss
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms

from models.model_crnn import CRNN
from models.model_unet import UNet
from datasets.img_dataset import ImgDataset
from utils import get_char_maps, set_bn_eval, pred_to_string
from utils import get_ocr_helper, compare_labels, save_img, create_dirs
from utils import random_subset
from transform_helper import PadWhite, AddGaussianNoice
import properties as properties
from selection_utils import datasampler_factory
from tracking_utils import call_crnn, generate_ctc_label, weighted_ctc_loss, generate_ctc_target_batches, add_labels_to_history
import wandb
import json
wandb.Table.MAX_ROWS = 50000
import csv
import pandas as pd
from pprint import pprint
wandb.init(project='ocr-calls-reduction', entity='tataganesh', tags=["VGG"])

minibatch_subset_methods = {"random": random_subset}

class TrainNNPrep():

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.lr_crnn = args.lr_crnn
        self.lr_prep = args.lr_prep
        self.max_epochs = args.epoch
        self.warmup_epochs = args.warmup_epochs
        self.inner_limit = args.inner_limit
        self.crnn_model_path = args.crnn_model
        self.prep_model_path = args.prep_model

        self.exp_base_path = args.exp_base_path
        self.ckpt_base_path = os.path.join(self.exp_base_path, properties.prep_crnn_ckpts)
        self.cers_base_path = os.path.join(self.exp_base_path, "cers")
        self.tracked_labels_path = os.path.join(self.exp_base_path, "tracked_labels")
        self.tensorboard_log_path = os.path.join(self.exp_base_path, properties.prep_tensor_board)
        self.img_out_path = os.path.join(self.exp_base_path, properties.img_out)
        create_dirs([self.exp_base_path, self.ckpt_base_path, self.tensorboard_log_path, self.img_out_path, self.cers_base_path, self.tracked_labels_path])

        self.sec_loss_scalar = args.scalar
        self.ocr_name = args.ocr
        self.std = args.std
        self.is_random_std = args.random_std
        self.iter_interval = args.print_iter
        self.inner_limit_skip = args.inner_limit_skip
        self.crnn_imputation = args.crnn_imputation
        # self.tensorboard_log_path = args.tb_log_path
        torch.manual_seed(42)
        self.train_set =  os.path.join(args.data_base_path, properties.vgg_text_dataset_train)
        self.validation_set =  os.path.join(args.data_base_path, properties.vgg_text_dataset_dev)
        self.start_epoch = args.start_epoch
        self.selection_method = args.minibatch_subset
        self.train_batch_size = self.batch_size
           
        self.train_batch_prop = 1 
        if args.minibatch_subset_prop is not None and self.selection_method:
            self.train_batch_prop = args.minibatch_subset_prop
        self.train_subset_size = args.train_subset_size
        self.val_subset_size = args.val_subset_size
        self.track_importance = pd.DataFrame()
        # self.model_preds_last = torch.zeros((self.train_subset_size, 50))
        
        self.cers = None
        if args.cers_ocr_path:
            with open(args.cers_ocr_path, 'r') as f:
                self.cers = json.load(f)
        
        if self.selection_method:
            self.cls_sampler = datasampler_factory(self.selection_method)
            if self.selection_method in ("uniformCER", "rangeCER"):
                self.sampler = self.cls_sampler(self.cers)
            else:
                self.sampler = self.cls_sampler()
        
        if self.cers:
            self.tracked_labels = {name: [] for name in self.cers.keys()}
            self.ctc_loss_weights_noocr = torch.tensor([0.5, 0.25, 0.15, 0.07, 0.03])
            self.ctc_loss_weights = torch.tensor([1, 0.7, 0.4, 0.2, 0.1])
            self.window_size = len(self.ctc_loss_weights)
            
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
        self.dataset = ImgDataset(
            self.train_set, transform=transform, include_name=True, include_index=True)
        self.validation_set = ImgDataset(
            self.validation_set, transform=transform, include_name=True)


        if not self.train_subset_size:
            self.train_subset_size = len(self.dataset)
        self.model_labels_last = ["" for _ in range(0, self.train_subset_size)]
        rand_indices = torch.randperm(len(self.dataset))[:self.train_subset_size]
        self.train_subset_index_mapping = torch.zeros(len(self.dataset))
        self.train_subset_index_mapping[rand_indices] = torch.arange(0, self.train_subset_size).float()
        self.dataset_subset = torch.utils.data.Subset(self.dataset, rand_indices)
        self.loader_train = torch.utils.data.DataLoader(
            self.dataset_subset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        
        if not self.val_subset_size:
            self.val_subset_size = len(self.validation_set)
        rand_indices = torch.randperm(len(self.validation_set))[:self.val_subset_size]
        validation_set_subset = torch.utils.data.Subset(self.validation_set, rand_indices)
        self.loader_validation = torch.utils.data.DataLoader(
            validation_set_subset, batch_size=self.batch_size, drop_last=True)

        self.train_set_size = len(self.loader_train.dataset)
        self.val_set_size = len(self.loader_validation.dataset)
        self.sample_importance = torch.ones(self.train_set_size)/10.0
        self.sample_frequency = torch.zeros((self.train_set_size, self.max_epochs, 2))

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
            self.scheduler_prep = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_prep, T_max=self.max_epochs)

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

    def log_gradients_in_model(self, model, writer, step):
        for tag, value in model.named_parameters():
            if value.grad is not None:
                writer.add_histogram(tag + "/grad", value.grad.cpu(), step)
                writer.add_histogram(tag + "/value", value.data.cpu(), step)
    
    def Rop(self, y, x, v):
        """Computes an Rop.
        
        Arguments:
            y (Variable): output of differentiated function
            x (Variable): differentiated input
            v (Variable): vector to be multiplied with Jacobian from the right
        """
        v.requires_grad = True
        w = torch.ones_like(y, requires_grad=True)
        return list(torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, v))

    def train(self):
        all_file_names = list()
        for images, labels, names, indices in self.loader_train:
            all_file_names.extend(names)

        noiser = AddGaussianNoice(
            std=self.std, is_stochastic=self.is_random_std, return_noise=True)
        writer = SummaryWriter(self.tensorboard_log_path)
        
        print(f"Batch size is {self.batch_size}")
        print(f"Train batch size is {self.train_batch_size}")
        validation_step = 0
        jvp_train_cer = 0
        total_bb_calls = 0
        total_crnn_updates = 0
        self.crnn_model.zero_grad()
        
        for epoch in range(self.start_epoch, self.max_epochs):
            epoch_bb_calls = 0
            epoch_crnn_updates = 0
            step = 0
            training_loss = 0
            jvp_loss = 0
            epoch_print_flag = True

            for images, labels, names, indices in self.loader_train:
                # self.sample_frequency[indices, epoch, 0] = 1
                self.crnn_model.train()
                self.prep_model.eval()
                self.prep_model.zero_grad()
                self.crnn_model.zero_grad()
                X_var = images.to(self.device)
                img_preds_all = self.prep_model(X_var)
                temp_loss = 0
                noisy_imgs_list = list()
                noisy_labels_list = list()
                # jitter_noise_list = list()
                indices_all = self.train_subset_index_mapping[indices].long()
                if self.selection_method and epoch >= self.warmup_epochs:
                    num_bb_samples = max(1, math.ceil(img_preds_all.shape[0]*(1 - self.train_batch_prop)))
                    img_preds, labels_gt, bb_sample_indices = self.sampler.query(img_preds_all, labels, num_bb_samples, names)

                    img_preds = img_preds.detach().cpu()
                    img_preds_names = [names[index] for index in bb_sample_indices]
                    skipped_mask = torch.ones(indices.shape[0], dtype=bool)
                    skipped_mask[bb_sample_indices] = False
                    
                else:
                    img_preds = img_preds_all.detach().cpu() 
                    img_preds_names = names
                    indices = indices_all
                    
                    # Compute Loss
                    # primary_loss_fn_sample_wise(scores, y, pred_size, y_size)
                if epoch_print_flag:
                    print(f"Total Samples - {img_preds_all.shape[0]}")
                    print(f"OCR Samples - {img_preds.shape[0]}")
                    epoch_print_flag = False
                if not(self.selection_method and int(self.train_batch_prop)==1):
                    for i in range(self.inner_limit):
                        self.prep_model.zero_grad()
                        if i == 0 and self.inner_limit_skip:
                            ocr_labels = self.ocr.get_labels(img_preds)
                            # Only required if OCR is called, to append to the existing history
                            add_labels_to_history(self, img_preds_names, ocr_labels)
                            
                            history_present_indices = [idx for idx, name in enumerate(names) if skipped_mask[idx] and name in self.tracked_labels and self.tracked_labels[name]]
                            loss_weights = None
                            if history_present_indices and self.crnn_imputation:
                                history_present_indices = random.sample(history_present_indices, min(len(ocr_labels), len(history_present_indices))) # Sample equal to number of ocr calls
                                extra_img_names = [names[idx] for idx in history_present_indices]
                                img_preds_names.extend(extra_img_names)
                                extra_imgs = img_preds_all[history_present_indices]
                                img_preds = torch.cat([img_preds.to(self.device), extra_imgs])
                                loss_weights = torch.zeros(img_preds.shape[0], self.window_size)
                                loss_weights[:len(ocr_labels), :] = self.ctc_loss_weights
                                loss_weights[len(ocr_labels):, :] = self.ctc_loss_weights_noocr
                                loss_weights = loss_weights.to(self.device)

                            # Peek at history of OCR labels for each strip and construct weighted CTC loss
                            target_batches = generate_ctc_target_batches(self, img_preds_names)
                            scores, pred_size = call_crnn(self, img_preds)
                            loss = weighted_ctc_loss(self, scores, pred_size, target_batches, loss_weights)
                            total_bb_calls += len(ocr_labels)
                            epoch_bb_calls += len(ocr_labels)
                            total_crnn_updates += len(history_present_indices)
                            epoch_crnn_updates += len(history_present_indices)
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
                self.optimizer_crnn.step()
                writer.add_scalar('CRNN Training Loss',
                                  CRNN_training_loss, step)

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

                #print(f"Preprocessor total samples - {img_preds.shape[0]}")

                # Update last seen prediction of image
                model_gen_labels = pred_to_string(scores, labels, self.index_to_char)
                
                for i, index in enumerate(indices_all):
                    self.model_labels_last[index.item()] = model_gen_labels[i]

                training_loss += loss.item()
                if step % self.iter_interval == 0:
                    print(f"Epoch: {epoch}, Iteration: {step} => {loss.item()}")
                step += 1 
                
                if self.selection_method and len(img_preds_names):
                    batch_cers = list()
                    for i in range(len(labels)):
                        _, batch_cer = compare_labels([model_gen_labels[i]], [labels[i]])
                        batch_cers.append(batch_cer)
                    self.sampler.update_cer(batch_cers, names)

            
            train_loss =  training_loss / (self.train_set_size//self.train_batch_size)
            writer.add_scalar('Training Loss', train_loss, epoch + 1) # Change to batch size if not randomly sampling from mini-batches

            if self.selection_method: 
                with open(os.path.join(self.cers_base_path, f"cers_{epoch}.json"), 'w') as f:
                    json.dump(self.sampler.cers, f)
                with open(os.path.join(self.tracked_labels_path, f"tracked_labels_{epoch}.json"), 'w') as f:
                    json.dump(self.tracked_labels, f)
            with open(os.path.join(self.tracked_labels_path, f"tracked_labels_current.json"), 'w') as f:
                json.dump(self.tracked_labels, f)
                
            if self.selection_method:
                with open(os.path.join(self.cers_base_path, f"cers_current.json"), 'w') as f:
                    json.dump(self.sampler.cers, f)

            wandb.save(os.path.join(self.tracked_labels_path, f"tracked_labels_current.json"))
            wandb.save(os.path.join(self.cers_base_path, f"cers_current.json"))


            if self.lr_scheduler:
                self.scheduler_crnn.step()
                self.scheduler_prep.step()
            self.prep_model.eval()
            self.crnn_model.eval()
            pred_correct_count = 0
            pred_CER = 0
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
                    pred_correct_count += crt
                    tess_accuracy += tess_crt
                    pred_CER += cer
                    tess_CER += tess_cer
                    validation_step += 1
            CRNN_accuracy = pred_correct_count/self.val_set_size
            OCR_accuracy = tess_accuracy/self.val_set_size
            CRNN_cer = pred_CER/self.val_set_size
            OCR_cer = tess_CER/self.val_set_size
            val_loss = validation_loss / (self.val_set_size//self.batch_size)
            writer.add_scalar('Accuracy/CRNN_output',
                              CRNN_accuracy, epoch + 1)
            writer.add_scalar('Accuracy/'+self.ocr_name+'_output',
                              OCR_accuracy, epoch + 1)
            writer.add_scalar('WER and CER/CRNN_CER',
                              CRNN_cer, epoch + 1)
            writer.add_scalar('WER and CER/'+self.ocr_name+'_CER',
                              OCR_cer, epoch + 1)
            writer.add_scalar('Validation Loss', val_loss, epoch + 1)

            # self.track_importance["File Name"] = [name for image, label, name, indice in self.dataset_subset]
            # self.track_importance["Importance (Loss)"] = self.sample_importance.detach().numpy()
            # self.track_importance.to_csv(os.path.join(self.exp_base_path, "Sample_Data_Importance_loss.csv"))
            # sample_importance_table = wandb.Table(dataframe=self.track_importance)
            # torch.save(self.sample_frequency, os.path.join(self.exp_base_path, "Sample_Frequency.pt"))

            wandb.log({"CRNN_accuracy": CRNN_accuracy, f"{self.ocr_name}_accuracy": OCR_accuracy, 
                        "CRNN_CER": CRNN_cer, f"{self.ocr_name}_cer": OCR_cer, "Epoch": epoch + 1,
                        "train_loss": train_loss, "val_loss": val_loss,
                        "Total Black-Box Calls": total_bb_calls, "Black-Box Calls":  epoch_bb_calls,
                        "Total CRNN Updates": total_crnn_updates, "CRNN Updates": epoch_crnn_updates})

            
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
            torch.save(self.prep_model,
                        os.path.join(self.ckpt_base_path, "Prep_model_"+str(epoch)))
            torch.save(self.crnn_model, os.path.join(self.ckpt_base_path,
                       "CRNN_model_" + str(epoch)))
        writer.flush()
        writer.close()

                



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
    parser.add_argument('--print_iter', type=int,
                        default=100, help='Interval for printing iterations per Epoch')
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
    parser.add_argument('--crnn_imputation', help="If true, crnn is updated using just the history for samples that do not have an OCR label ", action="store_true")

    
    
    args = parser.parse_args()
    # Conditions on arguments
    # if args.inner_limit < 1:
    #     parser.error("Minimum Value for Inner Limit is 1")
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

