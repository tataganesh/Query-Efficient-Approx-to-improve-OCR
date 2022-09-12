import torch
import random

def call_crnn(self, images):
    X_var = images.to(self.device)
    scores = self.crnn_model(X_var)
    out_size = torch.tensor(
        [scores.shape[0]] * images.shape[0], dtype=torch.int)
    return scores, out_size

def generate_ctc_label(self, labels):
    y_size = torch.tensor([len(l) for l in labels], dtype=torch.int)
    conc_label = ''.join(labels)
    y = [self.char_to_index[c] for c in conc_label]
    y_var = torch.tensor(y, dtype=torch.int)
    return y_var, y_size

def generate_ctc_target_batches(self, img_names):
    target_batches = list()
    for i in range(self.window_size):
        batch_labels = list()
        img_indices = list()
        for j, name in enumerate(img_names):
            label_history = self.tracked_labels[name]
            if i < len(label_history):
                ocr_label = label_history[-(i+1)] # ith index from the back
                batch_labels.append(ocr_label)
                img_indices.append(j)
        if len(img_indices):
            target, target_size = generate_ctc_label(self, batch_labels)
            target_batches.append([target, target_size, img_indices])
    return target_batches

def weighted_ctc_loss(self, scores, pred_size, target_batches, loss_weights=None):
    num_losses = min(len(target_batches), self.window_size)
    all_ctc_losses = list()
    for i in range(num_losses):
        target, target_size, img_indices = target_batches[i]
        scores_subset = scores[:, img_indices, :]
        pred_size_subset = pred_size[img_indices]
        if loss_weights is None:
            loss_weight = self.ctc_loss_weights[i]
            ctc_loss = self.primary_loss_fn(scores_subset, target, pred_size_subset, target_size)
            all_ctc_losses.append(loss_weight*ctc_loss)
        else:
            loss_weights_subset = loss_weights[img_indices, i]
            ctc_losses = self.primary_loss_fn_sample_wise(scores_subset, target, pred_size_subset, target_size)
            all_ctc_losses.append(torch.mean(loss_weights_subset*ctc_losses))
    return sum(all_ctc_losses)

def add_labels_to_history(self, image_keys, ocr_labels, epoch=None):
    for lbl_index, name in  enumerate(image_keys): 
        if name not in self.tracked_labels:
            self.tracked_labels[name] = list() # Why is this required?
        self.tracked_labels[name].append(ocr_labels[lbl_index])
        if epoch:
            if name not in self.tracked_labels:
                self.tracked_lbl_epochs[name] = list() # Why is this required?
            self.tracked_lbl_epochs[name].append(epoch)



# def history_imputation(self, img_pred_names, img_names_all,  img_preds, img_preds_all, skipped_mask, epoch=None):
#     ocr_labels = self.ocr.get_labels(img_preds)
#     # Only required if OCR is called, to append to the existing history
#     add_labels_to_history(self, img_pred_names, ocr_labels, epoch)
                            
#     history_present_indices = [idx for idx, name in enumerate(img_names_all) if skipped_mask[idx] and name in self.tracked_labels and self.tracked_labels[name]]
#     loss_weights = None
#     if history_present_indices and self.crnn_imputation:
#         history_present_indices = random.sample(history_present_indices, min(len(ocr_labels), len(history_present_indices))) # Sample equal to number of ocr calls
#         extra_img_names = [img_names_all[idx] for idx in history_present_indices]
#         img_pred_names.extend(extra_img_names)
#         extra_imgs = img_preds_all[history_present_indices]
#         img_preds = torch.cat([img_preds.to(self.device), extra_imgs])
#         loss_weights = torch.zeros(img_preds.shape[0], self.window_size)
#         loss_weights[:len(ocr_labels), :] = self.ctc_loss_weights
#         loss_weights[len(ocr_labels):, :] = self.ctc_loss_weights_noocr
#         if self.time_decay:
#             for index, strip_name in enumerate(img_pred_names):
#                 epoch_dist = [(epoch - element_epoch) for element_epoch in self.tracked_lbl_epochs[strip_name][::-1][:self.window_size]]
#                 for j in range(len(epoch_dist)):
#                     loss_weights[index, j] = loss_weights[index, j] * (self.time_decay**epoch_dist[j])
#         loss_weights = loss_weights.to(self.device)
        
#     return img_preds, img_pred_names, loss_weights