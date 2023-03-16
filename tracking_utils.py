import torch
from typing import List
import properties
import Levenshtein

def call_crnn(self, images):
    X_var = images.to(self.device)
    scores = self.crnn_model(X_var)
    out_size = torch.tensor(
        [scores.shape[0]] * images.shape[0], dtype=torch.int)
    return scores, out_size

def str_to_tensor(self, words: List[str]):
    """_summary_

    Args:
        words (List[str]): List of words in hisotry of a particular sample
        window_size (int): Size of history window 
    """
    vocab_size = len(properties.char_set)
    encoded_words = list()
    for word in words:
        char_mapping = [self.char_to_index[c] for c in word]
        empty_chr_slots = max(0, properties.max_char_len - len(word))
        char_mapping.extend([vocab_size] * empty_chr_slots)
        encoded_words.append(char_mapping)

    empty_word_slots = max(0, self.window_size - len(words))
    encoded_words.extend([[vocab_size] * properties.max_char_len] * empty_word_slots) # Better way - Generate random vectors
    
    encoded_words = torch.tensor(encoded_words).to(self.device)
    return encoded_words


def generate_loss_weights(self, img_names: list):
    """_summary_ 

    Args:
        attention_model (_type_): Attention model
        img_names (list): Image Names for inference
    """
    loss_weights = torch.zeros(len(img_names), self.window_size + 1).to(self.device)
    loss_weights[:, 0] = 1 # Weight of 1 for most recent label
    modified_weights_list = torch.empty(0).to(self.device)
    for img_index, name in enumerate(img_names):
        if name not in self.tracked_labels:
            continue
        label_history = self.tracked_labels[name][-self.window_size:][::-1] # Get labels in specific window from history, from most -> least recent
        history_len = len(label_history)
        if history_len:
            encoded_words = str_to_tensor(self, label_history) # Convert words to tensor representation
            weights = self.attention_model(encoded_words)
            loss_weights[img_index][1:history_len + 1] = weights[:history_len]
            modified_weights_list = torch.cat((modified_weights_list, weights[:history_len]))
    return loss_weights, modified_weights_list


def generate_weights_levenshtein(self, img_names: list):
    """Generate loss weights using levenshtein distance between history labels

    Args:
        img_names (list): List of selected images from current mini-batch

    Returns:
        tuple(torch.tensor, torch.tensor): Returns (loss weights, loss weights list). 
            loss weights - Tensor of shape (|img_names|, window_size + 1)
            loss weights list - 1-D tensor of all calculated loss weights. Used for wandb logging
    """
    hist_multiplier = 0.5  
    loss_weights = torch.zeros(len(img_names), self.window_size + 1).to(self.device)
    loss_weights[:, 0] = 1 # Weight of 1 for most recent label
    modified_weights_list = list()
    for img_index, name in enumerate(img_names):
        if name not in self.tracked_labels:
            continue
        label_history = self.tracked_labels[name][-self.window_size:][::-1]
        num_elements = max((len(label_history) - 1), 1)  # Omit ith element
        for i in range(len(label_history)):
            dist_sum = 0
            num_chars = max(1, len(label_history[i]))
            for j in range(len(label_history)):
                if i == j:
                    continue
                dist_sum += Levenshtein.distance(label_history[i], label_history[j])          
            dist_mean = dist_sum/num_elements # Mean of distances with all other labels
            normalized_dist_mean = hist_multiplier * (1 - min(dist_mean, num_chars)/num_chars) # Min-Max normalization using input string length. Smaller values are closer to 1
            loss_weights[img_index][i + 1] = normalized_dist_mean # i + 1 because first column is for label from current epoch
            modified_weights_list.append(normalized_dist_mean)
    modified_weights_list = torch.tensor(modified_weights_list).to(self.device)
    return loss_weights, modified_weights_list
            
    
    

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

def add_labels_to_history(self, image_keys, ocr_labels):
    for lbl_index, name in  enumerate(image_keys): 
        if name not in self.tracked_labels:
            self.tracked_labels[name] = list() # Why is this required?
        self.tracked_labels[name].append(ocr_labels[lbl_index])


if __name__ =="__main__":
    print("hey")
    
    def attn_model(param):
        return torch.tensor([1,2,3]).float()
    
    class temp:      
        def __init__(self):
            self.window_size = 4
            self.device = 'cpu'
            self.tracked_labels = {
                "a": ["a", "a", "a"],
                "b": ["i", " ", "o", "a"]
            }
    obj = temp()
    print(generate_weights_levenshtein(obj, obj.tracked_labels.keys()))
            