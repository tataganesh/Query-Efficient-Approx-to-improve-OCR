import tracking_utils
import torch
import Levenshtein
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import sys
sys.path.insert(0, "../")
from models.model_attention import HistoryAttention
import properties


class LossWeightGenerator(metaclass=ABCMeta):
    
    def __init__(self, tracking_args, device, char_to_index):
        pass
    
    @abstractmethod
    def gen_weights(self, tracked_labels, img_names):
        pass
        
    @abstractmethod
    def print_debug_statements(self):
        pass
    
    
class AttentionWeightGenerator(LossWeightGenerator):
    def __init__(self, tracking_args, device, char_to_index):
        self.window_size = tracking_args.window_size
        self.query_dim = tracking_args.query_dim
        self.emb_dim = tracking_args.emb_dim
        self.attn_activation = tracking_args.attn_activation
        self.device = device
        self.char_to_index = char_to_index
        
        self.attention_model = HistoryAttention(len(properties.char_set), self.emb_dim, self.query_dim, self.window_size, self.attn_activation).to(self.device)
        
    
    def print_debug_statements(self):
        pass
    
    def gen_weights(self, tracked_labels, img_names: list):
        """_summary_ 

        Args:
            attention_model (_type_): Attention model
            img_names (list): Image Names for inference
        """
        loss_weights = torch.zeros(len(img_names), self.window_size + 1).to(self.device)
        loss_weights[:, 0] = 1 # Weight of 1 for most recent label
        for img_index, name in enumerate(img_names):
            if name not in tracked_labels:
                continue
            label_history = tracked_labels[name][-self.window_size:][::-1] # Get labels in specific window from history, from most -> least recent
            history_len = len(label_history)
            if history_len:
                encoded_words = tracking_utils.str_to_tensor(self, label_history) # Convert words to tensor representation
                weights = self.attention_model(encoded_words)
                loss_weights[img_index][1:history_len + 1] = weights[:history_len]
        return loss_weights



class LevenshteinWeightGenerator(LossWeightGenerator):
    def __init__(self, tracking_args, device, char_to_index=None):
        self.args = tracking_args
        self.window_size = tracking_args.window_size
        self.device = device
    
    def print_debug_statements(self):
        pass
    
    def gen_weights(self, tracked_labels, img_names):
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
        loss_weights[:, 0] = 1  # Weight of 1 for most recent label
        for img_index, name in enumerate(img_names):
            if name not in  tracked_labels:
                continue
            label_history = tracked_labels[name][-self.window_size:][::-1]
            num_elements = max((len(label_history) - 1), 1)  # Omit ith element
            for i in range(len(label_history)):
                dist_sum = 0
                num_chars = max(1, len(label_history[i]))
                for j in range(len(label_history)):
                    if i == j:
                        continue
                    dist_sum += Levenshtein.distance(label_history[i], label_history[j])          
                dist_mean = dist_sum/num_elements  # Mean of distances with all other labels
                normalized_dist_mean = hist_multiplier * (1 - min(dist_mean, num_chars)/num_chars)  # Min-Max normalization using input string length. Smaller values are closer to 1
                loss_weights[img_index][i + 1] = normalized_dist_mean # i + 1 because first column is for label from current epoch
        return loss_weights



class DecayingWeightGenerator(LossWeightGenerator):  
    def __init__(self, tracking_args, device, char_to_index=None):
        self.decay_factor = tracking_args.decay_factor
        self.window_size = tracking_args.window_size
        self.device = device
    
    def print_debug_statements(self):
        pass
    
    def gen_weights(self, training_obj, img_names):
        return torch.tensor([self.decay_factor**i for i in range(0, self.window_size)]).to(self.device)

 
def weightgenerator_factory(method):
    weight_method_mapping = {
        "self_attention": AttentionWeightGenerator,
        "levenshtein": LevenshteinWeightGenerator,
        "decaying": DecayingWeightGenerator,
    }
    return weight_method_mapping[method]
