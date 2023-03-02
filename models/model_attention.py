import torch.nn.functional as F
import math
import torch.nn as nn
import torch


class HistoryAttention(nn.Module):
    def __init__(self, char_vocab_size, emb_size, Dq, window_size, activation='sigmoid', is_emb_train=False):
        super(HistoryAttention, self).__init__()
        self.Dq = Dq
        if is_emb_train:
            self.embedding = nn.Parameter(torch.normal(0, 1, (char_vocab_size + 1, emb_size)), requires_grad=True)
        else:
            embedding = torch.normal(0, 1, (char_vocab_size + 1, emb_size))
            embedding[char_vocab_size, :] = 0
            self.register_buffer('embedding', embedding)

        self.Wq = nn.Linear(emb_size, Dq)
        self.loss_coef_layer = nn.Linear(window_size, 1)
        self.activation = activation
        self.positional_encodings = nn.Parameter(torch.zeros(window_size, emb_size), requires_grad=True)
        
    def forward(self, char_indices):
        word_embs = self.embedding[char_indices].mean(dim=1)
        word_embs = word_embs + self.positional_encodings

        query = self.Wq(word_embs)
        attention_scores = F.softmax(torch.matmul(query, query.T)/math.sqrt(self.Dq), dim=1)
        if self.activation == "sigmoid":
            loss_weights = F.sigmoid(self.loss_coef_layer(attention_scores))
        elif self.activation == "softmax":
            loss_weights = F.softmax(self.loss_coef_layer(attention_scores), dim=0)
        elif self.activation == "relu":
            loss_weights = F.relu(self.loss_coef_layer(attention_scores))
            loss_weights = loss_weights/(loss_weights.sum() + 0.000001) # Normalize ReLU output

        loss_weights = loss_weights.squeeze(dim=1) # Convert from (window_size x 1) to (window_size)
        return loss_weights