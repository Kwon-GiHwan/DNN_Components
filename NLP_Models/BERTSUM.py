import torch
import torch.nn as nn
from torch import arange


class RNNEncoder(nn.Module):

    def __init__(self, bidirectional=True, num_layers=12, input_size=768,
                 hidden_size=768, dropout=0.0):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0

        hidden_size = self.hidden_size // num_directions

        # self.rnn = LayerNormLSTM( #author's  LSTM Model
        #     input_size=input_size,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     bidirectional=bidirectional)

        self.rnn = nn.LSTM(
          input_size=self.input_size,
          hidden_size=hidden_size,
          num_layers=self.num_layers,
          bidirectional=bidirectional
          )

        self.wo = nn.Linear(num_directions * hidden_size, 1, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        x = torch.transpose(x, 1, 0)
        memory_bank, _ = self.rnn(x)
        memory_bank = self.dropout(memory_bank) + x
        memory_bank = torch.transpose(memory_bank, 1, 0)

        sent_scores = self.sigmoid(self.wo(memory_bank))
        sent_scores = sent_scores.squeeze(-1) * mask.float()
        return sent_scores

class Classifier(nn.Module):
    def __init__(self, hidden_size=768):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear(x).squeeze(-1)
        # h = self.linear1(x)
        sent_scores = self.sigmoid(h) * mask_cls.float()

        return sent_scores

class Summarizer(nn.Module):
    def __init__(self, argument_train):
        super(Summarizer, self).__init__()

        self.bert = Bert(drop_rate_bert)

        if(encoder == "rnn"):
            self.encoder = RNNEncoder(bidirectional=True, num_layers=num_layers,
                                          input_size=input_size, hidden_size=hidden_size,
                                          dropout=drop_rate_encoder)

        elif(encoder == "linear"):
            self.encoder = Classifier(input_size)

    def forward(self, token_idx, attn_mask, cls_idx, cls_mask):

        top_vec = self.bert(token_idx, attn_mask)
        sents_vec = top_vec[arange(top_vec.size(0)).unsqueeze(1), cls_idx]

        sents_vec = sents_vec.detach().numpy() * cls_mask[:, :, None].float().detach().numpy()
        # sents_vec = sents_vec * cls_mask[:, :, None].float() #use if graphic card exists

        sent_scores = self.encoder(torch.from_numpy(sents_vec), cls_mask).squeeze(-1)
        # sent_scores = self.encoder(sents_vec, cls_mask).squeeze(-1)#use if graphic card exists

        return sent_scores, cls_mask
