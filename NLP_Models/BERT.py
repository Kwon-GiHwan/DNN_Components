from transformers import BertModel
import torch
import torch.nn as nn
class Bert(nn.Module):
    def __init__(self, dr_rate):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained('skt/kobert-base-v1')
        self.dropout = nn.Dropout(dr_rate)

    def forward(self, token_idx, attention_mask):

        hidden_staes, pooler= self.model(input_ids=token_idx, attention_mask=attention_mask.float().to(token_idx.device),
                              return_dict=False)

        top_layer = hidden_staes[:, 0, :]


        return self.dropout(hidden_staes)
