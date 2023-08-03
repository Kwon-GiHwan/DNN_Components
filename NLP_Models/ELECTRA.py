from transformers import ElectraModel
from torch import nn

class Electra(nn.Module):
    def __init__(self, dr_rate):
        super(Electra, self).__init__()
        self.model = ElectraModel.from_pretrained("monologg/koelectra-small-v3-discriminator")
        self.dropout = nn.Dropout(dr_rate)

    def forward(self, token_idx, attention_mask):

        hidden_staes = self.model(input_ids=token_idx, attention_mask=attention_mask.float().to(token_idx.device),
                              return_dict=False)

        # output = hidden_staes[:, 0, :] - for cls idx classifying(final states, pooler)


        return self.dropout(hidden_staes[0][:, 0, :])