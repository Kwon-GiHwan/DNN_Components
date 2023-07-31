import torch
from torch import nn
class RNNClassifier(nn.Module):

    def __init__(self, bidirectional=True, num_layers=1, input_size=768,
                 hidden_size=768, dropout=0.1, num_class = 5):
        super(RNNClassifier, self).__init__()

        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions

        self.num_class = num_class

        self.rnn = nn.LSTM(
          input_size=input_size,
          hidden_size=self.hidden_size,
          num_layers=num_layers,
          bidirectional=bidirectional
          )

        self.wo = nn.Linear(self.num_directions * self.hidden_size, self.num_class, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = 0)

    def forward(self, x):
        """See :func:`EncoderBase.forward()`"""
        # x = torch.transpose(x, 1, 0) #swap dim0 and dim1
        #dim transpost 필요성 재고
        output, _ = self.rnn(x)
        out_fow = output[range(len(output)),  :self.hidden_size]
        out_rev = output[:, self.hidden_size:]
        output = torch.cat((out_fow, out_rev), 1)
        output = self.dropout(output)

        out_cls = self.softmax(self.wo(torch.squeeze(output, 1)))

        return out_cls