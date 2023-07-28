class RNNCell_Encoder(nn.Module):  # 워드 임베딩 및 RNN cell 정의
    def __init__(self, input_dim, hidden_size):
        super(RNNCell_Encoder, self).__init__()
        self.rnn = nn.RNNCell(input_dim, hidden_size)  # rnn cell 구현

    def forward(self, inputs):
        bz = inputs.shape[1]
        ht = torch.zeros((bz, hidden_size)).to(device)  # 현재 상태(h_t)
        for word in inputs:  # word : 현재 입력 벡터(x_t)
            ht = self.rnn(word, ht)  # ht : 이전 상태(h_t-1)
        return ht


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.em = nn.Embedding(len(TEXT.vocab.stoi), embedding_dim)  # 임베딩
        self.rnn = RNNCell_Encoder(embedding_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.em(x)
        x = self.rnn(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x