import torch
from torch import nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        output, hidden = self.rnn(self.dropout(self.embedding(input)))
        return output, hidden


if __name__ == '__main__':
    encoder = EncoderRNN(100, 256)
    input = torch.randint(1, 10, (10, 10))                                                # 序列10个元素，每个元素10维
    output, hidden = encoder(input)
    print(output.size())
    print(hidden.size())
