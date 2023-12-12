import torch.nn as nn

class RNN(nn.Module):
    # 20231212 : num_layers=2, dropout=0.5 추가
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=0.5)

        # fully-connected layer
        self.fc = nn.Linear(in_features=hidden_size,
                            out_features=output_size)

    def forward(self, x, hidden):
        batch_size = x.size(0)

        r_out, hidden = self.rnn(x, hidden)
        r_out = r_out.contiguous().view(batch_size,-1)
        output = self.fc(r_out)

        return output, hidden