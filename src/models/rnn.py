import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)

        # fully-connected layer
        self.fc = nn.Linear(in_features=hidden_size,
                            out_features=output_size)

    def forward(self, x, hidden):
        batch_size = x.size(0)

        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)

        # shape output to the linear layer (batch_size, seq_length*hidden_dim)
        r_out = r_out.contiguous().view(batch_size,-1)

        # get final output
        output = self.fc(r_out)

        return output, hidden