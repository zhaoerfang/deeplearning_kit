from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """
    RNN model, in which layers can be rnn, gru or lstm.
    """

    def __init__(self, rnn_layer: Union[nn.RNN, nn.GRU, nn.LSTM], vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.rnn_layer = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn_layer.hidden_size
        if not self.rnn_layer.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs: torch.Tensor, state: torch.Tensor):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn_layer(X, state)
        output = nn.Linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device: torch.device, batch_size=1):
        if not isinstance(self.rnn_layer, nn.LSTM):
            return torch.zeros(
                (
                    self.num_directions * self.rnn_layer.num_layers,
                    batch_size,
                    self.num_hiddens,
                ),
                device=device,
            )
        else:
            return (
                torch.zeros(
                    (
                        self.num_directions * self.rnn_layer.num_layers,
                        batch_size,
                        self.num_hiddens,
                    ),
                    device=device,
                ),
                torch.zeros(
                    (
                        self.num_directions * self.rnn_layer.num_layers,
                        batch_size,
                        self.num_hiddens,
                    ),
                    device=device,
                ),
            )


