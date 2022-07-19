from typing import Union
import torch
import torch.nn as nn

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
            
