import torch
import torch.nn as nn
from encoderdecoder import Encoder


class Seq2SeqEncoder(Encoder):
    """
    The RNN encoder for seq2seq learning.
    The input source in not embedded.
    """

    def __init__(
        self,
        vocab_size,
        embed_size,
        num_hiddens,
        num_layers,
        bidirectional=False,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, bidirectional, dropout)

    def forward(self, src: torch.Tensor, *args):
        src = self.embedding(src)
        src = src.permute(1, 0, 2)
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        output, state = self.rnn(src)
        return output, state

