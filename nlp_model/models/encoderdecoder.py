import torch
import abc
from torch import nn


class Encoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # @abc.abstractmethod
    def forward(self, src: torch.Tensor, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_state(self, enc_outputs: torch.Tensor, *args):
        raise NotImplementedError

    def forward(self, tgt: torch.Tensor, state: torch.Tensor):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, *args):
        enc_outputs = self.encoder(src, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(tgt, dec_state)

