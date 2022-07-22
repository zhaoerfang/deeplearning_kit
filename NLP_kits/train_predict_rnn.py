import torch
import torch.nn as nn
import torch.optim
import torch.utils.data as udata
from typing import *
from nlp_model.tricks.nlp_tricks import grad_clipping

from rnn_model import RNNModel


def predict_rnn(
    prefix: List[str], num_preds: int, net: RNNModel, vocab, device: torch.device
):
    state = RNNModel.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.reshape(
        torch.tensor([outputs[-1]], device=device), (1, 1)
    ) #todo: rnn_input_shape
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(
            get_input(), state
        )  # y.shape: (num_steps, batch_size, embed_size * b), state.shape: (b * layers, batch_size, embed_size)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return "".join([vocab.idx_to_token[i] for i in outputs])

def train_rnn_epoch(net: RNNModel, train_iter: udata.dataloader.DataLoader, loss, updater: torch.optim.SGD, device=torch.device, use_random_iter=False):
    state = None
    for X, Y in train_iter:
        if state is None or not use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if issubclass(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), Y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long())
        updater.zero_grad()
        l.backward()
        grad_clipping(net, 1)
        updater.step()


def train_rnn(
    net: RNNModel,
    train_iter: udata.dataloader.DataLoader,
    lr: float,
    num_epochs: int,
    device: torch.device,
    vocab,
    num_preds=50,
    use_ramdom_iter=False,
):
    loss = nn.CrossEntropyLoss(reduction='mean')
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr=lr)
    predict = lambda prefix: predict_rnn(prefix, num_preds, net, vocab, device)
    for epoch in range(num_epochs):
        train_rnn_epoch(net, train_iter, loss, updater, device)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
    print(predict('time traveller'))

