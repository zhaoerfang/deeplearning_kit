import torch
import torch.nn as nn
import torch.optim
import torch.utils.data as udata


def train_rnn(
    net: nn.Module,
    train_iter: udata.dataloader.DataLoader,
    lr: float,
    num_epochs: int,
    device: torch.device,
    use_ramdom_iter=False,
):
    loss = nn.CrossEntropyLoss()
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr=lr)
        

