import torch
import torch.nn as nn


def sequence_mask(X: torch.Tensor, valid_len: torch.Tensor, value=0):
    """
    the input is batch first.
    """
    maxlen = X.size(1)
    mask = (
        torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :]
        < valid_len[:, None]
    )
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """
     `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
     `label` shape: (`batch_size`, `num_steps`)
     `valid_len` shape: (`batch_size`,)
    """

    def forward(self, pred: torch.Tensor, label: torch.Tensor, valid_len: torch.Tensor):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = "none"
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


if __name__ == "__main__":
    # X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # sequence_mask(X, torch.tensor([1, 2]))

    # loss = MaskedSoftmaxCELoss()
    # print(
    #     loss(
    #         torch.ones(3, 4, 10),
    #         torch.ones((3, 4), dtype=torch.long),
    #         torch.tensor([4, 2, 0]),
    #     )
    # )

    y_hat = torch.rand((2, 3, 4))
    y = torch.randint(0, 3, (2, 4))
    loss = nn.CrossEntropyLoss(reduction="none")
    l = loss(y_hat, y)
    print(y)

