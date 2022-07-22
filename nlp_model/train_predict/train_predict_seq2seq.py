import torch
import torch.nn as nn
import torch.optim
from torch.utils.data.dataloader import DataLoader
from tricks.seq_trick import MaskedSoftmaxCELoss, truncate_pad
from tricks.nlp_tricks import grad_clipping
from models.encoderdecoder import EncoderDecoder


def xavier_init_weights(m: nn.Module):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m._flat_weights_names:
            if "weights" in param:
                nn.init.xavier_uniform_(m._parameters[param])


def train_seq2seq(
    net: EncoderDecoder, data_iter: DataLoader, lr, num_epochs, tgt_vocab, device
):
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.1)
    loss = MaskedSoftmaxCELoss()
    net.train()

    for epoch in range(num_epochs):
        # todo: timer, metric
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor(
                [tgt_vocab["<bos>"]] * Y.shape[0], device=device
            ).reshape(
                -1, 1
            )  # [batch_size, num_step]
            dec_input = torch.cat([bos, Y[:, :-1]], dim=1)  # teaching forcing
            Y_hat, _ = net(X, dec_input, X_valid_len)
            # Y_hat: [batch_size, num_steps, vocab_size]
            # Y: [batch_size, num_step]
            # Y_valid_len = [batch_size,]
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            grad_clipping(net, theta=1)
            optimizer.step()
            scheduler.step()


def predict_seq2seq(
    net: EncoderDecoder,
    src_sentence: str,
    src_vocab,
    tgt_vocab,
    num_steps,
    device,
    save_attention_weights=False,
):
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(" ")] + [src_vocab["<eos>"]]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab["<pad>"])
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0
    )  # [1, num_steps] 1 represents one batch
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # add the batch axis
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab["<bos>"]], dtype=device), dim=0) 
    output_seq, attention_weights_seq = [], []
    for _ in range(num_steps):
        Y, dex_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item() 
        if save_attention_weights:
            attention_weights_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<eos>']:
            output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weights_seq




