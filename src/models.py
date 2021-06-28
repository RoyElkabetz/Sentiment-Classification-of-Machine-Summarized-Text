import random
import torch
from torch import nn


class TextClassificationModel(nn.Module):
    """Simple text classification model (taken from Pytorch.Tutorials)"""
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        # initializations
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
            dimensions:
            outputs = [src len, batch size, hid dim * n directions]
            hidden = [n layers * n directions, batch size, hid dim]
            cell = [n layers * n directions, batch size, hid dim]
        """
        embedded = self.dropout(self.embedding(src))  # [src len, batch size]
        outputs, (hidden, cell) = self.rnn(embedded)  # [src len, batch size, emb dim]
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, model_input, hidden, cell):
        """
            input = [batch size]
            hidden = [n layers * n directions, batch size, hid dim]
            cell = [n layers * n directions, batch size, hid dim]
            output = [seq len, batch size, hid dim * n directions]

            n directions in the decoder will both always be 1, therefore:
            hidden = [n layers, batch size, hid dim]
            context = [n layers, batch size, hid dim]
        """

        model_input = model_input.unsqueeze(0)  # [1, batch size]
        embedded = self.dropout(self.embedding(model_input))  # [1, batch size, emb dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))  # [batch size, output dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """A basic Encoder-Decoder for text modeling / summarizing network"""
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
            Dimensions:
            src = [src len, batch size]
            trg = [trg len, batch size]
            teacher_forcing_ratio is probability to use teacher forcing
            e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        """

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        decoder_input = trg[0, :]

        for t in range(1, trg_len):

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            decoder_input = trg[t] if teacher_force else top1
        return outputs
