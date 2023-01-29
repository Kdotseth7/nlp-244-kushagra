import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        in_embedding_dim,
        n_hidden,
        n_layers,
        bidirectional,
        rnn_type="elman", # can be elman, lstm, gru
        dropout=0.5  
    ):
        super(RNNModel, self).__init__()

        if rnn_type == "elman":
            self.rnn = nn.RNN(
                in_embedding_dim,
                n_hidden,
                n_layers,
                nonlinearity="tanh",
                dropout=dropout,
                bidirectional=bidirectional
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                in_embedding_dim,
                n_hidden,
                n_layers,
                dropout=dropout,
                bidirectional=bidirectional
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                in_embedding_dim,
                n_hidden,
                n_layers,
                dropout=dropout,
                bidirectional=bidirectional
        )
        else:
            raise NotImplementedError
        
        print('RNN Type: ', rnn_type)
        print('Bi: ', bidirectional)
        self.in_embedder = nn.Embedding(vocab_size, in_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_directions = 2 if bidirectional else 1
        self.pooling = nn.Linear(self.num_directions * n_hidden, vocab_size)
        self.init_weights()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.in_embedder.weight, -initrange, initrange)
        nn.init.zeros_(self.pooling.bias)
        nn.init.uniform_(self.pooling.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.dropout(self.in_embedder(input))
        if self.rnn_type == "elman" or self.rnn_type == "gru":
            output, hidden = self.rnn(emb, hidden)
        elif self.rnn_type == "lstm":
            output, (hidden, cell) = self.rnn(emb, (hidden, hidden))
        else:
            raise NotImplementedError
        output = self.dropout(output)
        pooled = self.pooling(output)
        pooled = pooled.view(-1, self.vocab_size)
        return F.log_softmax(pooled, dim=1), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros(self.num_directions * self.n_layers, batch_size, self.n_hidden)

    @staticmethod
    def load_model(path):
        with open(path, "rb") as f:
            model = torch.load(f)
            model.rnn.flatten_parameters()
            return model
