import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, 
                 input_dim, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers, 
                 bidirectional, 
                 dropout) -> None:
        super(LSTM, self).__init__()
        # Bidirectional LSTM or not
        self.bidirectional = bidirectional
        # Initialize Embedding Layer
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        # Initialzie LSTM layer to process the vector sequences 
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim,
                            num_layers = n_layers,
                            bidirectional = bidirectional,
                            dropout = dropout,
                            batch_first = True)
        num_directions = 2 if self.bidirectional else 1
        # Initialize Dense layer to predict
        self.fc = nn.Linear(hidden_dim * num_directions, output_dim)
        # Initialize dropout to improve with regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x, 
                x_lengths) -> torch.Tensor:
        # Embedding Layer
        embedded = self.embedding(x)
        # Dropout Layer before LSTM Layer
        embedded = self.dropout(embedded)
        # Packed Sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, 
                                                            x_lengths, 
                                                            batch_first = True, 
                                                            enforce_sorted = False)
        # LSTM Layer
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # Unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, 
                                                                  batch_first = False)
        # Concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers and Apply Dropout
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)) if self.lstm.bidirectional else self.dropout(hidden[-1,:,:])
        # Fully Connected Layer
        output = self.fc(hidden)
        return output.squeeze(1)