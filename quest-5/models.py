import torch
import torch.nn as nn

class LSTM(nn.Module):
    """LSTM Model Class"""
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
        # Initialize LSTM layer to process the vector sequences 
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
                                                                  batch_first = True)
        # Concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers and Apply Dropout
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)) if self.lstm.bidirectional else self.dropout(hidden[-1,:,:])
        # Fully Connected Layer
        output = self.fc(hidden)
        return output.squeeze(1)
    

class EncoderBlock(nn.Module):
    """Encoder block with self-attention and dense layer"""
    def __init__(self, input_dim, hidden_dim, n_heads, dropout):
        super(EncoderBlock, self).__init__()
        # Initialize Multihead Attention layer
        self.self_attn = nn.MultiheadAttention(input_dim, n_heads)
        # Initialize Dense layer
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        # Self-Attention layer
        attn_output, _ = self.self_attn(x, x, x)
        # Dropout and Residual connection
        x = self.norm1(x + self.dropout1(attn_output))
        # Dense layer
        fc_output = self.fc(x)
        # Dropout and Residual connection
        x = self.norm2(x + self.dropout2(fc_output))
        return x


class LSTM_With_Attention(nn.Module):
    """LSTM Model Class with Attention"""
    def __init__(self, 
                 input_dim, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers, 
                 bidirectional, 
                 dropout,
                 n_heads):
        super(LSTM_With_Attention, self).__init__()
        # Bidirectional LSTM or not
        self.bidirectional = bidirectional
        # Initialize Embedding Layer
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        # Initialize LSTM layer to process the vector sequences 
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim,
                            num_layers = n_layers,
                            bidirectional = bidirectional,
                            dropout = dropout,
                            batch_first = True)
        num_directions = 2 if self.bidirectional else 1
        # Initialize Encoder Block
        self.encoder = EncoderBlock(hidden_dim * num_directions, hidden_dim, n_heads, dropout)
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
                                                                  batch_first = True)
        # Apply Encoder Block
        output = self.encoder(output)
        # Concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers and Apply Dropout
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)) if self.lstm.bidirectional else self.dropout(hidden[-1,:,:])
        # Fully Connected Layer
        output = self.fc(hidden)
        return output.squeeze(1)
