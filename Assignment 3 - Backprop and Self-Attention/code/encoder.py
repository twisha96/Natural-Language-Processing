import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class Encoder(nn.Module):

    def __init__(self, embeddings, num_layers, dropout, bidirectional,
                 use_bridge):
        super(Encoder, self).__init__()
        dim = embeddings.embedding_dim
        num_directions = 2 if bidirectional else 1
        assert dim % num_directions == 0
        hidden_size = dim // num_directions

        self.embeddings = embeddings
        self.lstm = nn.LSTM(dim, hidden_size, num_layers,
                            dropout=dropout, bidirectional=bidirectional)

        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(hidden_size, num_layers)

    def forward(self, src, lengths):
        packed_emb = pack(self.embeddings(src), lengths)
        memory_bank, encoder_final = self.lstm(packed_emb)
        memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)

        #       T x B x d     L x B x d (2L x B x d/2 if bidir)
        return memory_bank, encoder_final

    def _initialize_bridge(self, hidden_size, num_layers):
        self.total_hidden_dim = hidden_size * num_layers
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(2)])

    def _bridge(self, hidden):
        def bottle_hidden(linear, states):
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        outs = tuple([bottle_hidden(layer, hidden[ix])
                      for ix, layer in enumerate(self.bridge)])
        return outs
