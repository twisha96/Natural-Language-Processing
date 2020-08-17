import attention
import stacked_rnn
import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, embeddings, num_layers, dropout, use_attention=False,
                 bidirectional_encoder=False):
        super(Decoder, self).__init__()
        self.embeddings = embeddings
        self.dim = embeddings.embedding_dim
        self.input_dim = embeddings.embedding_dim
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout)
        self.use_attention = use_attention
        self.bidirectional_encoder = bidirectional_encoder

        # Decoder state
        self.state = {}

        if use_attention:
            self.input_dim = 2 * self.dim
            self.attn = attention.GlobalAttention(self.dim)
            self.lstm = stacked_rnn.StackedLSTM(self.input_dim, self.dim,
                                                num_layers, dropout=dropout)
        else:
            self.lstm = nn.LSTM(self.input_dim, self.dim, num_layers,
                                dropout=dropout)

    def forward(self, rectangle_bptt, memory_bank=None, memory_lengths=None):
        if self.use_attention:
            assert isinstance(memory_bank, torch.Tensor)
            output, attns = self.run_attn(rectangle_bptt, memory_bank,
                                          memory_lengths)
        else:
            emb = self.embeddings(rectangle_bptt)
            emb = self.drop(emb)
            output, hidden = self.lstm(emb, self.state['hidden'])
            output = self.drop(output)
            self.update_state(hidden, None)
            attns = None

        return output, attns

    def run_attn(self, rectangle_bptt, memory_bank, memory_lengths):
        dec_outs = []
        attns = {"std": []}

        emb = self.embeddings(rectangle_bptt)  # T' x B x d
        emb = self.drop(emb)  # Embedding-level dropout added
        dec_state = self.state['hidden']
        input_feed = self.state['input_feed'].squeeze(0)  # B x d

        for _, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)  # B x d
            decoder_input = torch.cat([emb_t, input_feed], 1)  # B x 2d

            # (1 x B x d) (l x B x d)^2
            rnn_output, dec_state = self.lstm(decoder_input, dec_state)

            # (B x d)       (B x T)
            decoder_output, p_attn = self.attn(rnn_output,
                                               memory_bank.transpose(0, 1),
                                               memory_lengths=memory_lengths)
            decoder_output = self.drop(decoder_output)
            input_feed = decoder_output  # for next position
            dec_outs += [decoder_output]
            attns["std"] += [p_attn]

        output = dec_outs[-1]

        #               (L x B x d)^2       (1 x B x d)
        self.update_state(dec_state, output.unsqueeze(0))

        dec_outs = torch.stack(dec_outs)  # T' x B x d
        return dec_outs, attns

    def detach_state(self): # TODO: check
        def repackage(h):
            if isinstance(h, torch.Tensor):
                return h.detach()
            else:
                if h:
                    return tuple(repackage(v) for v in h)
                else:  # input_feed could be None
                    return None

        self.state['hidden'] = repackage(self.state['hidden'])
        self.state['input_feed'] = repackage(self.state['input_feed'])

    def update_state(self, state, input_feed):
        self.state['hidden'] = state
        self.state['input_feed'] = input_feed

    def init_state(self, batch_size=None, encoder_final=None):
        if encoder_final:
            def _fix_enc_hidden(hidden):
                # The encoder hidden is  (layers*directions) x batch x dim.
                # We need to convert it to layers x batch x (directions*dim).
                if self.bidirectional_encoder:
                    hidden = torch.cat([hidden[0:hidden.size(0):2],
                                        hidden[1:hidden.size(0):2]], 2)
                return hidden

            self.state['hidden'] = tuple([_fix_enc_hidden(enc_hid)
                                          for enc_hid in encoder_final])

            # Init the input feed.
            batch_size = self.state['hidden'][0].size(1)
            h_size = (batch_size, self.dim)
            self.state['input_feed'] = \
                self.state['hidden'][0].data.new(*h_size).zero_().unsqueeze(0)

        else:
            assert batch_size

            # Following PT ex.: get whatever Parameter object to create hidden
            weight = next(self.parameters())
            self.state['hidden'] = tuple([weight.new_zeros(self.num_layers,
                                                           batch_size, self.dim)
                                          for _ in range(2)])
            self.state['input_feed'] = None
