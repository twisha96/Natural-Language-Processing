import decoder
import encoder
import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, dropout,
                 is_conditional=False, bidirectional_encoder=False,
                 use_bridge=False, use_attention=False, logger=None):
        super(Seq2Seq, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.is_conditional = is_conditional
        self.bidirectional = bidirectional_encoder
        self.use_bridge = use_bridge
        self.use_attention = use_attention
        self.logger = logger

        self.wemb = nn.Embedding(self.vocab_size, dim, padding_idx=0)
        self.enc = None
        if self.is_conditional:
            self.enc = encoder.Encoder(self.wemb, num_layers, dropout,
                                       bidirectional=bidirectional_encoder,
                                       use_bridge=use_bridge)
        self.dec = decoder.Decoder(self.wemb, num_layers, dropout,
                                   use_attention=use_attention,
                                   bidirectional_encoder=bidirectional_encoder)
        self.linear = nn.Linear(self.dim, self.vocab_size)
        self.linear.weight = self.wemb.weight  # Parameter tying
        self.init_weights()

        num_params = sum(par.data.nelement() for par in self.parameters())
        if self.logger:
            self.logger.log('Seq2Seq')
            self.logger.log('      # parameters: %d' % num_params)
            self.logger.log('        vocab_size: %d' % self.vocab_size)
            self.logger.log('               dim: %d' % self.dim)
            self.logger.log('          # layers: %d' % self.num_layers)
            self.logger.log('    is_conditional: %d' % self.is_conditional)
            self.logger.log('     bidirectional: %d' % self.bidirectional)
            self.logger.log('        use_bridge: %d' % self.use_bridge)
            self.logger.log('     use_attention: %d' % self.use_attention)
            self.logger.log('')

    def forward(self, subblock, src=None, lengths=None, start=True):
        batch_size = subblock.size(1)
        memory_bank = None
        final = None
        if self.is_conditional and isinstance(src, torch.Tensor):
            memory_bank, final = self.enc(src, lengths)

        if start:
            self.dec.init_state(batch_size=batch_size, encoder_final=final)
        else:
            self.dec.detach_state()

        output, attns = self.dec(subblock, memory_bank=memory_bank,
                                 memory_lengths=lengths)

        decoded = self.linear(output.view(-1, output.size(2)))
        return decoded, attns

    def init_weights(self):
        self.wemb.weight.data.uniform_(-0.1, 0.1)
        self.wemb.weight.data[0, :].zero_()

        # Linear weight is tied with embedding so all set. Just set bias.
        self.linear.bias.data.zero_()
