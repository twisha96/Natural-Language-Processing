import encoder
import decoder
import torch
import torch.nn as nn
import unittest


class TestDecoder(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 200
        self.dim = 30
        self.wemb = nn.Embedding(self.vocab_size, self.dim, padding_idx=0)
        self.num_layers = 2
        self.dropout = 0

        # 1 6 9
        # 2 7 10
        # 3 8 0
        # 4 0 0
        # 5 0 0
        self.batch_size = 3
        self.T_src = 5
        self.src = torch.zeros((self.T_src, self.batch_size)).long()
        for row in range(5): self.src[row][0] = row + 1
        for row in range(3): self.src[row][1] = row + 6
        for row in range(2): self.src[row][2] = row + 9
        self.lengths = torch.IntTensor([5, 3, 2])

        # 9   3   1
        # 8   2   100
        # 7   100 0
        # 100 0   0
        self.T_tgt = 4
        self.tgt = torch.zeros((self.T_tgt, self.batch_size)).long()
        self.tgt[0][0] = 9
        self.tgt[1][0] = 8
        self.tgt[2][0] = 7
        self.tgt[3][0] = 100
        self.tgt[0][1] = 3
        self.tgt[1][1] = 2
        self.tgt[2][1] = 100
        self.tgt[0][2] = 1
        self.tgt[1][2] = 100

    def test_attention_conditioning_bidirectional(self):
        enc = encoder.Encoder(self.wemb, self.num_layers, self.dropout,
                              bidirectional=True, use_bridge=True)
        dec = decoder.Decoder(self.wemb, self.num_layers, self.dropout,
                              use_attention=True,
                              bidirectional_encoder=True)

        memory_bank, final = enc.forward(self.src, self.lengths)

        dec.init_state(batch_size=None, encoder_final=final)

        dec_outs, attns = dec(self.tgt, memory_bank=memory_bank,
                              memory_lengths=self.lengths)

        self.assertEqual(list(dec_outs.size()), [self.T_tgt, self.batch_size,
                                                 self.dim])

    def test_attention_conditioning(self):
        enc = encoder.Encoder(self.wemb, self.num_layers, self.dropout,
                              bidirectional=False, use_bridge=False)
        dec = decoder.Decoder(self.wemb, self.num_layers, self.dropout,
                              use_attention=True,
                              bidirectional_encoder=False)

        memory_bank, final = enc.forward(self.src, self.lengths)

        dec.init_state(batch_size=None, encoder_final=final)

        dec_outs, attns = dec(self.tgt, memory_bank=memory_bank,
                              memory_lengths=self.lengths)

        self.assertEqual(list(dec_outs.size()), [self.T_tgt, self.batch_size,
                                                 self.dim])

    def test_no_conditioning(self):
        dec = decoder.Decoder(self.wemb, self.num_layers, self.dropout,
                              use_attention=False,
                              bidirectional_encoder=False)

        dec.init_state(batch_size=self.batch_size, encoder_final=None)

        dec_outs, attns = dec(self.tgt, memory_bank=None, memory_lengths=None)

        self.assertEqual(list(dec_outs.size()), [self.T_tgt, self.batch_size,
                                                 self.dim])

    def test_simple_conditioning(self):
        enc = encoder.Encoder(self.wemb, self.num_layers, self.dropout,
                              bidirectional=False, use_bridge=False)
        dec = decoder.Decoder(self.wemb, self.num_layers, self.dropout,
                              use_attention=False,
                              bidirectional_encoder=False)

        memory_bank, final = enc.forward(self.src, self.lengths)

        dec.init_state(batch_size=None, encoder_final=final)

        dec_outs, attns = dec(self.tgt, memory_bank=None, memory_lengths=None)

        self.assertEqual(list(dec_outs.size()), [self.T_tgt, self.batch_size,
                                                 self.dim])


if __name__ == '__main__':
    unittest.main()
