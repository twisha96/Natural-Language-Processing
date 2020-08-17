import encoder
import torch
import torch.nn as nn
import unittest


class TestEncoder(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 50
        self.dim = 4
        self.wemb = nn.Embedding(self.vocab_size, self.dim, padding_idx=0)
        self.num_layers = 2
        self.dropout = 0

        # 1 6 9
        # 2 7 10
        # 3 8 0
        # 4 0 0
        # 5 0 0
        self.batch_size = 3
        self.T = 5
        self.src = torch.zeros((self.T, self.batch_size)).long()
        for row in range(5): self.src[row][0] = row + 1
        for row in range(3): self.src[row][1] = row + 6
        for row in range(2): self.src[row][2] = row + 9
        self.lengths = torch.IntTensor([5, 3, 2])

    def test_bidirectional(self):
        enc = encoder.Encoder(self.wemb, self.num_layers, self.dropout,
                              bidirectional=True, use_bridge=False)
        memory_bank, final = enc.forward(self.src, self.lengths)

        self.assertEqual(list(final[0].size()), [2 * self.num_layers,
                                                 self.batch_size,
                                                 self.dim // 2])
        self.assertEqual(list(memory_bank.size()),
                         [self.T, self.batch_size, self.dim])

    def test_bidirectional_bridge(self):
        enc = encoder.Encoder(self.wemb, self.num_layers, self.dropout,
                              bidirectional=True, use_bridge=True)
        memory_bank, final = enc.forward(self.src, self.lengths)

        self.assertEqual(list(final[0].size()), [2 * self.num_layers,
                                                 self.batch_size,
                                                 self.dim // 2])
        self.assertEqual(list(memory_bank.size()),
                         [self.T, self.batch_size, self.dim])

    def test_unidirectional(self):
        enc = encoder.Encoder(self.wemb, self.num_layers, self.dropout,
                              bidirectional=False, use_bridge=False)
        memory_bank, final = enc.forward(self.src, self.lengths)
        self.assertEqual(list(final[0].size()), [self.num_layers,
                                                 self.batch_size,
                                                 self.dim])
        self.assertEqual(list(memory_bank.size()),
                         [self.T, self.batch_size, self.dim])

    def test_unidirectional_bridge(self):
        enc = encoder.Encoder(self.wemb, self.num_layers, self.dropout,
                              bidirectional=False, use_bridge=True)
        memory_bank, final = enc.forward(self.src, self.lengths)
        self.assertEqual(list(final[0].size()), [self.num_layers,
                                                 self.batch_size,
                                                 self.dim])
        self.assertEqual(list(memory_bank.size()),
                         [self.T, self.batch_size, self.dim])


if __name__ == '__main__':
    unittest.main()
