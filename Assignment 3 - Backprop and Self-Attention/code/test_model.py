import model
import torch
import torch.nn as nn
import unittest


class TestModel(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 200
        self.dim = 30
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

    def test_monokuma_no_cond(self):
        mono = model.Monokuma(self.vocab_size, self.dim, self.num_layers,
                              self.dropout, is_conditional=False,
                              bidirectional_encoder=False,
                              use_bridge=False, use_attention=False)
        avgCE = nn.CrossEntropyLoss(ignore_index=0)
        mono.train()

        subblock = self.tgt[0:2,:]
        golds = self.tgt[1:3,:].view(-1)
        mono.zero_grad()
        output, attns = mono(subblock, src=None, lengths=None, start=True)
        self.assertEqual(attns, None)
        loss = avgCE(output, golds)
        loss.backward()

        nn.utils.clip_grad_norm_(mono.parameters(), 0.25)
        for p in mono.parameters():
            p.data.add_(-10, p.grad.data)

        subblock = self.tgt[2:3,:]
        golds = self.tgt[3:,:].view(-1)
        output, attns = mono(subblock, src=None, lengths=None, start=False)
        self.assertEqual(attns, None)
        loss = avgCE(output, golds)
        loss.backward()

        nn.utils.clip_grad_norm_(mono.parameters(), 0.25)
        for p in mono.parameters():
            p.data.add_(-10, p.grad.data)

    def test_monokuma(self):
        mono = model.Monokuma(self.vocab_size, self.dim, self.num_layers,
                              self.dropout, is_conditional=True,
                              bidirectional_encoder=False,
                              use_bridge=False, use_attention=False)
        avgCE = nn.CrossEntropyLoss(ignore_index=0)
        mono.train()

        subblock = self.tgt[0:2,:]
        golds = self.tgt[1:3,:].view(-1)
        mono.zero_grad()
        output, attns = mono(subblock, src=self.src, lengths=self.lengths,
                             start=True)
        self.assertEqual(attns, None)
        loss = avgCE(output, golds)
        loss.backward()

        nn.utils.clip_grad_norm_(mono.parameters(), 0.25)
        for p in mono.parameters():
            p.data.add_(-10, p.grad.data)

        subblock = self.tgt[2:3,:]
        golds = self.tgt[3:,:].view(-1)
        output, attns = mono(subblock, src=self.src, lengths=self.lengths,
                             start=False)
        self.assertEqual(attns, None)
        loss = avgCE(output, golds)
        loss.backward()

        nn.utils.clip_grad_norm_(mono.parameters(), 0.25)
        for p in mono.parameters():
            p.data.add_(-10, p.grad.data)

    def test_monokuma_attn(self):
        mono = model.Monokuma(self.vocab_size, self.dim, self.num_layers,
                              self.dropout, is_conditional=True,
                              bidirectional_encoder=False,
                              use_bridge=False, use_attention=True)
        avgCE = nn.CrossEntropyLoss(ignore_index=0)
        mono.train()

        #  src
        # 1 6 9
        # 2 7 10
        # 3 8 0
        # 4 0 0
        # 5 0 0
        #
        #
        #  subblock
        # 9   3   1                golds
        # 8   2   100            8   2   100
        #                        7   100 0
        subblock = self.tgt[0:2,:]
        golds = self.tgt[1:3,:].view(-1)
        mono.zero_grad()
        output, attns = mono(subblock, src=self.src, lengths=self.lengths,
                             start=True)
        loss = avgCE(output, golds)
        loss.backward()

        nn.utils.clip_grad_norm_(mono.parameters(), 0.25)
        for p in mono.parameters():
            p.data.add_(-10, p.grad.data)

        #  src
        # 1 6 9
        # 2 7 10
        # 3 8 0
        # 4 0 0
        # 5 0 0
        #
        #
        #  subblock
        # 7   100 0                golds
        #                      # 100 0   0
        subblock = self.tgt[2:3,:]
        golds = self.tgt[3:,:].view(-1)
        output, attns = mono(subblock, src=self.src, lengths=self.lengths,
                             start=False)
        loss = avgCE(output, golds)
        loss.backward()

        nn.utils.clip_grad_norm_(mono.parameters(), 0.25)
        for p in mono.parameters():
            p.data.add_(-10, p.grad.data)


if __name__ == '__main__':
    unittest.main()
