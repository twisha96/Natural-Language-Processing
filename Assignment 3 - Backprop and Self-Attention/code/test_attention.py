import attention
import torch
import torch.nn as nn
import unittest


class TestGlobalAttention(unittest.TestCase):

    def setUp(self):
        self.attn = attention.GlobalAttention(1)
        self.attn.linear_in.weight = nn.Parameter(torch.FloatTensor([[1]]))
        self.attn.linear_out.weight = nn.Parameter(torch.FloatTensor([[1, 0]]))

        self.mem = torch.FloatTensor([[0.3], [0.8], [1], [7]]).unsqueeze(0)
        self.lengths = torch.IntTensor([3])

    def test_global_attention_one_step(self):
        query = torch.FloatTensor([[0.5]])
        attn_h, align_vectors = self.attn(query, self.mem, self.lengths)

        self.assertAlmostEqual(attn_h[0][0].item(), 0.6301, places=4)

        self.assertAlmostEqual(align_vectors[0][0].item(), 0.2700, places=4)
        self.assertAlmostEqual(align_vectors[0][1].item(), 0.3467, places=4)
        self.assertAlmostEqual(align_vectors[0][2].item(), 0.3832, places=4)
        self.assertAlmostEqual(align_vectors[0][3].item(), 0.0000, places=4)

    def test_global_attention(self):
        queries = torch.FloatTensor([[0.5], [5]]).unsqueeze(0)
        attn_h, align_vectors = self.attn(queries, self.mem, self.lengths)

        self.assertAlmostEqual(attn_h[0][0][0].item(), 0.6301, places=4)
        self.assertAlmostEqual(attn_h[1][0][0].item(), 0.7316, places=4)

        self.assertAlmostEqual(align_vectors[0][0][0].item(), 0.2700, places=4)
        self.assertAlmostEqual(align_vectors[0][0][1].item(), 0.3467, places=4)
        self.assertAlmostEqual(align_vectors[0][0][2].item(), 0.3832, places=4)
        self.assertAlmostEqual(align_vectors[0][0][3].item(), 0.0000, places=4)

        self.assertAlmostEqual(align_vectors[1][0][0].item(), 0.0216, places=4)
        self.assertAlmostEqual(align_vectors[1][0][1].item(), 0.2631, places=4)
        self.assertAlmostEqual(align_vectors[1][0][2].item(), 0.7153, places=4)
        self.assertAlmostEqual(align_vectors[1][0][3].item(), 0.0000, places=4)


if __name__ == '__main__':
    unittest.main()
