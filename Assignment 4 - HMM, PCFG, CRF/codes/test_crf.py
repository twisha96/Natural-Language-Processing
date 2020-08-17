import crf
import itertools
import torch
import torch.nn as nn
import unittest


class TestCRF(unittest.TestCase):

    def setUp(self):
        self.Bs = [1, 2, 4]  # Batch sizes
        self.Ts = [1, 2, 4]  # Lengths
        self.Ls = [1, 2, 4]  # Label set sizes
        self.inits = [0.1, 1, 10]  # Init values

    def test_compute_normalizers(self):
        for B, T, L, init in itertools.product(self.Bs, self.Ts, self.Ls,
                                               self.inits):
            crfloss = crf.CRFLoss(L, init)
            scores = torch.randn(B, T, L)
            normalizers = crfloss.compute_normalizers(scores)
            normalizers_brute = crfloss.compute_normalizers_brute(scores)
            for i in range(B):
                self.assertAlmostEqual(normalizers[i].item(),
                                       normalizers_brute[i].item(),
                                       delta=1e-5)

    def test_decode(self):
        for B, T, L, init in itertools.product(self.Bs, self.Ts, self.Ls,
                                               self.inits):
            crfloss = crf.CRFLoss(L, init)
            scores = torch.randn(B, T, L)
            max_scores_brute, indices_brute = crfloss.decode_brute(scores)
            max_scores, indices = crfloss.decode(scores)
            for i in range(B):
                self.assertAlmostEqual(max_scores[i].item(),
                                       max_scores_brute[i].item(),
                                       delta=1e-5)
                for j in range(T):
                    self.assertEqual(indices[i, j], indices_brute[i, j])

    def test_score_targets(self):
        # B = 2, T = 3, L = 4
        crfloss = crf.CRFLoss(4, 0)
        crfloss.start \
            = nn.Parameter(torch.Tensor([0.7441, -1.1210, -0.0360, 2.2280]))
        crfloss.end \
            = nn.Parameter(torch.Tensor([-0.8736,  0.3495,  0.2857,  0.8824]))
        crfloss.T \
            = nn.Parameter(torch.Tensor([[ 0.3096,  0.3350,  0.9461,  1.1764],
                                         [-1.0077,  1.6397, -0.0194, -1.8316],
                                         [ 0.3731,  0.7105, -1.6131, -1.1109],
                                         [-0.4030, -1.3935, -0.3196,  0.2172]]))
        scores = torch.Tensor([[[-0.0243, -0.2344,  1.6577,  1.5445],
                                [-2.4474, -0.7471, -0.5287,  0.4362],
                                [ 0.5205, -0.4776,  1.0698, -0.0784]],
                               [[ 2.4130,  1.1780, -2.3695,  1.3947],
                                [-0.0052,  0.4720, -1.2306, -3.3132],
                                [ 0.2174, -0.8870,  0.0609,  0.5049]]])
        targets = torch.LongTensor([[2, 0, 1],
                                    [0, 3, 0]])

        target_scores = crfloss.score_targets(scores, targets)
        self.assertAlmostEqual(target_scores[0].item(), -1.0154, delta=0.000001)
        self.assertAlmostEqual(target_scores[1].item(), -0.0389, delta=0.000001)


if __name__ == '__main__':
    unittest.main()
