import nltk
import unittest

from bleu import compute_bleu


class TestBLEU(unittest.TestCase):

    def test_compute_bleu_small(self):
        h1 = ('It is a guide to action which ensures that the military always '
              'obeys the commands of the party').split()
        r1a = ('It is a guide to action that ensures that the military will '
               'forever heed Party commands').split()
        r1b = ('It is the guiding principle which guarantees the military '
               'forces always being under the command of the Party').split()
        r1c = ('It is the practical guide for the army always to heed the '
               'directions of the party').split()

        h2 = ('he read the book because he was interested in world '
              'history').split()
        r2a = ('he was interested in world history because he read the '
               'book').split()

        hyps = [h1, h2]
        reflists = [[r1a, r1b, r1c], [r2a]]

        bleu = compute_bleu(reflists, hyps)
        gold = nltk.translate.bleu_score.corpus_bleu(reflists, hyps)
        self.assertAlmostEqual(bleu, gold, places=10)

    def test_compute_bleu_large(self):
        paranmt_path = 'data/paranmt.txt'
        reflists = []
        hyps = []
        with open(paranmt_path) as f:
            for line in f:
                p1, p2 = line.split('\t')
                reflists.append([p1.split()])
                hyps.append(p2.split())

        bleu = compute_bleu(reflists, hyps)
        gold = nltk.translate.bleu_score.corpus_bleu(reflists, hyps)
        self.assertAlmostEqual(bleu, gold, places=10)


if __name__ == '__main__':
    unittest.main()
