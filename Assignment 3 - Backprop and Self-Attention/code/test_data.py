import data
import os
import pathlib
import shutil
import torch
import torch.nn as nn
import unittest


def print_block(block, i2w):
    for row in range(block.size(0)):
        print(' '.join(['{:15s}'.format(i2w[block[row][col]])
                        for col in range(block.size(1))]))

def verbalize_col(block, col, i2w):
    return ' '.join([i2w[block[row][col]] for row in range(block.size(0))])


def verbalize_golds(golds, i2w):
    return ' '.join([i2w[golds[i]] for i in range(golds.size(0))])


class TestData(unittest.TestCase):

    def setUp(self):
        """
                  a a               a b c
                  b b b             d e f g
                  z z z z           h i j
                  d                 k l m n o
        """
        self.dpath = '_testdir'
        self.text = 'a b c\n' + 'd e f g\n' + 'h i j\n' + 'k l m n o\n'
        self.src_text = 'a a\n' + 'b b b\n' + 'z z z z\n' + 'd\n'
        self.device = 'cpu'

        if os.path.exists(self.dpath):
            shutil.rmtree(self.dpath)
        os.mkdir(self.dpath)

        with open(os.path.join(self.dpath, 'train.txt'), 'w') as f:
            f.write(self.text)
        with open(os.path.join(self.dpath, 'src-train.txt'), 'w') as f:
            f.write(self.src_text)

        # Don't need valid/test data but the files need to be there.
        pathlib.Path(os.path.join(self.dpath, 'valid.txt')).touch()
        pathlib.Path(os.path.join(self.dpath, 'src-valid.txt')).touch()
        pathlib.Path(os.path.join(self.dpath, 'test.txt')).touch()
        pathlib.Path(os.path.join(self.dpath, 'src-test.txt')).touch()

    def tearDown(self):
        shutil.rmtree(self.dpath)

    def test_itr(self):
        dat = data.Data(self.dpath, 2, 'continuous', self.device)
        for batch_num, (subblock, golds) in enumerate(
                data.build_itr(dat.train, 3)):
            self.assertLess(batch_num, 3)

            if batch_num == 0:
                self.assertEqual(verbalize_col(subblock, 0, dat.i2w), 'a b c')
                self.assertEqual(verbalize_col(subblock, 1, dat.i2w), 'h i j')
                self.assertEqual(verbalize_golds(golds, dat.i2w),
                                 'b i c j %s %s' % (dat.EOS, dat.EOS))
            if batch_num == 1:
                self.assertEqual(verbalize_col(subblock, 0, dat.i2w),
                                 '%s d e' % (dat.EOS))
                self.assertEqual(verbalize_col(subblock, 1, dat.i2w),
                                 '%s k l' % (dat.EOS))
                self.assertEqual(verbalize_golds(golds, dat.i2w),
                                 'd k e l f m')
            if batch_num == 2:
                self.assertEqual(verbalize_col(subblock, 0, dat.i2w), 'f g')
                self.assertEqual(verbalize_col(subblock, 1, dat.i2w), 'm n')
                self.assertEqual(verbalize_golds(golds, dat.i2w),
                                 'g n %s o' % (dat.EOS))

    def test_continuous_batching(self):
        dat = data.Data(self.dpath, 2, 'continuous', self.device)

        self.assertEqual(list(dat.train.size()), [9, 2])

        self.assertEqual(verbalize_col(dat.train, 0, dat.i2w),
                         'a b c %s d e f g %s' % (dat.EOS, dat.EOS))
        self.assertEqual(verbalize_col(dat.train, 1, dat.i2w),
                         'h i j %s k l m n o' % (dat.EOS))

    def test_translation_batching_sort(self):
        dat = data.Data(self.dpath, 2, 'translation', self.device, sort=True,
                        is_conditional=True)

        tgt, src, src_lens = dat.train[0]

        """
                                     <sos>  <sos>
                      d     b            k      d
                  <pad>     b            l      e
                  <pad>     b            f      m
                                         g      n
                                     <eos>      o
                                     <pad>  <eos>

              But the columns switch order.
        """
        self.assertEqual(list(tgt.size()), [7, 2])
        self.assertEqual(list(src.size()), [3, 2])
        self.assertEqual(verbalize_col(tgt, 0, dat.i2w),
                         '%s d e f g %s %s' % (dat.SOS, dat.EOS, dat.PAD))
        self.assertEqual(verbalize_col(tgt, 1, dat.i2w),
                         '%s k l m n o %s' % (dat.SOS, dat.EOS))
        self.assertEqual(verbalize_col(src, 0, dat.i2w),
                         'b b b')
        self.assertEqual(verbalize_col(src, 1, dat.i2w),
                         'd %s %s' % (dat.PAD, dat.PAD))
        self.assertEqual(src_lens.tolist(), [3, 1])


        tgt, src, src_lens = dat.train[1]
        """

                  z z z z           h i j
                  a a               a b c

                                  <sos>     <sos>
                  a      z            a         h
                  a      z            b         i
              <pad>      z            c         j
              <pad>      z        <eos>     <eos>

              But the columns switch order.
        """

        self.assertEqual(list(tgt.size()), [5, 2])
        self.assertEqual(list(src.size()), [4, 2])
        self.assertEqual(verbalize_col(tgt, 0, dat.i2w),
                         '%s h i j %s' % (dat.SOS, dat.EOS))
        self.assertEqual(verbalize_col(tgt, 1, dat.i2w),
                         '%s a b c %s' % (dat.SOS, dat.EOS))
        self.assertEqual(verbalize_col(src, 0, dat.i2w),
                         'z z z z')
        self.assertEqual(verbalize_col(src, 1, dat.i2w),
                         'a a %s %s' % (dat.PAD, dat.PAD))
        self.assertEqual(src_lens.tolist(), [4, 2])

    def test_translation_batching(self):
        dat = data.Data(self.dpath, 3, 'translation', self.device, sort=False,
                        is_conditional=True)

        tgt, src, src_lens = dat.train[0]
        """
                  a a               a b c
                  b b b             d e f g
                  z z z z           h i j
        """
        self.assertEqual(list(tgt.size()), [6, 3])
        self.assertEqual(list(src.size()), [4, 3])
        self.assertEqual(verbalize_col(tgt, 0, dat.i2w),
                         '%s h i j %s %s' % (dat.SOS, dat.EOS, dat.PAD))
        self.assertEqual(verbalize_col(tgt, 1, dat.i2w),
                         '%s d e f g %s' % (dat.SOS, dat.EOS))
        self.assertEqual(verbalize_col(tgt, 2, dat.i2w),
                         '%s a b c %s %s' % (dat.SOS, dat.EOS, dat.PAD))
        self.assertEqual(verbalize_col(src, 0, dat.i2w),
                         'z z z z')
        self.assertEqual(verbalize_col(src, 1, dat.i2w),
                         'b b b %s' % (dat.PAD))
        self.assertEqual(verbalize_col(src, 2, dat.i2w),
                         'a a %s %s' % (dat.PAD, dat.PAD))
        self.assertEqual(src_lens.tolist(), [4, 3, 2])

        tgt, src, src_lens = dat.train[1]
        """
                  d                 k l m n o
        """
        self.assertEqual(list(tgt.size()), [7, 1])
        self.assertEqual(list(src.size()), [1, 1])
        self.assertEqual(verbalize_col(tgt, 0, dat.i2w),
                         '%s k l m n o %s' % (dat.SOS, dat.EOS))
        self.assertEqual(verbalize_col(src, 0, dat.i2w),
                         'd')
        self.assertEqual(src_lens.tolist(), [1])



if __name__ == '__main__':
    unittest.main()
