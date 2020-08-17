import control
import data
import model
import os
import shutil
import torch
import torch.nn as nn
import unittest


class TestControl(unittest.TestCase):

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

        with open(os.path.join(self.dpath, 'valid.txt'), 'w') as f:
            f.write(self.text)
        with open(os.path.join(self.dpath, 'src-valid.txt'), 'w') as f:
            f.write(self.src_text)

        with open(os.path.join(self.dpath, 'test.txt'), 'w') as f:
            f.write(self.text)
        with open(os.path.join(self.dpath, 'src-test.txt'), 'w') as f:
            f.write(self.src_text)

        self.dat_cont = data.Data(self.dpath, 2, 'continuous', self.device,
                                  is_conditional=False)
        self.dat_tran = data.Data(self.dpath, 3, 'translation', self.device,
                                  is_conditional=True, sort=True)

        self.dim = 10
        self.num_layers = 2
        self.dropout = 0
        self.lr = 20
        self.bptt = 2
        self.interval = 1

    def tearDown(self):
        shutil.rmtree(self.dpath)

    def test_epoch_translation_data_attn_bidir_bridge(self):
        mono = model.Monokuma(len(self.dat_tran.i2w), self.dim, self.num_layers,
                              self.dropout, is_conditional=True,
                              bidirectional_encoder=True,
                              use_bridge=True, use_attention=True)
        junko = control.Junko(mono, self.lr, self.bptt, self.interval)
        val_loss, _, epoch_time = junko.epoch_translation_data(self.dat_tran, 0,
                                                               shuffle=False)

    def test_epoch_translation_data_attn(self):
        mono = model.Monokuma(len(self.dat_tran.i2w), self.dim, self.num_layers,
                              self.dropout, is_conditional=True,
                              bidirectional_encoder=False,
                              use_bridge=False, use_attention=True)
        junko = control.Junko(mono, self.lr, self.bptt, self.interval)
        val_loss, _, epoch_time = junko.epoch_translation_data(self.dat_tran, 0,
                                                               shuffle=False)
    def test_epoch_translation_data_simple(self):
        mono = model.Monokuma(len(self.dat_tran.i2w), self.dim, self.num_layers,
                              self.dropout, is_conditional=True,
                              bidirectional_encoder=False,
                              use_bridge=False, use_attention=False)
        junko = control.Junko(mono, self.lr, self.bptt, self.interval)
        val_loss, _, epoch_time = junko.epoch_translation_data(self.dat_tran, 0,
                                                               shuffle=False)

    def test_epoch_continuous_data(self):
        mono = model.Monokuma(len(self.dat_cont.i2w), self.dim, self.num_layers,
                              self.dropout, is_conditional=False,
                              bidirectional_encoder=False,
                              use_bridge=False, use_attention=False)
        junko = control.Junko(mono, self.lr, self.bptt, self.interval)
        val_loss, _, epoch_time = junko.epoch_continuous_data(self.dat_cont, 0)

    def test_epoch_translation_data_dummy(self):
        mono = model.Monokuma(len(self.dat_tran.i2w), self.dim, self.num_layers,
                              self.dropout, is_conditional=False,
                              bidirectional_encoder=False,
                              use_bridge=False, use_attention=False)
        junko = control.Junko(mono, self.lr, self.bptt, self.interval)
        val_loss, _, epoch_time = junko.epoch_translation_data(self.dat_tran, 0,
                                                               shuffle=False)


if __name__ == '__main__':
    unittest.main()
