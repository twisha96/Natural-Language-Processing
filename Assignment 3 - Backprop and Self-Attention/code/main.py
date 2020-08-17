import argparse
import control
import data
import datetime
import mylogger
import math
import model
import os
import random
import sys
import time
import torch
import torch.nn as nn
import pdb

def make_model_path(args):
    if not os.path.exists('scratch'):
        os.makedirs('scratch')
    model_path = 'scratch/'
    model_path += '%s_' % os.path.basename(os.path.normpath(args.data))
    model_path += 'BS%d_' % args.batch_size
    model_path += 'BM:%s_' % args.batch_method
    model_path += 'bptt%d_' % args.bptt
    model_path += 'dim%d_' % args.dim
    model_path += 'nlayers%d_' % args.nlayers
    model_path += 'dropout%.2f_' % args.dropout
    if args.cond:
        model_path += 'cond_'
    if args.bidir:
        model_path += 'bidir_'
    if args.bridge:
        model_path += 'bridge_'
    if args.attn:
        model_path += 'attn_'
    if args.sort:
        model_path += 'sort_'
    if args.shuffle:
        model_path += 'shuffle_'
    model_path += 'lr%.2f_' % args.lr
    model_path += 'seed%d_' % args.seed
    model_path += 'epochs%d' % args.epochs
    return model_path


def main(args):
    main_start_time = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.cuda else 'cpu')

    model_path = args.model if args.model else make_model_path(args)
    logger = mylogger.Logger(model_path + '.log', args.train)
    logger.log(' '.join(sys.argv) + '\n')

    if args.batch_size_valid > 0:
        batch_size_valid = args.batch_size_valid
    else:
        batch_size_valid = 1 if args.batch_method == 'continuous' else 60

    dat = data.Data(args.data, args.batch_size, args.batch_method, device,
                    sort=args.sort, logger=logger, is_conditional=args.cond,
                    batch_size_valid=batch_size_valid)

    s2s = model.Seq2Seq(len(dat.i2w), args.dim, args.nlayers,
                        args.dropout, is_conditional=args.cond,
                        bidirectional_encoder=args.bidir,
                        use_bridge=args.bridge, use_attention=args.attn,
                        logger=logger).to(device)

    ctrl = control.Control(s2s, args.lr, args.bptt, args.interval,
                           model_path=model_path, logger=logger)

    if args.train:
        ctrl.train(dat, args.epochs, args.shuffle)
        logger.log(time.strftime("%H:%M:%S", time.gmtime(time.time()
                                                         - main_start_time)))
    else:
        ctrl.load_s2s()
        train_loss, train_sqxent = ctrl.evaluate(dat.train)
        valid_loss, valid_sqxent = ctrl.evaluate(dat.valid)
        print('train ppl: %.2f       train sqxent: %.2f' %
              (math.exp(train_loss), train_sqxent))
        print('valid ppl: %.2f       valid sqxent: %.2f' %
              (math.exp(valid_loss), valid_sqxent))

    # pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='',
                        help='model path [%(default)s]')
    parser.add_argument('--data', type=str, default='./data',
                        help='data directory [%(default)s]')
    parser.add_argument('--train', action='store_true',
                        help='train?')
    parser.add_argument('--batch_size', type=int, default=20, metavar='BS',
                        help='batch size [%(default)d]')
    parser.add_argument('--batch_size_valid', type=int, default=0,
                        metavar='BSV',
                        help='validation batch size (0 if auto) [%(default)d]')
    parser.add_argument('--batch_size_test', type=int, default=0,
                        metavar='BST',
                        help='test batch size (0 if auto) [%(default)d]')
    parser.add_argument('--batch_method', type=str, default='continuous',
                        metavar='BM',
                        help='batch method (continuous, translation) '
                        '[%(default)s]')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length [%(default)d]')
    parser.add_argument('--dim', type=int, default=100,
                        help='dimension of input/hidden states [%(default)d]')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of LSTM layers [%(default)d]')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout) '
                        '[%(default)f]')
    parser.add_argument('--cond', action='store_true',
                        help='conditional language model?')
    parser.add_argument('--bidir', action='store_true',
                        help='bidirectional encoder?')
    parser.add_argument('--bridge', action='store_true',
                        help='use bridge?')
    parser.add_argument('--attn', action='store_true',
                        help='use attention?')
    parser.add_argument('--sort', action='store_true',
                        help='sort by target lengths before batching? '
                        '(only for translation data)')
    parser.add_argument('--shuffle', action='store_true',
                        help='shuffle bundles? (only for translation data)')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate [%(default)f]')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed [%(default)d]')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit [%(default)d]')
    parser.add_argument('--interval', type=int, default=20,
                        help='logging interval [%(default)d]')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA?')

    args = parser.parse_args()
    main(args)
