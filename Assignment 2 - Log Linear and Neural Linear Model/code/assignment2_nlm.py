# Instructor: Karl Stratos
# Student: Twisha Naik

import math
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import util

from collections import OrderedDict


class FF(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output, num_layers,
                 activation='relu', dropout_rate=0, layer_norm=False,
                 residual_connection=False):
        super(FF, self).__init__()
        assert num_layers >= 0  # 0 = Linear
        assert (not residual_connection) or (dim_hidden == dim_input)
        self.residual_connection = residual_connection

        self.stack = nn.ModuleList()
        for l in range(num_layers):
            layer = []

            if layer_norm:
                layer.append(nn.LayerNorm(dim_input if l == 0 else dim_hidden))

            layer.append(nn.Linear(dim_input if l == 0 else dim_hidden,
                                   dim_hidden))
            layer.append({'tanh': nn.Tanh(), 'relu': nn.ReLU()}[activation])

            if dropout_rate > 0:
                layer.append(nn.Dropout(dropout_rate))

            self.stack.append(nn.Sequential(*layer))

        self.out = nn.Linear(dim_input if num_layers < 1 else dim_hidden,
                             dim_output)

    def forward(self, x):
        for layer in self.stack:
            x = x + layer(x) if self.residual_connection else layer(x)
        return self.out(x)


# https://github.com/pytorch/examples/blob/0c1654d6913f77f09c0505fb284d977d89c17c1a/word_language_model/main.py#L80
def batchify(toks, bsz):
    data = torch.tensor(toks)
    nbatch = len(toks) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).contiguous()
    return data


# Adapted from https://discuss.pytorch.org/t/pairwise-cosine-distance/30961/4.
def cosine_similarity(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def get_init_weights(init_value):
    def init_weights(m):
        if init_value > 0.0:
            if hasattr(m, 'weight') and hasattr(m.weight, 'uniform_'):
                nn.init.uniform_(m.weight, a=-init_value, b=init_value)
            if hasattr(m, 'bias') and hasattr(m.bias, 'uniform_'):
                nn.init.uniform_(m.bias, a=-init_value, b=init_value)

    return init_weights


class FFLM(nn.Module):

    def __init__(self, model, vocab, unk_symbol, init, lr, check_interval, seed,
                 nhis, wdim, hdim, nlayers, batch_size):
        super(FFLM, self).__init__()

        self.model = model
        self.token_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
        self.token_to_idx[unk_symbol] = 0
        self.i2x = [None for _ in range(len(self.token_to_idx))]
        for x in self.token_to_idx:
            self.i2x[self.token_to_idx[x]] = x
        self.V = len(self.token_to_idx)
        self.batch_size = batch_size
        self.nhis = nhis

        # Seed needed for reproducibility.
        random.seed(seed)
        torch.manual_seed(args.seed)

        self.E = nn.Embedding(self.V, wdim)

        # TODO: define feedfoward layer with correct dimensions.        
        # Parameters to be defined: dim_input, dim_hidden, dim_output, num_layers
        self.FF = FF(nhis*wdim, hdim, self.V, nlayers)

        self.apply(get_init_weights(init))
        self.lr = lr
        self.check_interval = check_interval
        self.best_val_ppl = float('inf')

        self.mean_ce = nn.CrossEntropyLoss(reduction='mean')
        self.sum_ce = nn.CrossEntropyLoss(reduction='sum')

        self.loss = []

    def forward(self, X, Y, mean=True):  # X (B x nhis), Y (B)
        # TODO: calculate logits (B x V) s.t.
        #       softmax(logits[i,:]) = distribution p(:|X[i]) under the model.
        word_embedding = self.E(X).view(self.batch_size, -1)
        logits = self.FF.forward(word_embedding)

        loss = self.mean_ce(logits, Y) if mean else self.sum_ce(logits, Y)
        return loss

    def train_epochs(self, train_toks, val_toks, epochs):
        T = batchify([self.token_to_idx[x] for x in train_toks],
                     self.batch_size)
        V = batchify([self.token_to_idx[x] for x in val_toks], self.batch_size)

        optim = torch.optim.Adam(self.parameters(), lr=self.lr)

        for ep in range(epochs):
            total_loss = 0
            batch_inds = list(range(self.nhis, T.size(1)))
            random.shuffle(batch_inds)
            for batch_num, i in enumerate(batch_inds):
                X = T[:, i - self.nhis: i]
                Y = T[:, i]
                loss = self.forward(X, Y)
                total_loss += loss.item() * T.size(0)
                loss.backward() # This computes all gradients.
                optim.step()  # This updates all weights.
                if (batch_num + 1) % self.check_interval == 0:
                    print('%d/%d batches, batch avg loss %g' %
                          (batch_num + 1, len(batch_inds), loss))

            train_ppl = math.exp(total_loss / len(batch_inds) / T.size(0))
            print("Training Loss: {:8.4f}".format(total_loss/len(batch_inds)/T.size(0)))
            self.loss.append(total_loss / len(batch_inds) / T.size(0))


            val_ppl = self.test(V)
            print('Epoch {:3d} | running train ppl {:8.4f} '
                  '| val ppl {:8.4f}'.format(ep + 1, train_ppl, val_ppl),
                  end='    ')

            if val_ppl < self.best_val_ppl:
                self.best_val_ppl = val_ppl
                print('***new best val ppl***')
                torch.save(self, self.model)
                num_bad_epochs = 0
            else:
                print()
                num_bad_epochs += 1

            if num_bad_epochs >= 7:
                break


    def test(self, V):
        self.eval()

        batch_inds = list(range(self.nhis, V.size(1)))
        loss = 0
        with torch.no_grad():
            for batch_num, i in enumerate(batch_inds):
                loss += self.forward(V[:, i - self.nhis: i], V[:, i],
                                     mean=False)
        ppl = math.exp(loss / len(batch_inds) / V.size(0))

        self.train()
        return ppl

    def nearest_neighbors(self, K):
        K = min(K, self.E.weight.size(0) - 1)
        D = cosine_similarity(self.E.weight, self.E.weight)
        scores, nn_inds = D.topk(K + 1, dim=1, largest=True)

        nns = {}
        for x in self.token_to_idx:
            nns[x] = [(self.i2x[i], scores[self.token_to_idx[x]][nind].item())
                      for nind, i in enumerate(nn_inds[self.token_to_idx[x]])]

        return nns


def main(args):
    tokenizer = util.Tokenizer(tokenize_type='nltk', lowercase=True)

    train_toks = tokenizer.tokenize(open(args.train_file).read())
    num_train_toks = int(args.train_fraction * len(train_toks))
    print('-' * 79)
    print('Using %d tokens for training (%g%% of %d)' %
          (num_train_toks, 100 * args.train_fraction, len(train_toks)))
    train_toks = train_toks[:int(args.train_fraction * len(train_toks))]
    val_toks = tokenizer.tokenize(open(args.val_file).read())
    num_val_toks = int(args.val_fraction * len(val_toks))
    print('Using %d tokens for validation (%g%% of %d)' %
          (num_val_toks, 100 * args.val_fraction, len(val_toks)))
    val_toks = val_toks[:int(args.val_fraction * len(val_toks))]

    train_ngram_counts = tokenizer.count_ngrams(train_toks)


    # Get vocab and threshold.
    print('Using vocab size %d (excluding UNK) (original %d)' %
          (min(args.vocab, len(train_ngram_counts[0])),
           len(train_ngram_counts[0])))
    vocab = [tup[0] for tup, _ in train_ngram_counts[0].most_common(args.vocab)]
    train_toks = tokenizer.threshold(train_toks, vocab, args.unk)
    val_toks = tokenizer.threshold(val_toks, vocab, args.unk)

    lm = FFLM(args.model, vocab, args.unk, args.init, args.lr,
                  args.check_interval, args.seed, args.nhis, args.wdim, args.hdim,
                  args.nlayers, args.B)
    print(lm)

    if not args.test:
        # Estimate parameters.
        lm.train_epochs(train_toks, val_toks, args.epochs)

    lm = torch.load(args.model)
    val_ppl = lm.test(batchify([lm.token_to_idx[x] for x in val_toks], args.B))
    print('Optimized Perplexity: %f' %(val_ppl))

    nns = lm.nearest_neighbors(args.K)
    for x in random.choices(list(nns.keys()), k=args.K):
        print('%s:' % x, end=' ')
        for z, _ in nns[x][1:]:
            print('%s' % z, end=' ')
        print()
    
    ############################################################################
    # Test the convergence of the linear and non-linear models
    # Epochs = 1000 --> To test the convergence and termination
    # wdim = hdim = 30
    # nlayers_list = [0, 1]
    # loss_values = []
    # for nlayers in nlayers_list:
    #     lm = FFLM(args.model, vocab, args.unk, args.init, 0.00003,
    #                   args.check_interval, args.seed, args.nhis, 30, 30,
    #                   nlayers, args.B)
    #     print(lm)

    #     if not args.test:
    #         # Estimate parameters.
    #         lm.train_epochs(train_toks, val_toks, args.epochs)

    #     lm = torch.load(args.model)
    #     val_ppl = lm.test(batchify([lm.token_to_idx[x] for x in val_toks], args.B))
    #     print('Optimized Perplexity: %f' %(val_ppl))
    #     loss_values.append(lm.loss)

    #     nns = lm.nearest_neighbors(args.K)
    #     for x in random.choices(list(nns.keys()), k=args.K):
    #         print('%s:' % x, end=' ')
    #         for z, _ in nns[x][1:]:
    #             print('%s' % z, end=' ')
    #         print()
    # print(loss_values)

    ############################################################################
    # # Varying learning rates
    # learning_rates = [0.00001,0.00003,0.0001,0.0003,0.001]
    # print("########################################")
    # for lr in learning_rates:
    #     print("Learning rate is: " + str(lr))

    #     # Varying Dimensions
    #     dimensions = [1, 5, 10, 100, 200]
    #     for dim in dimensions:
    #         print ("Dimension is: " + str(dim))
    #         lm = FFLM(args.model, vocab, args.unk, args.init, lr,
    #                   args.check_interval, args.seed, args.nhis, dim, dim,
    #                   1, args.B)    
    #         print(lm)

    #         if not args.test:
    #             # Estimate parameters.
    #             lm.train_epochs(train_toks, val_toks, args.epochs)

    #         lm = torch.load(args.model)
    #         val_ppl = lm.test(batchify([lm.token_to_idx[x] for x in val_toks], args.B))
    #         print('Optimized Perplexity: %f' %(val_ppl))

    #         nns = lm.nearest_neighbors(args.K)
    #         for x in random.choices(list(nns.keys()), k=args.K):
    #             print('%s:' % x, end=' ')
    #             for z, _ in nns[x][1:]:
    #                 print('%s' % z, end=' ')
    #             print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.pt',
                        help='model file [%(default)s]')
    parser.add_argument('--test', action='store_true',
                        help='do not train, load trained model?')
    parser.add_argument('--train_file', type=str,
                        default='data/gigaword_subset.val',
                        help='corpus for training [%(default)s]')
    parser.add_argument('--val_file', type=str,
                        default='data/gigaword_subset.val',
                        help='corpus for validation [%(default)s]')
    parser.add_argument('--init', type=float, default=0.0,
                        help='init range (default if 0) [%(default)g]')
    parser.add_argument('--lr', type=float, default=0.00003,
                        help='learning rate [%(default)g]')
    parser.add_argument('--vocab', type=int, default=1000,
                        help='max vocab size [%(default)d]')
    parser.add_argument('--nhis', type=int, default=3,
                        help='number of previous words to condition on '
                        '[%(default)d]')
    parser.add_argument('--wdim', type=int, default=30,
                        help='word embedding dimension [%(default)d]')
    parser.add_argument('--hdim', type=int, default=30,
                        help='hidden state dimension [%(default)d]')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers [%(default)d]')
    parser.add_argument('--B', type=int, default=16,
                        help='batch size [%(default)d]')
    parser.add_argument('--train_fraction', type=float, default=0.1,
                        help='use this fraction of training data [%(default)g]')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='use this fraction of val data [%(default)g]')
    parser.add_argument('--K', type=int, default=10,
                        help='K in top K [%(default)d]')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs [%(default)d]')
    parser.add_argument('--check_interval', type=int, default=2000,
                        metavar='CH',
                        help='number of updates for a check [%(default)d]')
    parser.add_argument('--unk', type=str, default='<?>',
                        help='unknown token symbol [%(default)s]')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [%(default)d]')
    args = parser.parse_args()
    main(args)
