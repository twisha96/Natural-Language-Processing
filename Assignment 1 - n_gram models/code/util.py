import os
import matplotlib.pyplot as plt
import nltk
import numpy as np

from collections import Counter
from transformers import BertTokenizer, RobertaTokenizer


class BigramLanguageModel:

    def __init__(self, vocab, unk_symbol, smoothing, alpha=0.001, beta=0.7):
        self.token_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
        self.token_to_idx[unk_symbol] = 0
        self.n = len(self.token_to_idx)  # Num token types (including UNK)
        self.smoothing = smoothing
        self.alpha = alpha
        self.beta = beta
        self.bi_counts = None      # Holds the bigram counts
        self.bi_prob = None        # Holds the computed bigram probabilities.

    """Train a basic n-gram language model"""
    def train(self, corpus):
        self.bi_counts = np.zeros((self.n, self.n), dtype=float)

        # Convert to token indices.
        corpus_idx = []
        for w in corpus:
            assert w in self.token_to_idx
            corpus_idx.append(self.token_to_idx[w])

        # Gather counts
        for i, idx in enumerate(corpus_idx[:-1]):
            self.bi_counts[idx][corpus_idx[i + 1]] += 1

        # Pre-compute the probabilities.
        if not self.smoothing:
            self.compute_bigram_prob()
        elif self.smoothing == 'laplace':
            self.compute_bigram_prob_laplace(self.alpha)
        elif self.smoothing == 'interpolation':
            self.compute_bigram_prob_interpolation(self.beta, self.alpha)
        else:
            raise ValueError('Unknown smoothing type.')

    def compute_bigram_prob(self):
        self.bi_prob = self.bi_counts.copy()

        for i, _ in enumerate(self.bi_prob):
            cnt = np.sum(self.bi_prob[i])
            if cnt > 0:
                self.bi_prob[i] /= cnt

    def compute_bigram_prob_laplace(self, alpha):
        # TODO: Implement here
        self.bi_prob = self.bi_counts.copy()

        for i, _ in enumerate(self.bi_prob):
            cnt = np.sum(self.bi_prob[i])
            if cnt > 0:
                self.bi_prob[i] = (self.bi_prob[i]+alpha)/(cnt+alpha*self.n)

        # Do not remove the following normalization check.
        for i in range(self.bi_prob.shape[0]):
            assert abs(1 - np.sum(self.bi_prob[i])) < 1e-5


    def compute_bigram_prob_interpolation(self, beta, alpha):
        # TODO: Implement here
        self.compute_bigram_prob_laplace(alpha)
        unigram_counts = np.sum(self.bi_counts, axis=0)
        corpus_length = np.sum(self.bi_counts)

        for i, _ in enumerate(self.bi_prob):
            self.bi_prob[i] = beta*(self.bi_prob[i]) + (1-beta)*(unigram_counts+alpha)/(corpus_length+alpha*self.n)

        # Do not remove the following normalization check.
        for i in range(self.bi_prob.shape[0]):
            assert abs(1 - np.sum(self.bi_prob[i])) < 1e-5

    def test(self, corpus):
        logprob = 0.
        # Convert to token indices.

        corpus = [self.token_to_idx[w] for w in corpus]

        for i, idx in enumerate(corpus[:-1]):
            logprob += np.log(self.bi_prob[idx, corpus[i+1]])

        logprob /= len(corpus[:-1])

        # Compute perplexity
        ppl = np.exp(-logprob)

        return ppl


class Tokenizer:

    def __init__(self, tokenize_type='basic', lowercase=False):
        self.tokenize_type = tokenize_type
        self.lowercase = lowercase

        if self.tokenize_type == 'wp':
            self.wptok = BertTokenizer.from_pretrained('bert-base-cased')

        if self.tokenize_type == 'bpe':
            self.bpetok = RobertaTokenizer.from_pretrained('roberta-base')

    def tokenize(self, string):
        if self.lowercase:
            string = string.lower()

        print("Type of Tokenizer: ", self.tokenize_type)

        if self.tokenize_type == 'basic':
            tokens = string.split()
        elif self.tokenize_type == 'nltk':
            tokens = nltk.tokenize.word_tokenize(string)
        elif self.tokenize_type == 'wp':
            tokens = self.wptok.tokenize(string)
        elif self.tokenize_type == 'bpe':
            tokens = self.bpetok.tokenize(string)
        else:
            raise ValueError('Unknown tokenization type.')

        return tokens

    def count_ngrams(self, toks, n=3):
        ngram_counts = [Counter() for _ in range(n)]

        for i in range(len(toks)):
            for j in range(n):
                if i - j >= 0:
                    ngram = tuple(toks[(i-j):(i+1)])  # TODO: implement (in one line)
                    ngram_counts[j][ngram] += 1

        return ngram_counts

    def threshold(self, toks, vocab, unk_symbol):
        V = set(vocab)
        assert unk_symbol not in V
        return [w if w in V else unk_symbol for w in toks]


def test_ngram_counts(tokenizer):
    test_ngram_counts = tokenizer.count_ngrams('the dog saw the cat '
                                               'the cat saw the rat '
                                               'the rat saw the dog '.split())
    assert test_ngram_counts[0][('the',)] == 6
    assert test_ngram_counts[0][('saw',)] == 3
    assert test_ngram_counts[0][('dog',)] == 2
    assert test_ngram_counts[1][('saw', 'the')] == 3
    assert test_ngram_counts[1][('the', 'cat')] == 2
    assert test_ngram_counts[1][('rat', 'the')] == 1
    assert test_ngram_counts[2][('the', 'dog', 'saw')] == 1
    assert test_ngram_counts[2][('rat', 'the', 'rat')] == 1


def show_ngram_information(ngram_counts, k, figure_file, quiet=False):
    topk_unigrams = ngram_counts[0].most_common(k)
    topk_bigrams = ngram_counts[1].most_common(k)
    topk_trigrams = ngram_counts[2].most_common(k)

    if not quiet:
        print('-' * 79)
        print('Most frequent n-grams for n=1,2,3')
        for i in range(k):
            if i < len(ngram_counts[0]):
                print('{:30s}\t{:30s}\t{:30s}'.format(str(topk_unigrams[i]),
                                                      str(topk_bigrams[i]),
                                                      str(topk_trigrams[i])))

    plt.figure(figsize=(20,5))
    labels, y = zip(*ngram_counts[0].most_common(100))
    zipf = [y[0]/i for i in range(1, len(y) + 1)]
    labels = [label_[0] for label_ in labels]
    plt.plot(labels, y, '-r', label='Frequency', linewidth=1)
    plt.plot(labels, zipf, '-b', label='Zipf', linewidth=1)
    plt.xticks(rotation=90)
    plt.legend(loc="upper right")
    plt.savefig(figure_file, bbox_inches='tight')
    print('Saved plot at %s' % figure_file)
    print('-' * 79)
