import torch.nn as nn

from collections import Counter


def get_init_weights(init_value):
    def init_weights(m):
        if init_value > 0.0:
            if hasattr(m, 'weight') and hasattr(m.weight, 'uniform_'):
                nn.init.uniform_(m.weight, a=-init_value, b=init_value)
            if hasattr(m, 'bias') and hasattr(m.bias, 'uniform_'):
                nn.init.uniform_(m.bias, a=-init_value, b=init_value)

    return init_weights


def get_boundaries(bio):
    """
    Extracts an ordered list of boundaries. BIO label sequences can be either
    -     Raw BIO: B     I     I     O => {(0, 2, None)}
    - Labeled BIO: B-PER I-PER B-LOC O => {(0, 1, "PER"), (2, 2, "LOC")}
    """
    boundaries= []
    i = 0

    while i < len(bio):
        if bio[i][0] == 'O': i += 1
        else:
            s = i
            entity = bio[s][2:] if len(bio[s]) > 2 else None
            i += 1
            while i < len(bio) and bio[i][0] == 'I':
                if len(bio[i]) > 2 and bio[i][2:] != entity: break
                i += 1
            boundaries.append((s, i - 1, entity))

    return boundaries


def load_vertical_tagged_data(data_path, sort_by_length=True):
    wordseqs = []
    tagseqs = []
    charseqslist = []
    wordseq = []
    tagseq = []
    charseqs = []
    wordcounter = Counter()
    tagcounter = Counter()
    charcounter = Counter()
    with open(data_path) as f:
        for line in f:
            toks = line.split()
            if toks:
                assert len(toks) == 2
                word, tag = toks
                wordcounter[word] += 1
                tagcounter[tag] += 1
                for c in word:
                    charcounter[c] += 1
                wordseq.append(word)
                tagseq.append(tag)
                charseqs.append([c for c in word])

            else:
                if wordseq:
                    wordseqs.append(wordseq)
                    tagseqs.append(tagseq)
                    charseqslist.append(charseqs)
                wordseq = []
                tagseq = []
                charseqs = []

    if wordseq:  # Last sequence when no space at the end
        wordseqs.append(wordseq)
        tagseqs.append(tagseq)
        charseqslist.append(charseqs)

    if sort_by_length:
        wordseqs, tagseqs, charseqslist \
            = (list(t) for t in
               zip(*sorted(zip(wordseqs, tagseqs, charseqslist),
                           key=lambda x: len(x[0]), reverse=True)))

    return wordseqs, tagseqs, charseqslist, wordcounter, tagcounter, charcounter
