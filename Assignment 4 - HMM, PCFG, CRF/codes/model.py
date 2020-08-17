import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter
from crf import GreedyLoss, CRFLoss
from torch.nn.utils.rnn import pack_padded_sequence
from util import get_boundaries


class BiLSTMTagger(nn.Module):

    def __init__(self, num_word_types, num_tag_types, num_char_types, dim_word,
                 dim_char, dim_hidden, dropout, num_layers, nochar, loss_type,
                 init):
        super(BiLSTMTagger, self).__init__()
        self.PAD_ind = 0
        self.nochar = nochar
        self.wemb = nn.Embedding(num_word_types, dim_word,
                                 padding_idx=self.PAD_ind)
        if not nochar:
            self.cemb = nn.Embedding(num_char_types, dim_char,
                                     padding_idx=self.PAD_ind)
            self.wlstm = BiLSTMOverCharacters(self.cemb, num_layers)

        dim_input = dim_word if nochar else dim_word + 2 * dim_char
        self.slstm = nn.LSTM(dim_input, dim_hidden, num_layers,
                             bidirectional=True)
        self.scorer = nn.Linear(2 * dim_hidden, num_tag_types)
        self.drop = nn.Dropout(p=dropout)
        self.loss = GreedyLoss() if loss_type == 'greedy' else \
                    CRFLoss(num_tag_types, init)

    def score(self, X, Y, C, C_lengths):
        B, T = X.size()
        wembs = self.wemb(X)  # B x T x d_w
        if not self.nochar:
            creps = self.wlstm(C, C_lengths).view(B, T, -1)  # B x T x 2d_c
            wembs = torch.cat([wembs, creps], dim=2)  # B x T x (d_w + 2d_c)

        output, _ = self.slstm(wembs)  # B x T x 2d_h
        scores = self.drop(self.scorer(output))  # B x T x L
        return scores

    def forward(self, X, Y, C, C_lengths):
        scores = self.score(X, Y, C, C_lengths)   # B x T x L
        loss = self.loss(scores, Y)
        return {'loss': loss}

    def do_epoch(self, epoch_num, train_batches, clip, optim, logger=None,
                 check_interval=200):
        self.train()

        output = {}
        for batch_num, (X, Y, C, C_lengths) in enumerate(train_batches):
            optim.zero_grad()

            forward_result = self.forward(X, Y, C, C_lengths)

            loss = forward_result['loss']
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), clip)
            optim.step()

            for key in forward_result:
                output[key] = forward_result[key] if not key in output else \
                              output[key] + forward_result[key]

            if logger and (batch_num + 1) % check_interval == 0:
                logger.log('Epoch {:3d} | Batch {:5d}/{:5d} | '
                           'Average Loss {:8.4f}'.format(
                               epoch_num, batch_num + 1, len(train_batches),
                               output['loss'] / (batch_num + 1)))

            if math.isnan(output['loss']):
                if logger:
                    logger.log('Stopping training since objective is NaN')
                break

        for key in output:
            output[key] /= (batch_num + 1)

        return output

    def evaluate(self, eval_batches, tag2y=None):
        self.eval()
        if 'O' in tag2y:
            y2tag = [None for tag in tag2y]
            for tag in tag2y:
                y2tag[tag2y[tag]] = tag
            tp = Counter()
            fp = Counter()
            fn = Counter()

        num_preds = 0
        num_correct = 0
        gold_entities = {}
        for (X, Y, C, C_lengths) in eval_batches:
            B, T = Y.size()
            scores = self.score(X, Y, C, C_lengths)   # B x T x L
            _, preds = self.loss.decode(scores)   # B x T
            num_preds += B * T
            num_correct += (preds == Y).sum().item()

            if 'O' in tag2y:
                for i in range(B):
                    gold_bio_labels = [y2tag[Y[i, j].item()]
                                       for j in range(T)]
                    pred_bio_labels = [y2tag[preds[i, j].item()]
                                       for j in range(T)]
                    gold_boundaries = set(get_boundaries(gold_bio_labels))
                    pred_boundaries = set(get_boundaries(pred_bio_labels))
                    for (s, t, entity) in gold_boundaries:
                        gold_entities[entity] = True
                        if (s, t, entity) in pred_boundaries:
                            tp[entity] += 1
                            tp['<all>'] += 1
                        else:
                            fn[entity] += 1
                            fn['<all>'] += 1
                    for (s, t, entity) in pred_boundaries:
                        if not (s, t, entity) in gold_boundaries:
                            fp[entity] += 1
                            fp['<all>'] += 1

        output = {'acc': num_correct / num_preds * 100}

        if 'O' in tag2y:
            for e in list(gold_entities) + ['<all>']:
                p_denom = tp[e] + fp[e]
                r_denom = tp[e] + fn[e]
                p_e = 100 * tp[e] / p_denom if p_denom > 0 else 0
                r_e = 100 * tp[e] / r_denom if r_denom > 0 else 0
                f1_denom = p_e + r_e
                f1_e = 2 * p_e * r_e / f1_denom if f1_denom > 0 else 0
                output['p_%s' % e] = p_e
                output['r_%s' % e] = r_e
                output['f1_%s' % e] = f1_e

        return output

class BiLSTMOverCharacters(nn.Module):

    def __init__(self, cemb, num_layers):
        super(BiLSTMOverCharacters, self).__init__()
        self.cemb = cemb
        self.bilstm = nn.LSTM(cemb.embedding_dim, cemb.embedding_dim,
                              num_layers, bidirectional=True)

    def forward(self, padded_chars, char_lengths):
        B = len(char_lengths)

        packed = pack_padded_sequence(self.cemb(padded_chars), char_lengths,
                                      batch_first=True, enforce_sorted=False)
        _, (final_h, _) = self.bilstm(packed)

        final_h = final_h.view(self.bilstm.num_layers, 2, B,
                               self.bilstm.hidden_size)[-1]       # 2 x BT x d_c
        cembs = final_h.transpose(0, 1).contiguous().view(B, -1)  # BT x 2d_c
        return cembs
