import data
import math
import random
import time
import torch
import torch.nn as nn
import pdb

class Control(nn.Module):

    def __init__(self, s2s, lr, bptt, interval, model_path=None,
                 logger=None):
        super(Control, self).__init__()
        self.s2s = s2s
        self.lr = lr
        self.bptt = bptt
        self.interval = interval
        self.model_path = model_path
        self.logger = logger

        self.avgCE = nn.CrossEntropyLoss(ignore_index=0)
        self.sumCE = nn.CrossEntropyLoss(reduction='sum',
                                         ignore_index=0)
        self.nbatches = 0

        if self.logger:
            self.logger.log('Control')
            self.logger.log('            lr: %.2f' % lr)
            self.logger.log('          bptt: %d' % bptt)
            self.logger.log('')

    def train(self, dat, epochs, shuffle=False):
        self.nbatches = self.count_batches(dat.train)
        best_val_loss = None

        try:
            for epoch in range(1, epochs + 1):
                val_loss, val_sqxent, epoch_time = self.do_epoch(dat, epoch,
                                                                 shuffle)

                if not best_val_loss or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if self.model_path:
                        with open(self.model_path, 'wb') as f:
                            torch.save(self.s2s, f)
                else:
                    self.lr /= 4.0

        except KeyboardInterrupt:
             if self.logger:
                self.logger.log('-' * 89)
                self.logger.log('Exiting from training early')

        if self.model_path:
            self.load_s2s()

            final_loss, final_sqxent = self.evaluate(dat.valid)
            if self.logger:
                self.logger.log('=' * 89)
                self.logger.log('| End of training | final loss {:5.2f} '
                                '| final ppl {:8.2f} | final sqxent {:8.2f}'.\
                                format(final_loss, math.exp(final_loss),
                                       final_sqxent))
                self.logger.log('=' * 89)

    def load_s2s(self):
        with open(self.model_path, 'rb') as f:
            self.s2s = torch.load(f)
            if self.s2s.is_conditional:
                self.s2s.enc.lstm.flatten_parameters()
            if type(self.s2s.dec.lstm).__name__ == 'LSTM':
                self.s2s.dec.lstm.flatten_parameters()

    def do_epoch(self, dat, epoch, shuffle=False):
        val_loss, val_sqxent, epoch_time \
            = self.epoch_continuous_data(dat, epoch) \
            if dat.batch_method == 'continuous' else \
               self.epoch_translation_data(dat, epoch, shuffle)

        if self.logger:
            self.logger.log('-' * 89)
            self.logger.log('| end of epoch {:3d} | time: {:5.2f}s '
                            '| valid loss {:5.2f} | valid ppl {:8.2f} '
                            '| valid sqxent {:8.2f}'.format(
                                epoch, epoch_time, val_loss,
                                math.exp(val_loss), val_sqxent))
            self.logger.log('-' * 89)
        return val_loss, val_sqxent, epoch_time

    def count_batches(self, training_source):
        if isinstance(training_source, torch.Tensor):
            nbatches = (training_source.size(0) - 1) // self.bptt
            if (training_source.size(0) - 1) % self.bptt != 0:
                nbatches += 1
        else:
            nbatches = 0
            for block, _, _, in training_source:
                for _ in data.build_itr(block, self.bptt):
                    nbatches += 1

        return nbatches

    def epoch_translation_data(self, dat, epoch, shuffle=False):
        self.s2s.train()
        epoch_start_time = time.time()
        log_start_time = time.time()
        sum_avgCEs = 0
        if self.nbatches == 0:
            self.nbatches = self.c
            count_batches(dat.train)
        batch_num = 0

        if shuffle:
            random.shuffle(dat.train)

        for block, block_src, src_lens in dat.train:

            # pdb.set_trace()
            itr = data.build_itr(block, self.bptt)
            starting = True
            # pdb.set_trace()
            for subblock, golds in itr:
                loss = self.step_on_batch(subblock, golds, src=block_src,
                                          lengths=src_lens, start=starting)
                starting = False
                sum_avgCEs += loss
                batch_num += 1

                if batch_num % self.interval == 0:
                    self.report(epoch, batch_num, sum_avgCEs / self.interval,
                                time.time() - log_start_time)
                    sum_avgCEs = 0
                    log_start_time = time.time()

        val_loss, val_sqxent = self.evaluate_translation_data(dat.valid)
        epoch_time = time.time() - epoch_start_time

        return val_loss, val_sqxent, epoch_time

    def epoch_continuous_data(self, dat, epoch):
        self.s2s.train()
        epoch_start_time = time.time()
        log_start_time = time.time()
        sum_avgCEs = 0
        if self.nbatches == 0:
            self.nbatches = self.count_batches(dat.train)
        self.s2s.dec.init_state(batch_size=dat.train.size(1),
                                 encoder_final=None)

        itr = data.build_itr(dat.train, self.bptt)
        for batch_num_, (subblock, golds) in enumerate(itr):
            # pdb.set_trace()
            loss = self.step_on_batch(subblock, golds, src=None,
                                      lengths=None, start=False)
            sum_avgCEs += loss

            if (batch_num_ + 1) % self.interval == 0:
                self.report(epoch, batch_num_ + 1, sum_avgCEs / self.interval,
                            time.time() - log_start_time)
                sum_avgCEs = 0
                log_start_time = time.time()

        val_loss, dummy = self.evaluate_continuous_data(dat.valid)
        epoch_time = time.time() - epoch_start_time

        return val_loss, dummy, epoch_time

    def step_on_batch(self, subblock, golds, src=None, lengths=None,
                      start=True):
        self.s2s.zero_grad()
        output, attn = self.s2s(subblock, src=src, lengths=lengths,
                                 start=start)
        loss = self.avgCE(output, golds)
        loss.backward()

        nn.utils.clip_grad_norm_(self.s2s.parameters(), 0.25)
        # Update using the calculated gradients
        for p in self.s2s.parameters():
            p.data.add_(-self.lr, p.grad.data)

        return loss.item()

    def evaluate(self, data_source):
        if isinstance(data_source, torch.Tensor):
            return self.evaluate_continuous_data(data_source)
        else:
            return self.evaluate_translation_data(data_source)

    def evaluate_translation_data(self, bundles):
        self.s2s.eval()
        totalCE = 0.
        ngolds = 0
        nseqs = 0

        with torch.no_grad():
            for block, block_src, src_lens in bundles:
                nseqs += block.size(1)
                itr = data.build_itr(block, self.bptt)
                starting = True
                for subblock, golds in itr:
                    output, _ = self.s2s(subblock, src=block_src,
                                          lengths=src_lens, start=starting)
                    starting = False
                    totalCE += self.sumCE(output, golds).item()
                    ngolds += len(golds.nonzero())  # Ignore padders!

        return (totalCE / ngolds if ngolds > 0 else float('inf'),
                totalCE / nseqs if nseqs > 0 else float('inf'))  # seq cross-ent


    def evaluate_continuous_data(self, block):
        self.s2s.eval()
        self.s2s.dec.init_state(batch_size=block.size(1), encoder_final=None)
        totalCE = 0.

        with torch.no_grad():
            itr = data.build_itr(block, self.bptt)
            for subblock, golds in itr:
                output, _ = self.s2s(subblock, src=None, lengths=None,
                                      start=False)
                totalCE += len(subblock) * self.avgCE(output, golds).item()

        # There's no padding symbol, so this division works.
        return (totalCE / (len(block) - 1),
                -1.0) # dummy value for sqxent

    def report(self, epoch, batch_num, avg_avgCEs, elapsed):
        assert self.nbatches > 0
        if self.logger:
            self.logger.log('| epoch {:3d} | {:5d}/{:5d} batches '
                            '| lr {:02.2f} | ms/batch {:5.2f} '
                            '| loss {:5.2f} | ppl {:8.2f}'.format(
                                epoch, batch_num, self.nbatches, self.lr,
                                elapsed * 1000 / self.interval,
                                avg_avgCEs, math.exp(avg_avgCEs)))
