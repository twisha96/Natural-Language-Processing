import os
import random
import torch


def build_itr(block, bptt):
    i = 0
    while i < block.size(0) - 1:
        T = min(bptt, block.size(0) - 1 - i)
        subblock = block[i : i + T]
        golds = block[i + 1 : i + 1 + T].view(-1)
        i += bptt
        yield subblock, golds


class Data(object):

    def __init__(self, dpath, batch_size, batch_method, device, sort=True,
                 logger=None, is_conditional=False, batch_size_valid=1):
        """
               dpath:      data directory
          batch_size:      number of sequences in one batch during training
        batch_method:      'continuous' or 'translation'
              device:      'cpu' or 'cuda'
                sort:      sort by target sequence lengths before batching
                           (only applicable for batch_method='translation')
        """
        self.dpath = dpath
        self.batch_size = batch_size
        if batch_method == 'continuous':
            assert not is_conditional
        self.batch_method = batch_method
        self.device = device
        self.sort = sort
        self.logger = logger
        self.is_conditional = is_conditional
        if self.is_conditional:
            assert self.batch_method == 'translation'

        self.w2i = {}
        self.i2w = []
        self.PAD = '<pad>'
        self.EOS = '<eos>'
        self.SOS = '<sos>'
        self.add_word(self.PAD)  # Maintain index 0 for the padding symbol!
        self.add_word(self.EOS)
        self.add_word(self.SOS)

        if logger:
            logger.log('Building data from %s...' %(self.dpath))
            logger.log('      batch_size: %s' %(self.batch_size))
            logger.log('batch_size_valid: %s' %(batch_size_valid))
            logger.log('    batch_method: %s   ' %(self.batch_method), False)
            if self.batch_method == 'translation':
                if self.sort:
                    logger.log('   (sorted by target lengths)')
                else:
                    logger.log('   (no sorting by target lengths)')
            else:
                logger.log('')
            logger.log('          device: %s' %(self.device))
            logger.log('  is_conditional: %s' %(self.is_conditional))
            logger.log('')

        self.train = self.process_file('train.txt', batch_size)
        self.valid = self.process_file('valid.txt', batch_size_valid)

        if logger:
            logger.log('')
            logger.log('vocab_size: %d' % len(self.w2i))
            logger.log('')
            logger.log('train.txt')
            logger.log('              # words: %d' % self.num_train_words)
            logger.log('               # seqs: %d' % self.num_train_seqs)
            logger.log('  avg/max/min lengths: %d/%d/%d' %
                       (self.avg_train_len, self.max_train_len,
                        self.min_train_len))
            logger.log('')

            if self.is_conditional:
                logger.log('src-train.txt')
                logger.log('              # words: %d' % self.num_src_words)
                logger.log('               # seqs: %d' % self.num_src_seqs)
                logger.log('  avg/max/min lengths: %d/%d/%d' %
                           (self.avg_src_len, self.max_src_len,
                            self.min_src_len))
                logger.log('')

    def process_file(self, fname, batch_size):
        if self.batch_method == 'continuous':
            return self.build_single_block(fname, batch_size)

        elif self.batch_method == 'translation':
            return self.build_bundles(fname, batch_size)

        else:
            raise ValueError('Unknown batch method \'%s\'' % self.batch_method)

    def build_bundles(self, fname, batch_size):
        tgts = []
        with open(os.path.join(self.dpath, fname), 'r', encoding='utf8') as f:
            num_words = 0
            num_seqs = 0
            max_len = 0
            min_len = 100000000
            for line in f:
                words = [self.SOS] + line.split() + [self.EOS]
                num_words += len(words)
                num_seqs += 1
                max_len = max(max_len, len(words))
                min_len = min(min_len, len(words))
                tgts.append([self.add_word(word) for word in words])
            if fname == 'train.txt':
                self.avg_train_len = num_words / num_seqs
                self.max_train_len = max_len
                self.min_train_len = min_len
                self.num_train_seqs = num_seqs
                self.num_train_words = num_words

        srcs = []
        fpath_src  = os.path.join(self.dpath, 'src-' + fname)
        if self.is_conditional:
            assert os.path.isfile(fpath_src)
            with open(fpath_src, 'r', encoding='utf8') as f:
                num_words = 0
                num_seqs = 0
                max_len = 0
                min_len = 100000000
                for line in f:
                    words = line.split()  # No SOS or EOS for src
                    num_words += len(words)
                    num_seqs += 1
                    max_len = max(max_len, len(words))
                    min_len = min(min_len, len(words))
                    srcs.append([self.add_word(word) for word in words])
                if fname == 'train.txt':
                    self.avg_src_len = num_words / num_seqs
                    self.max_src_len = max_len
                    self.min_src_len = min_len
                    self.num_src_seqs = num_seqs
                    self.num_src_words = num_words

            assert len(srcs) == len(tgts)

        perm = sorted(range(len(tgts)),
                      key=lambda x: len(tgts[x]), reverse=True) \
                      if self.sort else range(len(tgts))

        bundles = []
        for i in range(0, len(tgts), batch_size):
            bundle = self.build_bundle(tgts, srcs, i, perm, batch_size)
            bundles.append(bundle)

        if self.logger:
            self.logger.log('%d batches' % len(bundles))
        return bundles

    def build_bundle(self, tgts, srcs, i, perm, batch_size):
        ncols = min(batch_size, len(tgts) - i)
        nrows = max([len(tgts[perm[j]]) for j in range(i, i + ncols)])
        nrows_src = max([len(srcs[perm[j]]) for j in range(i, i + ncols)]) \
                    if srcs else 0

        block = torch.zeros((nrows, ncols)).long()
        block_src = torch.zeros((nrows_src, ncols)).long() if srcs \
                    else None
        src_lens = []
        for j in range(i, i + ncols):
            sent_num = perm[j]
            if srcs:
                src_lens.append(len(srcs[sent_num]))
            col = j - i

            for row in range(nrows):
                if row < len(tgts[sent_num]):
                    block[row][col] = tgts[sent_num][row]

            if srcs:
                for row in range(nrows_src):
                    if row < len(srcs[sent_num]):
                        block_src[row][col] = srcs[sent_num][row]

        block = block.contiguous().to(self.device)
        assert len(block) > 1

        block_src = block_src.contiguous().to(self.device) \
                    if srcs else None
        src_lens = torch.IntTensor(src_lens).contiguous().to(self.device)

        if srcs:
            # Sort columns of block & block_src by lengths for packing.
            src_lens, perm_index = src_lens.sort(0, descending=True)
            block_src = block_src[:,perm_index]
            block = block[:,perm_index]

        return (block, block_src, src_lens)

    def build_single_block(self, fname, batch_size):
        fpath = os.path.join(self.dpath, fname)

        with open(fpath, 'r', encoding='utf8') as f:
            num_words = 0
            num_seqs = 0
            max_len = 0
            min_len = 100000000
            for line in f:
                words = line.split() + [self.EOS]  # No SOS (as in convention)
                num_words += len(words)
                num_seqs += 1
                max_len = max(max_len, len(words))
                min_len = min(min_len, len(words))
                for word in words:
                    self.add_word(word)
            if fname == 'train.txt':
                self.avg_train_len = num_words / num_seqs
                self.max_train_len = max_len
                self.min_train_len = min_len
                self.num_train_seqs = num_seqs
                self.num_train_words = num_words

        if num_words == 0: return None

        ids = torch.LongTensor(num_words)
        with open(fpath, 'r', encoding='utf8') as f:
            token = 0
            for line in f:
                words = line.split() + [self.EOS]
                for word in words:
                    ids[token] = self.w2i[word]
                    token += 1

        if batch_size > num_words // 2:
            batch_size = num_words // 2  # batch_size <= # words / 2
            if self.logger:
                self.logger.log('Readjusted batch size to %d' % batch_size)


        T = num_words // batch_size
        ndumped = num_words - T * batch_size
        if self.logger:
            self.logger.log('%s: %d x %d (%d words trimmed)' %
                            (fname, T, batch_size, ndumped))
        ids = ids.narrow(0, 0, T * batch_size)
        ids = ids.view(batch_size, -1).t().contiguous().to(self.device)

        assert len(ids) > 1
        return ids

    def add_word(self, word):
        if word not in self.w2i:
            self.i2w.append(word)
            self.w2i[word] = len(self.i2w) - 1
        return self.w2i[word]
