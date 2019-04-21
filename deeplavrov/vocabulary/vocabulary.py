import json
import time
from collections import Counter
from itertools import islice
from multiprocessing.pool import Pool

import nltk

__tokenizer__ = {
    'nltk': nltk.word_tokenize
}


class WordVocabEncoder:
    def __init__(self, source_vocab=None,
                 target_vocab=None,
                 source_tokenizing_method='nltk', target_tokenizing_method='nltk',
                 max_seq_len=120, padding_symbol='<pad>', unknown_symbol='<unk>',
                 start_symbol='#start#', end_symbol='#end#', encoding='utf-8', word_min_freq=2):

        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        self.source_reverse_vocab = WordVocabEncoder._get_reverse_vocab(self.source_vocab)
        self.target_reverse_vocab = WordVocabEncoder._get_reverse_vocab(self.target_vocab)

        self.source_tokenizing_method = source_tokenizing_method
        self.target_tokenizing_method = target_tokenizing_method

        self.max_seq_len = max_seq_len
        self.padding_symbol = padding_symbol
        self.unknown_symbol = unknown_symbol
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
        self.encoding = encoding
        self.word_min_freq = word_min_freq

    def build(self, source_corpora, target_corpora, n_jobs=4):
        """
        builds vocabulary for both corporas (they must be parallel)
        :param source_corpora: txt file path where every instance is separated with \n
        :param target_corpora: txt file path where every instance is separated with \n
        :param n_jobs: cores for multiprocessing
        :return: None
        """
        self.build_from_corpora(source_corpora, is_source=True, n_jobs=n_jobs)
        self.build_from_corpora(target_corpora, is_source=False, n_jobs=n_jobs)

    def build_from_corpora(self, corpora, is_source=True, n_jobs=4):
        """
        builds vocab only for one corpora, by default assuming it is source
        :param corpora:  txt file path where every instance is separated with \n
        :param is_source: is corpora a source or target
        :param n_jobs : cores for multiprocessing
        :return: None
        """
        if is_source:
            self.source_vocab = self._build_vocab_from_file(corpora,
                                                            __tokenizer__.get(self.source_tokenizing_method),
                                                            n_jobs)

            self.source_reverse_vocab = WordVocabEncoder._get_reverse_vocab(self.source_vocab)
        else:
            self.target_vocab = self._build_vocab_from_file(corpora,
                                                            __tokenizer__.get(self.target_tokenizing_method),
                                                            n_jobs)

            self.target_reverse_vocab = WordVocabEncoder._get_reverse_vocab(self.target_vocab)

    def save(self, filename):
        fields = {k: v for k, v in self.__dict__.items() if k not in ['source_reverse_vocab', 'target_reverse_vocab']}
        # do not save reverse vocab as it is redundant
        with open(filename, 'w') as f:
            json.dump(fields, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            fields = json.load(f)
        return WordVocabEncoder(**fields)

    def _build_vocab_from_file(self, filename, tokenizing_method, n_jobs=4):
        start = time.time()
        frequencies = Counter()
        with open(filename, encoding=self.encoding) as f:
            while True:
                batch = list(islice(f, 100000))
                if not batch:
                    break
                with Pool(n_jobs) as pool:
                    for processed in pool.imap(func=tokenizing_method, iterable=batch):
                        frequencies.update([self.start_symbol] + processed + [self.end_symbol])
        end = time.time()
        print('Building vocab from {} took {} second(s)'.format(filename, end - start))

        vocab = {token: index for index, token in enumerate(frequencies.keys(), start=1)  # 0 and -1 are reserved
                 if frequencies[token] >= self.word_min_freq}
        vocab[self.padding_symbol] = 0  # index for padding is 0
        vocab[self.unknown_symbol] = -1  # index for unknown is -1

        return vocab

    @classmethod
    def _get_reverse_vocab(cls, vocab):
        if not vocab:
            return None
        return {idx: key for key, idx in vocab.items()}

# if __name__ == '__main__':
#     a = WordVocabEncoder()
#     a.save('a')
