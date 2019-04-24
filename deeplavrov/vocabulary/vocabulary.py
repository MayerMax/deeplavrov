import json
import time
from collections import Counter
from itertools import islice, chain
from multiprocessing.pool import Pool

import nltk
from keras.preprocessing.sequence import pad_sequences

_tokenizer = {
    'nltk': nltk.word_tokenize
}


class WordVocabEncoder:
    def __init__(self, source_vocab=None,
                 target_vocab=None,
                 source_tokenizing_method='nltk', target_tokenizing_method='nltk',
                 max_source_len=120, max_target_len=120, padding_symbol='<pad>', pad_source_pre=True,
                 pad_target_pre=False, unknown_symbol='<unk>',
                 start_symbol='#start#', end_symbol='#end#', encoding='utf-8', word_min_freq=2, **kwargs):

        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        self.source_reverse_vocab = WordVocabEncoder._get_reverse_vocab(self.source_vocab)
        self.target_reverse_vocab = WordVocabEncoder._get_reverse_vocab(self.target_vocab)

        self.source_tokenizing_method = source_tokenizing_method
        self.target_tokenizing_method = target_tokenizing_method

        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

        self.padding_symbol = padding_symbol
        self.pad_source_pre = pad_source_pre
        self.pad_target_pre = pad_target_pre

        self.unknown_symbol = unknown_symbol
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
        self.encoding = encoding
        self.word_min_freq = word_min_freq

    def to_indices(self, text, is_source=True):
        if is_source:
            return [self.source_vocab.get(x, -1) for x in chain([self.start_symbol],
                                                                _tokenizer.get(self.source_tokenizing_method)(text),
                                                                [self.end_symbol])]
        return [self.target_vocab.get(x, -1) for x in chain([self.start_symbol],
                                                            _tokenizer.get(self.target_tokenizing_method)(text),
                                                            [self.end_symbol])]

    def to_text(self, indices, is_source=True):
        if is_source:
            exclude = [0, self.source_vocab[self.start_symbol], self.source_vocab[self.end_symbol]]
            return ' '.join([self.source_reverse_vocab[x] for x in indices if x not in exclude])
        else:
            exclude = [0, self.target_vocab[self.start_symbol], self.target_vocab[self.end_symbol]]
            return ' '.join([self.target_reverse_vocab[x] for x in indices if x not in exclude])

    def pad_indices(self, indices, is_source=True):
        if is_source:
            strategy = 'pre' if self.pad_source_pre else 'post'
            return pad_sequences([indices], maxlen=self.max_source_len, padding=strategy)[0]
        else:
            strategy = 'pre' if self.pad_target_pre else 'post'
            return pad_sequences([indices], maxlen=self.max_target_len, padding=strategy)[0]

    def text_to_indices(self, text, is_source=True):
        indices = self.to_indices(text, is_source)
        return self.pad_indices(indices, is_source)

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
                                                            _tokenizer.get(self.source_tokenizing_method),
                                                            n_jobs)

            self.source_reverse_vocab = WordVocabEncoder._get_reverse_vocab(self.source_vocab)
        else:
            self.target_vocab = self._build_vocab_from_file(corpora,
                                                            _tokenizer.get(self.target_tokenizing_method),
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
