import pickle
import time

import nltk
from keras.preprocessing.sequence import pad_sequences

_tokenizer = {
    'nltk': nltk.word_tokenize
}


class Index:
    def __init__(self, corpora=None, tokenizer='nltk', padding='post', to_lower=True,
                 word2idx=None, idx2word=None, vocab=None, max_len=0):
        self.word2idx = word2idx if word2idx else {}
        self.idx2word = idx2word if idx2word else {}
        self.vocab = vocab if vocab else set()
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.padding = padding
        self.to_lower = to_lower

        if corpora:
            self.build(corpora)

    def build(self, corpora):
        start = time.time()
        with open(corpora, 'r', encoding='utf-8') as f:
            for phrase in f:
                if not phrase:
                    continue
                if self.to_lower:
                    phrase = phrase.lower().strip()
                tokens = ['<start>'] + _tokenizer[self.tokenizer](phrase) + ['<end>']
                self.max_len = max(self.max_len, len(tokens))
                self.vocab.update(tokens)

        self.vocab = sorted(self.vocab)
        self.word2idx['<pad>'] = 0
        self.idx2word[0] = '<pad>'

        for index, word in enumerate(self.vocab, 1):
            self.word2idx[word] = index
            self.idx2word[index] = word
        end = time.time()
        print('Building vocab took {} second(s)'.format(end - start))

    def text_to_sequence(self, text):
        if self.to_lower:
            text = text.lower()
        tokens = ['<start>'] + _tokenizer[self.tokenizer](text) + ['<end>']
        indices = [self.word2idx[t] for t in tokens]
        return pad_sequences([indices], maxlen=self.max_len, padding=self.padding)[0]

    def sequence_to_text(self, sequence, return_as_tokens=True):
        tokens = [self.idx2word[x] for x in sequence if self.idx2word[x] not in ['<start>', '<end>', '<pad>', ' ']]
        if return_as_tokens:
            return tokens
        return ' '.join(tokens).strip()

    def tokenize(self, text):
        return _tokenizer[self.tokenizer](text)

    def save(self, filename):
        fields = {k: v for k, v in self.__dict__.items()}
        with open(filename, 'wb') as f:
            pickle.dump(fields, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            fields = pickle.load(f)
        return Index(**fields)
