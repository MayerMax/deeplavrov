import os

from deeplavrov.vocabulary.vocabulary import WordVocabEncoder

import tensorflow as tf
import keras
import numpy as np
from keras.layers import Embedding, GRU, Dense, CuDNNGRU


def create_gru(units):
    if tf.test.is_gpu_available():
        return CuDNNGRU(units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    else:
        return GRU(units, return_sequences=True, return_state=True, recurrent_activation='sigmoid',
                   recurrent_initializer='glorot_uniform')


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


class Encoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = create_gru(self.enc_units)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = create_gru(self.dec_units)
        self.fc = Dense(vocab_size)

        self.W1 = Dense(self.dec_units)
        self.W2 = Dense(self.dec_units)
        self.V = Dense(1)

    def call(self, x, hidden, enc_output):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, state, attention_weights

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))


class WordLevelRNNTranslator:
    def __init__(self, source_embed_size=256, target_embed_size=256, num_units=1024, batch_size=64, **kwargs):
        """
        Write the docs!
        """
        self.source_embed_size = source_embed_size
        self.target_embed_size = target_embed_size
        self.num_units = num_units
        self.batch_size = batch_size

        self.vocab_encoder = WordVocabEncoder(**kwargs)
        self.params_to_save = kwargs

        self.encoder = None
        self.decoder = None

    def load(self, filename):
        pass

    def save(self, filename):
        pass

    def fit_from_file(self, source_corpora, target_corpora, n_jobs=4, checkpoint_dir='./training_checkpoints'):
        self.vocab_encoder.build(source_corpora, target_corpora, n_jobs=n_jobs)
        self.vocab_encoder.save('vocabulary_index.json')
        self.params_to_save['index_path'] = 'vocabulary_index.json'  # fix later

        self.encoder = Encoder(len(self.vocab_encoder.source_vocab),
                               self.source_embed_size, self.num_units, self.batch_size)
        self.decoder = Decoder(len(self.vocab_encoder.target_vocab),
                               self.target_embed_size, self.num_units, self.batch_size)
        optimizer = tf.train.AdamOptimizer()

        checkpoint_dir = checkpoint_dir
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         encoder=self.encoder,
                                         decoder=self.decoder)

