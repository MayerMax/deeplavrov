import json
import os
import time

from itertools import islice
from multiprocessing.pool import Pool

import tqdm

from deeplavrov.vocabulary.vocabulary import WordVocabEncoder

import tensorflow as tf

import numpy as np
from keras.layers import Embedding, GRU, Dense, CuDNNGRU


def create_gru(units):
    # if tf.test.is_gpu_available():
    #     return CuDNNGRU(units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    # else:
    return GRU(units, return_sequences=True, return_state=True, recurrent_activation='sigmoid',
               recurrent_initializer='glorot_uniform')


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


class Encoder(tf.keras.Model):
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


class Decoder(tf.keras.Model):
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
    def __init__(self, source_embed_size=256, target_embed_size=256, num_units=1024, batch_size=64, num_epochs=20,
                 **kwargs):
        """
        Write the docs!
        """
        self.source_embed_size = source_embed_size
        self.target_embed_size = target_embed_size
        self.num_units = num_units
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.vocab_encoder = WordVocabEncoder(**kwargs)
        self.params_to_save = kwargs

        self.encoder = None
        self.decoder = None

        self.checkpoint_dir = None
        self.checkpoint_prefix = None
        self.checkpoint = None

    def load(self, filename):
        pass

    def save(self, filename):
        pass

    def _process_pair(self, source_target_pair):
        source_text, target_text = source_target_pair
        return (self.vocab_encoder.text_to_indices(source_text.strip(), is_source=True),
                self.vocab_encoder.text_to_indices(target_text.strip(), is_source=False))

    def _file_generator(self, source_filename, target_filename, n_jobs=4):
        with open(source_filename, encoding='utf-8') as source_f, open(target_filename, encoding='utf-8') as target_f:
            with Pool(n_jobs) as pool:
                while True:
                    source_batch = list(islice(source_f, 100000))
                    target_batch = list(islice(target_f, 100000))
                    if not source_batch or not target_batch:
                        break

                    yield from pool.imap(func=self._process_pair, iterable=zip(source_batch, target_batch))

    def generator(self, source_filename, target_filename):
        with open(source_filename, encoding='utf-8') as source_f, open(target_filename, encoding='utf-8') as target_f:
            eng_line, ru_line = source_f.readline().strip(), target_f.readline().strip()
            while eng_line:
                yield self.vocab_encoder.text_to_indices(eng_line, True), \
                      self.vocab_encoder.text_to_indices(ru_line, False)

    def fit_from_file(self, source_corpora, target_corpora, n_jobs=4, checkpoint_dir='./training_checkpoints'):
        # self.vocab_encoder.build(source_corpora, target_corpora, n_jobs=n_jobs)
        # self.vocab_encoder.save('vocabulary_index.json')
        self.vocab_encoder = WordVocabEncoder.load('../vocabulary_index.json')
        self.params_to_save['index_path'] = 'vocabulary_index.json'  # fix later

        self.encoder = Encoder(len(self.vocab_encoder.source_vocab),
                               self.source_embed_size, self.num_units, self.batch_size)
        self.decoder = Decoder(len(self.vocab_encoder.target_vocab),
                               self.target_embed_size, self.num_units, self.batch_size)
        optimizer = tf.train.AdamOptimizer()

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                              encoder=self.encoder,
                                              decoder=self.decoder)

        gen = lambda: self.generator(source_corpora, target_corpora)

        dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32))
        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        try:
            for epoch in tqdm.tqdm(range(self.num_epochs)):
                start = time.time()

                hidden = self.encoder.initialize_hidden_state()
                total_loss = 0

                for (batch_index, (source, target)) in enumerate(dataset):
                    cur_loss = 0
                    with tf.GradientTape() as tape:
                        encoder_output, encoder_hidden = self.encoder(source, hidden)

                        decoder_hidden = encoder_hidden
                        decoder_input = tf.expand_dims([self.vocab_encoder.get_index_of_word(
                            self.vocab_encoder.start_symbol)] * self.batch_size, 1)

                        # right now using teacher forcing: feeding the target as the next input
                        # add code to change this option ...

                        for timestamp in range(1, target.shape[1]):
                            predictions, dec_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)
                            cur_loss += loss_function(target[:, timestamp], predictions)

                            decoder_input = tf.expand_dims(target[:, timestamp], 1)
                    batch_loss = (cur_loss / int(target.shape[1]))
                    total_loss += batch_loss

                    variables = self.encoder.variables + self.decoder.variables
                    gradients = tape.gradient(cur_loss, variables)

                    optimizer.apply_gradients(zip(gradients, variables))

                    if batch_index % 100 == 0:
                        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch_index, batch_loss.numpy()))

                if (epoch + 1) % 2 == 0:
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix)

                print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                    total_loss / batch_index))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        except KeyboardInterrupt as _:
            print('Saving before interruption')
            self._save()
            print('Saved')
        print('Saving after all epochs')
        self._save()
        print('Saved')

    def _save(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        path_to_model = tf.train.latest_checkpoint(self.checkpoint_dir)

        fields = {k: v for k, v in self.__dict__.items() if k not in ['source_reverse_vocab',
                                                                      'target_reverse_vocab',
                                                                      'encoder', 'decoder',
                                                                      'checkpoint']}
        fields['path_to_model'] = path_to_model
        fields['path_to_index'] = 'vocabulary_index.json'
        with open('model.json', 'w') as f:
            json.dump(fields, f)


if __name__ == '__main__':
    tf.enable_eager_execution()
    params = {
        'max_source_len': 49,
        'max_target_len': 51
    }
    translator = WordLevelRNNTranslator(**params)
    translator.fit_from_file('../data/eng_anki_corpora.txt', '../data/ru_anki_corpora.txt')
