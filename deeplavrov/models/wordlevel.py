import json
import os
import pickle
import time

from itertools import islice
from multiprocessing.pool import Pool

import tqdm
from nltk.translate.bleu_score import sentence_bleu

from deeplavrov.models.attention.layers import BahdanauAttention, LuongMultiplicativeStyle
from deeplavrov.vocabulary.vocabulary import Index

import tensorflow as tf
import numpy as np


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        if tf.test.is_gpu_available():
            self.gru = tf.keras.layers.CuDNNGRU(self.enc_units, return_sequences=True, return_state=True,
                                                recurrent_initializer='glorot_uniform')
        else:
            self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True,
                                           recurrent_initializer='glorot_uniform')

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
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        if tf.test.is_gpu_available():
            self.gru = tf.keras.layers.CuDNNGRU(self.dec_units, return_sequences=True, return_state=True,
                                                recurrent_initializer='glorot_uniform')
        else:
            self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True,
                                           recurrent_initializer='glorot_uniform')

        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))

        logits = self.fc(output)

        return logits, state, attention_weights


class AttentionRNNTranslator:
    def __init__(self, input_embed_size=256, target_embed_size=256, num_units=1024, batch_size=16, num_epochs=20,
                 input_tokenizer='nltk', target_tokenizer='nltk', input_padding='pre', target_padding='post',
                 to_lower=True, input_index=None, target_index=None, checkpoint_dir=None):
        self.input_embed_size = input_embed_size
        self.target_embed_size = target_embed_size
        self.num_units = num_units
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.input_index = input_index if input_index else Index(tokenizer=input_tokenizer, padding=input_padding,
                                                                 to_lower=to_lower)

        self.target_index = target_index if target_index else Index(tokenizer=target_tokenizer, padding=target_padding,
                                                                    to_lower=to_lower)

        self.encoder = None
        self.decoder = None

        self.checkpoint_dir = checkpoint_dir if checkpoint_dir else './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = None
        self.optimizer = None

        self._fields_to_save = {k: v for k, v in self.__dict__.items() if k not in ['encoder',
                                                                                    'decoder',
                                                                                    'checkpoint',
                                                                                    'optimizer',
                                                                                    'checkpoint_prefix']}

    def _step(self, inp, targ, enc_hidden):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([self.target_index.word2idx['<start>']] * self.batch_size, 1)

            # teacher forcing
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)

                loss += loss_function(targ[:, t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (loss / int(targ.shape[1]))

            variables = self.encoder.trainable_variables + self.decoder.trainable_variables

            gradients = tape.gradient(loss, variables)

            self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    def _batch_generator(self, input_file, target_file):
        with open(input_file, 'r', encoding='utf-8') as first, open(target_file, 'r', encoding='utf-8') as second:
            inp, targ = first.readline(), second.readline()
            while inp:
                yield (self.input_index.text_to_sequence(inp),
                       self.target_index.text_to_sequence(targ))
                inp, targ = first.readline(), second.readline()

    def fit_from_file(self, input_file, target_file, val_input_file=None, val_target_file=None):
        self.input_index.build(input_file)
        self.target_index.build(target_file)

        gen = lambda: self._batch_generator(input_file, target_file)
        dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32))
        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        self._init_model()

        for epoch in range(self.num_epochs):
            start = time.time()
            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(dataset):
                batch_loss = self._step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))

            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / batch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def translate(self, sentence, return_as_tokens=True):
        inputs = [self.input_index.text_to_sequence(sentence)]
        inputs = tf.convert_to_tensor(inputs)

        result = []

        hidden = [tf.zeros((1, self.num_units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.target_index.word2idx['<start>']], 0)

        for t in range(self.target_index.max_len):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_out)

            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(predicted_id)

            if self.target_index.idx2word[predicted_id] == '<end>':
                return self.target_index.sequence_to_text(result)

            dec_input = tf.expand_dims([predicted_id], 0)

        return self.target_index.sequence_to_text(result, return_as_tokens)

    def get_bleu_score(self, input_file, target_file):
        score = 0
        counter = 0
        with open(input_file, 'r', encoding='utf-8') as first, open(target_file, 'r', encoding='utf-8') as second:
            inp, targ = first.readline(), second.readline()
            while inp:
                translation = self.translate(inp.strip())
                reference = [self.target_index.tokenize(targ.strip())]
                score += sentence_bleu(reference, translation)
                counter += 1
                inp, targ = first.readline(), second.readline()
        return (score / counter) * 100

    def save(self, filename):
        print('Saved!')
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        with open(filename, 'wb') as f:
            pickle.dump(self._fields_to_save, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            fields = pickle.load(f)
        obj = AttentionRNNTranslator(**fields)
        obj._init_model()
        obj.checkpoint.restore(tf.train.latest_checkpoint(obj.checkpoint_dir))
        return obj

    def _init_model(self):
        self.encoder = Encoder(len(self.input_index.word2idx), self.input_embed_size, self.num_units, self.batch_size)
        self.decoder = Decoder(len(self.target_index.word2idx), self.target_embed_size, self.num_units, self.batch_size)
        self.optimizer = tf.train.AdamOptimizer()
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)
