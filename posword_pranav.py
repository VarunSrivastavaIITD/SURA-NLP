#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
import argparse

from keras.preprocessing.text import Tokenizer

from nltk.corpus import brown

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.iterators import SerialIterator
from chainer.links import BinaryHierarchicalSoftmax
from nltk import FreqDist
import cupy
from itertools import islice, cycle


# Definition of a recurrent net for language modeling
class RNNForLM(chainer.Chain):
    def __init__(self, n_vocab, n_units):
        super(RNNForLM, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.l1 = L.LSTM(n_units, n_units)
            self.l2 = L.LSTM(n_units, n_units)
            self.l3 = L.Linear(n_units, n_vocab)

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, ratio=.25))
        h2 = self.l2(F.dropout(h1, ratio=.25))
        y = self.l3(F.dropout(h2, ratio=.25))
        return y


class RNNForBrown(chainer.Chain):
    def __init__(self, n_vocab, n_units, n_layers):
        super(RNNForBrown, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.lstm = StackedLSTMLayers(n_units, n_layers)
            # self.lstm = L.NStepLSTM(n_layers, n_units, n_units, 0.25)
            self.l3 = L.Linear(n_units, n_vocab)

        for param in self.l3.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def reset_state(self):
        self.lstm.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.lstm(F.dropout(h0, ratio=.25))
        y = self.l3(F.dropout(h1, ratio=.25))
        return y


class StackedLSTMLayers(chainer.ChainList):
    def __init__(self, n_units, n_layers):
        super(StackedLSTMLayers, self).__init__(
            * [L.LSTM(n_units, n_units) for i in range(n_layers)])
        self.add_persistent('n_layers', n_layers)

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def reset_state(self):
        for l in self:
            l.reset_state()

    def __call__(self, x):
        h = self[0](x)

        for i in range(1, self.n_layers - 1):
            h = self[i](h)

        return h


class StackedLinearLayers(chainer.ChainList):
    def __init__(self, n_units, n_layers, dropout_ratio=0.5):
        super(StackedLinearLayers, self).__init__(
            * [L.Linear(n_units, n_units) for i in range(n_layers)])
        self.add_persistent('n_layers', n_layers)
        self.add_persistent('dropout_ratio', dropout_ratio)

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def __call__(self, x):
        h = self[0](x)

        for i in range(1, self.n_layers - 1):
            h = self[i](F.dropout(h, ratio=self.dropout_ratio))

        return h


class RecNetwork(chainer.Chain):
    def __init__(self,
                 n_vocab,
                 n_word_units,
                 n_word_layers,
                 n_pos,
                 n_pos_units,
                 n_pos_layers,
                 n_lin_layers,
                 lin_dropout=0.5):
        super(RecNetwork, self).__init__()

        n_lin_units = n_vocab + n_pos

        with self.init_scope():
            self.vocab_layer = RNNForBrown(n_vocab, n_word_units,
                                           n_word_layers)
            self.pos_layer = RNNForBrown(n_pos, n_pos_units, n_pos_layers)

            if n_lin_layers > 0:
                self.dense = StackedLinearLayers(n_lin_units, n_lin_layers,
                                                 lin_dropout)

            self.linear = L.Linear(n_lin_units, n_vocab)

        self.add_persistent('n_vocab', n_vocab)
        self.add_persistent('n_pos', n_pos)
        self.add_persistent('n_lin_layers', n_lin_layers)

        for param in self.linear.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def reset_state(self):
        self.vocab_layer.reset_state()
        self.pos_layer.reset_state()

    def __call__(self, x):
        # x1 = x[:self.n_vocab]
        # x2 = x[self.n_vocab:]

        x1 = x[:, 0]
        x2 = x[:, 1]

        y1 = F.softmax(self.vocab_layer(x1))
        y2 = F.softmax(self.pos_layer(x2))

        print(y1.shape)
        print(y2.shape)

        y = F.concat((y1, y2))

        if self.n_lin_layers > 0:
            h = self.dense(y)
        else:
            h = y

        return self.linear(h)


# Dataset iterator to create a batch of sequences at different positions.
# This iterator returns a pair of current words and the next words. Each
# example is a part of sequences starting from the different offsets
# equally spaced within the whole sequence.
class ParallelSequentialIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        # Offsets maintain the position of each sequence in the mini-batch.
        self.offsets = [i * length // batch_size for i in range(batch_size)]

        # print("iterations per epoch ", len(self.dataset)/batch_size)
        # NOTE: this is not a count of parameter updates. It is just a count of
        # calls of ``__next__``.
        self.iteration = 0
        # use -1 instead of None internally
        self._previous_epoch_detail = -1.

    def __next__(self):
        # This iterator returns a list representing a mini-batch. Each item
        # indicates a different position in the original sequence. Each item is
        # represented by a pair of two word IDs. The first word is at the
        # "current" position, while the second word at the next position.
        # At each iteration, the iteration count is incremented, which pushes
        # forward the "current" position.
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            # If not self.repeat, this iterator stops at the end of the first
            # epoch (i.e., when all words are visited once).
            raise StopIteration
        cur_words = self.get_words()
        self._previous_epoch_detail = self.epoch_detail
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration * self.batch_size / len(self.dataset)

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def get_words(self):
        # It returns a list of current words.
        return [
            self.dataset[(offset + self.iteration) % len(self.dataset)]
            for offset in self.offsets
        ]

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                (self.current_position - self.batch_size) / len(self.dataset)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(self._previous_epoch_detail,
                                                  0.)
            else:
                self._previous_epoch_detail = -1.


# Custom updater for truncated BackProp Through Time (BPTT)
class BPTTUpdater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(train_iter, optimizer, device=device)
        self.bprop_len = bprop_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        loss = 0
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(self.bprop_len):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()

            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = self.converter(batch, self.device)

            # Compute the loss at this time step and accumulate it
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters


# Routine to rewrite the result dictionary of LogReport to add perplexity
# values
def compute_perplexity(result):
    result['perplexity'] = np.exp(result['main/loss'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchsize',
        '-b',
        type=int,
        default=20,
        help='Number of examples in each mini-batch')
    parser.add_argument(
        '--bproplen',
        '-l',
        type=int,
        default=35,
        help='Number of words in each mini-batch '
        '(= length of truncated BPTT)')
    parser.add_argument(
        '--epoch',
        '-e',
        type=int,
        default=39,
        help='Number of sweeps over the dataset to train')
    parser.add_argument(
        '--gpu',
        '-g',
        type=int,
        default=-1,
        help='GPU ID (negative value indicates CPU)')
    parser.add_argument(
        '--gradclip',
        '-c',
        type=float,
        default=5,
        help='Gradient norm threshold to clip')
    parser.add_argument(
        '--out', '-o', default='result', help='Directory to output the result')
    parser.add_argument(
        '--resume', '-r', default='', help='Resume the training from snapshot')
    parser.add_argument(
        '--test',
        action='store_true',
        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument(
        '--unit',
        '-u',
        type=int,
        default=650,
        help='Number of LSTM units in each layer')
    parser.add_argument(
        '--model',
        '-m',
        default='model.npz',
        help='Model file name to serialize')
    parser.add_argument(
        '--trainsplit', default=0.7, help='Fraction of dataset to train')
    parser.add_argument(
        '--testsplit',
        default=0.2,
        help='Fraction of dataset to be used for training')
    args = parser.parse_args()

    dataset = np.array(construct_dataset(7900)).astype(np.int32)
    len_dataset = len(dataset)
    train = dataset[:int(args.trainsplit * len_dataset)]
    val = dataset[int(args.trainsplit * len_dataset):int((
        args.trainsplit + args.testsplit) * len_dataset)]
    test = dataset[int((args.trainsplit + args.testsplit) * len_dataset):]
    n_vocab = max(train) + 1  # train is just an array of integers
    print('#vocab =', n_vocab)

    if args.test:
        train = train[:100]
        val = val[:100]
        test = test[:100]

    train_iter = ParallelSequentialIterator(train, args.batchsize)
    val_iter = ParallelSequentialIterator(val, 1, repeat=False)
    test_iter = ParallelSequentialIterator(test, 1, repeat=False)

    # Prepare an RNNLM model
    rnn = RNNForLM(n_vocab, args.unit)
    tree = BinaryHierarchicalSoftmax.create_huffman_tree(FreqDist(dataset))
    lossfn = BinaryHierarchicalSoftmax(n_vocab, tree)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        lossfn.to_gpu()

    model = L.Classifier(rnn, lossfun=lossfn)
    # model = L.Classifier(rnn)
    model.compute_accuracy = False  # we only want the perplexity
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Set up an optimizer
    optimizer = chainer.optimizers.RMSprop()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    # Set up a trainer
    updater = BPTTUpdater(train_iter, optimizer, args.bproplen, args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    eval_model = model.copy()  # Model with shared params and distinct states
    eval_rnn = eval_model.predictor
    trainer.extend(
        extensions.Evaluator(
            val_iter,
            eval_model,
            device=args.gpu,
            # Reset the RNN state at the beginning of each evaluation
            eval_hook=lambda _: eval_rnn.reset_state()))

    interval = 10 if args.test else 1000
    trainer.extend(
        extensions.LogReport(
            postprocess=compute_perplexity, trigger=(interval, 'iteration')))
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'perplexity', 'val_perplexity']),
        trigger=(interval, 'iteration'))
    trainer.extend(
        extensions.ProgressBar(update_interval=1 if args.test else 10))
    trainer.extend(extensions.snapshot(), trigger=(interval, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
        trigger=(interval, 'iteration'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, model)

    trainer.run()

    # Evaluate the final model
    print('test')
    eval_rnn.reset_state()
    evaluator = extensions.Evaluator(test_iter, eval_model, device=args.gpu)
    result = evaluator()
    print('test perplexity:', np.exp(float(result['main/loss'])))

    # Serialize the final model
    chainer.serializers.save_npz(args.model, model)


def construct_dataset(nvocab):
    # tokenizer = Tokenizer(num_words=nvocab)
    # texts = [' '.join(w.lower() for s in p for w in s ) for p in brown.paras()]
    # tokenizer.fit_on_texts(texts)
    # sequences = tokenizer.texts_to_sequences(texts)
    tokenizer = Tokenizer(num_words=nvocab)
    texts = [' '.join(w.lower() for w in s) for s in brown.sents()]
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    return [word for item in sequences for word in item]


def iterate_inf(dataset, n):
    pair_dataset = zip(dataset, dataset[1:])
    infinite_counter = cycle(pair_dataset)
    while True:
        yield take(n, infinite_counter)


def iterate(dataset, n):
    pair_dataset = zip(dataset, dataset[1:])
    for i in pair_dataset:
        yield take(n, pair_dataset)


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


if __name__ == '__main__':
    main()
