#!/usr/bin/env python

from __future__ import division, print_function

import argparse
from collections import Counter
from itertools import cycle, islice

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import training
from chainer.links import BinaryHierarchicalSoftmax
from chainer.training import extensions
from nltk import FreqDist
from nltk.corpus import brown

from custom_classifier import CustomClassifier

# from keras.preprocessing.text import Tokenizer


# Subnetwork for Rec2Network class
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

# Subnetwork for RecNetwork class


class RNNForBrown(chainer.Chain):
    def __init__(self, n_vocab, n_units, n_layers):
        super(RNNForBrown, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.lstm = StackedLSTMLayers(n_units, n_layers)
            # self.lstm = L.NStepLSTM(n_layers, n_units, n_units, 0.25)
            # self.l3 = L.Linear(n_units, n_vocab)

        # for param in self.l3.params():
        #     param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def reset_state(self):
        self.lstm.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.lstm(F.dropout(h0, ratio=.25))
        # y = self.l3(F.dropout(h1, ratio=.25))
        y = h1
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
    def __init__(self, n_units, n_layers, dropout_ratio=0.25):
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

#


class RecNetwork(chainer.Chain):
    def __init__(self,
                 n_vocab,
                 n_word_units,
                 n_word_layers,
                 n_pos,
                 n_pos_units,
                 n_pos_layers,
                 n_lin_layers,
                 lin_dropout=0.2,
                 use_hsmax=False,
                 n_lin_units=None):
        super(RecNetwork, self).__init__()

        if n_lin_units is None:
            n_lin_units = n_vocab + n_pos

        with self.init_scope():
            self.vocab_layer = RNNForBrown(n_vocab, n_word_units,
                                           n_word_layers)
            self.pos_layer = RNNForBrown(n_pos, n_pos_units, n_pos_layers)

            self.compress = L.Linear(in_size=None, out_size=n_lin_units)

            if n_lin_layers > 0:
                self.dense = StackedLinearLayers(n_lin_units, n_lin_layers,
                                                 lin_dropout)

            self.linear = L.Linear(n_lin_units, n_vocab)

        self.add_persistent('n_vocab', n_vocab)
        self.add_persistent('n_pos', n_pos)
        self.add_persistent('n_lin_layers', n_lin_layers)
        self.add_persistent('use_hsmax', use_hsmax)

        for param in self.linear.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def reset_state(self):
        self.vocab_layer.reset_state()
        self.pos_layer.reset_state()

    def __call__(self, x):
        # x1 = x[:self.n_vocab]
        # x2 = x[self.n_vocab:]

        if self.n_lin_layers > 0:
            h = self.dense(
                self.compress(
                    F.concat((F.softmax(self.vocab_layer(x[:, 0])), F.softmax(
                        self.pos_layer(x[:, 1]))))))
        else:
            h = self.compress(
                F.concat((F.softmax(self.vocab_layer(x[:, 0])), F.softmax(
                    self.pos_layer(x[:, 1])))))

        out = h if self.use_hsmax else self.linear(h)

        # if chainer.config.train is False:
        #     return F.softmax(out)

        return out


class Rec2Network(chainer.Chain):
    def __init__(self,
                 n_vocab,
                 n_word_units,
                 n_pos,
                 n_pos_units,
                 lin_dropout=0.25):
        super(Rec2Network, self).__init__()

        n_lin_units = n_vocab + n_pos

        self.add_persistent('n_vocab', n_vocab)
        self.add_persistent('n_pos', n_pos)

        with self.init_scope():
            self.wordlm = RNNForLM(n_vocab, n_word_units)
            self.poslm = RNNForLM(n_pos, n_pos_units)
            self.lin1 = L.Linear(n_lin_units, n_lin_units)
            self.lin2 = L.Linear(n_lin_units, n_lin_units)
            self.soft = L.Linear(n_lin_units, n_vocab)

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def reset_state(self):
        self.wordlm.reset_state()
        self.poslm.reset_state()

    def __call__(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        y1 = F.softmax(self.wordlm(x1))
        y2 = F.softmax(self.poslm(x2))

        y = F.concat((y1, y2))

        h0 = self.lin1(y)
        h1 = self.lin2(h0)

        out = self.soft(h1)

        return out


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
        '--vocabulary',
        '-V',
        type=int,
        default=7900,
        help='Size of Vocabulary')
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
        default=200,
        help='Warning: This Parameter is unused, \
         kept for compatibility \
        , Number of LSTM units in each layer')
    parser.add_argument(
        '--r1units',
        '-r1',
        type=int,
        default=100,
        help='Number of LSTM units in each word layer')
    parser.add_argument(
        '--r2units',
        '-r2',
        type=int,
        default=50,
        help='Number of LSTM units in each pos layer')
    parser.add_argument(
        '--dunits',
        '-d',
        type=int,
        default=150,
        help='Number of units in Dense Layer')
    parser.add_argument(
        '--use_hsmax',
        '-hs',
        type=bool,
        default=False,
        help='True if using hierarchical softmax')
    parser.add_argument(
        '--r1layers',
        '-r1l',
        type=int,
        default=1,
        help='Number of LSTM word layers')
    parser.add_argument(
        '--r2layers',
        '-r2l',
        type=int,
        default=1,
        help='Number of LSTM pos layers')
    parser.add_argument(
        '--dlayers', '-dl', type=int, default=0, help='Number of dense layers')
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

    dataset = np.array(construct_dataset(args.vocabulary)).astype(np.int32)
    len_dataset = len(dataset)
    train = dataset[:int(args.trainsplit * len_dataset)]
    val = dataset[int(args.trainsplit * len_dataset):int((
        args.trainsplit + args.testsplit) * len_dataset)]
    test = dataset[int((args.trainsplit + args.testsplit) * len_dataset):]
    # n_vocab = max(train) + 1  # train is just an array of integers
    word_set, pos_set = list(zip(*dataset))
    n_vocab = max(word_set) + 1
    n_pos = max(pos_set) + 1
    print('#vocab =', n_vocab)
    print('#pos =', n_pos)

    if args.test:
        train = train[:100]
        val = val[:100]
        test = test[:100]

    train_iter = ParallelSequentialIterator(train, args.batchsize)
    val_iter = ParallelSequentialIterator(val, 1, repeat=False)
    test_iter = ParallelSequentialIterator(test, 1, repeat=False)

    # Prepare an RNNLM model

    assert (args.r1units + args.r2units == args.dunits)

    # rnn = RecNetwork(n_vocab, args.r1units, args.r1layers, n_pos, args.r2units,
    #                  args.r2layers, args.dlayers, 0.3, args.use_hsmax)

    rnn = Rec2Network(n_vocab, args.r1units, n_pos, args.r2units)

    tree = BinaryHierarchicalSoftmax.create_huffman_tree(FreqDist(word_set))
    lossfun = BinaryHierarchicalSoftmax(n_vocab, tree)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        lossfun.to_gpu()

    if args.use_hsmax:
        model = CustomClassifier(rnn, lossfun=lossfun)
    else:
        model = CustomClassifier(rnn)
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


# def construct_dataset(nvocab):
# def num_there(s):
#     return any(i.isdigit() for i in s)

# tag_list = [
#     t[1] for t in brown.tagged_words()
#     if t[0] not in '!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n'
#     and not num_there(t[0])
# ]

# tokenizer = Tokenizer(
#     num_words=nvocab, filters='!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n')
# texts = [
#     ' '.join(w.lower() for w in s if not num_there(w))
#     for s in brown.sents()
# ]

# if __debug__:
#     pass
#     # print(len([w for t in texts for w in t.split()]))
#     # print(len(set([w for t in texts for w in t.split()])))

# tokenizer.fit_on_texts(texts)
# sequences = tokenizer.texts_to_sequences(texts)
# word_list = [word for item in sequences for word in item]

# if __debug__:
#     # from collections import Counter
#     print('Tag Len ', len(tag_list))
#     print('Word Len ', len(word_list))
#     # a = [
#     #     w for t in texts for w in t.split()
#     #     if w not in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
#     # ]
#     # sa = set(a)
#     # sb = set(word_list)
#     # c = Counter(a)
#     # csort = sorted(((b, a) for a, b in c.items()), reverse=True)

#     # ckeys = [i for i, (b, a) in enumerate(csort)]

#     # print(set(ckeys) - set(word_list))

# assert (len(tag_list) == len(word_list))

# return list(zip(word_list, tag_list))


def num_there(s):
    return any(i.isdigit() for i in s)


def construct_dataset(nvocab):

    dataset = [(a.lower(), b)
               for a, b in brown.tagged_words(tagset='universal')
               if a not in '!"#$%&()*+-,./:;<=>?@[\\]^_`{|}~\t\n']

    word_list, tag_list = zip(*dataset)
    word_set = set(word_list)
    tag_set = set(tag_list)

    tag_dict = {tag: i for i, tag in enumerate(tag_set)}

    c = Counter(word_list)
    c = dict(c.most_common(nvocab))

    inv_c = [(count, word) for word, count in c.items()]
    inv_c = sorted(inv_c, reverse=True)

    _, sorted_words = zip(*inv_c)

    word_dict = {word: i for i, word in enumerate(sorted_words)}

    result = [(word_dict[word], tag_dict[tag]) for word, tag in dataset
              if word in c]
    if __debug__:
        print('Num Words : {}, Total Words : {}'.format(
            len(word_set), len(word_list)))

    return result


if __name__ == '__main__':
    main()
