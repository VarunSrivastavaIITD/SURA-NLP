from __future__ import division, print_function

import argparse
import copy
import sys
from collections import Counter
from itertools import cycle, islice

import chainer
import chainer.computational_graph as c
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import serializers, training
from chainer.dataset import convert
from chainer.links import BinaryHierarchicalSoftmax
from chainer.training import extensions
from keras.preprocessing.text import Tokenizer
from nltk import FreqDist
from nltk.corpus import brown

from custom_classifier import CustomClassifier
from posmod import (BPTTUpdater, ParallelSequentialIterator, RecNetwork,
                    compute_perplexity)


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
        default=6500,
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
        default=50,
        help='Number of LSTM units in each word layer')
    parser.add_argument(
        '--r2units',
        '-r2',
        type=int,
        default=20,
        help='Number of LSTM units in each pos layer')
    parser.add_argument(
        '--dunits',
        '-d',
        type=int,
        default=40,
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
        '--dlayers', '-dl', type=int, default=1, help='Number of dense layers')
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

    n_lin_units = args.dunits

    dataset = np.array(construct_dataset(args.vocabulary)).astype(np.int32)
    # dataset = dataset[:int(0.4 * len(dataset))]
    # wordset = np.random.randint(1, 7901, size=(1000, ))
    # tagset = np.random.randint(1, 11, size=(1000, ))
    # dataset = np.dstack((wordset, tagset)).reshape(-1, 2).astype(np.int32)

    len_dataset = len(dataset)
    train = dataset[:int(args.trainsplit * len_dataset)]
    val = dataset[int(args.trainsplit * len_dataset):int((
        args.trainsplit + args.testsplit) * len_dataset)]
    test = dataset[int((args.trainsplit + args.testsplit) * len_dataset):]
    # n_vocab = max(train) + 1  # train is just an array of integers
    word_set, pos_set = list(zip(*dataset))
    n_vocab = max(word_set)
    n_pos = max(pos_set)
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

    # assert (args.r1units + args.r2units == args.dunits)

    rnn = RecNetwork(n_vocab, args.r1units, args.r1layers, n_pos, args.r2units,
                     args.r2layers, args.dlayers, 0.3, args.use_hsmax,
                     n_lin_units)

    # batch = train_iter.__next__()
    # x, t = convert.concat_examples(batch, args.gpu)

    # vs = rnn(chainer.Variable(x))
    # g = c.build_computational_graph(vs)
    # with open('Visualization/rec_network_rnn', 'w') as o:
    #     o.write(g.dump())

    # sys.exit(0)

    if args.use_hsmax:
        tree = BinaryHierarchicalSoftmax.create_huffman_tree(
            FreqDist(word_set))
        lossfun = BinaryHierarchicalSoftmax(n_lin_units, tree)

    if args.use_hsmax:
        model = CustomClassifier(rnn, lossfun=lossfun)
    else:
        model = CustomClassifier(rnn)

    model.compute_accuracy = False  # we only want the perplexity
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        if args.use_hsmax:
            lossfun.to_gpu()
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

    interval = 10 if args.test else 500
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

    dataset = [(a.lower(), b)
               for a, b in brown.tagged_words(tagset='universal')
               if a not in '!"#$%&()*+-,./:;<=>?@[\\]^_`{|}~\t\n']

    word_list, tag_list = zip(*dataset)
    word_set = set(word_list)
    tag_set = set(tag_list)

    tag_dict = {tag: i + 1 for i, tag in enumerate(tag_set)}

    c = Counter(word_list)
    c = dict(c.most_common(nvocab))

    inv_c = [(count, word) for word, count in c.items()]
    inv_c = sorted(inv_c, reverse=True)

    _, sorted_words = zip(*inv_c)

    word_dict = {word: i + 1 for i, word in enumerate(sorted_words)}

    result = [(word_dict[word], tag_dict[tag])
              if word in c else (0, tag_dict[tag]) for word, tag in dataset]
    if __debug__:
        print('Num Words : {}, Total Words : {}'.format(
            len(word_set), len(word_list)))

    return result


def evaluate(model, iter, gpu):
    # Evaluation routine to be used for validation and test.
    model.predictor.train = False
    evaluator = model.copy()  # to use different state
    evaluator.predictor.reset_state()  # initialize state
    evaluator.predictor.train = False  # dropout does nothing
    sum_perp = 0
    data_count = 0
    for batch in copy.copy(iter):
        x, t = convert.concat_examples(batch, gpu)
        loss = evaluator(x, t)
        sum_perp += loss.data
        data_count += 1
    model.predictor.train = True
    return np.exp(float(sum_perp) / data_count)


if __name__ == "__main__":
    main()
