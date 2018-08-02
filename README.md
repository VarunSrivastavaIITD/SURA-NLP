# SURA-NLP
Analysis and Optimisation of Text Generating Models

## Setup

- python 3.4+
- chainer
- numpy
- keras
- argparse
- nltk (with brown corpus data)

## File Synopsis

1. [`train_brown.py`](train_brown.py): Code for running base model (no parts of speech information)
    1. *class* **RNNForLM**: Main recurrent network taking one hot word vectors as input
    2. *class* **ParallelSequentialIterator**: Dataset iterator to create a batch of sequences at different positions
    3. *class* **BPTTUpdater**: Custom updater for truncated BackProp Through Time (BPTT)
    4. *function* **compute_perplexity**: Routine to rewrite the result dictionary of LogReport to add perplexity values