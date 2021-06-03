import json
import yaml
import pickle
from pprint import pp
from typing import List

import numpy as np
from tensorflow import keras
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace


def learn_tokenizer(training_sentences: List[str], vocab_size, max_seq_length) -> Tokenizer:
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"])
    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", length=max_seq_length)
    tokenizer.enable_truncation(max_length=max_seq_length)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    tokenizer.train_from_iterator(training_sentences, trainer)
    tokenizer.save("data/tokenizer.json", pretty=True)

    return tokenizer


def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)


def encode_dataset(message_pairs, tokenizer, vocab_size) -> np.array:
    encoder_inputs = []
    decoder_inputs = []
    decoder_outputs = []
    for pair in message_pairs:
        encoder_inputs.append(pair[0])
        decoder_inputs.append(" ".join(["[BOS]", pair[1]]))
        decoder_outputs.append(" ".join([pair[1], "[EOS]"]))

    encoded_encoder_inputs = keras.utils.to_categorical(
        np.array([encoding.ids for encoding in tokenizer.encode_batch(encoder_inputs)]),
        num_classes=vocab_size,
        dtype='uint8'
    )

    encoded_decoder_inputs = keras.utils.to_categorical(
        np.array([encoding.ids for encoding in tokenizer.encode_batch(decoder_inputs)]),
        num_classes=vocab_size,
        dtype='uint8'
    )

    encoded_decoder_outputs = keras.utils.to_categorical(
        np.array([encoding.ids for encoding in tokenizer.encode_batch(decoder_outputs)]),
        num_classes=vocab_size,
        dtype='uint8'
    )

    return encoded_encoder_inputs, encoded_decoder_inputs, encoded_decoder_outputs


if __name__ == '__main__':

    with open('config.yaml') as f:
        params = yaml.load(f.read(), Loader=yaml.CLoader)

    with open('data/message_pairs.json', 'r') as fd:
        msg_pairs = json.load(fd)

    flattened_msg = [sentence for pair in msg_pairs for sentence in pair]
    learned_tokenizer = learn_tokenizer(flattened_msg, params['Vocab']['size'], params['Vocab']['max_seq_length'])

    dataset = encode_dataset(msg_pairs, learned_tokenizer, params['Vocab']['size'])

    with open("data/encoder_inputs", 'wb') as fd:
        pickle.dump(dataset[0], fd)

    with open("data/decoder_inputs", 'wb') as fd:
        pickle.dump(dataset[1], fd)

    with open("data/decoder_outputs", 'wb') as fd:
        pickle.dump(dataset[2], fd)

    print(dataset[0].shape)
    print(dataset[1].shape)
    print(dataset[2].shape)
