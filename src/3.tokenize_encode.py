import json
import pickle
from pprint import pp
from typing import List

import numpy as np
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

VOCAB_SIZE = 5000
MAX_SEQ_LENGTH = 128


def learn_tokenizer(training_sentences: List[str]) -> Tokenizer:
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    trainer = WordPieceTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"])
    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", length=MAX_SEQ_LENGTH)
    tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    tokenizer.train_from_iterator(training_sentences, trainer)
    tokenizer.save("data/tokenizer.json", pretty=True)

    return tokenizer


def encode_dataset(message_pairs, tokenizer) -> np.array:
    encoder_inputs = []
    decoder_inputs = []
    decoder_outputs = []
    for pair in message_pairs:
        encoder_inputs.append(pair[0])
        decoder_inputs.append(" ".join(["[BOS]", pair[1]]))
        decoder_outputs.append(" ".join([pair[1], "[EOS]"]))

    encoded_encoder_inputs = np.expand_dims(np.array([encoding.ids for encoding in tokenizer.encode_batch(encoder_inputs)]), axis=-1)
    encoded_decoder_inputs = np.expand_dims(np.array([encoding.ids for encoding in tokenizer.encode_batch(decoder_inputs)]), axis=-1)
    encoded_decoder_outputs = np.expand_dims(np.array([encoding.ids for encoding in tokenizer.encode_batch(decoder_outputs)]), axis=-1)

    return encoded_encoder_inputs, encoded_decoder_inputs, encoded_decoder_outputs


if __name__ == '__main__':
    with open('data/message_pairs.json', 'r') as fd:
        msg_pairs = json.load(fd)

    flattened_msg = [sentence for pair in msg_pairs for sentence in pair]
    learned_tokenizer = learn_tokenizer(flattened_msg)

    dataset = encode_dataset(msg_pairs, learned_tokenizer)

    with open("data/encoder_inputs", 'wb') as fd:
        pickle.dump(dataset[0], fd)

    with open("data/decoder_inputs", 'wb') as fd:
        pickle.dump(dataset[1], fd)

    with open("data/decoder_outputs", 'wb') as fd:
        pickle.dump(dataset[2], fd)

    print(dataset[0].shape)
    print(dataset[1].shape)
    print(dataset[2].shape)

    '''
    output = learned_tokenizer.encode("(vous l'avez reçu ? 😗 on a reçu le vôtre en plus du nôtre)")
    print(output.tokens)
    print(output.ids)
    print(learned_tokenizer.decode(output.ids))
    print(learned_tokenizer.encode("a b").ids)
    '''
