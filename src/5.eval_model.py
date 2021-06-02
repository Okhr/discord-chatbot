import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
from tensorflow.keras import Model
from tokenizers import Tokenizer
from tensorflow import keras

VOCAB_SIZE = 1000
MAX_SEQ_LENGTH = 32
RUN_NAME = "VOCAB1000-SEQ32-CELL8-BS16-1622667261"
CELL_SIZE = 8


def make_prediction(full_model, tokenizer, input_string: str):
    encoder_inputs = full_model.get_layer('encoder_inputs').input
    _, enc_state_h, enc_state_c = full_model.get_layer('encoder_lstm').output
    encoder_states = [enc_state_h, enc_state_c]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = full_model.get_layer('decoder_inputs').input
    decoder_state_input_h = keras.Input(shape=(CELL_SIZE,), name='input_h')
    decoder_state_input_c = keras.Input(shape=(CELL_SIZE,), name='input_c')
    decoder_lstm = full_model.get_layer('decoder_lstm')
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs,
                                                             initial_state=[decoder_state_input_h,
                                                                            decoder_state_input_c])
    decoder_dense = full_model.get_layer('dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model([decoder_inputs, decoder_state_input_h, decoder_state_input_c],
                                [decoder_outputs, state_h_dec, state_c_dec])

    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", length=MAX_SEQ_LENGTH)

    print(f"Raw input => {input_string}")
    input_encoding = tokenizer.encode(input_string)
    print(f"Tokenized input => {input_encoding.tokens[:input_encoding.tokens.index('[PAD]')]}")

    encoded_input = keras.utils.to_categorical(
        np.array([input_encoding.ids]),
        num_classes=VOCAB_SIZE,
        dtype='float16'
    )

    bos_id = tokenizer.encode('[BOS]').ids[0]
    eos_id = tokenizer.encode('[EOS]').ids[0]
    pad_id = tokenizer.encode('[PAD]').ids[0]

    input_token_id = np.zeros((1, 1, VOCAB_SIZE))
    input_token_id[0, 0, bos_id] = 1

    answer_token_ids = []

    states_value = encoder_model.predict([encoded_input])

    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([input_token_id, states_value])

        # predict next token
        next_token_id = np.argmax(output_tokens[0, -1, :])
        answer_token_ids.append(next_token_id)

        # end the inference if we hit a [EOF] token or max length
        if next_token_id == eos_id or len(answer_token_ids) > MAX_SEQ_LENGTH:
            stop_condition = True

        input_token_id = np.zeros((1, 1, VOCAB_SIZE))
        input_token_id[0, 0, next_token_id] = 1

        # update states
        states_value = [h, c]

    print(f"Model output => {tokenizer.decode(answer_token_ids)}")


if __name__ == '__main__':
    model = keras.models.load_model(f"models/{RUN_NAME}/final-model.hdf5")
    wordpiece_tokenizer = Tokenizer.from_file("data/tokenizer.json")

    make_prediction(model, wordpiece_tokenizer, "Salut à tous")
