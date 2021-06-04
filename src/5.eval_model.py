import os
import time
from stat import ST_MTIME

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from termcolor import colored
import yaml
import numpy as np
from tensorflow.keras import Model
from tokenizers import Tokenizer
from tensorflow import keras


def make_prediction(full_model, tokenizer, input_string: str, vocab_size, max_seq_length, cell_size):
    encoder_inputs = full_model.get_layer('encoder_inputs').input
    _, enc_state_h, enc_state_c = full_model.get_layer('encoder_lstm').output
    encoder_states = [enc_state_h, enc_state_c]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = full_model.get_layer('decoder_inputs').input
    decoder_embeddings_outputs = full_model.get_layer('decoder_embeddings').output
    decoder_state_input_h = keras.Input(shape=(cell_size,), name='input_h')
    decoder_state_input_c = keras.Input(shape=(cell_size,), name='input_c')
    decoder_lstm = full_model.get_layer('decoder_lstm')
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_embeddings_outputs,
                                                             initial_state=[decoder_state_input_h,
                                                                            decoder_state_input_c])
    decoder_dense = full_model.get_layer('dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model([decoder_inputs, decoder_state_input_h, decoder_state_input_c],
                                [decoder_outputs, state_h_dec, state_c_dec])

    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", length=max_seq_length)

    # print(f"Raw input => {input_string}")
    input_encoding = tokenizer.encode(input_string)
    # print(f"Tokenized input => {input_encoding.tokens[:input_encoding.tokens.index('[PAD]')]}")

    encoded_input = np.array([input_encoding.ids])

    bos_id = tokenizer.encode('[BOS]').ids[0]
    eos_id = tokenizer.encode('[EOS]').ids[0]
    pad_id = tokenizer.encode('[PAD]').ids[0]

    input_token_id = np.array([[bos_id]])
    answer_token_ids = []

    states_value = encoder_model.predict([encoded_input])

    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([input_token_id] + states_value)

        # predict next token
        next_token_id = np.argmax(output_tokens[0, -1, :])
        answer_token_ids.append(next_token_id)

        # end the inference if we hit a [EOF] token or max length
        if next_token_id == eos_id or len(answer_token_ids) > max_seq_length:
            stop_condition = True

        input_token_id = np.array([[next_token_id]])

        # update states
        states_value = [h, c]

    return tokenizer.decode(answer_token_ids)


if __name__ == '__main__':

    with open('config.yaml') as f:
        params = yaml.load(f.read(), Loader=yaml.CLoader)

    if params['Evaluation']['run_name'] == 'latest':
        entries = [(os.path.join('models', mod), mod) for mod in os.listdir('models')]
        entries = sorted([(os.stat(path)[ST_MTIME], path, mod) for path, mod in entries],
                         key=lambda x: x[0],
                         reverse=True)
        run_name = entries[0][2]
    else:
        run_name = params['Evaluation']['run_name']

    # print(f"Run name => {run_name}")

    model = keras.models.load_model(f"models/{run_name}/final-model.hdf5")
    wordpiece_tokenizer = Tokenizer.from_file("data/tokenizer.json")

    while True:
        try:
            user_input = str(input(colored("#>", 'cyan')))
            model_output = make_prediction(model,
                                           wordpiece_tokenizer,
                                           user_input,
                                           params['Vocab']['size'],
                                           params['Vocab']['max_seq_length'],
                                           params['Model']['cell_size'])
            print(colored(model_output, 'yellow'))
            print()
        except KeyboardInterrupt:
            print(colored('\nBye !', 'red'))
            exit(0)
