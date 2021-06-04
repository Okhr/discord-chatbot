import os
import pickle
import time
import yaml

from tensorflow import keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LSTM, Embedding, TimeDistributed
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def build_model(vocab_size, sequence_size, embedding_size, cell_size):
    encoder_inputs = Input(shape=(sequence_size,), name="encoder_inputs")
    encoder_embeddings = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=sequence_size,
                                   name="encoder_embeddings")(encoder_inputs)
    _, state_h, state_c = LSTM(cell_size, return_state=True, name="encoder_lstm")(encoder_embeddings)

    decoder_inputs = Input(shape=(sequence_size,), name="decoder_inputs")
    decoder_embeddings = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=sequence_size,
                                   name="decoder_embeddings")(decoder_inputs)
    decoder_outputs, _, _ = LSTM(cell_size, return_sequences=True, return_state=True, name="decoder_lstm")(
        decoder_embeddings, initial_state=[state_h, state_c])

    outputs = Dense(vocab_size, activation='softmax', name="dense")(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], outputs)

    opt = keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(opt, loss=keras.losses.sparse_categorical_crossentropy, metrics=['acc'])

    print(model.summary())
    # keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    return model


if __name__ == '__main__':

    with open('config.yaml') as f:
        params = yaml.load(f.read(), Loader=yaml.CLoader)

    with open('data/encoder_inputs', 'rb') as fd:
        enc_inputs = pickle.load(fd)

    with open('data/decoder_inputs', 'rb') as fd:
        dec_inputs = pickle.load(fd)

    with open('data/decoder_outputs', 'rb') as fd:
        dec_outputs = pickle.load(fd)

    print(f"Encoder inputs shape : {enc_inputs.shape}")
    print(f"Decoder inputs shape : {dec_inputs.shape}")
    print(f"Decoder outputs shape : {dec_outputs.shape}")

    RUN_NAME = f"VOCAB{params['Vocab']['size']}-SEQ{params['Vocab']['max_seq_length']}-EMBEDDING{params['Model']['embedding_size']}-CELL{params['Model']['cell_size']}-BS{params['Training']['batch_size']}-" + str(
        int(time.time()))

    full_model = build_model(params['Vocab']['size'], params['Vocab']['max_seq_length'],
                             params['Model']['embedding_size'], params['Model']['cell_size'])

    # callbacks
    if not os.path.exists(f"models/{RUN_NAME}"):
        os.makedirs(f"models/{RUN_NAME}")
    tensorboard = TensorBoard(log_dir="logs/" + RUN_NAME,
                              histogram_freq=1,
                              write_graph=True,
                              write_images=True,
                              embeddings_freq=1)
    model_checkpoint = ModelCheckpoint(filepath="models/" + RUN_NAME + "/epoch{epoch:04d}-val{val_acc:.4f}.hdf5",
                                       monitor="val_acc",
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode="max",
                                       verbose=1)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor="val_acc",
                                             factor=0.6,
                                             patience=5,
                                             min_lr=10e-6,
                                             verbose=0)
    early_stopping = EarlyStopping(monitor="val_acc",
                                   patience=8,
                                   verbose=0)

    full_model.fit([enc_inputs, dec_inputs],
                   dec_outputs,
                   batch_size=params['Training']['batch_size'],
                   epochs=params['Training']['epochs'],
                   validation_split=0.2,
                   callbacks=[tensorboard, model_checkpoint, reduce_lr_on_plateau, early_stopping])

    full_model.save(f"models/{RUN_NAME}/final-model.hdf5")
