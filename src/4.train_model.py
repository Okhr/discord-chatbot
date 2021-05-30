import pickle
import time

from tensorflow import keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LSTM, Embedding, TimeDistributed
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

VOCAB_SIZE = 5000
MAX_SEQ_LENGTH = 128
EMBEDDING_SIZE = 200
CELL_SIZE = 128

BATCH_SIZE = 32

RUN_NAME = f"VOCAB{VOCAB_SIZE}-SEQ{MAX_SEQ_LENGTH}-EMBEDDING{EMBEDDING_SIZE}-CELL{CELL_SIZE}-BS{BATCH_SIZE}-" + str(
    int(time.time()))


def build_model(vocab_size, embedding_size, sequence_size, cell_size):
    encoder_inputs = Input(shape=(sequence_size,))
    encoder_embeddings = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=sequence_size)(
        encoder_inputs)
    _, state_h, state_c = LSTM(cell_size, return_state=True)(encoder_embeddings)

    decoder_inputs = Input(shape=(sequence_size,))
    decoders_embeddings = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=sequence_size)(
        decoder_inputs)
    decoder_outputs = LSTM(cell_size, return_sequences=True)(decoders_embeddings, initial_state=[state_h, state_c])

    outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], outputs)

    opt = keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(opt, loss=keras.losses.sparse_categorical_crossentropy, metrics=['acc'])

    return model


if __name__ == '__main__':
    with open('data/encoder_inputs', 'rb') as fd:
        enc_inputs = pickle.load(fd)

    with open('data/decoder_inputs', 'rb') as fd:
        dec_inputs = pickle.load(fd)

    with open('data/decoder_outputs', 'rb') as fd:
        dec_outputs = pickle.load(fd)

    print(f"Encoder inputs shape : {enc_inputs.shape}")
    print(f"Decoder inputs shape : {dec_inputs.shape}")
    print(f"Decoder outputs shape : {dec_outputs.shape}")

    full_model = build_model(VOCAB_SIZE, EMBEDDING_SIZE, MAX_SEQ_LENGTH, CELL_SIZE)

    # callbacks
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
                                       verbose=0)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor="val_acc",
                                             factor=0.6,
                                             patience=5,
                                             min_lr=10e-6,
                                             verbose=0)
    early_stopping = EarlyStopping(monitor="val_acc",
                                   patience=8,
                                   verbose=0)

    full_model.fit([enc_inputs, dec_inputs], dec_outputs, batch_size=BATCH_SIZE, epochs=5, validation_split=0.2,
                   callbacks=[tensorboard, model_checkpoint, reduce_lr_on_plateau, early_stopping])
