from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, CuDNNLSTM, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
import datetime
import random
import sys
import os
import re

def Train(data_text, seq_length=40, to_gen=600, learn_rate=0.001, epoch_num=500, batch_sz=1024, use_cuda=True):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20480)])
        except RuntimeError as e:
            print(e)

    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    print('Initializing:')

    checkpoint_dir = './data/checkpoints/'      #set the checkpoint directory

    with open(data_text, 'r',encoding='utf-8') as doc:      #open the text file
        raw_text = doc.read()
        doc.close()

    print('Text Loaded')

    raw_text = ''.join(c for c in raw_text if c.isascii() and not c.isdigit())      #strip text for use in processing
    chars = sorted(list(set(raw_text)))

    char_to_int = dict((c, i)for i, c in enumerate(chars))
    int_to_char = {v: k for k, v in char_to_int.items()}

    n_chars = len(raw_text)
    n_vocab = len(chars)

    step = 1
    sentences = []
    next_chars = []

    for i in range(0, len(raw_text) - seq_length, step):
        sentences.append(raw_text[i: i+seq_length])
        next_chars.append(raw_text[i+seq_length])

    x = np.zeros((len(sentences), seq_length, n_vocab), dtype=bool)
    y = np.zeros((len(sentences), n_vocab), dtype=bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_to_int[char]] = 1
        y[i, char_to_int[next_chars[i]]] = 1

    print('Text Processed')

    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

    filepath = './data/checkpoints/best_model-{epoch:02d}.hdf5'

    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )

    def findGCP():          #function to find the greatest checkpoint
        filenums = []
        for file in os.listdir(checkpoint_dir):
            filenums.append(int(re.findall(r'\d+',file)[0]))
        model_GCP = './data/checkpoints/best_model-'+str(max(filenums)) + '.hdf5'
        print('Loading model ' + model_GCP)
        return model_GCP

    if len(os.listdir(checkpoint_dir)):
        model = load_model(findGCP())

    if not len(os.listdir(checkpoint_dir)):
            model = Sequential([
                CuDNNLSTM(512, input_shape=(seq_length, n_vocab),  return_sequences=True),
                Dropout(0.2),
                CuDNNLSTM(512,return_sequences=True),
                Dropout(0.1),
                CuDNNLSTM(512,return_sequences=True),
                Dropout(0.05),
                CuDNNLSTM(512),
                Dense(n_vocab, activation='softmax')
                ])

    print('Training...')

    optimizer = Adam(learning_rate=learn_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks_list = [checkpoint, tensorboard_callback]
    history = model.fit(
        x, y,
        batch_size=batch_sz,
        epochs=epoch_num,
        callbacks=callbacks_list
        )


if __name__ == '__main__':
    data_text = r'.\data\ao3_best.txt'
    Train(data_text)
