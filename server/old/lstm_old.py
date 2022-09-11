from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
import sys
import os


train = True
seq_length = 40
learn_rate = 0.001
batch_sz = 4096
epoch_num = 10000
to_gen = 6000

with open(r"ao3.txt", "r",encoding='utf-8') as doc:
    raw_text = doc.read()
    doc.close()
# print(raw_text)

# it's a long preprocessed string.
raw_text = "".join(c for c in raw_text if c.isascii() and not c.isdigit())
# Taking only the unique characters with other special characters also.
chars = sorted(list(set(raw_text)))


char_to_int = dict((c, i)for i, c in enumerate(chars))
int_to_char = {v: k for k, v in char_to_int.items()}

n_chars = len(raw_text)
n_vocab = len(chars)


step = 1  # How far we will take the steps.
sentences = []  # x
next_chars = []  # y


# Creating list of sentences and next_chars (x and y)
for i in range(0, len(raw_text) - seq_length, step):
    sentences.append(raw_text[i: i+seq_length])
    next_chars.append(raw_text[i+seq_length])


# Preparing the text to be sutiable as an input to the model (matrix representation).
x = np.zeros((len(sentences), seq_length, n_vocab), dtype=np.bool)
y = np.zeros((len(sentences), n_vocab), dtype=np.bool)


for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_int[char]] = 1
    y[i, char_to_int[next_chars[i]]] = 1


# Model checkpoint

filepath = "checkpoints/best_model-{epoch:02d}.hdf5"

checkpoint = ModelCheckpoint(filepath,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
callbacks_list = [checkpoint]

toLoad = 0
greatest_checkpoint = 0
use_pre = False

def remz(thing):
    thing = str(thing)
    if thing.startswith("0"):
        thing.replace("0","")
        return int(thing)
    else:
        return int(thing)


if len(os.listdir("checkpoints/")) >= 1:
    for checkpoint in os.listdir("checkpoints/"):
        checkpoint = checkpoint.replace("best_model-","")
        checkpoint = checkpoint.replace(".hdf5","")
        pre = checkpoint
        if checkpoint.startswith("0"):
            checkpoint.replace("0","")
            use_pre = True
        checkpoint = int(checkpoint)
        if checkpoint > remz(greatest_checkpoint):
            greatest_checkpoint = checkpoint
            if use_pre:
                greatest_checkpoint = pre


if len(os.listdir("checkpoints/")) >= 1:
    model = load_model("checkpoints/best_model-"+str(greatest_checkpoint)+".hdf5")

if train:
    if len(os.listdir("checkpoints/")) == 0:
        model = Sequential([
            LSTM(256, input_shape=(seq_length, n_vocab),  return_sequences=True),
            Dropout(0.2),
            LSTM(256),
            Dense(n_vocab, activation='softmax')
        ])

    optimizer = Adam(learning_rate=learn_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)

    history = model.fit(x, y,
                        #Batch size
                        batch_size=batch_sz,
                        epochs=epoch_num,
                        callbacks=callbacks_list)


# For testing the trained model.

start_index = random.randint(0, n_chars - seq_length - 1)
generated = ''
# Getting a random sentence from the text
sentence = raw_text[start_index: start_index + seq_length]
generated += sentence
print('Input sequence:"' + sentence + '"\n')

# For inducing stochasticty.


def sample(preds):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


np.seterr(divide='ignore')  # ignore the warning of divide by zero.


for i in range(to_gen):   # Number of characters including spaces
    x_pred = np.zeros((1, seq_length, n_vocab))
    for t, char in enumerate(sentence):
        # Preparing the x we want to predict as we have done for training. The full sentence is in shape of (1, 60, 69)
        x_pred[0, t, char_to_int[char]] = 1.

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds)  # Providing stochasticty
    # next_index = np.random.choice(y.shape[1], 1, p=preds)[0] # Another way to choose index with stochasticty and providing the probability distribution of the preds
    next_char = int_to_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char  # move the window by 1 character

    sys.stdout.write(next_char)
    sys.stdout.flush()
print()
