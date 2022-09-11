from keras.models import load_model
import tensorflow as tf
import numpy as np
import websockets
import asyncio
import json
import os
import re

checkpoint_dir = './data/checkpoints/'      #set the checkpoint directory
def findGCP():          #function to find the greatest checkpoint
    filenums = []
    for file in os.listdir(checkpoint_dir):
        filenums.append(int(re.findall(r'\d+',file)[0]))
    model_GCP = './data/checkpoints/best_model-'+str(max(filenums)) + '.hdf5'
    print('Loading model ' + model_GCP)
    return model_GCP

def Server(data_text, data_model, seq_length=40, to_gen=600):   #define the main server function
    async def sample(preds):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    async def Generate(ws, prompt):     #define algorithm to generate FanFictions based on model
        to_gen = int(prompt[0])
        prompt = prompt[1]
        if len(prompt) < seq_length: truncated = prompt.rjust(seq_length)
        elif len(prompt) > seq_length: truncated = prompt[-seq_length:]
        else: truncated = prompt

        generated = ''
        sentence = truncated
        generated += sentence

        for i in range(to_gen):
            x_pred = np.zeros((1, seq_length, n_vocab))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_int[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = await sample(preds)  # Providing stochasticty
            next_char = int_to_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char  # move the window by 1 character

            await ws.send(json.dumps(next_char))    #send generated character accross network
            await asyncio.sleep(0)

        await ws.send(json.dumps('Done'))           #send the done command

    async def ClientCount(ws, path):                #detect number of clients?
        while True:
            running = int((len(asyncio.all_tasks())-1) / 5)
            await ws.send(json.dumps(running))
            await asyncio.sleep(1)

    # Listen for web client messages
    async def ClientRead(ws, path):                         #get prompt from web server
        running = int((len(asyncio.all_tasks())-1) / 5)
        print(f'{running} Clients Connected')

        while True:
            message = await ws.recv()           #wait for message to be recieved
            prompt = json.loads(message)        #unpack message

            if prompt != 'Connected': print(prompt)
            if type(prompt) == list: await Generate(ws, prompt)

    # Handle read/write on client connection
    async def SocketHandler(ws, path):
        read = asyncio.create_task(ClientRead(ws, path))
        count = asyncio.create_task(ClientCount(ws, path))
        done, pending = await asyncio.wait([read, count])
        error = read.exception() and count.exception()
        if error:
            if isinstance(error, websockets.exceptions.WebSocketException):
                if type(error) == websockets.exceptions.ConnectionClosedOK: print('Client Disconnected')
                else: print('Unexpected Disconnect!')
            else: raise error

    print('Initializing:')

    with open(data_text, 'r',encoding='utf-8') as doc:              #load text to generate off of (sample)
        raw_text = doc.read()
        doc.close()
    print('Text Loaded')

    raw_text = ''.join(c for c in raw_text if c.isascii() and not c.isdigit())
    chars = sorted(list(set(raw_text)))

    char_to_int = dict((c, i)for i, c in enumerate(chars))
    int_to_char = {v: k for k, v in char_to_int.items()}

    n_chars = len(raw_text)
    n_vocab = len(chars)
    print('Text Processed')

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

    model = load_model(data_model)
    print('Model Loaded')

    np.seterr(divide='ignore')  # Ignore div by zero.

    server = websockets.serve(SocketHandler, '0.0.0.0', 7654)       #start web socket
    print('Ready')

    asyncio.get_event_loop().run_until_complete(server)             #run server
    asyncio.get_event_loop().run_forever()


if __name__ == '__main__':
    data_text = r'.\data\ao3_best.txt'
    data_model = findGCP()
    Server(data_text, data_model)
