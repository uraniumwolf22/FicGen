from train import Train
from server import findGCP
from server import Server
import numpy as np
import sys
import os
checkpoint_dir = './data/checkpoints/'      #set the checkpoint directory
data_text = r'.\data\ao3_best.txt'      #set the directory of text to train off


to_gen = 600                #how many characters of text to generate
seq_length = 100             #sequence length for processing of text (larger = more context,  smaller = more precision)
learn_rate = 0.001          #how quickly the algorithm will make changes
epoch_num = 10000           #how many itterations
batch_sz = 64             #number of sequences to process concurrently
use_cuda = False             #should the algorithm use the optimized CUDNN algorithm

if '-train' in sys.argv:        #if we are training,  then start train algorithm
    print("Initializing train")
    Train(
        data_text=data_text,
        seq_length=seq_length,
        to_gen=to_gen,
        learn_rate=learn_rate,
        epoch_num=epoch_num,
        batch_sz=batch_sz,
        use_cuda=use_cuda
    )

else:                           #if we are not training then start the ficgen server
    data_model = findGCP()    #set directory of current NN model
    print("Initializing FicGenServer")
    Server(
        data_text=data_text,
        data_model=data_model,
        seq_length=seq_length,
        to_gen=to_gen
    )
