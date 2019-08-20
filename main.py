print("Loading Libraries ... ")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import display, clear_output
from sklearn.model_selection import train_test_split as tts
import sys
import time
import os
import argparse
# %matplotlib inline
import pandas as pd
import matplotlib.image as img
from model import *
from model_trainer import *
from utils import *


# Parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-lr', '--lr', default = 0.01)
parser.add_argument('-tree_depth', '--tdepth', default = 2)
parser.add_argument('-conv_depth', '--cdepth', default = 2)
parser.add_argument('-sparsity', '--sty', default = 0.995)
parser.add_argument('-regularization', '--reg', default = 0.000001)

args = parser.parse_args()

print("Loading the Cifar 10 Dataset")

(Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.cifar10.load_data()

"""
one hot encoder for converting integer values to one hot vector
"""
from sklearn.preprocessing import LabelEncoder as LE 
from sklearn.preprocessing import OneHotEncoder as OHE
mo1 = LE()
mo2 = OHE()
Ytrain = mo2.fit_transform(mo1.fit_transform((Ytrain.ravel())).reshape(-1,1)).todense()
Ytest = mo2.transform(mo1.transform((Ytest.ravel())).reshape(-1,1)).todense()

# N, dDims = X_train.shape
N,W,H,C = Xtrain.shape
dDims = W*H*C
# nClasses = len(np.unique(Y_train))
nClasses = Ytrain.shape[1]
print('Training Size:',N,',Data Dims:', dDims,',No. Classes:', nClasses)

print("Preprocessing the dataset...")
Xtrain = preprocess(Xtrain)
Xtest = preprocess(Xtest)

writer = tf.summary.FileWriter('convbonsai')

"""
kernelsshp : kernel sizes for convolutional filters at each level in tree
strides : strides for convolutional layers at each level in tree
tDepth : bonsai tree depth after conv tree ends
cDepth : depth on convolutional tree
ch : number of channels in image
lW, lT, lV, lZ : regularization params
lr : learning rate
sZ,sW,sV,sT : sparsity constraints on params Z,W,V,T
"""
tf.reset_default_graph()
print("Creating the model graph and training graph..")
kernelsshp = [[4,4,3,3],[4,4,3,2],[3,3,2,1],[2,2,5,1],[3,3,1,1],[3,3,1,1],[3,3,1,1]]
strides = [[1,2,2,1],[1,2,2,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]

cdepth = args.cdepth
tdepth = args.tdepth
tree = ConvBonsai(nClasses = nClasses, dDims = dDims, pDims = 28, tDepth = tdepth, sigma = 1,
              kernelsshp = kernelsshp, strides = strides , cDepth = cdepth, ch = 3)

X = tf.placeholder("float32", [None, dDims])
Y = tf.placeholder("float32", [None, nClasses])
reg = args.reg
sty = args.sty
lrm = args.lr
bonsaiTrainer = ConvBonsaiTrainer(tree, lW = reg, lT = reg, lV = reg, lZ = reg, lr = lrm, X = X, Y = Y,
                              sZ = sty, sW = sty, sV = sty, sT = sty)
init_op = tf.global_variables_initializer()
print("Done Creating the graphs.")


print("Restoring/Initializing the model state...")
directory = "./bonsaiconv"
filename = directory + "/model"  #filename to save model
try:
    os.stat(directory)
except:
    os.mkdir(directory) 
with tf.name_scope('hidden') as scope:
    with tf.Session() as sess:
        saver = tf.train.Saver()
        try:
            saver.restore(sess, filename)
        except:
            sess.run(init_op)
            
###   __ uncomment if using tensorboard__
#         writer = tf.summary.FileWriter('convbonsai')
#         writer.add_graph(sess.graph)
        
        saver.save(sess, filename)
        totalEpochs = 10
        batchSize = np.maximum(1000, int(np.ceil(np.sqrt(Ytrain.shape[0]))))
        for i in range(5):
            bonsaiTrainer.train(batchSize, totalEpochs, sess, Xtrain, Xtest, Ytrain, Ytest, saver, filename,1)
            saver.save(sess, filename + str(i))
            print("Done sequence ", i)


image = Xtrain[:10000,:]
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, filename)
    var = calc_zero_ratios(tree)
    _feed_dict = {bonsaiTrainer.X:(image*255).astype(int).reshape(-1,dDims),bonsaiTrainer.sigmaI:float(1)}
    
    start = time.time()
    val = sess.run([tree.prediction, tree.wts, tree.bs, tree.convs, tree.kernelsT, tree.cnodeProb], feed_dict=_feed_dict)
    end = time.time()
    
    size = 0
    for i in range(len(val[1])):
        size += np.sum(val[1][i]>0.0000000001)
        size += np.sum(val[2][i]>0.0000000001)

    print('Number of non_zero paramters : ',var + size,' Time taken : ', end-start)


print("The super classification effect are as follow ..")
LL = []
LR = []
RR = []
RL = []
for i in range(len(val[0])):

    if(np.round(val[-1][:,:,i])[0] == 1 and np.round(val[-1][:,:,i])[2] == 1):
        LL.append(val[0][i])
    elif(np.round(val[-1][:,:,i])[0] == 1 and np.round(val[-1][:,:,i])[3] == 1):
        LR.append(val[0][i])
    elif(np.round(val[-1][:,:,i])[1] == 1 and np.round(val[-1][:,:,i])[4] == 1):
        RL.append(val[0][i])
    elif(np.round(val[-1][:,:,i])[1] == 1 and np.round(val[-1][:,:,i])[5] == 1):
        RR.append(val[0][i])
    else:
        pass


np.unique(RL, return_counts = True)
np.unique(LL, return_counts = True)
np.unique(RR, return_counts = True)
np.unique(LR, return_counts = True)