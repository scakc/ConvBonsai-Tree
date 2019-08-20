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

class ConvBonsai():
    def __init__(self, nClasses, dDims, pDims, tDepth, sigma, kernelsshp, strides, cDepth = 2, W=None, T=None, V=None, Z=None, ch = None):
        '''
        dDims : data Dimensions
        pDims : projected Dimesions
        nClasses : num Classes
        tDepth : tree Depth
        
        Expected Dimensions:
        --------------------
        Bonsai Params // Optional
        
        W [numClasses*totalNodes, projectionDimension]
        V [numClasses*totalNodes, projectionDimension]
        Z [projectionDimension, dataDimension + 1]
        T [internalNodes, projectionDimension]

        internalNodes = 2**treeDepth - 1
        totalNodes = 2*internalNodes + 1

        sigma - tanh non-linearity
        sigmaI - Indicator function for node probabilities
        sigmaI - has to be set to infinity(1e9 for practicality)
        
        while doing testing/inference
        numClasses will be reset to 1 in binary case
        '''
        
        # Initialization of parameter variables
        
        self.dDims = dDims
        self.pDims = pDims
        
        # If number of classes is two we dont need to calculate other class probability
        if nClasses == 2:
            self.nClasses = 1
        else:
            self.nClasses = nClasses

        self.tDepth = tDepth
        self.sigma = sigma
        self.iNodes = 2**self.tDepth - 1
        self.tNodes = 2*self.iNodes + 1
        
        self.cDepth = cDepth
        self.ciNodes = 2**self.cDepth - 1
        self.ctNodes = 2*self.ciNodes + 1
        
        
        self.kernelsT = []
        
        self.strides = []
        
        if(ch is None):
            ch = 3
        
        self.channels = ch    
        var = int(np.sqrt(self.dDims/self.channels))
        self.d1 = var
        self.d2 = var
        d1 = self.d1
        d2 = self.d2
        
        assert d1*d2*ch == self.dDims, " Dimension mismatch, doesn't seem like it's a image or set channel(ch) = 1"
        
        oD1 = d1
        oD2 = d2
        
        self.wts = []
        self.wts1 = []
        self.wts2 = []
        self.bs = []
        
        
        h = 0
        h_old = 0
        Codims1 = self.d1
        Codims2 = self.d2
        
        with tf.name_scope("Params"):
            for i in range(self.ctNodes):

                h = int(np.floor(np.log(i+1)/np.log(2)))

                self.kernelsT.append(
                    tf.get_variable('kernelT'+str(i), kernelsshp[h], 
                                 initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32),
                                 dtype=tf.float32)
                )

                self.strides.append(strides[h])


            for i in range(self.cDepth+1):
                Codims1 = np.floor((Codims1 - kernelsshp[i][0])/(strides[i][1])) + 1
                Codims2 = np.floor((Codims2 - kernelsshp[i][1])/(strides[i][2])) + 1


            self.CoDims = int(Codims1*Codims2) + 1
            self.pDims = self.CoDims
            self.Z = tf.Variable(tf.random_normal([2,2]), name='Z', dtype=tf.float32) 

            self.W = tf.Variable(tf.random_normal([self.ctNodes - self.ciNodes, self.nClasses * self.tNodes, self.pDims]), name='W', dtype=tf.float32)
            self.V = tf.Variable(tf.random_normal([self.ctNodes - self.ciNodes, self.nClasses * self.tNodes, self.pDims]), name='V', dtype=tf.float32)
            self.T = tf.Variable(tf.random_normal([self.ctNodes - self.ciNodes, self.iNodes, self.pDims]), name='T', dtype=tf.float32)
        
        
        self.score = None
        self.X_ = None
        self.prediction = None
        self.convs = []
        self.cnodeProb = []
        self.nodeProb = []
        self.scores = []
    
    def __call__(self, X, sigmaI):
        '''
        Function to build the Bonsai Tree graph
        
        Expected Dimensions
        -------------------
        X is [_, self.dDims]
        X_ is [_, self.pDims]
        '''
        errmsg = "Dimension Mismatch, X is [_, self.dataDimension]"
        assert (len(X.shape) == 2 and int(X.shape[1]) == self.dDims), errmsg
        
        sigmaI = tf.reshape(sigmaI, [1,1])
        
        # return score, X_ if exists where X_ is the projected X, i.e X_ = (Z.X)/(D^)
        if self.score is not None:
            return self.score, self.X_
        
        
        Ximg = tf.reshape(X, [-1,self.d1,self.d2,self.channels])
        
        self.convs = []
        
        
        # For Root Node score...
        self.__cnodeProb = [] # node probability list
        self.__cnodeProb.append(1) # probability of x passing through root is 1.
        
        with tf.name_scope('ConvNode'+str(0)):
        # All score sums variable initialized to root score... for each tree (Note: can be negative)
            convT = 0.1*tf.nn.leaky_relu(tf.nn.conv2d(Ximg,
                self.kernelsT[0],
                padding="VALID",
                strides = self.strides[0]), name = 'convT0')

            self.convs.append(convT)

            flatConv = tf.layers.Flatten()(convT)
            b = tf.squeeze(flatConv.shape[1])
            self.wts.append(tf.Variable(tf.random_normal([1, b]), name='wts' + str(0), dtype=tf.float32))
            self.bs.append(tf.Variable(tf.random_normal([1, 1]), name='bs' + str(0), dtype=tf.float32))

            finalImg =  None


            fscore_ = None
            fX_ = None
            self.__nodeProbs = []
        
        for i in range(1,self.ctNodes):
            with tf.name_scope('ConvNode'+str(i)):
                
                parent_id = int(np.ceil(i / 2.0) - 1.0)

                convTprev = self.convs[parent_id]
                flatConvP = tf.layers.Flatten()(convTprev)


                cscore = tf.multiply(sigmaI, tf.matmul(self.wts[parent_id], flatConvP, transpose_b = True) + self.bs[parent_id])# 1 x _

                # Calculating probability that x should come to this node next given it is in parent node...
                cprob = tf.divide((1 + ((-1)**(i + 1))*tf.tanh(cscore)),2.0) # : scalar 1 x_
                cprob = self.__cnodeProb[parent_id] * cprob # : scalar 1 x _


                # adding prob to node prob list
                self.__cnodeProb.append(cprob)

                convT = 0.1*tf.nn.leaky_relu(tf.nn.conv2d(convTprev,
                    self.kernelsT[i],
                    padding="VALID",
                    strides = self.strides[i]), name = 'convT' + str(i))

                self.convs.append(convT)

                flatConv = tf.layers.Flatten()(convT)
                b = tf.squeeze(flatConv.shape[1])

                self.wts.append(tf.Variable(tf.random_normal([1, b]), name='wts' + str(i), dtype=tf.float32))
                self.bs.append(tf.Variable(tf.random_normal([1, 1]), name='bs' + str(i), dtype=tf.float32))
            

            
            if(i+1 > self.ciNodes):
                # projected output of convolutional layers....
                
                iinum = i - self.ciNodes
                
                a,b = flatConv.shape
                onesmat = flatConv[:,0:1]*0 + 1

                flat_imgs = tf.concat([flatConv, onesmat], axis = 1)
   
                X_ = tf.transpose(flat_imgs)#tf.matmul(self.Z, flat_imgs, transpose_b = True)

                # For Root Node score...
                tnodeProb = [] # node probability list
                tnodeProb.append(cprob) # probability of x passing through root is 1.
                W_ = self.W[iinum, 0:(self.nClasses),:]# first K trees root W params : KxD^
                V_ = self.V[iinum, 0:(self.nClasses),:]# first K trees root V params : KxD^

                # All score sums variable initialized to root score... for each tree (Note: can be negative)
                score_ = tnodeProb[0]*tf.multiply(tf.matmul(W_, X_), tf.tanh(self.sigma * tf.matmul(V_, X_))) # : Kx_
                self.scores.append(flat_imgs)

                for t in range(1, self.tNodes):
                    with tf.name_scope('BonNode'+str(i)+str(t)):
                    # current node is i
                    # W, V of K different trees for current node
                        W_ = self.W[iinum,t * self.nClasses:((t + 1) * self.nClasses),:]# : KxD^
                        V_ = self.V[iinum,t * self.nClasses:((t + 1) * self.nClasses),:]# : KxD^


                        # i's parent node shared theta param reshaping to 1xD^
                        T_ = tf.reshape(self.T[iinum,int(np.ceil(t / 2.0) - 1.0),:],[-1, self.pDims])# : 1xD^

                        # Calculating probability that x should come to this node next given it is in parent node...
                        prob = tf.divide((1 + ((-1)**(t + 1))*tf.tanh(tf.multiply(sigmaI, tf.matmul(T_, X_)))),2.0) # : scalar 1x_

                        # Actual probability that x will come to this node...p(parent)*p(this|parent)...
                        prob = tnodeProb[int(np.ceil(t / 2.0) - 1.0)] * prob # : scalar 1x_

                        # adding prob to node prob list
                        tnodeProb.append(prob)
                        # New score addes to sum of scores...
                        score_ += tnodeProb[t]*tf.multiply(tf.matmul(W_, X_), tf.tanh(self.sigma * tf.matmul(V_, X_))) # Kx_

                self.scores.append(score_)
                self.__nodeProbs.append(tnodeProb[1:])

                if(fscore_ is None):
                    fscore_ = score_
                    fX_ = tf.matmul(T_, X_)*cprob
                else:
                    fscore_ = fscore_ + score_
                    fX_ = fX_ + tf.matmul(T_, X_)*cprob
            else:
                pass
                
        self.score = fscore_ 
        self.X_ = fX_
        self.nodeProb = tf.convert_to_tensor(self.__nodeProbs[:])
        self.cnodeProb = tf.convert_to_tensor(self.__cnodeProb[1:])
        self.layers = self.convs
        return self.score, self.X_
                
   
        
    
    def predict(self):
        '''
        Takes in a score tensor and outputs a integer class for each data point
        '''
        if self.prediction is not None:
            return self.prediction
        if self.nClasses > 2:
            self.prediction = tf.argmax(tf.transpose(self.score), 1) # score is 1xk
        else:
            self.prediction = tf.argmax(tf.concat([tf.transpose(self.score),0*tf.transpose(self.score)], 1), 1)
        return self.prediction

    def assert_params(self):
        
        # Asserting Initializaiton
        
        errRank = "All Parameters must has only two dimensions shape = [a, b]"
        assert len(self.W.shape) == len(self.Z.shape), errRank
        assert len(self.W.shape) == len(self.T.shape), errRank
        assert len(self.W.shape) == 2, errRank
        msg = "W and V should be of same Dimensions"
        assert self.W.shape == self.V.shape, msg
        errW = "W and V are [numClasses*totalNodes, projectionDimension]"
        assert self.W.shape[0] == self.nClasses * self.tNodes, errW
        assert self.W.shape[1] == self.pDims, errW
        errZ = "Z is [projectionDimension, dataDimension]"
        assert self.Z.shape[0] == self.pDims, errZ
        assert self.Z.shape[1] == self.dDims, errZ
        errT = "T is [internalNodes, projectionDimension]"
        assert self.T.shape[0] == self.iNodes, errT
        assert self.T.shape[1] == self.pDims, errT
        assert int(self.nClasses) > 0, "numClasses should be > 1"
        msg = "# of features in data should be > 0"
        assert int(self.dDims) > 0, msg
        msg = "Projection should be  > 0 dims"
        assert int(self.pDims) > 0, msg
        msg = "treeDepth should be >= 0"
        assert int(self.tDepth) >= 0, msg