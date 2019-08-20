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

class ConvBonsaiTrainer():
    
    def __init__(self, tree, lW, lT, lV, lZ, lr, X, Y, sW, sV, sZ, sT):
        
        '''
        bonsaiObj - Initialised Bonsai Object and Graph...
        lW, lT, lV and lZ are regularisers to Bonsai Params...
        sW, sT, sV and sZ are sparsity factors to Bonsai Params...
        lr - learningRate fro optimizer...
        X is the Data Placeholder - Dims [_, dataDimension]
        Y - Label placeholder for loss computation
        useMCHLoss - For choice between HingeLoss vs CrossEntropy
        useMCHLoss - True - MultiClass - multiClassHingeLoss
        useMCHLoss - False - MultiClass - crossEntropyLoss
        '''
        #  Intializations of training parameters
        self.tree = tree
        
        # regularization params lambdas(l) (all are scalars)
        self.lW = lW
        self.lV = lV
        self.lT = lT
        self.lZ = lZ

        # sparsity parameters (scalars all...) will be used to calculate percentiles to make other cells zero
        self.sW = sW 
        self.sV = sV
        self.sT = sT
        self.sZ = sZ

        # placeholders for inputs and labels
        self.Y = Y # _ x nClasses
        self.X = X # _ x D
        
        # learning rate
        self.lr = lr
        
        # Asserting initialization
        self.assert_params()
        
        # place holder for path selection parameter sigmaI
        self.sigmaI = tf.placeholder(tf.float32, name='sigmaI')
        # invoking __call__ of tree getting initial values of score and projected X
        self.score, self.X_ = self.tree(self.X, self.sigmaI)
        # defining loss function tensorflow graph variables.....
        self.loss, self.marginLoss, self.regLoss = self.lossGraph()
        # defining single training step graph process ...
        self.tree.TrainStep = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.trainStep = self.tree.TrainStep
        # defining accuracy and prediction graph objects
        self.accuracy = self.accuracyGraph()
        self.prediction = self.tree.predict()
        
        
        # set all parameters above 0.99 if dont want to use IHT
        if self.sW > 0.99 and self.sV > 0.99 and self.sZ > 0.99 and self.sT > 0.99:
            self.isDenseTraining = True
        else:
            self.isDenseTraining = False
            
        # setting the hard thresholding graph obejcts
        self.hardThrsd()
        
    def hardThrsd(self):
        '''
        Set up for hard Thresholding Functionality
        '''
        with tf.name_scope("IHT"):
            # place holders for sparse parameters....
            self.__Wth = tf.placeholder(tf.float32, name='Wth')
            self.__Vth = tf.placeholder(tf.float32, name='Vth')
            self.__Zth = tf.placeholder(tf.float32, name='Zth')
            self.__Tth = tf.placeholder(tf.float32, name='Tth')

            # assigning the thresholded values to params as a graph object for tensorflow....
            self.__Woph = self.tree.W.assign(self.__Wth)
            self.__Voph = self.tree.V.assign(self.__Vth)
            self.__Toph = self.tree.T.assign(self.__Tth)
            self.__Zoph = self.tree.Z.assign(self.__Zth)

            # grouping the graph objects as one object....
            self.hardThresholdGroup = tf.group(
                self.__Woph, self.__Voph, self.__Toph, self.__Zoph)
        
    def hardThreshold(self, A, s):
        '''
        Hard thresholding function on Tensor A with sparsity s
        '''
        # copying to avoid errors....
        A_ = np.copy(A)
        # flattening the tensor...
        A_ = A_.ravel()
        if len(A_) > 0:
            # calculating the threshold value for sparse limit...
            th = np.percentile(np.abs(A_), (1 - s) * 100.0, interpolation='higher')
            # making sparse.......
            A_[np.abs(A_) < th] = 0.0
        # reconstructing in actual shape....
        A_ = A_.reshape(A.shape)
        return A_

    def accuracyGraph(self):
        '''
        Accuracy Graph to evaluate accuracy when needed
        '''
        with tf.name_scope("ACC"):
            if (self.tree.nClasses > 2):
                correctPrediction = tf.equal(tf.argmax(tf.transpose(self.score), 1), tf.argmax(self.Y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
            else:
                # some accuracy functional analysis for 2 classes could be different from this...
                y_ = self.Y * 2 - 1
                correctPrediction = tf.multiply(tf.transpose(self.score), y_)
                correctPrediction = tf.nn.relu(correctPrediction)
                correctPrediction = tf.ceil(tf.tanh(correctPrediction)) # final predictions.... round to(0 or 1)
                self.accuracy = tf.reduce_mean(
                    tf.cast(correctPrediction, tf.float32))

        return self.accuracy
        
    
    def lossGraph(self):
        '''
        Loss Graph for given tree
        '''
        with tf.name_scope("Loss"):
            # regularization losses.....
            self.regLoss = 0.5 * (self.lZ * tf.square(tf.norm(self.tree.Z)) +
                              self.lW * tf.square(tf.norm(self.tree.W)) +
                              self.lV * tf.square(tf.norm(self.tree.V)) +
                              self.lT * tf.square(tf.norm(self.tree.T)))

            llen = self.tree.ciNodes
            var = 0
            for i in range(llen):
                var = var +  self.lT * tf.square(tf.norm(self.tree.wts[i]))

            self.regLoss = self.regLoss + var

            # emperical actual loss.....
            if (self.tree.nClasses > 2):
                '''
                Cross Entropy loss for MultiClass case in joint training for
                faster convergence
                '''
                # cross entropy loss....
                self.marginLoss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.transpose(self.score),
                                                                   labels=tf.stop_gradient(self.Y)))
            else:
                # sigmoid loss....
                self.marginLoss = tf.reduce_mean(tf.nn.relu(1.0 - (2 * self.Y - 1) * tf.transpose(self.score)))

            # adding the losses...
            self.loss = self.marginLoss + self.regLoss
        return self.loss, self.marginLoss, self.regLoss
        
    def assert_params(self):
        # asserting the initialization....
        err = "sparsity must be between 0 and 1"
        assert self.sW >= 0 and self.sW <= 1, "W " + err
        assert self.sV >= 0 and self.sV <= 1, "V " + err
        assert self.sZ >= 0 and self.sZ <= 1, "Z " + err
        assert self.sT >= 0 and self.sT <= 1, "T " + err
        errMsg = "Dimension Mismatch, Y has to be [_, " + str(self.tree.nClasses) + "]"
        errCont = " numClasses are 1 in case of Binary case by design"
        assert (len(self.Y.shape) == 2 and self.Y.shape[1] == self.tree.nClasses), errMsg + errCont
        
        

        
    def train(self, batchSize, totalEpochs, sess, Xtrain, Xval, Ytrain, Yval, saver, filename,valsig):
        iht = 0 # to keep a note if thresholding has been started ...
        numIters = Xtrain.shape[0] / batchSize # number of batches at a time...
        totalBatches = numIters * totalEpochs # total number of batch operations...
        treeSigmaI = valsig # controls the fidelity of the approximation too high can saturate tanh.
            
        maxTestAcc = -10000
        itersInPhase = 0
        
        for i in range(totalEpochs):
            print("\nEpoch Number: " + str(i))
            # defining training acc and loss
            trainAcc = 0.0
            trainAccOld = 0.0
            trainLoss = 0.0
            trainBest = 0.0
            
            numIters = int(numIters)
            
            for j in range(numIters):
                # creating batch.....sequentiall could be done randomly using choice function...
                mini_batchX = Xtrain[j*batchSize:(j+1)*batchSize,:] # B x D
                mini_batchY = Ytrain[j*batchSize:(j+1)*batchSize] # B x 
            
                # feed for training using tensorflow graph based gradient descent approach......
                _feed_dict = {self.X: mini_batchX, self.Y: mini_batchY,
                                  self.sigmaI: treeSigmaI}
                
                # training the tensorflow graph
                _, batchLoss, batchAcc = sess.run(
                    [self.trainStep, self.loss, self.accuracy],
                    feed_dict=_feed_dict)
                
                # calculating acc....
                trainAcc += batchAcc
                trainLoss += batchLoss
                
                
                
                # to update sigmaI.....
                if ((itersInPhase) % 100 == 0):
                    
                    # Making a random batch....
                    indices = np.random.choice(Xtrain.shape[0], 100)
                    rand_batchX = Xtrain[indices, :]
                    rand_batchY = Ytrain[indices, :]
                    rand_batchY = np.reshape(rand_batchY, [-1, self.tree.nClasses])

                    _feed_dict = {self.X: rand_batchX,
                                  self.sigmaI: treeSigmaI}
                    # Projected matrix...
                    Xcapeval = self.X_.eval(feed_dict=_feed_dict) # D^ x 1
                    sum_tr = 0.0 
                    for k in range(0, self.tree.iNodes):
                        sum_tr += (np.sum(np.abs(Xcapeval)))

                    
                    if(self.tree.iNodes > 0):
                        sum_tr /= (self.tree.iNodes) # normalizing all sums
                        sum_tr = 1 / sum_tr # inverse of average sum
                    else:
                        sum_tr = 0.1
                    # thresholding inverse of sum as min(1000, sum_inv*2^(cuurent batch number / total bacthes / 30))
                    sum_tr = min(
                        1000, sum_tr * (2**(float(itersInPhase) /
                                            (float(totalBatches) )))*valsig/30)
                    # assiging higher values as convergence is reached...
                    treeSigmaI = max(sum_tr, treeSigmaI)
                    
                itersInPhase+=1
                
                
                # to start hard thresholding after half_time(could vary) ......
                if((itersInPhase//numIters > (1/2)*totalEpochs) and (not self.isDenseTraining)):
                    if(iht == 0):
                        print('\n\nHard Thresolding Started\n\n')
                        iht = 1
                    
                    # getting the current estimates of  W,V,Z,T...
                    currW = self.tree.W.eval()
                    currV = self.tree.V.eval()
                    currZ = self.tree.Z.eval()
                    currT = self.tree.T.eval()

                    # Setting a method to make some values of matrix zero....
                    self.__thrsdW = self.hardThreshold(currW, self.sW)
                    self.__thrsdV = self.hardThreshold(currV, self.sV)
                    self.__thrsdZ = self.hardThreshold(currZ, self.sZ)
                    self.__thrsdT = self.hardThreshold(currT, self.sT)

                    # runnign the hard thresholding graph....
                    fd_thrsd = {self.__Wth: self.__thrsdW, self.__Vth: self.__thrsdV,
                                self.__Zth: self.__thrsdZ, self.__Tth: self.__thrsdT}
                    sess.run(self.hardThresholdGroup, feed_dict=fd_thrsd)
                    
            
            
            print("Train Loss: " + str(trainLoss / numIters) +
                  " Train accuracy: " + str(trainAcc / numIters))
            print("SigmaI :",treeSigmaI,":LR:",self.lr)
            
            # calculating the test accuracies with sigmaI as expected -> inf.. = 10^9
            oldSigmaI = treeSigmaI
            treeSigmaI = 1e9
            
            # test feed for tf...
            _feed_dict = {self.X: Xval, self.Y: Yval,
                                  self.sigmaI: treeSigmaI}
            
            # calculating losses....
            testAcc, testLoss, regTestLoss = sess.run([self.accuracy, self.loss, self.regLoss], feed_dict=_feed_dict)
            
            
            if maxTestAcc <= testAcc:
                maxTestAccEpoch = i
                maxTestAcc = testAcc
                saver.save(sess, filename + "/model_best")
                
            
            print("Test accuracy %g" % testAcc)
            print("MarginLoss + RegLoss: " + str(testLoss - regTestLoss) +
                  " + " + str(regTestLoss) + " = " + str(testLoss) + "\n", end='\r')
            
            
            treeSigmaI = oldSigmaI
            
        # sigmaI has to be set to infinity to ensure
        # only a single path is used in inference
        treeSigmaI = 1e9
        print("\nMaximum Test accuracy at compressed" +
              " model size(including early stopping): " +
              str(maxTestAcc) + " at Epoch: " +
              str(maxTestAccEpoch + 1) + "\nFinal Test" +
              " Accuracy: " + str(testAcc))