# Preprocessing files
import numpy as np
import tensorflow as tf

def preprocess(x):
    z = (x - x.mean(axis=(0,1,2), keepdims=True)) / x.std(axis=(0,1,2), keepdims=True)
    N, W, H, X = z.shape
    return z.reshape(N, -1)


def calc_zero_ratios(tree):
    zs = np.sum(np.abs(tree.Z.eval())>0.000000000000001)
    ws = np.sum(np.abs(tree.W.eval())>0.000000000000001)
    vs = np.sum(np.abs(tree.V.eval())>0.000000000000001)
    ts = np.sum(np.abs(tree.T.eval())>0.000000000000001)
    print('Number of non zeros achieved...\nW:',ws,'\nV:',vs,'\nT:',ts,'\nZ:',zs)
    var = (ws+vs+ts)
    return var