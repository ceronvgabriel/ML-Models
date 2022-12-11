#Gabriel Ceron gaboceron10@gmail.com

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as splin

def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

# Covariance matrix
def covM(Data):
    DC = Data - vcol(Data.mean(1))
    C = (1/DC.shape[1]) * np.dot(DC, DC.T)
    return C

# PCA

class pca:
  def __init__(self,m):
    self.m=m
    self.P=[]
    self.accum=None

  def fit(self,Data_):
    Data=Data_.copy().T
    C = covM(Data)
    s, U = np.linalg.eigh(C)
    self.P = U[:, ::-1][:, 0:self.m]
    sm=s[::-1][0:self.m]
    #accumulated variance for the chosen m
    self.accum=sum(sm)/sum(s)
    return self.accum

  def transform(self,Data_):
    DP = np.dot(self.P.T, Data_.T)
    return DP.T


