#Gabriel Ceron gaboceron10@gmail.com

import numpy
import numpy as np
import scipy
import pdb

#this functions shall be used for binary classification only
#DTR and DTE shall be passed as transposed to all functions

g=[]

def GAU_ND_log(XND, mu, C):
    M = XND.shape[0]
    _, determinant = numpy.linalg.slogdet(C)
    determinant = numpy.log(numpy.linalg.det(C))
    inverse = numpy.linalg.inv(C)
    dot = []
    XND_mu = XND - mu
    for x in XND_mu.T:
        dot.append(numpy.dot(x.T, numpy.dot(inverse, x)))
    return -M/2*numpy.log(2*numpy.pi) - 1/2*determinant - 1/2*numpy.hstack(dot).flatten()

def vcol(x):
    return x.reshape(x.shape[0], 1)

def MVG(DTR, LTR, DTE):
   nc = len(numpy.unique(LTR))
   DTR_class = [DTR[:, LTR == i] for i in range(nc)]
   means = []
   covariances = []
   for class_data in DTR_class:
      means.append(class_data.mean(1))
      covariances.append(numpy.cov(class_data, bias=True))
   S = numpy.zeros((nc, DTE.shape[1]))
   for i in range(nc):
      for j, sample in enumerate(DTE.T):
         S[i, j] = numpy.exp(GAU_ND_log(vcol(sample), vcol(means[i]), covariances[i]))
   lr=S[1,:]/S[0,:]
   llr=numpy.log(lr)
   return llr


def MVG_naive(DTR, LTR, DTE):
    nc = len(numpy.unique(LTR))
    DTR_class = [DTR[:, LTR == i] for i in range(nc)]
    means = []
    covariances = []
    for class_data in DTR_class:
        means.append(class_data.mean(1))
        covariances.append(numpy.cov(class_data, bias=True)*numpy.identity(class_data.shape[0])) #Diagonal only
    S = numpy.zeros((nc, DTE.shape[1]))
    for i in range(nc):
        for j, sample in enumerate(DTE.T):
            S[i, j] = GAU_ND_log(vcol(sample), vcol(means[i]), covariances[i])
    
    llr = S[1,:] - S[0,:]
    return llr


def MVG_tied(DTR, LTR, DTE):
    nc = len(numpy.unique(LTR))
    DTR_class = [DTR[:, LTR == i] for i in range(nc)]
    means = []
    covariances = []
    for class_data in DTR_class:
        means.append(class_data.mean(1))
        covariances.append(numpy.cov(class_data, bias=True))
    #Within class covariance:sum of the weighted covariances of each class
    SS = 0
    for i in range(nc):
        SS += DTR_class[i].shape[1]*covariances[i]
    SS = SS / DTR.shape[1]
    S = numpy.zeros((nc, DTE.shape[1]))
    for i in range(nc):
        for j, sample in enumerate(DTE.T):
            S[i, j] = GAU_ND_log(vcol(sample), vcol(means[i]), SS)
    llr = S[1,:] - S[0,:]
    return llr


