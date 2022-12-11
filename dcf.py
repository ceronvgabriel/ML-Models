#Gabriel Ceron gaboceron10@gmail.com

import numpy
import numpy as np
import matplotlib.pyplot as plt
g=[]

def sig(z):
    return(1 / (1 + numpy.exp(-z)))

#Use this function if you want to set the threshold from the scores or logLikelihoodRatio LLR
def confMatrBinLLR(thresh, scores, labels):
    HT = scores > thresh
    HF = scores <= thresh
    label1 = (labels == 1)
    label0 = (labels == 0)
    confM = numpy.zeros((2, 2))
    confM[0][0] = (HF*label0).sum()
    confM[0][1] = (HF*label1).sum()
    confM[1][0] = (HT*label0).sum()
    confM[1][1] = (HT*label1).sum()
    return confM

# get DCF unnormalized
def getDCFu(triple, confM):
    pi,Cfn,Cfp = triple
    FNR = confM[0][1] / (confM[0][1] + confM[1][1])
    FPR = confM[1][0] / (confM[1][0] + confM[0][0])
    DCFu = (pi*Cfn*FNR) + ((1-pi)*Cfp*FPR)
    return DCFu

# get DCF normalized
def getDCFn(triple, DCFu):
    pi,Cfn,Cfp = triple
    dummy = min(pi * Cfn, (1 - pi) * Cfp)
    DCFn = DCFu/dummy
    return DCFn

# get minDCFn
def getMinDCFn(scores,labels,triple=[0.5,1,1]):
  scores_s=scores.copy()
  scores_s.sort()
  DCFn_v=[]
  for thresh in scores_s:
    confMatr = confMatrBinLLR(thresh,scores,labels)
    DCFu=getDCFu(triple,confMatr)
    DCFn=getDCFn(triple,DCFu)
    DCFn_v.append(DCFn)
  min_DCFn=min(DCFn_v)
  i_min=DCFn_v.index(min_DCFn)
  min_thresh=scores_s[i_min]
  return min_DCFn , DCFn_v, min_thresh,scores_s

# get ROC
# return FPR, TPR
def getROC(confMatr):
  FNR = confMatr[0][1] / (confMatr[0][1] + confMatr[1][1])
  FPR = confMatr[1][0] / (confMatr[1][0] + confMatr[0][0])
  return FPR, 1-FNR

# get MinDCFn and also calculate ROC
def getMinDCFn_ROC(scores,labels,triple=[0.5,1,1]):
  scores_c=scores.copy()
  scores_c.sort()
  DCFn_v=[]
  FPR_v=[]
  TPR_v=[]
  for thresh in scores_c:
    confMatr = confMatrBinLLR(thresh,scores,labels)
    DCFu=getDCFu(triple,confMatr)
    DCFn=getDCFn(triple,DCFu)
    DCFn_v.append(DCFn)
    FPR,TPR=getROC(confMatr)
    FPR_v.append(FPR)
    TPR_v.append(TPR)
  min_DCFn=min(DCFn_v)
  i_min=DCFn_v.index(min_DCFn)
  min_thresh=scores_c[i_min]
  return min_DCFn , DCFn_v, min_thresh, scores_c, [FPR_v,TPR_v]

#Draw ROC curve
def drawROC(FPR, TPR):
    ig, ax = plt.subplots()
    ax.plot(FPR, TPR, label="ROC Curve")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    plt.show()

def getActDCFn(scores,labels,triple=[0.5,1,1]):
  #for efective prior:
  pi=triple[0]
  thresh=np.log((pi)/(1-pi))
  confMatr = confMatrBinLLR(thresh,scores,labels)
  DCFu=getDCFu(triple,confMatr)
  DCFn=getDCFn(triple,DCFu)
  return DCFn













