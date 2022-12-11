#Gabriel Ceron gaboceron10@gmail.com

import numpy
from scipy.stats import norm
import matplotlib.pyplot as plt

def acc(predL,LTE):
  match=predL==LTE
  return sum(match)/len(match)

def llr_acc(scores,LTE,thresh=0):
  predL=scores>thresh
  match=predL==LTE
  return sum(match)/len(match)
#
def vcol(x):
    return x.reshape(x.shape[0], 1)


#Standarization
#Note you have to enter transposed data
class Stand:
  ms=None
  ss=None

  def fit2D(self,x):
    m=numpy.mean(x,1)
    s=numpy.std(x,1)
    self.ms=m
    self.ss=s

  def trans2D(self,x):
    f_v=[]
    for i,f in enumerate(x):
      f_v.append((f-self.ms[i])/self.ss[i])
    return f_v

  def stand1D(self,x):
    m=numpy.mean(x)
    s=numpy.std(x)
    f_v=[]
    return (x-m)/s

#Gaussanization
class Gauss_trans:
  DTR=[]
  N=None

  def fit(self,DTR):
    self.N=DTR.shape[0]
    self.DTR=DTR.copy().T
    self.DTR.sort()
    

  def trans(self,D):
    gauss_data=[]
    for x in D:
      r_v=[]
      for i,d in enumerate(self.DTR):
        accum=numpy.searchsorted(d,x[i])
        r=(accum + 1)/(self.N+2)
        r_v.append(r)
      gauss_d=norm.ppf(r_v)
      gauss_data.append(gauss_d)
    return numpy.array(gauss_data)


#Use hist functions with transposed Data
#For unsupervised:
def hist_feat(feature, feat_name="feature", b=100, d=True, a=1, edgec="black"):
    fig, ax = plt.subplots()
    ax.hist(feature, bins=b, density=d, alpha=a, edgecolor=edgec)
    ax.set_xlabel(feat_name)
    ax.legend()
    plt.show()

def hist_feat_all(DTRT, feat_name=None, b=100, d=True, a=1, edgec="black"):
    if feat_name is None:
      feat_name=["Feature_"+str(i) for i in range(1,len(DTRT)+1)]
    for i,feat in enumerate(DTRT):
      hist_feat(feat,feat_name=feat_name[i],b=b)

#histograms for binary classification problems:
def hist_class(feature,LTR, feat_name="", classes=["0","1"], b=100, d=True, a=1, edgec="black"):
    nc = len(numpy.unique(LTR))
    fig, ax = plt.subplots()
    for i in range(nc):
      feature_class = feature[LTR == i]
      ax.hist(feature_class, bins=b, density=d, label=classes[i], alpha=a, edgecolor=edgec)
    ax.set_xlabel(feat_name)
    ax.set_ylabel("Density")
    ax.legend()
    plt.show()

def hist_class_all(DTRT,LTR,classes=["0","1"],feat_names=None, b=100):
  if feat_names is None:
    feat_names=["Feature_"+str(i) for i in range(1,len(DTRT)+1)]
  for i,feat in enumerate(DTRT):
    hist_class(feat,LTR,feat_name=feat_names[i],classes=classes,b=b)

# Scatter plot function for two dimentions and two classes
#Pass transposed data
def scatter_classes_2D(DTR, LTR):
    labels=numpy.unique(LTR)
    nc = len(numpy.unique(LTR))
    fig, ax = plt.subplots()
    feat_1_class_1 = DTR[0][LTR == labels[0]]
    feat_1_class_2 = DTR[0][LTR == labels[1]]
    feat_2_class_1 = DTR[1][LTR == labels[0]]
    feat_2_class_2 = DTR[1][LTR == labels[1]]
    ax.scatter(feat_1_class_1,feat_2_class_1)
    ax.scatter(feat_1_class_2,feat_2_class_2)
    ax.set_xlabel("Feature_1")
    ax.set_ylabel("Feature_2")
    ax.legend()
    plt.show()