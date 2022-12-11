#Gabriel Ceron gaboceron10@gmail.com

import dcf
import numpy
import numpy as np
import logistic_def
import gaussian_def
import pdb
import svm_k_def as svm
import gmm_log_def as gmm
import pca
import utils


np.random.seed(0)

#Beware that if K is very high, confMatrix has few values and division by zero may occur, resulting in nan.
#Pass model = "logReg", "gauss", "svm" or "gmm"
#cv with minDCFn metric, for a given working point
#pass diferent params for each type of model

def cv(DTR,LTR,model,K=10,t=[0.5,1,1],prepro=None,params={}):
  
  N = int(DTR.shape[0]/K)

  ind = numpy.random.permutation(DTR.shape[0]) # indexes

  if prepro=="pca3":
    myPca=pca.pca(3)
  if prepro=="gauss":
    myGauss=utils.Gauss_trans()

  minDCFn_v = []
  thresh_v=[]
  preds_pool=[]
  for i in range(K):
      #print("\r",'CV fold: ', i, "/", K, end='')
      indTE = ind[i*N:(i+1)*N]
      if i > 0:
          indTR_L = ind[0:i*N]
      elif (i+1) < K:
          indTR_R = ind[(i+1)*N:]
      if i == 0:
          indTR = indTR_R
      elif i == K-1:
          indTR = indTR_L
      else:
          indTR = numpy.hstack([indTR_L, indTR_R])
      DTR_k = DTR[indTR,:]
      LTR_k = LTR[indTR]
      DTE_k = DTR[indTE,:]
      LTE_k = LTR[indTE]

      #Preprocess each fold
      if prepro=="pca3":
        myPca.fit(DTR_k)
        DTR_k = myPca.transform(DTR_k)
        DTE_k = myPca.transform(DTE_k)
      if prepro=="gauss":
        myGauss.fit(DTR_k)
        DTR_k = myGauss.trans(DTR_k)
        DTE_k = myGauss.trans(DTE_k)
      
      #predict each fold
      if model=="logReg":
        #scores are inverted, so taking negative
        llr=-logistic_def.logReg(DTR_k,LTR_k,DTE_k,**params)
      #for gaussian models DTR and DTE shall be passed transposed
      elif model=="gaussFull":
        llr=gaussian_def.MVG(DTR_k.T, LTR_k, DTE_k.T)
      elif model=="gaussTied":
        llr=gaussian_def.MVG_tied(DTR_k.T, LTR_k, DTE_k.T)
      elif model=="gaussNaive":
        llr=gaussian_def.MVG_naive(DTR_k.T, LTR_k, DTE_k.T)
      elif model=="svm":
        #we actually get the scores
        llr=svm.svm(DTR_k, LTR_k, DTE_k,**params)
      elif model=="gmm":
        gmm2=gmm.GMM_2C(DTR_k, LTR_k,**params)
        gmm2.run()
        #we also invert scores
        llr=-gmm2.predict(DTE_k)
      #Pool the predictions to take minDCF
      preds_pool.extend(llr)
  #Once we have the llr or posterior probs take minDCFn
  minDCFn,DCFn_v,thresh, s=dcf.getMinDCFn(preds_pool,LTR[ind[:K*N]],triple=t)
  actDCFn = dcf.getActDCFn(preds_pool,LTR[ind[:K*N]],triple=t)

  return minDCFn, DCFn_v, thresh,actDCFn

#Single fold evaluation, can be used to split training data or use evaluation data (pass DTE and LTE)
def singleFold(DTR,LTR,DTE=None,LTE=None,model="logReg",prepro=None,split=0.2,t=[0.5,1,1],params={}):



  #If we want to do a single split from training
  if DTE is None and LTE is None:
    ind = numpy.random.permutation(DTR.shape[0])
    trainSplit=int(len(LTR)*(1-split))
    indTR=ind[:trainSplit]
    indTE=ind[trainSplit:]
    DTR_k = DTR[indTR,:]
    LTR_k = LTR[indTR]
    DTE_k = DTR[indTE,:]
    LTE_k = LTR[indTE]
  #"Single fold evaluation, 100% of training data"
  elif DTE is not None and LTE is not None:
    #print("Single fold evaluation, 100% of training data")
    DTR_k = DTR
    LTR_k = LTR
    DTE_k = DTR
    LTE_k = LTR

  if prepro=="pca3":
    myPca=pca.pca(3)
  if prepro=="gauss":
    myGauss=utils.Gauss_trans()
  
  if prepro=="pca3":
    myPca.fit(DTR_k)
    DTR_k = myPca.transform(DTR_k)
    DTE_k = myPca.transform(DTE_k)
  if prepro=="gauss":
    myGauss.fit(DTR_k)
    DTR_k = myGauss.trans(DTR_k)
    DTE_k = myGauss.trans(DTE_k)

  #pdb.set_trace()
  if model=="logReg":
    #scores are inverted, so taking negative
    llr=-logistic_def.logReg(DTR_k,LTR_k,DTE_k,**params)
  #for gaussian models DTR and DTE shall be passed transposed
  elif model=="gaussFull":
    llr=gaussian_def.MVG(DTR_k.T, LTR_k, DTE_k.T)
  elif model=="gaussTied":
    llr=gaussian_def.MVG_tied(DTR_k.T, LTR_k, DTE_k.T)
  elif model=="gaussNaive":
    llr=gaussian_def.MVG_naive(DTR_k.T, LTR_k, DTE_k.T)
  elif model=="svm":
    #we actually get the scores
    llr=svm.svm(DTR_k, LTR_k, DTE_k,**params)
  elif model=="gmm":
    gmm2=gmm.GMM_2C(DTR_k, LTR_k,**params)
    gmm2.run()
    #we also invert scores
    llr=-gmm2.predict(DTE_k)

  minDCFn,DCFn_v,thresh, s=dcf.getMinDCFn(llr,LTE_k,triple=t)
  actDCFn = dcf.getActDCFn(llr,LTE_k,triple=t)

  return minDCFn, DCFn_v, thresh,actDCFn
