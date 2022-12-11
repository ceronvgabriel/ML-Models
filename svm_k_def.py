#Gabriel Ceron gaboceron10@gmail.com

import numpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize
from numpy import linalg

#Kernels:

#Globals for kernels, a class can also be implemented
C=0
d=3
b=0
gamma=10

def linear_kernel( x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y):
    return (np.dot(x, y) + b) ** d

def gaussian_kernel( x, y):
    return np.exp(-gamma*linalg.norm(x - y) ** 2 )

def rbf_kernel(x,y):
  return np.exp(-1.0*gamma*np.dot(np.subtract(x,y).T,np.subtract(x,y)))

#utils
def mrow(x):
    return x.reshape((1,x.size))
def mcol(x):
    return x.reshape((x.size, 1))

#svm
def lagr_wrap(H):
    def lagr(alpha):
        ones = numpy.ones(alpha.size)
        LD=0.5*numpy.linalg.multi_dot([alpha.T,H,alpha]) - numpy.dot(alpha.T,mcol(ones))
        LD_d= numpy.dot(H,alpha)-ones #gradient
        return LD, LD_d.flatten() 
    return lagr

def svm(DTR,LTR,DTE,C_=0.1,kernel_="rbf",d_=3,b_=0,gamma_=10):
  """Pass data as rows: samples, columns: features"""
  #set values for kernels, a class may also be implemented
  global C
  C=C_
  global gamma
  gamma = gamma_

  #Select kernel:
  if kernel_ =="linear":
    kernel=linear_kernel
  elif kernel_ == "polynomial":
    global d
    d=d_
    global b
    b=b_
    kernel=polynomial_kernel
  elif kernel_ == "gaussian":
    kernel=gaussian_kernel
  elif kernel_ == "rbf":    
    kernel=rbf_kernel

  X=DTR.copy()

  #Get G hat for kernel version of SVM
  G_h=np.apply_along_axis(lambda x1 : np.apply_along_axis(lambda x2:  kernel(x1, x2), 1, X),1, X)  

  #Classes labels -1,+1
  z = numpy.copy(LTR)
  z[z==0]=-1
  y=z
  z= mcol(z)

  #Get H hat for kernel version of SVM
  H_h= z*z.T*G_h

  #Dual solution:
  lagrangian=lagr_wrap(H_h)
  #initialize  
  x0=numpy.zeros(LTR.size)
  #set bounds:
  bounds=[(0,C) for i in range(LTR.size)]
  #optimize
  alpha,_,_= scipy.optimize.fmin_l_bfgs_b(lagrangian, approx_grad=False,x0=x0, iprint=0, bounds=bounds, factr=1.0)

  #Use all support vectors:
  supportVectors = X
  supportAlphaY = y*alpha

  #Predict for kernel SVM
  def predict(x):
      x1 = np.apply_along_axis(lambda s: kernel(s, x), 1, supportVectors)
      x2 = x1 * supportAlphaY
      return np.sum(x2)
  scores = np.apply_along_axis(predict, 1, DTE) #scores

  return scores
    