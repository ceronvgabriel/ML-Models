#Gabriel Ceron gaboceron10@gmail.com

import numpy
from scipy.optimize import fmin_l_bfgs_b

def aux(z):
  return (1+numpy.exp(-z))

def logReg(DTR,LTR,DTE,reg=10**(-3)):
  x0 = numpy.zeros(DTR.shape[1]+1)
  y=LTR
  m = y.size

  def costFunctionRegB3(wB):
    w,b=wB[:-1],wB[-1]
    #@ behaves as .dot for 2D arrays
    # This is required to make the optimize function work
    a1 = aux(-((DTR@w) + b))
    a2 = aux( ((DTR@w) + b))
    first = numpy.log1p(a1).T @ y
    second = numpy.log1p(a2).T @ (1 - y)
    J = (1 / m) * (first + second) + (reg / (2 * m)) * numpy.sum(numpy.square(w))
    return J

  x,f,d=fmin_l_bfgs_b(func=costFunctionRegB3,x0=x0,approx_grad = True)
  ws,bs=x[:-1],x[-1]
  p=ws@ DTE.T+bs #predict
  return p



