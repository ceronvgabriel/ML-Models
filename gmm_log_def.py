#Gabriel Ceron gaboceron10@gmail.com

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pdb
import numpy

np.random.seed(0)

#Gaussian Utils
def GAU_1D(X, mu, var):
    return -0.5*numpy.log(2*numpy.pi) - 0.5*numpy.log(var) - (X-mu)**2/(2*var)


def GAU_ND(XND, mu, C):
    '''Working just as numpy.multivariate_normal, but slower'''
    #pdb.set_trace()
    mu=vcol(mu)
    XND=XND.T
    if len(XND.shape) == 1:
      M=1
    else:
      M=len(XND)
    #pdb.set_trace()

    _,determinant = numpy.linalg.slogdet(C)
    determinant = numpy.log(numpy.linalg.det(C))
    inverse = numpy.linalg.inv(C)
    dot = []
    XND_mu = XND - mu
    #pdb.set_trace()
    for x in XND_mu.T:
        dot.append(numpy.dot(x.T, numpy.dot(inverse, x)))
    return np.exp(-M/2*numpy.log(2*numpy.pi) - 1/2*determinant - 1/2*numpy.hstack(dot).flatten())


def pdf_GMM_1D(X, gmm) :
  '''
  X should be rows=features, columns=samples
  '''
  n_comp=len(gmm)
  W,M,S, =gmm.T
  likelyhood=0; #Sum of likelyhoods of each component
  for comp in range(n_comp):
    comp_like=numpy.exp(GAU_1D(X,M[comp],S[comp]))
    likelyhood+=comp_like*W[comp] #sum of the likelyhoods of each component by it's weight
  return likelyhood

def vcol(x):
    return x.reshape(x.shape[0], 1)

def mcol(x):
    return x.reshape(x.size, 1)

#Supervised GMM, 2 class classification
class GMM_2C:

  def __init__(self,DTR,LTR,n_comp=10,iterations=100):
    print("gmm")
    self.models_v=[]
    self.n_comp=n_comp
    self.iterations=iterations
    self.nc = len(numpy.unique(LTR))
    DTRT=DTR.T
    DTR_c = [DTRT[:, LTR == i] for i in range(self.nc)]

    #Our supervised model as a composition of unsupervised models
    for m in range(self.nc):
      self.models_v.append(GMM(DTR_c[m].T,n_comp,iterations))

  def run(self):
    for m in range(self.nc):
      self.models_v[m].run()
  
  def predict(self,DTR):
    predictions=[]
    for m in range(self.nc):
      #we could use prior here
      predictions.append(np.array(self.models_v[m].predict(DTR)))
    llr=np.log(predictions[0]/predictions[1])
    return llr

#Unsupervised GMM
class GMM:

  def __init__(self,X,num_comp,iterations):
    self.iterations = iterations
    self.num_comp = num_comp
    self.X = X
    self.mu = None
    self.w = None
    self.cov = None
      
  #Train
  def run(self):
                
    #Initialize means randomly, from num components and features
    self.mu = np.random.randint(min(self.X[:,0]),max(self.X[:,0]),size=(self.num_comp,len(self.X[0])))

    #Initialize covariances
    self.cov = np.zeros((self.num_comp,len(self.X[0]),len(self.X[0])))
    for dim in range(len(self.cov)):
        np.fill_diagonal(self.cov[dim],5)

    #Initialize weights
    self.w = np.ones(self.num_comp)/self.num_comp
    log_likelihoods = [] #log likelihoods for each iteration
        
    for i in range(self.iterations):               
      #E Step
      r_i = np.zeros((len(self.X),len(self.cov)))

      for m,co,we,r in zip(self.mu,self.cov,self.w,range(len(r_i[0]))):
          r_i[:,r] = we*GAU_ND(self.X,m,co)/np.sum([w_c*GAU_ND(self.X,mu_c,cov_c) for w_c,mu_c,cov_c in zip(self.w,self.mu,self.cov)],axis=0) 

      #M Step
      self.mu = []
      self.cov = []
      self.w = []
      log_likelihood = []

      for c in range(len(r_i[0])):
        Z = np.sum(r_i[:,c],axis=0)
        #flatten
        mu_c = (1/Z)*np.sum(self.X*r_i[:,c].reshape(len(self.X),1),axis=0)
        self.mu.append(mu_c)

        #newCov=((1/Z)*np.dot((np.array(r_i[:,c]).reshape(len(self.X),1)*(self.X)).T,(self.X)) - np.dot(mu_c,mu_c.T))# here
        newCov=((1/Z)*np.dot((np.array(r_i[:,c]).reshape(len(self.X),1)*(self.X-mu_c)).T,(self.X-mu_c)))
        #Avoid degenerate solutions:
        psi=0.01
        U, s, _ = numpy.linalg.svd(newCov)
        s[s<psi] = psi
        newCov = numpy.dot(U, mcol(s)*U.T)
        
        self.cov.append(newCov)#reg

        self.w.append(Z/np.sum(r_i))

      l_like = np.log(np.sum([k*GAU_ND(self.X,self.mu[i],self.cov[j]) for k,i,j in zip(self.w,range(len(self.mu)),range(len(self.cov)))]))
      #print("Likelihood: ",l_like,end="\r")

      log_likelihoods.append(l_like)
    
    #Uncomment if you want to show the graph
    # fig, ax=plt.subplots()
    # ax.plot(log_likelihoods)
    # ax.set(xlabel='Iterations', ylabel='Log-likelihood')


  #Predict likelihood, used by GMM_2C
  def predict(self,Y):
    l_like = [k*GAU_ND(Y,self.mu[i],self.cov[j]) for k,i,j in zip(self.w,range(len(self.mu)),range(len(self.cov)))]
    return sum(l_like)



#Show graphs for the estimated pdf of every feature
def showGraphEst(gmmLogModel,DTR):
  start=10
  end=10
  n_points=1000
  points=np.array(range(-start*n_points,end*n_points,1))/n_points
  points_md=np.array([points for i in range(4)]).T
  gmm_log=np.array([[k,vcol(gmmLogModel.mu[i]),gmmLogModel.cov[j]] for k,i,j in zip(gmmLogModel.pi,range(len(gmmLogModel.mu)),range(len(gmmLogModel.cov)))])
  for f in range(len(DTR.T)):
    gmm_1D=np.array([[gmm_log[c][0],gmm_log[c][1][f],gmm_log[c][2][f][f]] for c in range(gmmLogModel.num_comp)])
    gmmPDF_1D=pdf_GMM_1D(points_md.T[f],gmm_1D)
    fig, ax = plt.subplots()
    ax.plot(points_md.T[f],gmmPDF_1D,"r")
    _=ax.hist(DTR.T[f],color="b",bins=50,density=True)

