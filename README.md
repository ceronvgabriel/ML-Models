# ML-Models
Group of ML models wrote in python for the final project of the course Machine Learning and Pattern Recognition in the Politecnico di Torino University

# Dataset
A Banknote authentication dataset is used. 2 class classification problem, Label 0 if the Banknote is authentic, Label 1 if it is false.

# Models used
- Logistic Regression
- Gaussian Classifier
- Gaussian Mixture Model
- Support Vector Machine

# Dataset splits
DTR: Data Training
DTE: Data Test
LTR: Label Training
LTE: Label Test

# DCF
DCF stands for Detection Cost Function, is a kind of Cost Function / Error Rate based on a threshold, more about it here:
https://drive.google.com/file/d/1uhZxjL-xipmtddl3vpBmX2SczXpvL0eu/view?usp=sharing

# LLR
LLR stands for Log Likelihood Ratio. It is basically the score of an inference.
If we want to get the accuracy from the scores we do as shown in the utils.py file:

def llr_acc(scores,LTE,thresh=0):
  predL=scores>thresh
  match=predL==LTE
  return sum(match)/len(match)

As shown by default the threshold for the scores is 0, so if positive the prediction/inference correspond to Label A and if negative to label B

LLR is equal to:
LLR=ln(P(x|HT)/P(x|HF))
Where P(x|HT) is the probability that the sample x belongs to class HT (True)
And with P(x|HT) = 1 - P(x|HF), we can clear the equation for P(x|HT) obtaning the Sigmoid function, this means that we can convert from LLR/Score to probability using the sigmoid function.

