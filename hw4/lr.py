#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import sys
import re
from math import log
from math import exp
from math import sqrt

MAX_ITERS = 100

# Load data from a file
def read_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  varnames = p.split(header)
  namehash = {}
  for l in f:
    example = [int(x) for x in p.split(l.strip())]
    x = example[0:-1]
    y = example[-1]
    data.append( (x,y) )
  return (data, varnames)

# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
  numvars = len(data[0][0])
  w = [0.0] * numvars
  b = 0.0
  
  #100
  for i in range(MAX_ITERS):
    #defining different variables than above so I can do calculations
    gradientW = [0.0] * numvars
    gradientB = 0

    #using x,y values later
    for (x,y) in data:
      a = 0
      for i in range(numvars):
        a += w[i] * x[i]
      a += b

      #using 0.02 for eta
      gradientB -= eta * (y / (1 + exp(y * a)))
      
      for j in range(numvars):
        gradientW[j] -= eta * (y * x[j] / (1 + exp(y * a)))

    for k in range(numvars):
      #using .9 for lambda
      gradientW[k] += eta * (l2_reg_weight * w[k])
  
    #update bias
    b -= gradientB

    for num in range(numvars):
      #update weights
      w[num] -= gradientW[num]

  return (w,b)

# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
  (w,b) = model

  result = 0
  for i in range(len(x)):
    result += x[i] * w[i]

  return 1 / (1 + exp(-(result + b)))

# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  if (len(argv) != 5):
    print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
    sys.exit(2)
  (train, varnames) = read_data(argv[0])
  (test, testvarnames) = read_data(argv[1])
  eta = float(argv[2])
  lam = float(argv[3])
  modelfile = argv[4]

  # Train model
  (w,b) = train_lr(train, eta, lam)

  # Write model file
  f = open(modelfile, "w+")
  f.write('%f\n' % b)
  for i in range(len(w)):
    f.write('%s %f\n' % (varnames[i], w[i]))

  # Make predictions, compute accuracy
  correct = 0
  for (x,y) in test:
    prob = predict_lr( (w,b), x )
    #print(prob)
    if (prob - 0.5) * y > 0:
      correct += 1
  acc = float(correct)/len(test)
  #print("Accuracy: ",acc)

if __name__ == "__main__":
  main(sys.argv[1:])
