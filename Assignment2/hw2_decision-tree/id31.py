#!/usr/bin/python
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
#
#for entropy calcs
from __future__ import division
import sys
import re
# Node class for the decision tree
import node

#for entropy calcs
import math

train=None
varnames=None
test=None
testvarnames=None
root=None

# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
        if p == 0 or p == 1:
            return 0
        lf = p
        rh = 1 - p
        res = -(rh * math.log(rh, 2)) - (lf*math.log(lf, 2))
        return res;

# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):

        entotal = entropy(py/total)
        gain = 0
        if pxi == total:
            gain = entotal - ((pxi/total) * entropy(py_pxi/pxi))
        elif pxi != 0:
            gain = entotal - ((pxi/total)*entropy(py_pxi/pxi)) - (((total-pxi)/total) * entropy((py-py_pxi)/(total-pxi)))
        elif pxi == 0:
            gain = entotal - ((total-pxi)/total)  *entropy((py -py_pxi)/(total - pxi))
        return gain

# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable    

# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)

# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)

def findSuccess(data):
#helper function: given data, find all y in the last field s.t y = 1
    succ = 0
    for item in data:
        if item[-1] == 1:
            succ += 1
    return succ

def splitr(pred, varnames):
#helper function: given a prediction, creates either a 1 or 0 leaf
    if pred >= 0.5:
        return node.Leaf(varnames,1)
    else:
        return node.Leaf(varnames,0)

def findattr(data, place):
#helper function: given an index, counts all entries with a 1 in the attribute in that index
#and also counts all entries with a one in their last index as well
    posx = 0
    pos_x_y = 0
    for item in data:
        if item[place] == 1:
            posx += 1
            if item[-1] == 1:
                pos_x_y += 1

    return (posx, pos_x_y)

# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames):
    succ = findSuccess(data)  #grabs the initial success rate

    pred = (succ / len(data)) #predicts based on the ratio of passes / failures

    if len(varnames) == 1:    #if there's only one attribute left, split on it automaticall
        return splitr(pred, varnames)

    info = 0      #information gain, tallied per attribute
    for i in range(len(varnames) - 1):
        posx = 0      #count of the total entries with a 1 in their ith index
        pos_x_y = 0   #and of the total entries with a 1 in their last and ith index

        res = findattr(data, i)
        posx = res[0]
        pos_x_y = res[1]

        calced = infogain(pos_x_y, posx, succ, len(data))
        if calced > info:  #find largest gain to split on
            info = calced
            place = i

    if info == 0:
        return splitr(pred, varnames)

    left = []
    right = []
    for i in range(len(data)):   #split each branch's possible values into left and right
        if data[i][place] == 0:
            left.append(data[i])
        else:
            right.append(data[i])

    #recursive call, continues the tree
    return node.Split(varnames, place, build_tree(left, varnames), build_tree(right, varnames))

# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS,testS,modelS):
    global train
    global varnames
    global test
    global testvarnames
    global root
    (train, varnames) = read_data(trainS)
    (test, testvarnames) = read_data(testS)
    modelfile = modelS

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)

def runTest():
    correct = 0
    # The position of the class label is the last element in the list.
    yi = len(test[0]) - 1
    for x in test:
        # Classification is done recursively by the node class.
        # This should work as-is.
        pred = root.classify(x)
        if pred == x[yi]:
            correct += 1
    acc = float(correct)/len(test)
    return acc  
    
    
# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
        print 'Usage: id3.py <train> <test> <model>'
        sys.exit(2)
    loadAndTrain(argv[0],argv[1],argv[2]) 
                    
    acc = runTest()             
    print "Accuracy: ",acc                      

if __name__ == "__main__":
    main(sys.argv[1:])