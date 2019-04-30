#!/usr/bin/python
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
#
import sys
import re
# Node class for the decision tree
import node
import math

train=None
varnames=None
test=None
testvarnames=None
root=None

# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
	# >>>> YOUR CODE GOES HERE <<<<
    # For now, always return "0":

	symmetric_frac = (1-p)
	if p == 0 or symmetric_frac == 0:
		return 0
	else:
		return -(p*math.log(p,2)) - (symmetric_frac*math.log(symmetric_frac,2));

# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
	# >>>> YOUR CODE GOES HERE <<<<
    # For now, always return "0":

	s = 0
	s_c1 = 0
	s_c2 = 0

	s = entropy(float(py)/float(total))
	if pxi == 0:
		s_c1 = 0
	else:
		s_c1 = (float(pxi)/float(total))*entropy(float(py_pxi)/float(pxi))

	if(total - pxi) == 0:
		s_c2 = 0
	else:
		s_c2 = (float(total - pxi)/float(total))*entropy(float(py-py_pxi)/float(total - pxi))

	return s - (s_c1 + s_c2)

# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable

def partition(data, varnames, var):
	node_left = []
	node_right = []
	for item in data:
		if item[var] == 0:
			node_left.append(item)
		else:
			node_right.append(item)
	return node_left, node_right

# Decide which variable to split on
def split_on(data, varnames):
	largest_gain = 0
	best_gain = None

	for i in range(len(varnames)-1):
		attr_label, attr, label, tot = var_counts(data, i)
		temp = infogain(attr_label, attr, label, tot)
		if temp > largest_gain:
			largest_gain = temp
			best_gain = i

	return best_gain

# Retreives py_pxi, pxi, py for the infogain function
def var_counts(data, var_col):
	var_count = 0
	var_count_label = 0
	label = 0
	total = 0

	for item in data:
		total += 1
		if item[var_col] == 1:
			var_count += 1
		if item[-1] == 1:
			label += 1
		if item[var_col] == 1 and item[-1] == 1:
			var_count_label += 1

	return var_count_label, var_count, label, total

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

# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames):
    # >>>> YOUR CODE GOES HERE <<<<
    # For now, always return a leaf predicting "1":

	for i in range(len(varnames) - 1):
		counts = var_counts(data, i)

	if counts[2] == counts[3]:
		return node.Leaf(varnames, 1)
	elif counts[2] == 0:
		return node.Leaf(varnames, 0)
	else:
		best_split = split_on(data, varnames)
		if best_split == None:
			return node.Leaf(varnames,1)
		left_split, right_split = partition(data, varnames, best_split)
		best_node = node.Split(varnames, best_split, build_tree(left_split, varnames), build_tree(right_split, varnames))

		return best_node

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