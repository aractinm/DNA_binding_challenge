import numpy as np 
import argparse
import os
import sys
import csv
import cvxopt
from cvxopt import matrix
from cvxopt.solvers import qp
from joblib import Parallel, delayed
import multiprocessing
from fit import fit
from Bag_of_words import build_BOW
from substring_kernel import create_matrix
cvxopt.solvers.options['show_progress'] = False
	
def training():
	filename = "Results/" + "Substring_L_0.01.csv"

	result = open(filename, 'w+')
	result.write(",Bound\n")
	weight = [0.3, 0.7, 0.05]
	sublen = [6, 6, 5]

	for k in range(3):
		filename = "data_work/" + "Xtr" + str(k) + ".csv"	
		print(filename)
		data = open(filename, 'r')
		X_orig = []
		for line in data:
			X_orig.append(line.strip('\n'))

		# labels
		filename = "data_work/" + "Ytr" + str(k) + ".csv"
		print(filename)
		data = open(filename, 'r')
		y_train = []
		next(data)
		for line in data:
			temp = [np.float(i.strip()) for i in line.split(',')]
			if temp[1] == 0:
				y_train.append(-1.0)
			else:
				y_train.append(1.0)

		y_train = np.array(y_train)
		print("Y train shape = ", y_train.shape)

		# Creating substring kernel
		bag = []
		build_BOW(str(k), bag, sublen[k])
		bag = list(set(bag))

		X_train = create_matrix(weight[k], X_orig, bag)
		K_train = np.dot(X_train, X_train.T)


		# Test data
		filename = "data_work/" + "Xte" + str(k) + ".csv"	
		print(filename)
		data = open(filename, 'r')
		X_orig = []
		for line in data:
			X_orig.append(line.strip('\n'))


		X_test = create_matrix(weight[k], X_orig, bag)

		######## Training #################
		alpha = fit(K_train, y_train, 0.01)
		print(alpha)

		########## Computing bias #############
		bias = 0
		for i in range(len(y_train)):
			if y_train[i]*alpha[i] > 0 and y_train[i]*alpha[i] < 1/(2*0.01*len(y_train)):
				bias = y_train[i] - np.dot(K_train, alpha)[i]
				break

		print("Bias = {0}".format(bias))
		
		y_pred = np.zeros(X_test.shape[0])
		for i in range(X_test.shape[0]):
			if np.dot(alpha.T, np.dot(X_train, X_test[i,:].T) + bias) > 0:
				y_pred[i] = 1
			else:
				y_pred[i] = 0

			temp = str(k*X_test.shape[0] + i) +',' + str(int(y_pred[i]))
			result.write(temp)
			result.write('\n')

		y_pred = np.zeros(y_train.shape)
		res = 0
		for i in range(len(y_train)):
			if np.dot(alpha.T, np.dot(X_train, X_train[i,:].T) + bias) > 0:
				y_pred[i] = 1.0
			else:
				y_pred[i] = -1.0

			if y_pred[i] == y_train[i]:
				res += 1

		print("Train Accuracy for {1} dataset = {0}".format(res/len(y_train), k))
	result.close()

def main(args):
	training()

if __name__ == "__main__":
    main(sys.argv[1:])
