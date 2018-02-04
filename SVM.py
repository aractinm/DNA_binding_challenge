import numpy as np 
import argparse
import os
import sys
import csv
from cvxopt import matrix
from cvxopt.solvers import qp
from joblib import Parallel, delayed
import multiprocessing

lamb = 0.01
sigma = 0.1

def gauss_kernel(data, store=0):
	row, col = data.shape
	gauss_kernel = np.zeros([row, row])

	for i in range(row):
		print("Gaussian row = ", i)
		for j in range(row):
			gauss_kernel[i, j] = np.exp(-np.linalg.norm(data[i,:] - data[j, :])**2/ (2*sigma**2))

	if store == 1:
		np.save("pretrained/gaussian_kernel_{0}_{1}.npy".format(row, sigma), gauss_kernel)
	return gauss_kernel

def gauss_kernel_sample(data, example):
	row = data.shape[0]
	gauss_kernel_sample = np.zeros([row,1])

	for i in range(row):
		gauss_kernel_sample[i,0] = np.exp(-np.linalg.norm(data[i,:] - example)**2/ (2*sigma**2))
		#print(gauss_kernel_sample[i,0])

	return(gauss_kernel_sample)

def normalize(X):
	mean = np.tile(np.mean(X, axis =0),(X.shape[0],1))
	std  = np.tile(np.std(X, axis =0),(X.shape[0],1))
	
	X_norm = np.subtract(X, mean)
	#X_norm = np.divide(X_norm, std)
	return(X_norm, mean[0,:])

def read_data(args):
	# Training data
	filename = "data_work/" + "Xtr" + args.dataset + "_mat50_train.csv"	
	print(filename)
	data = open(filename, 'r')
	data_train = []
	for line in data:
		temp = [np.float(i.strip()) for i in line.split(' ')]
		data_train.append(temp)

	X_train = np.array(data_train)
	X_train, mean = normalize(X_train)
	K_train = gauss_kernel(X_train)

	# labels
	filename = "data_work/" + "Ytr" + args.dataset + "_train.csv"
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
	print(y_train.shape)

	########### CV data ########################
	filename = "data_work/" + "Xtr" + args.dataset + "_mat50_cv.csv"	
	print(filename)
	data = open(filename, 'r')
	data_cv = []
	for line in data:
		temp = [np.float(i.strip()) for i in line.split(' ')]
		data_cv.append(temp)

	X_cv = np.array(data_cv)
	X_cv = np.subtract(X_cv, np.tile(mean,(X_cv.shape[0],1)))
	
	# labels
	filename = "data_work/" + "Ytr" + args.dataset + "_cv.csv"
	print(filename)
	data = open(filename, 'r')
	y_cv = []
	for line in data:
		temp = [np.float(i.strip()) for i in line.split(',')]
		if temp[1] == 0:
			y_cv.append(-1.0)
		else:
			y_cv.append(1.0)
	
	y_cv = np.array(y_cv)
	print(y_cv.shape)

	return X_train, K_train, y_train, X_cv, y_cv

def read_data_all(k):
	# Training data
	filename = "data_work/" + "Xtr" + k + "_mat50.csv"	
	print(filename)
	data = open(filename, 'r')
	data_train = []
	for line in data:
		temp = [np.float(i.strip()) for i in line.split(' ')]
		data_train.append(temp)

	X_train = np.array(data_train)
	K_train = gauss_kernel(X_train)

	# labels
	filename = "data_work/" + "Ytr" + k + ".csv"
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
	print(y_train.shape)

	########### Test data ########################
	filename = "data_work/" + "Xte" + k + "_mat50.csv"	
	print(filename)
	data = open(filename, 'r')
	data_cv = []
	for line in data:
		temp = [np.float(i.strip()) for i in line.split(' ')]
		data_cv.append(temp)

	X_test = np.array(data_cv)
	
	return X_train, K_train, y_train, X_test

def read_data_combined():
	################ Reading data #########################
	data_train = []
	for k in range(3):
		filename = "data_work/" + "Xtr" + str(k) + "_mat50.csv"	
		print(filename)
		data = open(filename, 'r')
		for line in data:
			temp = [np.float(i.strip()) for i in line.split(' ')]
			data_train.append(temp)

		data.close()

	X_train = np.array(data_train)
	print(X_train.shape)

	if os.path.isfile("pretrained/gaussian_kernel_{0}_{1}.npy".format(X_train.shape[0], sigma)):
		K_train = np.load("pretrained/gaussian_kernel_{0}_{1}.npy".format(X_train.shape[0], sigma))
	else:
		K_train = gauss_kernel(X_train, 1)

	############### labels ##########################
	y_train = []
	for k in range(3):
		filename = "data_work/" + "Ytr" + str(k) + ".csv"
		print(filename)
		data = open(filename, 'r')
		next(data)
		for line in data:
			temp = [np.float(i.strip()) for i in line.split(',')]
			if temp[1] == 0:
				y_train.append(-1.0)
			else:
				y_train.append(1.0)
		data.close()

	y_train = np.array(y_train)
	print(y_train.shape)

	########### Test data ########################
	data_cv = []
	for k in range(3):
		filename = "data_work/" + "Xte" + str(k) + "_mat50.csv"	
		print(filename)
		data = open(filename, 'r')
		for line in data:
			temp = [np.float(i.strip()) for i in line.split(' ')]
			data_cv.append(temp)

		data.close()

	X_test = np.array(data_cv)
	X_test = normalize(X_test)
	print(X_test.shape)

	return X_train, K_train, y_train, X_test

def read_data_combined_finetune():
	################ Reading data #########################
	data_train = []
	for k in range(3):
		filename = "data_work/" + "Xtr" + str(k) + "_mat50_train.csv"	
		print(filename)
		data = open(filename, 'r')
		for line in data:
			temp = [np.float(i.strip()) for i in line.split(' ')]
			data_train.append(temp)

		data.close()

	X_train = np.array(data_train)
	X_train, mean = normalize(X_train)
	print(X_train.shape)

	if os.path.isfile("pretrained/gaussian_kernel_{0}_{1}.npy".format(X_train.shape[0], sigma)):
		K_train = np.load("pretrained/gaussian_kernel_{0}_{1}.npy".format(X_train.shape[0], sigma))
	else:
		K_train = gauss_kernel(X_train, 1)

	############### labels ##########################
	y_train = []
	for k in range(3):
		filename = "data_work/" + "Ytr" + str(k) + "_train.csv"
		print(filename)
		data = open(filename, 'r')
		next(data)
		for line in data:
			temp = [np.float(i.strip()) for i in line.split(',')]
			if temp[1] == 0:
				y_train.append(-1.0)
			else:
				y_train.append(1.0)
		data.close()

	y_train = np.array(y_train)
	print(y_train.shape)

	########### CV data ########################
	data_cv = []
	for k in range(3):
		filename = "data_work/" + "Xtr" + str(k) + "_mat50_cv.csv"	
		print(filename)
		data = open(filename, 'r')
		for line in data:
			temp = [np.float(i.strip()) for i in line.split(' ')]
			data_cv.append(temp)

		data.close()

	X_cv = np.array(data_cv)
	X_cv = np.subtract(X_cv, np.tile(mean,(X_cv.shape[0],1)))
	print(X_cv.shape)

	y_cv = []
	for k in range(3):
		filename = "data_work/" + "Ytr" + str(k) + "_cv.csv"
		print(filename)
		data = open(filename, 'r')
		for line in data:
			temp = [np.float(i.strip()) for i in line.split(',')]
			if temp[1] == 0:
				y_cv.append(-1.0)
			else:
				y_cv.append(1.0)
		data.close()

	y_cv = np.array(y_cv)
	print(y_cv.shape)

	return X_train, K_train, y_train, X_cv, y_cv

def fit(K_train, y_train):
	print("Training ....")
	N = y_train.shape[0]
	P = matrix(K_train)
	Q = matrix(np.expand_dims(-2*y_train, axis =1))
	G = matrix(np.concatenate((np.diag(-1*y_train), np.diag(y_train)), axis = 0))
	#print(G)
	temp1 = np.zeros([N, 1])
	temp2 = np.multiply(np.ones([N, 1]), 1/(2*lamb*N))
	H = matrix(np.concatenate((temp1, temp2), axis = 0))
	A = matrix(np.ones([1,N]))
	b = matrix(np.zeros([1])) 
	sol = qp(P,Q,G,H,A,b)
	alpha = np.array(sol['x'])
	return(alpha)

def single_dataset_finetuning(args):
	######## Generating Kernel ################
	X_train, K_train, y_train, X_cv, y_cv = read_data(args)

	######## Training #################
	alpha = fit(K_train, y_train)
	print(alpha)

	########## Computing bias #############
	bias = 0
	for i in range(len(y_train)):
		if y_train[i]*alpha[i] > 0 and y_train[i]*alpha[i] < 1/(2*lamb*len(y_train)):
			bias = y_train[i] - np.dot(K_train, alpha)[i]
			break

	print("Bias = {0}".format(bias))
	# for i in range(len(y_train)):
	# 	print(y_train[i] - np.dot(alpha.T, gauss_kernel_sample(X_train, X_train[i,:])))
	y_pred = np.zeros(y_cv.shape)
	res = 0
	for i in range(len(y_cv)):
		if np.dot(alpha.T, gauss_kernel_sample(X_train, X_cv[i,:]) + bias) > 0:
			y_pred[i] = 1.0
		else:
			y_pred[i] = -1.0

		if y_pred[i] == y_cv[i]:
			res += 1

	print("CV Accuracy = {0}".format(res/len(y_cv)))

	y_pred = np.zeros(y_train.shape)
	res = 0
	for i in range(len(y_train)):
		if np.dot(alpha.T, gauss_kernel_sample(X_train, X_train[i,:]) + bias) > 0:
			y_pred[i] = 1.0
		else:
			y_pred[i] = -1.0

		if y_pred[i] == y_train[i]:
			res += 1

	print("Train Accuracy = {0}".format(res/len(y_train)))

def single_dataset_training():
	filename = "Results/" + "SVM_" + str(lamb) + "L_" + str(sigma) + "S_norm.csv"

	result = open(filename, 'w+')
	result.write(",Bound\n")
	for k in range(3):
		######## Generating Kernel ################
		X_train, K_train, y_train, X_test = read_data_all(str(k))

		######## Training #################
		alpha = fit(K_train, y_train)
		print(alpha)

		########## Computing bias #############
		bias = 0
		for i in range(len(y_train)):
			if y_train[i]*alpha[i] > 0 and y_train[i]*alpha[i] < 1/(2*lamb*len(y_train)):
				bias = y_train[i] - np.dot(K_train, alpha)[i]
				break

		print("Bias = {0}".format(bias))
		
		y_pred = np.zeros(X_test.shape[0])
		for i in range(X_test.shape[0]):
			if np.dot(alpha.T, gauss_kernel_sample(X_train, X_test[i,:]) + bias) > 0:
				y_pred[i] = 1
			else:
				y_pred[i] = 0

			temp = str(k*X_test.shape[0] + i) +',' + str(int(y_pred[i]))
			result.write(temp)
			result.write('\n')

		y_pred = np.zeros(y_train.shape)
		res = 0
		for i in range(len(y_train)):
			if np.dot(alpha.T, gauss_kernel_sample(X_train, X_train[i,:]) + bias) > 0:
				y_pred[i] = 1.0
			else:
				y_pred[i] = -1.0

			if y_pred[i] == y_train[i]:
				res += 1

		print("Train Accuracy for {1} dataset = {0}".format(res/len(y_train), k))
	result.close()

def combined_dataset_training():
	filename = "Results/" + "SVM_" + str(lamb) + "L_" + str(sigma) + "S_full.csv"

	result = open(filename, 'w+')
	result.write("Id,Bound\n")
	######## Generating Kernel ################
	X_train, K_train, y_train, X_test = read_data_combined()

	######## Training #################
	alpha = fit(K_train, y_train)

	########## Computing bias #############
	bias = 0
	for i in range(len(y_train)):
		if y_train[i]*alpha[i] > 0 and y_train[i]*alpha[i] < 1/(2*lamb*len(y_train)):
			bias = y_train[i] - np.dot(K_train, alpha)[i]
			break

	print("Bias = {0}".format(bias))
	
	y_pred = np.zeros(X_test.shape[0])
	for i in range(X_test.shape[0]):
		if np.dot(alpha.T, gauss_kernel_sample(X_train, X_test[i,:]) + bias) > 0:
			y_pred[i] = 1
		else:
			y_pred[i] = 0

		temp = str(i) +',' + str(int(y_pred[i]))
		result.write(temp)
		result.write('\n')

	y_pred = np.zeros(y_train.shape)
	res = 0
	for i in range(len(y_train)):
		if np.dot(alpha.T, gauss_kernel_sample(X_train, X_train[i,:]) + bias) > 0:
			y_pred[i] = 1.0
		else:
			y_pred[i] = -1.0

		if y_pred[i] == y_train[i]:
			res += 1

	print("Train Accuracy for combined datasets = {0}".format(res/len(y_train)))
	result.close()

def combined_dataset_finetuning():
	######## Generating Kernel ################
	X_train, K_train, y_train, X_cv, y_cv = read_data_combined_finetune()

	######## Training #################
	alpha = fit(K_train, y_train)

	########## Computing bias #############
	bias = 0
	for i in range(len(y_train)):
		if y_train[i]*alpha[i] > 0 and y_train[i]*alpha[i] < 1/(2*lamb*len(y_train)):
			bias = y_train[i] - np.dot(K_train, alpha)[i]
			break

	print("Bias = {0}".format(bias))
	
	y_pred = np.zeros(X_cv.shape[0])
	res = 0
	for i in range(X_cv.shape[0]):
		if np.dot(alpha.T, gauss_kernel_sample(X_train, X_cv[i,:]) + bias) > 0:
			y_pred[i] = +1
		else:
			y_pred[i] = -1

		if y_pred[i] == y_train[i]:
			res += 1
	print("CV Accuracy for combined datasets = {0}".format(res/len(y_cv)))

	y_pred = np.zeros(y_train.shape)
	res = 0
	for i in range(len(y_train)):
		if np.dot(alpha.T, gauss_kernel_sample(X_train, X_train[i,:]) + bias) > 0:
			y_pred[i] = 1.0
		else:
			y_pred[i] = -1.0

		if y_pred[i] == y_train[i]:
			res += 1

	print("Train Accuracy for combined datasets = {0}".format(res/len(y_train)))

def main(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", help="dataset")
	args = parser.parse_args()

	#single_dataset_finetuning(args)
	#single_dataset_training()
	#combined_dataset_training()
	combined_dataset_finetuning()

	

	



if __name__ == "__main__":
    main(sys.argv[1:])

