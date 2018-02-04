from cvxopt import matrix
from cvxopt.solvers import qp
import numpy as np

def fit(K_train, y_train, lamb):
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