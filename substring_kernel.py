import numpy as np

def create_matrix(weight, data, bag):
	mat = np.zeros((len(data), len(bag)))
	row, col = mat.shape
	for i in range(row):
		print(i)
		for j in range(col):
			mat[i,j] = find(data[i], bag[j], weight)
	return mat


def find(main, sub, weight):
	i = 0
	j = 0
	start_ind = 0
	stop_ind = 0
	acc = 0
	while i < len(main):
		if main[i] == sub[j]:
			if j == 0:
				start_ind = i + 1
				j += 1
				i += 1
			elif j == len(sub)-1:
				stop_ind = i + 1
				acc += weight**(stop_ind - start_ind + 1)
				j = 0
				i = start_ind
			else:
				j += 1
				i +=1
		else: 
			i += 1

	return(acc)
