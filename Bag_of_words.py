import numpy as np

def build_BOW(dataset, bag, k):
	# Read files and build bow of words of length k		
	filename = "data_work/" + "Xtr" + dataset + ".csv"	
	print(filename)
	print(k)
	data = open(filename, 'r')
	for line in data:
		line = line.strip('\n')
		bag_of_words(bag, line, k)

# Finding substrings of length k
def bag_of_words(bag, string, k):
	n = len(string)
	for i in range(0, n-k+1):
		bag.append(string[i:i+k])

def build_BOW_entire(dataset, bag, k):
	# Read files and build bow of words of length k		
	for line in dataset:
		bag_of_words(bag, line, k)

