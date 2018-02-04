from __future__ import division
import numpy as np

##returns dictionary of sequences
def find_sequences(X,len_list):
    dic = {}
    for k in len_list:
        for i in np.arange(X.shape[0]):
            x = X[i,0]
            for l in np.arange(len(x)-k):
                mainSeq = x[l:l+k]
                if(mainSeq not in dic.keys()):
                    dic[mainSeq] = 1
                else:
                    dic[mainSeq]+= 1
                
    return dic
##given a dictionary and X return dataset of frequencies wrt to the sequences
def seq_to_mat(X,seq):
    mat = np.zeros((X.shape[0],len(seq)))
    for i in np.arange(X.shape[0]):
        x = X[i,0]
        for j in np.arange(len(seq)):
            subSeq = seq.keys()[j] 
            k = len(subSeq)
            for l in np.arange(len(x)):
                if(subSeq == x[l:l+k]):
                    mat[i,j] += 1
    return mat

def trim_sequences(X,seq,ratio):
    new_dic = {}
    m = X.shape[0]
    for j in seq.keys():
        count = 0
        for i in np.arange(m):
            if(j in X[i,0]):
                count += 1
        if(count >= ratio*m):
            new_dic[j] = seq[j]
    return new_dic

