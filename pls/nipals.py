import pandas as pd
import numpy as np
from numpy import linalg as ln

####################################################################
# FUNCTIONS FOR FILTERING AND SORTING THE DATASET

def filter (data, i_start, i_end, j_start, j_end):
    return [[data [i][j] for j in range(j_start,j_end)]for i in range(i_start,i_end)]

def _sort (x):
    for i in range (len(x)):
        for j in range(len(x[0])-1):
            idx=j+1
            while x[i][idx] < x[i][idx-1]:
                print("x idx=", idx,";","idx -1=",idx-1)
                tmp=x[i][idx]
            
                x[i][idx]=x[i][idx-1]
            
                x[i][idx-1]=tmp
            
                if idx >1:
                
                    idx=idx-1



    return(x)

####################################################################
#BASIC STATS (Note: the covariance function is commented because actually it is not used in this script. I included here just in case).

def avg (x,row,col):
    return [sum(x[i])/row for i in range(col)]

def sd (x, col):
    return [np.std(x[i]) for i in range(col)]


####################################################################
#MATRIX OPERATIONS

def transpose (matrix, col, row):
    return [[matrix [i][j] for i in range(col)] for j in range(row)]

def mul(xx_matrix, xy_matrix):
    mul=[[0 for j in range(len(xx_matrix[0]))] for i in range(len(xy_matrix))]
    for i in range(len(xy_matrix)):
        for j in range(len(xx_matrix[0])):
            for k in range(len(xx_matrix)):
                mul[i][j]+= xx_matrix[k][j]*xy_matrix[i][k]
    return(mul)


if __name__ == "__main__":
    
   
