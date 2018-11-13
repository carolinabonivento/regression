import pandas as pd
import numpy as np
from numpy import linalg as ln
from numpy.linalg import inv
import sys
import scipy
from scipy import stats

#Select columns and rows

def filter (data, i_start, i_end, j_start, j_end):
    return [[data [i][j] for j in range(j_start,j_end)]for i in range(i_start,i_end)]

#Select columns

def sel_col (data, idx):
    return [data[i] for i in range(idx+1)]

#Sorting algorithm (insert sort)

def _sort (x):
    for i in range (len(x)):
        for j in range(len(x[0])-1):
            idx=j+1
            while x[i][idx] < x[i][idx-1]:
                
                tmp=x[i][idx]
            
                x[i][idx]=x[i][idx-1]
            
                x[i][idx-1]=tmp
            
                if idx >1:
                
                    idx=idx-1
    return(x)

# Stats

def avg (x,n,vars):
    return [sum(x[i])/n for i in range(vars)]

def sd (x, vars):
    return [np.std(x[i]) for i in range(vars)]

def sum_squares (x, mean):
    sq=[0]
    for i in range(len(x)):
        for j in range(len(x[0])):
            sq += (x[i][j]-mean)**2
    return(sq)

def f_stat (R2f,R2r, dfr, dff):
    return[((R2f-R2r)/(dfr-dff))/((1-R2f)/dff)]

#Matrix transpose and matrix-matrix multiplication

def transpose (matrix, col, row):
    return [[matrix [i][j] for i in range(col)] for j in range(row)]

def mul(xx_matrix, xy_matrix):
    mul=[[0 for j in range(len(xx_matrix[0]))] for i in range(len(xy_matrix))]
    for i in range(len(xy_matrix)):
        for j in range(len(xx_matrix[0])):
            for k in range(len(xx_matrix)):
                mul[i][j]+= xx_matrix[k][j]*xy_matrix[i][k]
    return(mul)

#Divide matrix elements by a scalar

def divide(matrix,scalar):
    return[[matrix[i][j]/scalar for j in range(len(matrix[0]))] for i in range(len(matrix))]

# Subtract element by element of two matrices

def diff(mat_1, mat_2):
    return [[mat_1[i][j]-mat_2[i][j] for j in range(len(mat_1[0]))] for i in range(len(mat_1))]

#Printing with %8.12f numbers' format

def printing (matrix):
    print([["%8.5f" % matrix[i][j] for j in range(len(matrix[0]))]for i in range(len(matrix))])

if __name__ == "__main__":
    
    argv=sys.argv[:]
    
    if len(argv)<2:
        print("2 arguments required. Provide data file name")
        sys.exit(0)

    data=pd.read_csv(argv[1],header= None)

    print("** dataset dimensions **")
    print("number of subjects= ", data.shape[0])
    print("number of variables= ", data.shape[1])
    print("")

    print(data)
    # test squared matrix

    x_squared= filter (data, 0, 3, 0, data.shape[0]-3)
    print("select data for squared matrix")
    print(x_squared)

    tx_squared= transpose (x_squared, 3, 3)
    print("transposed squared matrix")
    print(tx_squared)

    tx_x_squared=mul (x_squared, tx_squared)
    print("mat mul")
    print(tx_x_squared)

    eig_tx_x_squared= ln.eig(tx_x_squared)
    print("Eigen not sorted")
    print(eig_tx_x_squared)

    tmp=[[eig_tx_x_squared[i][j] for j in range(len(eig_tx_x_squared[0]))] for i in range(1)]
    print("** maximum eigenvector **")
    printing(tmp)

    print("")
    tmp=_sort(tmp)
    printing(tmp)














    # test rectangular matrix
    '''
    x_rect= filter (data, 0, 4, 0, data.shape[0]-3)
    print("select data for squared matrix")
    print(x_rect)
    '''


