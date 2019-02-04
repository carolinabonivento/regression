import pandas as pd
import numpy as np
from numpy import linalg as ln
from numpy.linalg import inv
import sys
import scipy
from scipy import stats

#Select columns and rows

def filter (data, i_start, i_end, j_start, j_end):
    tmp=[[data [i][j] for j in range(j_start,j_end)]for i in range(i_start,i_end)]
    printing(tmp)
    return(tmp)

#Select columns

def sel_col (data, idx):
    tmp=[data[i] for i in range(idx+1)]
    print(tmp)
    return(tmp)

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
    tmp=[sum(x[i])/n for i in range(vars)]
    print(tmp)
    return(rmp)

def sd (x, vars):
    tmp= [np.std(x[i]) for i in range(vars)]
    print(tmp)
    return(tmp)

def sum_squares (x, mean):
    sq=[0]
    for i in range(len(x)):
        for j in range(len(x[0])):
            sq += (x[i][j]-mean)**2

    printing(sq)
    return(sq)

def f_stat (R2f,R2r, dfr, dff):
    tmp=[((R2f-R2r)/(dfr-dff))/((1-R2f)/dff)]
    print(tmp)
    return(tmp)

#Matrix transpose and matrix-matrix multiplication

def transpose (matrix, col, row):
    tmp=[[matrix [j][i] for j in range(col)] for i in range(row)]
    printing(tmp)
    return(tmp)

def mul(xx_matrix, xy_matrix):
    mul=[[0 for j in range(len(xx_matrix[0]))] for i in range(len(xy_matrix))]
    for i in range(len(xy_matrix)):
        for j in range(len(xx_matrix[0])):
            for k in range(len(xx_matrix)):
                mul[i][j]+= xx_matrix[k][j]*xy_matrix[i][k]
    printing(mul)
    return(mul)

#Eigen vector

def eigen (matrix):
    
    e_tmp= ln.eig(matrix)
    tmp=[[e_tmp[i][j] for j in range(len(e_tmp[0]))] for i in range(1)]
    print("** maximum eigenvector - not sorted **")
    printing(tmp)
    print("")
    s_tmp=_sort(tmp)
    print("** maximum eigenvector sorted **")
    printing (s_tmp)
    return(s_tmp)

#Divide matrix elements by a scalar

def divide(matrix,scalar):
    tmp=[[matrix[i][j]/scalar for j in range(len(matrix[0]))] for i in range(len(matrix))]
    printing(tmp)
    return(tmp)

# Subtract element by element of two matrices

def diff(mat_1, mat_2):
    tmp=[[mat_1[i][j]-mat_2[i][j] for j in range(len(mat_1[0]))] for i in range(len(mat_1))]
    printing(tmp)
    return(tmp)

#Printing with %8.12f numbers' format

def printing (matrix):
    print([["%8.3f" % matrix[i][j] for j in range(len(matrix[0]))]for i in range(len(matrix))])

if __name__ == "__main__":
    
    argv=sys.argv[:]
    
    if len(argv)<2:
        print(" 1 arguments required. Provide data file name")
        sys.exit(0)

    data=pd.read_csv(argv[1],header= None)

    print("** dataset dimensions **")
    print("number of subjects= ", data.shape[0])
    print("number of variables= ", data.shape[1])
    print("")

    print("X  matrix")
    x= filter (data, 0, data.shape[1], 0, data.shape[0]-3)
    print("")

    print("Y  matrix")
    y= filter (data, 0, 2, 3, data.shape[0])
    print("")

    print(len(x), len(x[0]))
    print(len(y), len(y[0]))
    #transpose x
    print("transposed x matrix")
    tx= transpose (x, len(x),len(x[0]))
    print("")
    #matrix multiplication
    print("s= mat mul xt * y")
    s=mul (tx,y)
    print("")

    #transpose s
    print("transposed s matrix")
    st= transpose (s, len(s), len(s[0]))
    print("")

    #s*st
    print("ss= s * st")
    ss=mul (s,st)
    print("")

    #eigen vector
    print("w = eigen max (ss)")
    w=eigen(ss)
    print("")

    '''
    print("type SS", type(ss[0][0]))
    print("")

    print("t= x * w")
    t=mul (x,w)
    print("type t", type(t), type(t[0][0]))
    print("")

    print("SS")
    x_tx=mul (x,tx)
    print("")
    #eigen vector
    print("eigen tx x")
    eig_tx_x= eigen (tx_x)
    print("")
    print("eigen x tx")
    eig_x_tx= eigen (x_tx)
    print("")
    print("x * eigen vector tx_t")
    t=mul(x,eig_tx_x)
    '''

