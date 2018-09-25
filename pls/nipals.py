import pandas as pd
import numpy as np
from numpy import linalg as ln
import sys

# Select columns and rows
def filter (data, i_start, i_end, j_start, j_end):
    return [[data [i][j] for j in range(j_start,j_end)]for i in range(i_start,i_end)]

# Sorting algorithm (insert sort)
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

#Basic Stats

def avg (x,row,col):
    return [sum(x[i])/row for i in range(col)]

def sd (x, col):
    return [np.std(x[i]) for i in range(col)]

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

#Divide matrix element by a scalar

def divide(matrix,scalar):
    return[[matrix[i][j]/scalar for j in range(len(matrix[0]))] for i in range(len(matrix))]

def printing (matrix):
    print([["%8.12f" % matrix[i][j] for j in range(len(matrix[0]))] for i in range(len(matrix))])

if __name__ == "__main__":
    
    argv=sys.argv[:]
    
    if len(argv)<2:
        print("1 argument required. Provide data file name")
        sys.exit(0)

    # load the data
    data=pd.read_csv(argv[1],header= None)
    row=data.shape[0]
    col=data.shape[1]
    print("** dataset dimensions **")
    print("number of subjects= ", row)
    print("number of variables= ", col)
    print("")
    # select the columns with the independent variables and build the 'x' matrix
    _x=filter (data, 0, 3, 0, row)
    # select the columns with the dependent variables and build the 'y' matrix
    _y=filter (data, 3,col , 0, row)


    x_col=len(_x)
    x_row=(len(_x[0]))
    print("** x matrix dimension **")
    print("number of independent variables= ",x_col)
    print("number of subjects= ",x_row)
    print("")
    y_col=len(_y)
    y_row=(len(_y[0]))
    print("** y matrix dimension **")
    print("number of dependent variables= ",y_col)
    print("number of subjects= ",y_row)
    print("")

    # do the transpose of 'x' and 'y' matrices
    _xT=transpose (_x, x_col, x_row)

    _yT=transpose (_y, y_col, y_row)

    # covariance matrix - this will be a subsequent, iterative procedure (i.e. the covariance will be computed each time with deflated 'x' and 'y' matrices

    s=mul(_xT,_y)
    s_col=len(s)
    s_row=(len(s[0]))
    # transpose the 's' matrix
    st=transpose (s, s_col, s_row)

    # multiply 's' and 'st' matrices - 1st step for calculating the principal component
    ss=mul(s,st)

    # find the maximum eigenvector
    _eig=ln.eig(ss)
    w=[[_eig[i][j] for j in range(len(_eig[0]))] for i in range(1)]
    print("** maximum eigenvector **")
    printing(w)
    print("")
    sorted_w=_sort(w)
    print("** maximum eigenvector sorted **")
    printing(sorted_w)
    print("")
    
    # T scores' column
    t=mul(_x,sorted_w)

    # Transpose t
    t_col=len(t)
    t_row=len(t[0])
    tt=transpose(t, t_col, t_row)

    # Compute the loading vectors 'c' and 'p'
    tt_t=mul(tt,t)
    xT_t=mul(_xT,t)
    yT_t=mul(_yT,t)

    p=divide(xT_t,tt_t)
    c=divide(yT_t,tt_t)
    print("** loading vector 'p' **")
    printing(p)
    print("")
    print("** loading vector 'c' **")
    printing(c)
    print("")

    print("todo: compute the regressors matrix X+ and betas; F contrasts; do a for loop so that the code tests more than 1 component")



