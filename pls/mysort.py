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
            while x[i][idx] < x[i][j]:
                
                tmp=x[i][idx]
            
                x[i][idx]=x[i][j]
            
                x[i][j]=tmp
            
                if idx >1:
                
                    idx=j
    return(x)

def m_sort (x,col):
    tmp=[0]
    for i in range (len(x)):
        if i<len(x)-1:
            idx=i+1
            print(x[idx])


def transpose (matrix, col, row):
    return [[matrix [i][j] for i in range(col)] for j in range(row)]


#Printing with %8.12f numbers' format

def printing (matrix):
    print([["%8.2f" % matrix[i][j] for j in range(len(matrix[0]))]for i in range(len(matrix))])

if __name__ == "__main__":
    
    argv=sys.argv[:]
    
    if len(argv)<3:
        print("2 arguments required. Provide data file name, starting column y")
        sys.exit(0)

    data=pd.read_csv(argv[1],header= None)
    row=data.shape[0]
    col=data.shape[1]
    print("** dataset dimensions **")
    print("number of subjects= ", row)
    print("number of variables= ", col)
    print("")

    # Select the columns with the independent variables and build the 'x' matrix

    id_col=int(argv[2])
    _x=filter (data, 0, id_col, 0, row)
    print(_x)
    print(len(_x), len(_x[0]))
#   s=_sort(_x)
    print("")
#   print(s)
    print("")
    t_data=transpose(data,col,row)
    printing(t_data)
    print("len x",len(t_data))
    print("check point")
    print(t_data[0+1][1],t_data[0][1])
    if t_data[0+1][1]>t_data[0][1]:
        print(t_data[0][1])


    ss=m_sort (t_data,1)
    print("m sort")
    print(ss)
    print("")

