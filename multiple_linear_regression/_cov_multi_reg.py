import pandas as pd
import numpy as np
from numpy.linalg import inv
import math as m
import scipy
from scipy import stats


def avg (x):
    return [sum(x[i])/row for i in range(col)]

def sd (x):
    return [np.std(x[i]) for i in range(col)]

def cov (x, md_x):
    cov_xy=[[0 for r in range(col)] for c in range(col)]
    for i in range(col):
        for j in range (col):
            for k in range (row):
                cov_xy[i][j]+=((data[i][k]-md_x[i])*(data[j][k]-md_x[j]))/(row)
    return(cov_xy)

def cor (cov, sd_x):
    cor_xy=[[0 for r in range(col)] for c in range(col)]
    for i in range(col):
        for j in range (col):
            cor_xy[i][j] = cov[i][j]/(sd_x[i]*sd_x[j])
    return(cor_xy)

def filter (data, i_start, i_end, j_start, j_end):
    return [[data [i][j] for j in range(j_start,j_end)]for i in range(i_start,i_end)]

def v_filter (data, i_start, i_end, j_start, j_end):
    return [[data [j] for j in range(j_start,j_end)] for i in range(i_start,i_end)]

def conv_list(x):
    return [[x[i][j] for j in range(len(x[0]))] for i in range(len(x))]

def transpose (matrix, col, row):
    return [[matrix [i][j] for i in range(col)] for j in range(row)]

def mul(xx_matrix, xy_matrix):
    mul=[[0 for j in range(len(xy_matrix[0]))] for i in range(len(xx_matrix))]
    for i in range(len(xx_matrix)):
        for j in range(len(xy_matrix[0])):
            for k in range(len(xy_matrix)):
                mul[i][j]+= xx_matrix[i][k]*xy_matrix[k][j]
    return(mul)


def estimate(xx_matrix, xy_matrix, intercept):
    mul=[[intercept for j in range(len(xx_matrix[0]))] for i in range(len(xy_matrix[0]))]
    for i in range(len(xy_matrix[0])):
        for j in range(len(xx_matrix[0])):
            for k in range(len(xx_matrix)):
                mul[i][j]+= xx_matrix[k][j]*xy_matrix[k][i]
    return(mul)

def sum_squares (x, mean):
    sq=[0]
    for i in range(len(x)):
        for j in range(len(x[0])):
            sq += (x[i][j]-mean)**2
    return(sq)

def r2 (ssr,sst,N,k):
    return[1-((1-(ssr/sst))**2)*((N-1)/(N-k-1))]

def f_stat (R2f,R2r, dfr, dff):
    return[((R2f-R2r)/(dfr-dff))/((1-R2f)/dff)]


if __name__ == "__main__":
    
  
