import pandas as pd
import numpy as np
from numpy.linalg import inv
import math as m
import sys

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
            print("cov= ",cov[i][j],"sd i", sd_x[i], " sd k", sd_x[j],"cov/sd", cov[i][j]/(sd_x[i]*sd_x[j]))
    return(cor_xy)


if __name__ == "__main__":
    
    argv=sys.argv[:]
    
    if len(argv)<2:
        print("1 argument required. Provide data file name")
        sys.exit(0)
    
    data=pd.read_csv(argv[1],header= None)
    row=data.shape[0]
    col=data.shape[1]
    print("** dataset dimensions **")
    print(row)
    print(col)
    mean=avg(data)
    stdev=sd(data)
    print(stdev)
    
    covar=cov(data, mean)
    correl=cor(covar, stdev)
    print("---------CORRELATION MATRIX---------")
    print(correl)
   

