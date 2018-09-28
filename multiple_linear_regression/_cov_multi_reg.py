import sys
import pandas as pd
import numpy as np
from numpy.linalg import inv
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
    
    
    # covariance matrix
    
    covar=cov(data, mean)

    # filter columns with dependent variable
    y_col=0
    y=data[y_col]
    y=v_filter (y, y_col,1, y_col,len(data))
    
    # filter columns with regressors
    i_start=y_col+1
    i_end=col
    j_start=0
    j_end=len(data[1])
    x= filter(data, i_start,i_end, j_start, j_end)

    # select covariances between regressors
    
    j_start=i_start
    j_end=len(covar[0])
    c_xx= filter (covar, i_start,i_end, j_start, j_end)
    
    # inverse of the matrix with covariances between regressors
    
    i_cxx=inv(c_xx)
    i_cxx= conv_list(i_cxx)
    
    # select covariances between y and regressors
    
    i_start=y_col
    i_end=i_start+1
    j_start=i_start+1
    j_end=len(covar[0])
    c_xy= filter (covar, i_start,i_end, j_start, j_end)
    c_xy=transpose(c_xy,len(c_xy), len(c_xy[0]))
    
    # do beta - matrix product between inverse of the matrix of covariance between regressors and matrix of covariance between y and regressors
    
    b=mul(i_cxx, c_xy)
    
    # estimate intercept
    
    x_means=v_filter (mean, i_start,i_end, j_start,j_end)
    y_means=mean[y_col]
    s=mul(x_means, b)
    intercept=y_means-s
    
    print("coefficients")
    print("intercept =", intercept[0][0])
    print("betas vector = ", b)
    print("betas = ", b[0][0], b[1][0])
    
    # estimate expected value of y
    
    e=estimate(x, b, intercept[0][0])
    
    # sum of squares (y estimated - y mean) and (y observed - y mean) - R2
    
    ssr=sum_squares (e,mean[y_col])
    sst=sum_squares(y,mean[y_col])
    print("Sum of Squares regression= ",ssr[0])
    print("Sum of Squares total= ", sst[0])
    R2=ssr/sst
    print("R2= ", R2[0])
    
    # degrees of freedom
    # tot df (df of the B0 condition)
    dfr=len(x[0])-1
    # error df= n of subj - n of indip var -1
    dff=len(x[0])-(len(x)+1)
    
    # F statistic
    
    F=f_stat (R2,0, dfr, dff)
    p_val= 1-(scipy.stats.f.cdf(F,dfr,dff))
    print("F(",dfr,",",dff,") = ", F[0][0], "p = ", p_val[0][0])
    print("")
    
    ########################################################################################################
    
    i_start=1
    i_end=len(b)
    j_start=0
    print("regressors matrix starts at index :", i_start, "and ends at index :", i_end-1)
    j_end=len(x[0])
    
    x_i=filter(x,i_start,i_end, j_start, j_end)
    print("x_i= ",x_i, len(x_i[0]))
    # test effect of x1 = multi regregression - regression with only effect of x2
    # estimate expected value of y
    
    i_start= 2
    i_end= 3
    j_start= 2
    j_end= 3
    c_xx= filter (covar, i_start,i_end, j_start, j_end)
    
    # inverse of the matrix with covariances between regressors
    
    i_cxx=inv(c_xx)
    i_cxx= conv_list(i_cxx)
    
    # select covariances between y and regressors
    
    i_start=y_col
    i_end=i_start+1
    
    c_xy= filter (covar, i_start,i_end, j_start, j_end)
    c_xy=transpose(c_xy,len(c_xy), len(c_xy[0]))
    
    b_i=mul(i_cxx, c_xy)
    
    # estimate intercept
    
    x_means=v_filter (mean, i_start,i_end, j_start,j_end)
    y_means=mean[y_col]
    s=mul(x_means, b_i)
    intercept=y_means-s
    
    print("coefficients")
    print("intercept =", intercept[0][0])
    print("betas vector = ", b_i, len(b_i))
    
    # estimate expected values of y
    
    e=estimate(x_i, b_i, intercept[0][0])
    
    # sum of squares (y estimated - y mean) and (y observed - y mean) - R2
    
    ssr=sum_squares (e,mean[y_col])
    sst=sum_squares(y,mean[y_col])
    print("Sum of Squares regression= ",ssr[0])
    print("Sum of Squares total= ", sst[0])
    
    # compute R2
    
    R2r=ssr/sst
    print("R2r = ", R2r[0])
    
    # degrees of freedom for new reference model
    
    dfr_i=len(x_i[0])- (len(b_i)+1)
    
    # F statistic
    
    F_i=f_stat (R2,R2r, dfr_i, dff)
    p_val_i= 1- (scipy.stats.f.cdf(F_i,dfr_i,dff))
    print("F (", dfr_i,",", dff, ") = ",F_i[0][0], "p = ", p_val_i [0][0])











