import pandas as pd
import numpy as np
from numpy import linalg as ln
from numpy.linalg import inv
import sys

#Select columns and rows

def filter (data, i_start, i_end, j_start, j_end):
    return [[data [i][j] for j in range(j_start,j_end)]for i in range(i_start,i_end)]

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

#Basic stats

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

#Divide matrix elements by a scalar

def divide(matrix,scalar):
    return[[matrix[i][j]/scalar for j in range(len(matrix[0]))] for i in range(len(matrix))]

# Subtract element by element of two matrices

def diff(mat_1, mat_2):
    return [[mat_1[i][j]-mat_2[i][j] for j in range(len(mat_1[0]))] for i in range(len(mat_1))]

#Printing with %8.12f numbers' format

def printing (matrix):
    print([["%8.12f" % matrix[i][j] for j in range(len(matrix[0]))]for i in range(len(matrix))])

if __name__ == "__main__":
    
    argv=sys.argv[:]
    
    if len(argv)<2:
        print("1 argument required. Provide data file name")
        sys.exit(0)

    data=pd.read_csv(argv[1],header= None)
    row=data.shape[0]
    col=data.shape[1]
    print("** dataset dimensions **")
    print("number of subjects= ", row)
    print("number of variables= ", col)
    print("")

    # Select the columns with the independent variables and build the 'x' matrix

    _x=filter (data, 0, 3, 0, row)

    # Select the columns with the dependent variables and build the 'y' matrix

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

    w=[]
    t=[]
    c=[]
    p=[]

    for d in range(x_col):
        print("")
        print("*******************************")
        print("THIS IN THE ITERATION N°-->",d+1)
        print("*******************************")
        print("")
        
        '''
        print("** x matrix **"")
        printing(_x)
        print("** y matrix **"")
        printing(_y)
        '''
        
        # Transpose 'x' and 'y' matrices
        
        _xT=transpose (_x, x_col, x_row)
        _yT=transpose (_y, y_col, y_row)

        # Covariance matrix - computed with deflated 'x' and 'y' matrices at each iteration
        
        s=mul(_xT,_y)
        s_col=len(s)
        s_row=(len(s[0]))
        
        # Transpose the 's' matrix
        
        st=transpose (s, s_col, s_row)

        # Multiply 's' and 'st' matrices - 1st step for calculating the principal component
        
        ss=mul(s,st)
        
        # Find the maximum eigenvector
        
        _eig=ln.eig(ss)
        tmp=[[_eig[i][j] for j in range(len(_eig[0]))] for i in range(1)]
        print("** maximum eigenvector **")
        printing(tmp)
        print("")
        tmp=_sort(tmp)
        w.append(tmp)
        print("** maximum eigenvector sorted **")
        printing(w[d])
        print("")
    
        # T scores' column
        
        t_tmp=mul(_x,w[d])
        t.append(t_tmp)
 
        # Transpose t
        
        t_col=len(t_tmp)
        t_row=len(t_tmp[0])
        tt_tmp=transpose(t_tmp, t_col, t_row)

        # Compute the loading vectors 'c' and 'p'
        
        tt_t=mul(tt_tmp,t_tmp)
        xT_t=mul(_xT,t_tmp)
        yT_t=mul(_yT,t_tmp)
        
        # 'p'
        
        tmp=divide(xT_t,tt_t[0][0])
        p.append(tmp)
        pt=transpose(p[d], len(p[d]), len(p[d][0]))
        print("** loading vector 'p' **")
        print(p[d])
        print("")
        
        # 'c'
        
        tmp=divide(yT_t,tt_t[0][0])
        c.append(tmp)
        ct=transpose(c[d], len(c[d]), len(c[d][0]))
        print("** loading vector 'c' **")
        print(c[d])
        print("")
        
        # Deflate '_x' and '_y'
        
        t_pt=mul(t[d],pt)
        _x=diff(_x,t_pt)
        
        t_ct=mul(t[d],ct)
        _y=diff(_y,t_ct)
        
        d+=1

    # Prepare P W T and tranpose

    P=[p[i][j] for j in range(len(p[0])) for i in range(len(p))]
    Pt=transpose(P,len(P),len(P[0]))
    W=[w[i][j] for j in range(len(w[0])) for i in range(len(w))]
    Wt=transpose(W,len(W),len(W[0]))
    T=[t[i][j] for j in range(len(t[0])) for i in range(len(t))]
    Tt=transpose(T,len(T),len(T[0]))

    # Low-rank PLS decomposition of the input matrix X

    Pt_W=mul(Pt,W)
    inv_Pt_W=inv(Pt_W)
    tmp=mul(W,inv_Pt_W)
    X_pls=mul(tmp,Tt)

    # Estimate regression parameters

    B_pls=mul(X_pls,_y)
    print("")
    print("** Estimated regression parameters (Bpls) for rank= ", len(W))
    printing(B_pls)
    print("")

    print("todo: loop for computing betas; F contrasts; ")






