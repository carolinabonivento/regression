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
    print([["%8.12f" % matrix[i][j] for j in range(len(matrix[0]))]for i in range(len(matrix))])

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
    #print(_x)
    # Select the columns with the dependent variables and build the 'y' matrix

    _y=filter (data, id_col,col , 0, row)
    y_mean=avg(_y, len(_y[0]), len(_y))
    #print(_y)
    print("** x matrix dimension **")
    print("number of independent variables= ",len(_x))
    print("number of subjects= ",len(_x[0]))
    print("")
    print("** y matrix dimension **")
    print("number of dependent variables= ",len(_y))
    print("number of subjects= ",len(_y[0]))
    print("")

    w=[]
    t=[]
    c=[]
    p=[]

    # Initialize input matrices x_pls and y_pls that will be used to calculate W, T, P and C - x_pls and y_pls will be "deflated" at each loop

    x_pls=_x
    y_pls=_y

    for d in range(len(_x)):
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
        
        x_plsT=transpose (x_pls, len(x_pls), len(x_pls[0]))
        y_plsT=transpose (y_pls,len(y_pls), len(y_pls[0]))

        # Covariance matrix - computed with deflated 'x' and 'y' matrices at each iteration
        
        s=mul(x_plsT,y_pls)

        # Transpose the 's' matrix
        
        st=transpose (s, len(s), len(s[0]))

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
        print(type(w[d]),type(w[d][0]))
        print("")
    
        # T scores' column
        
        t_tmp=mul(x_pls,w[d])
        t.append(t_tmp)
 
        # Transpose t

        tt_tmp=transpose(t_tmp, len(t_tmp), len(t_tmp[0]))

        # Compute the loading vectors 'c' and 'p'
        
        tt_t=mul(tt_tmp,t_tmp)
        x_plsT_t=mul(x_plsT,t_tmp)
        y_plsT_t=mul(y_plsT,t_tmp)
        
        # 'p'
        
        tmp=divide(x_plsT_t,tt_t[0][0])
        p.append(tmp)
        pt=transpose(p[d], len(p[d]), len(p[d][0]))
        print("** loading vector 'p' **")
        print(p[d])
        print("")
        
        # 'c'
        
        tmp=divide(y_plsT_t,tt_t[0][0])
        c.append(tmp)
        ct=transpose(c[d], len(c[d]), len(c[d][0]))
        print("** loading vector 'c' **")
        print(c[d])
        print("")
        
        # Deflate '_x' and '_y'
        
        t_pt=mul(t[d],pt)
        x_pls=diff(x_pls,t_pt)
        
        t_ct=mul(t[d],ct)
        y_pls=diff(y_pls,t_ct)
        
        d+=1

    # Prepare P W T and tranpose

    P=[p[i][j] for j in range(len(p[0])) for i in range(len(p))]
    W=[w[i][j] for j in range(len(w[0])) for i in range(len(w))]
    T=[t[i][j] for j in range(len(t[0])) for i in range(len(t))]

    for d in range(len(W)):
        print("")
        print("*******************************")
        print("THIS IS THE ITERATION N°-->",d+1)
        print("*******************************")
        print("")
        
        selected_p=sel_col(P,d)
        selected_w=sel_col(W,d)
        selected_t=sel_col(T,d)
        
      
        print("P[",d+1,"]= ",selected_p)
        print("W[",d+1,"]= ",selected_w)
        print("T[",d+1,"]= ",selected_t)
      
        trans_sel_p=transpose(selected_p,len(selected_p),len(selected_p[0]))
        trans_sel_w=transpose(selected_w,len(selected_w),len(selected_w[0]))
        trans_sel_t=transpose(selected_t,len(selected_t),len(selected_t[0]))
        
        '''
        print("Pt[",d+1,"]= ",trans_sel_p)
        print("Wt[",d+1,"]= ",trans_sel_w)
        print("Tt[",d+1,"]= ",trans_sel_t)
        '''
        
        # Low-rank PLS decomposition of the input matrix X
        
        Pt_W=mul(trans_sel_p,selected_w)
        inv_Pt_W=inv(Pt_W)
        tmp=mul(selected_w,inv_Pt_W)
        XX_pls=mul(tmp,trans_sel_t)
        print("")
        print("** Input matrix X decomposed at rank= ", d+1)
        printing(XX_pls)
        print("")
    
        # Estimate regression parameters
        
        B_pls=mul(XX_pls,_y)
        print("PLS BETAs")
        printing(B_pls)
        # Estimate Y (Y =XB - see Stott et al. 2017)
        
        Y_est=mul(_x,B_pls)
        Ye_mean=avg(Y_est, len(Y_est[0]), len(Y_est))
        
        #print("")
        #print("** Estimated Y (Y=X*Bpls) for rank= ", d+1)
        #printing(Y_est)

       
        # Test difference
        for var in range(len(_y)):
            print("")
            print("******************************************************************")
            print("R2 and F for dependent variable N°-->",var+1,"for RANK N°-->", d+1)
            print("******************************************************************")
            print("")
            selected_Ye=sel_col(Y_est, var)
            selected_y=sel_col(_y, var)
            ssr=sum_squares (selected_Ye,Ye_mean[var])
            sst=sum_squares(selected_y,y_mean[var])
            print("Sum of Squares regression= ",ssr[0])
            print("Sum of Squares total= ", sst[0])
            R2=ssr/sst
            print("R2= ", R2[0])
            
            
            # Degrees of freedom
            
            dfr=len(_x[0])-1
            dff=len(_x[0])-(len(_x)+1)
            
            # F statistic
            
            F=f_stat (R2,0, dfr, dff)
            p_val= 1- (scipy.stats.f.cdf(F,dfr,dff))
            print("F(",dfr,",",dff,") = ", F[0][0], "p = ", p_val[0][0])
            print("")

    d+=1
