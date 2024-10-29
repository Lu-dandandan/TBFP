import numpy as np 
from numba import jit  

# 这个改完了 

@jit(nopython=True)
def fun(f):
    return f
@jit(nopython=True)
def sig_order2(x_cut):
    # 输入一维x横向量
    X_sig_cut = (x_cut**2).reshape((len(x_cut),1))
    return X_sig_cut 
@jit(nopython=True)
def sig_order4(x_cut):
    # 输入一维x横向量  
    X_sig_cut = np.empty((len(x_cut),2)) # 有两列
    X0 = x_cut**2
    X_sig_cut[:,0] = np.copy(X0)
    X_sig_cut[:,1] = X0**2
    return X_sig_cut
@jit(nopython=True)
def sele_sig(sig_order): 
    # 返回函数
    if sig_order == 2:
        return fun(sig_order2)
    elif sig_order == 4:
        return fun(sig_order4)
    else:  
        raise ValueError("The sig_order inputted is out of range, Please input an int in [2,4]")
    
@jit(nopython=True)
def trend_order0(t_cut,X0):
    # 返回函数
    return X0
@jit(nopython=True)
def trend_order1(t_cut,X0):  
    # 输入t一维列，X0一维列向量 
    return np.hstack((X0,t_cut))
@jit(nopython=True)
def trend_order2(t_cut,X0):
    # 输入t一维列向量，X0一维列向量
    X2 = t_cut**2
    return np.hstack((X0,t_cut,X2))
@jit(nopython=True)
def trend_order3(t_cut,X0):
    # 输入t一维列向量，X0一维列向量
    X2 = t_cut**2
    X3 = t_cut**3 
    return np.hstack((X0,t_cut,X2,X3))
@jit(nopython=True)
def trend_order4(t_cut,X0):
    # 输入t一维列向量，X0一维列向量
    X2 = t_cut**2
    X3 = t_cut**3
    X4 = t_cut**4
    return np.hstack((X0,t_cut,X2,X3,X4))
@jit(nopython=True)
def trend_order5(t_cut,X0):
    # 输入t一维列向量，X0一维列向量
    X2 = t_cut**2
    X3 = t_cut**3
    X4 = t_cut**4
    X5 = t_cut**5
    return np.hstack((X0,t_cut,X2,X3,X4,X5))
@jit(nopython=True)
def trend_order6(t_cut,X0):
    # 输入t一维列向量，X0一维列向量
    X2 = t_cut**2
    X3 = t_cut**3
    X4 = t_cut**4
    X5 = t_cut**5
    X6 = t_cut**6
    return np.hstack((X0,t_cut,X2,X3,X4,X5,X6))
@jit(nopython=True)
def trend_order7(t_cut,X0):
    # 输入t一维列向量，X0一维列向量
    X2 = t_cut**2
    X3 = t_cut**3
    X4 = t_cut**4
    X5 = t_cut**5
    X6 = t_cut**6
    X7 = t_cut**7  
    return np.hstack((X0,t_cut,X2,X3,X4,X5,X6,X7))
@jit(nopython=True)
def trend_order8(t_cut,X0):
    # 输入t一维列向量，X0一维列向量
    X2 = t_cut**2
    X3 = t_cut**3
    X4 = t_cut**4
    X5 = t_cut**5
    X6 = t_cut**6
    X7 = t_cut**7  
    X8 = t_cut**8 
    return np.hstack((X0,t_cut,X2,X3,X4,X5,X6,X7,X8))
@jit(nopython=True)
def sele_trend(trend_order): # 返回函数
    if trend_order == 0:
        return fun(trend_order0)
    elif trend_order == 1:
        return fun(trend_order1)
    elif trend_order == 2:
        return fun(trend_order2)
    elif trend_order == 3:
        return fun(trend_order3)
    elif trend_order == 4:
        return fun(trend_order4) 
    elif trend_order == 5:
        return fun(trend_order5)
    elif trend_order == 6:
        return fun(trend_order6)
    elif trend_order == 7:
        return fun(trend_order7)
    elif trend_order == 8:
        return fun(trend_order8) 
    else:
        raise ValueError("The trend_order inputted is out of range, Please input an int in [0,1,2,3,4,5,6,7,8]") 



# .....................global fitting用到的trend  
@jit(nopython=True)
def global_trend_order0(phase_cut,X0):
    return X0
@jit(nopython=True)
def global_trend_order1(phase_cut,X0): 
    # 输入phase_cut是纵的
    X1 = X0*phase_cut
    return np.hstack((X0,X1))
@jit(nopython=True)
def global_trend_order2(phase_cut,X0):
    X1 = X0*phase_cut
    X2 = X1**2
    return np.hstack((X0,X1,X2))
@jit(nopython=True)
def global_trend_order3(phase_cut,X0):
    X1 = X0*phase_cut
    X2 = X1**2
    X3 = X1**3
    return np.hstack((X0,X1,X2,X3))
@jit(nopython=True)
def global_trend_order4(phase_cut,X0):
    X1 = X0*phase_cut
    X2 = X1**2
    X3 = X1**3
    X4 = X1**4
    return np.hstack((X0,X1,X2,X3,X4))
@jit(nopython=True)
def global_trend_order5(phase_cut,X0):
    X1 = X0*phase_cut
    X2 = X1**2
    X3 = X1**3
    X4 = X1**4
    X5 = X1**5
    return np.hstack((X0,X1,X2,X3,X4,X5))
@jit(nopython=True)
def global_trend_order6(phase_cut,X0):
    X1 = X0*phase_cut
    X2 = X1**2
    X3 = X1**3
    X4 = X1**4
    X5 = X1**5
    X6 = X1**6
    return np.hstack((X0,X1,X2,X3,X4,X5,X6))
@jit(nopython=True)
def global_trend_order7(phase_cut,X0):
    X1 = X0*phase_cut
    X2 = X1**2
    X3 = X1**3
    X4 = X1**4
    X5 = X1**5
    X6 = X1**6
    X7 = X1**7
    return np.hstack((X0,X1,X2,X3,X4,X5,X6,X7))
@jit(nopython=True)
def global_trend_order8(phase_cut,X0):
    X1 = X0*phase_cut
    X2 = X1**2
    X3 = X1**3
    X4 = X1**4
    X5 = X1**5
    X6 = X1**6
    X7 = X1**7
    X8 = X1**8
    return np.hstack((X0,X1,X2,X3,X4,X5,X6,X7,X8))  
@jit(nopython=True)
def sele_global_trend(trend_order): # 返回函数
    if trend_order == 0:
        return fun(global_trend_order0)
    elif trend_order == 1:
        return fun(global_trend_order1)
    elif trend_order == 2:
        return fun(global_trend_order2)
    elif trend_order == 3:
        return fun(global_trend_order3)
    elif trend_order == 4:
        return fun(global_trend_order4) 
    elif trend_order == 5:
        return fun(global_trend_order5)
    elif trend_order == 6:
        return fun(global_trend_order6)
    elif trend_order == 7:
        return fun(global_trend_order7)
    elif trend_order == 8:
        return fun(global_trend_order8)
    else:
        raise ValueError("The trend_order inputted is out of range, Please input an int in [0,1,2,3,4,5,6,7,8]")    





#..............AR 
@jit(nopython=True) 
def ar_order1(cut_idx,f_new): 
    X1 = f_new[cut_idx-1] 
    return X1.reshape((len(cut_idx),1))  

@jit(nopython=True)
def ar_order2(cut_idx,f_new): 
    X1 = f_new[cut_idx-1]
    X2 = f_new[cut_idx-2]
    return np.vstack((X1,X2)).T

@jit(nopython=True)
def ar_order3(cut_idx,f_new): 
    X1 = f_new[cut_idx-1]
    X2 = f_new[cut_idx-2]
    X3 = f_new[cut_idx-3]
    return np.vstack((X1,X2,X3)).T

@jit(nopython=True)
def ar_order4(cut_idx,f_new): 
    X1 = f_new[cut_idx-1]
    X2 = f_new[cut_idx-2]
    X3 = f_new[cut_idx-3]
    X4 = f_new[cut_idx-4]
    return np.vstack((X1,X2,X3,X4)).T

@jit(nopython=True)
def ar_order5(cut_idx,f_new): 
    X1 = f_new[cut_idx-1]
    X2 = f_new[cut_idx-2]
    X3 = f_new[cut_idx-3]
    X4 = f_new[cut_idx-4]
    X5 = f_new[cut_idx-5]
    return np.vstack((X1,X2,X3,X4,X5)).T 

@jit(nopython=True)
def sele_ar(ar_order): 
    # 返回函数 
    if ar_order == 1:
        return fun(ar_order1)
    elif ar_order == 2:
        return fun(ar_order2)
    elif ar_order == 3:
        return fun(ar_order3)
    elif ar_order == 4:
        return fun(ar_order4)
    elif ar_order == 5:
        return fun(ar_order5)
    else:  
        raise ValueError("The ar_order inputted is out of range, Please input an int in [1,2,3,4,5]")
