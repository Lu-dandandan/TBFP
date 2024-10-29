import numpy as np
from TBFP704_2_add_ar.constants import dmin_ori,Msun,Rsun,d_to_s,G 
from numba import jit

A_up = 0.31
A_down = 0.009

def d_grid(P,d_sam):
    d_up = A_up*P**(1/3.0) 
    dmax2 = np.min(np.array([d_up,d_sam[-1]]))
    d_down = A_down*P**(1/3.0)
    dmin2 = np.max(np.array([d_sam[0],d_down]))
    idx_d = np.where((d_sam>=dmin2)&(d_sam<=dmax2))[0]
    return d_sam[idx_d],idx_d



from numpy import pi
def dp_fun(Rs,Ms,S,OS_P,P):
    # P输入的是天
    Ms *= Msun 
    Rs *= Rsun  
    S = S*d_to_s
    P = P*d_to_s
    A = (2.0*pi)**(2/3.0)/pi*Rs/(G*Ms)**(1/3.0)/S/OS_P
    dp = A*P**(4/3.0)
    dp = dp/d_to_s
    return dp  

def P_grid(Rs,Ms,S,OS_P,Pmin,Pmax): 
    # 输入的Rs,Ms单位是太阳单位，S是day, Pmin和Pmax都是day

    # 把恒星半径，质量，时间跨度转成国际单位
    # 我没有加一些限制，以防止出现一些别的情况。
    Ms *= Msun 
    Rs *= Rsun  
    S = S*d_to_s

    # fre最小和最大
    fre_min = 2.0/S
    fre_max = (G*Ms/(3*Rs)**3)**0.5/2.0/pi

    # cubic sampling of frequency
    A = (2.0*pi)**(2/3.0)/pi*Rs/(G*Ms)**(1/3.0)/S/OS_P
    C = fre_min**(1/3.0)-A/3.0
    N = (fre_max**(1/3.0)-fre_min**(1/3.0)+A/3.0)*3.0/A
    x = np.arange(N)+1
    fre = (A/3.0*x+C)**3
    Per = 1/fre/d_to_s # 单位是s，转成天
    return Per[(Per<=Pmax)&(Per>=Pmin)]