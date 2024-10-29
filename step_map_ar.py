# 改完了
import numpy as np 
from TBFP704_2_add_ar.constants import dmax 
from numba import jit

# 弄变换的window 

def fun(f):
    return f

def sele_fitting_2D_ar(limit,window_mode): 
    # 返回函数
    if window_mode == 'fixed':
        if limit == 'no':
            return fun(fitting_2D_fixed_window_no_limit_ar)
        elif limit == 'yes':
            return fun(fitting_2D_fixed_window_limit_ar)
        else:
            raise ValueError("The limit inputted is out of range, Please input an int in ['no','yes']")
    elif window_mode == 'varied':
        if limit == 'no':
            return fun(fitting_2D_varied_window_no_limit_ar)
        elif limit == 'yes': 
            return fun(fitting_2D_varied_window_limit_ar)
        else:
            raise ValueError("The limit inputted is out of range, Please input an int in ['no','yes']")
    else:
        raise ValueError("The window_mode inputted is out of range, Please input an int in ['fixed','varied']")

@jit(nopython=True)
def get_wX(X_sig_cut,X_trend_cut,inv_df_cut):
    # 输入sig和trend的列向量，1/sigma的一维列向量
    X_cut = np.hstack((X_sig_cut,X_trend_cut))
    wX_cut = X_cut*inv_df_cut 
    return wX_cut 

@jit(nopython=True)
def get_covar(wX_cut):
    xtx = wX_cut.T@wX_cut
    return np.linalg.inv(xtx)

@jit(nopython=True)
def get_L(wX_cut,wF_cut): 
    # wX_cut是列矩阵，wF_cut一维横列都行
    var = get_covar(wX_cut)
    L = var@wX_cut.T@wF_cut
    return L

@jit(nopython=True)
def get_chi2_base(X_trend_cut,inv_df_cut,wF_cut):
    wX_base = X_trend_cut*inv_df_cut
    L = get_L(wX_base,wF_cut)
    wR_base = wF_cut-wX_base@L
    return wR_base.T@wR_base

def fitting_2D_fixed_window_no_limit_ar(i,seg_ts,seg_fs,seg_dfs,size_d,d_sam,tm_sam,e,
               sig_order,trend_order,get_X_sig_cut,get_X_trend_cut,ar_order,get_X_ar):
    seg_t0 = seg_ts[i]  
    seg_f0 = seg_fs[i]
    seg_df0 = seg_dfs[i]
    inv_df = (1/seg_df0)[:,None]
    wF = seg_f0/seg_df0
    f_new = np.concatenate((np.zeros(5),seg_f0))

    idx_tm2 = np.where((tm_sam>=seg_t0[0])&(tm_sam<=seg_t0[-1]))[0]
    tm_sam_here = tm_sam[idx_tm2]
    dlnL_here = np.full((size_d,len(idx_tm2)),np.nan) 

    # 先竖着d撒点 
    for k in range(size_d):
        d = d_sam[k]
        for j in range(len(tm_sam_here)):
            tm = tm_sam_here[j]
            # 只有内部条件，和外部trend个数的限制 
            cut_idx = np.where((seg_t0>=(tm-e*dmax))&(seg_t0<=(tm+e*dmax)))[0]
            cut_num = len(cut_idx)

            inbool0_left = ((seg_t0>=(tm-e*dmax))&(seg_t0<(tm-d/2.0)))  
            left_num = np.sum(inbool0_left)  
            inbool0 = ((seg_t0>(tm-d/2.0))&(seg_t0<(tm+d/2.0)))  
            in_num = np.sum(inbool0) 
            if in_num < (sig_order/2.0+2):
                continue
            if left_num < trend_order+1+ar_order:#np.maximum(trend_order+1,in_num*3):
                continue 
            if (cut_num-in_num-left_num) < trend_order+1+ar_order:#np.maximum(trend_order+1,in_num*3):
                continue 
            
            t_cut_ori = seg_t0[cut_idx] # 这个没有伸缩
            # 开始拟合
            # 伸缩
            t_cut = 2*(t_cut_ori-tm)/d  # 这个伸缩过了
            inv_df_cut = inv_df[cut_idx]
            wF_cut = wF[cut_idx]             

            # 构建X,wX,wF,L,chi2       
            inbool = (t_cut>=-1)&(t_cut<=1)
            x_cut = np.where(inbool,t_cut,1)
            X_sig_cut = get_X_sig_cut(x_cut) 

            X0 = np.ones(cut_num)[:,None]
            X_trend_cut = get_X_trend_cut(t_cut[:,None],X0)

            X_ar_cut = get_X_ar(cut_idx,f_new)
            X_base_cut = np.hstack((X_trend_cut,X_ar_cut)) 
            wX_cut = get_wX(X_sig_cut,X_base_cut,inv_df_cut)

            wF_cut = np.asarray(wF_cut, dtype=np.float64)
            L = get_L(wX_cut,wF_cut)
            # chi2_trend
            chi2_base = get_chi2_base(X_base_cut,inv_df_cut,wF_cut)
            wR = wF_cut-wX_cut@L
            chi2 = wR.T@wR 
            dlnL_here[k,j] = 0.5*(chi2_base-chi2)
    if i != 0: 
        idx_tm1 = np.where((tm_sam>seg_ts[i-1][-1])&(tm_sam<seg_t0[0]))[0]
        dlnL_add = np.full((size_d,len(idx_tm1)),np.nan)
        dlnL_here = np.hstack((dlnL_add,dlnL_here))
    return dlnL_here 



def fitting_2D_fixed_window_limit_ar(i,seg_ts,seg_fs,seg_dfs,size_d,d_sam,tm_sam,e,
               sig_order,trend_order,get_X_sig_cut,get_X_trend_cut,ar_order,get_X_ar):
    seg_t0 = seg_ts[i]  
    seg_f0 = seg_fs[i]
    seg_df0 = seg_dfs[i]
    inv_df = (1/seg_df0)[:,None]
    wF = seg_f0/seg_df0
    f_new = np.concatenate((np.zeros(5),seg_f0))

    idx_tm2 = np.where((tm_sam>=seg_t0[0])&(tm_sam<=seg_t0[-1]))[0]
    tm_sam_here = tm_sam[idx_tm2]
    dlnL_here = np.full((size_d,len(idx_tm2)),np.nan) 

    # 先竖着d撒点 
    for k in range(size_d):
        d = d_sam[k]
        for j in range(len(tm_sam_here)):
            tm = tm_sam_here[j]

            # 只有内部条件，和外部trend个数的限制 
            cut_idx = np.where((seg_t0>=(tm-e*dmax))&(seg_t0<=(tm+e*dmax)))[0]
            cut_num = len(cut_idx)

            inbool0_left = ((seg_t0>=(tm-e*dmax))&(seg_t0<(tm-d/2.0)))  
            left_num = np.sum(inbool0_left)  

            inbool0 = ((seg_t0>(tm-d/2.0))&(seg_t0<(tm+d/2.0)))  
            in_num = np.sum(inbool0) 
            if in_num < (sig_order/2.0+2):
                continue
            if left_num < trend_order+1+ar_order:#np.maximum(trend_order+1,in_num*3):
                continue 
            if (cut_num-in_num-left_num) < trend_order+1+ar_order:#np.maximum(trend_order+1,in_num*3):
                continue 
            
            t_cut_ori = seg_t0[cut_idx] # 这个没有伸缩
            # 开始拟合
            # 伸缩
            t_cut = 2*(t_cut_ori-tm)/d  # 这个伸缩过了
            inv_df_cut = inv_df[cut_idx]
            wF_cut = wF[cut_idx]             

            # 构建X,wX,wF,L,chi2       
            inbool = (t_cut>=-1)&(t_cut<=1)
            x_cut = np.where(inbool,t_cut,1)
            X_sig_cut = get_X_sig_cut(x_cut) 

            X0 = np.ones(cut_num)[:,None]
            X_trend_cut = get_X_trend_cut(t_cut[:,None],X0)

            X_ar_cut = get_X_ar(cut_idx,f_new)
            X_base_cut = np.hstack((X_trend_cut,X_ar_cut)) 
            wX_cut = get_wX(X_sig_cut,X_base_cut,inv_df_cut)            

            wF_cut = np.asarray(wF_cut, dtype=np.float64)
            L = get_L(wX_cut,wF_cut)
            # chi2_trend
            chi2_base = get_chi2_base(X_base_cut,inv_df_cut,wF_cut)


            if sig_order == 2:
                if L[0]<0:
                    continue
                else:
                    wR = wF_cut-wX_cut@L
                    chi2 = wR.T@wR
            if sig_order == 4: 
                if (L[0]>0) and (L[1]>0):
                    wR = wF_cut-wX_cut@L 
                    chi2 = wR.T@wR
                elif (L[0]>0) and (L[0]/L[1]<-2):  
                    wR = wF_cut-wX_cut@L
                    chi2 = wR.T@wR 
                elif (L[0]/L[1]>-2) and (L[0]/L[1]<0):
                    # 换方程 
                    xin = -2*t_cut**2+t_cut**4
                    xout = -1 
                    x2 = np.where(inbool,xin,xout)
                    wX_cut = get_wX(x2[:,None],X_base_cut,inv_df_cut)
                    L = get_L(wX_cut,wF_cut) # 只有A4 
                    if L[0]>0:
                        continue
                    wR = wF_cut-wX_cut@L
                    chi2 = wR.T@wR
                else:
                    continue
            dlnL_here[k,j] = 0.5*(chi2_base-chi2)
        
    if i != 0: 
        idx_tm1 = np.where((tm_sam>seg_ts[i-1][-1])&(tm_sam<seg_t0[0]))[0]
        dlnL_add = np.full((size_d,len(idx_tm1)),np.nan)
        dlnL_here = np.hstack((dlnL_add,dlnL_here))
    return dlnL_here 

def fitting_2D_varied_window_no_limit_ar(i,seg_ts,seg_fs,seg_dfs,size_d,d_sam,tm_sam,e,
               sig_order,trend_order,get_X_sig_cut,get_X_trend_cut,ar_order,get_X_ar):
    seg_t0 = seg_ts[i]  
    seg_f0 = seg_fs[i]
    seg_df0 = seg_dfs[i]
    inv_df = (1/seg_df0)[:,None]
    wF = seg_f0/seg_df0
    f_new = np.concatenate((np.zeros(5),seg_f0))

    idx_tm2 = np.where((tm_sam>=seg_t0[0])&(tm_sam<=seg_t0[-1]))[0]
    tm_sam_here = tm_sam[idx_tm2]
    dlnL_here = np.full((size_d,len(idx_tm2)),np.nan) 

    # 先竖着d撒点 
    for k in range(size_d):
        d = d_sam[k]
        for j in range(len(tm_sam_here)):
            tm = tm_sam_here[j]
            # 只有内部条件，和外部trend个数的限制 
            cut_idx = np.where((seg_t0>=(tm-e*d))&(seg_t0<=(tm+e*d)))[0]
            cut_num = len(cut_idx)

            inbool0_left = ((seg_t0>=(tm-e*d))&(seg_t0<(tm-d/2.0)))  
            left_num = np.sum(inbool0_left)  

            inbool0 = ((seg_t0>(tm-d/2.0))&(seg_t0<(tm+d/2.0)))  
            in_num = np.sum(inbool0) 
            if in_num < (sig_order/2.0+2):
                continue
            if left_num < trend_order+1+ar_order:#np.maximum(trend_order+1,in_num*3):
                continue 
            if (cut_num-in_num-left_num) < trend_order+1+ar_order:#np.maximum(trend_order+1,in_num*3):
                continue
            
            t_cut_ori = seg_t0[cut_idx] # 这个没有伸缩
            # 开始拟合
            # 伸缩
            t_cut = 2*(t_cut_ori-tm)/d  # 这个伸缩过了
            inv_df_cut = inv_df[cut_idx]
            wF_cut = wF[cut_idx]             

            # 构建X,wX,wF,L,chi2       
            inbool = (t_cut>=-1)&(t_cut<=1)
            x_cut = np.where(inbool,t_cut,1)
            X_sig_cut = get_X_sig_cut(x_cut) 
            X0 = np.ones(cut_num)[:,None]
            X_trend_cut = get_X_trend_cut(t_cut[:,None],X0)

            X_ar_cut = get_X_ar(cut_idx,f_new)
            X_base_cut = np.hstack((X_trend_cut,X_ar_cut)) 
            wX_cut = get_wX(X_sig_cut,X_base_cut,inv_df_cut)

            wF_cut = np.asarray(wF_cut, dtype=np.float64)
            L = get_L(wX_cut,wF_cut)
            # chi2_trend
            chi2_base = get_chi2_base(X_base_cut,inv_df_cut,wF_cut)
            wR = wF_cut-wX_cut@L
            chi2 = wR.T@wR 
            dlnL_here[k,j] = 0.5*(chi2_base-chi2)
    if i != 0: 
        idx_tm1 = np.where((tm_sam>seg_ts[i-1][-1])&(tm_sam<seg_t0[0]))[0]
        dlnL_add = np.full((size_d,len(idx_tm1)),np.nan)
        dlnL_here = np.hstack((dlnL_add,dlnL_here))
    return dlnL_here 



def fitting_2D_varied_window_limit_ar(i,seg_ts,seg_fs,seg_dfs,size_d,d_sam,tm_sam,e,
               sig_order,trend_order,get_X_sig_cut,get_X_trend_cut,ar_order,get_X_ar):
    seg_t0 = seg_ts[i]  
    seg_f0 = seg_fs[i]
    seg_df0 = seg_dfs[i]
    inv_df = (1/seg_df0)[:,None]
    wF = seg_f0/seg_df0
    f_new = np.concatenate((np.zeros(5),seg_f0))

    idx_tm2 = np.where((tm_sam>=seg_t0[0])&(tm_sam<=seg_t0[-1]))[0]
    tm_sam_here = tm_sam[idx_tm2]
    dlnL_here = np.full((size_d,len(idx_tm2)),np.nan) 

    # 先竖着d撒点 
    for k in range(size_d):
        d = d_sam[k]
        for j in range(len(tm_sam_here)):
            tm = tm_sam_here[j]

            # 只有内部条件，和外部trend个数的限制 
            cut_idx = np.where((seg_t0>=(tm-e*d))&(seg_t0<=(tm+e*d)))[0]
            cut_num = len(cut_idx)

            inbool0_left = ((seg_t0>=(tm-e*d))&(seg_t0<(tm-d/2.0)))  
            left_num = np.sum(inbool0_left) 

            inbool0 = ((seg_t0>(tm-d/2.0))&(seg_t0<(tm+d/2.0)))  
            in_num = np.sum(inbool0) 
            if in_num < (sig_order/2.0+2):
                continue
            if left_num < trend_order+1+ar_order:#np.maximum(trend_order+1,in_num*3):
                continue 
            if (cut_num-in_num-left_num) < trend_order+1+ar_order:#np.maximum(trend_order+1,in_num*3):
                continue 
            
            t_cut_ori = seg_t0[cut_idx] # 这个没有伸缩
            # 开始拟合
            # 伸缩
            t_cut = 2*(t_cut_ori-tm)/d  # 这个伸缩过了
            inv_df_cut = inv_df[cut_idx]
            wF_cut = wF[cut_idx]             

            # 构建X,wX,wF,L,chi2       
            inbool = (t_cut>=-1)&(t_cut<=1)
            x_cut = np.where(inbool,t_cut,1)
            X_sig_cut = get_X_sig_cut(x_cut) 
            X0 = np.ones(cut_num)[:,None]
            X_trend_cut = get_X_trend_cut(t_cut[:,None],X0)

            X_ar_cut = get_X_ar(cut_idx,f_new)
            X_base_cut = np.hstack((X_trend_cut,X_ar_cut)) 
            wX_cut = get_wX(X_sig_cut,X_base_cut,inv_df_cut)

            wF_cut = np.asarray(wF_cut, dtype=np.float64)
            L = get_L(wX_cut,wF_cut)
            # chi2_trend
            chi2_base = get_chi2_base(X_base_cut,inv_df_cut,wF_cut)


            if sig_order == 2:
                if L[0]<0:
                    continue
                else:
                    wR = wF_cut-wX_cut@L
                    chi2 = wR.T@wR
            if sig_order == 4: 
                if (L[0]>0) and (L[1]>0):
                    wR = wF_cut-wX_cut@L 
                    chi2 = wR.T@wR
                elif (L[0]>0) and (L[0]/L[1]<-2):  
                    wR = wF_cut-wX_cut@L
                    chi2 = wR.T@wR 
                elif (L[0]/L[1]>-2) and (L[0]/L[1]<0):
                    # 换方程 
                    xin = -2*t_cut**2+t_cut**4
                    xout = -1 
                    x2 = np.where(inbool,xin,xout)
                    wX_cut = get_wX(x2[:,None],X_base_cut,inv_df_cut)
                    L = get_L(wX_cut,wF_cut) # 只有A4 
                    if L[0]>0:
                        continue
                    wR = wF_cut-wX_cut@L
                    chi2 = wR.T@wR
                else:
                    continue
            dlnL_here[k,j] = 0.5*(chi2_base-chi2)
        
    if i != 0: 
        idx_tm1 = np.where((tm_sam>seg_ts[i-1][-1])&(tm_sam<seg_t0[0]))[0]
        dlnL_add = np.full((size_d,len(idx_tm1)),np.nan)
        dlnL_here = np.hstack((dlnL_add,dlnL_here))
    return dlnL_here