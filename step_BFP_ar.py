import numpy as np
from TBFP704_2_add_ar.grid import d_grid
from TBFP704_2_add_ar.constants import dmax
from numba import jit 

def fun(f):
    return f
def running_median(data, kernel):
    # 如果数据中有nan,
    idx = np.arange(kernel) + np.arange(len(data) - kernel + 1)[:, None] # 二维矩阵
    idx = idx.astype(np.int64) 
    med = []
    for kk in range(idx.shape[0]): 
        arr = data[idx[kk]]
        if np.all(np.isnan(arr)):
            med.append(np.nan)
        else:
            med.append(np.nanmedian(arr))
    med = np.array(med)
    # 补充前后的数据点
    first_values = med[0]
    last_values = med[-1]
    missing_values = len(data) - len(med) # 缺失的数据量，全都用第一个和最后一个表示
    values_front = int(missing_values * 0.5)
    values_end = missing_values - values_front
    med = np.append(np.full(values_front, first_values), med)
    med = np.append(med, np.full(values_end, last_values)) 
    return med
def sele_fitting_global_ar(limit,window_mode): 
    # 返回函数
    if window_mode == 'fixed':
        if limit == 'no':
            return fun(fitting_global_fixed_window_no_limit_ar)
        elif limit == 'yes':
            return fun(fitting_global_fixed_window_limit_ar)
        else:
            raise ValueError("The limit inputted is out of range, Please input an int in ['no','yes']")
    elif window_mode == 'varied':
        if limit == 'no':
            return fun(fitting_global_varied_window_no_limit_ar)
        elif limit == 'yes':
            return fun(fitting_global_varied_window_limit_ar)
        else:
            raise ValueError("The limit inputted is out of range, Please input an int in ['no','yes']")
    else:   
        raise ValueError("The window_mode inputted is out of range, Please input an int in ['fixed','varied']")

def sele_result_fitting_ar(limit):
    if limit == 'no':
        return fun(result_fitting_no_limit_ar)
    else:
        return fun(result_fitting_limit_ar) 

# 一个P下找到最佳的参数们
# 改完了
def fold_map_tm0d(P,d_sam,tm_sam,dlnL,tm_gap,OS_tm):
    # 这个P下的d_sam2 
    d_sam2,idx_dd = d_grid(P,d_sam)
    dlnL2 = dlnL[idx_dd,:]  # 后面基于这个做  
    row = int(P//tm_gap) #几个tm0

    tm_gap_new = d_sam2[0]/OS_tm
    nn = tm_gap_new//tm_gap

    col = int(dlnL2.shape[1]//row+1) # 每个tm0有多少个叠加
    num_d = dlnL2.shape[0]
    # 补上这些nan
    add = int(row-dlnL2.shape[1]%row)  
    dlnL2 = np.hstack((dlnL2,np.full((num_d,add),np.nan)))
    dlnL2 = np.reshape(dlnL2, (num_d, col, row)) 
    if (nn != 0) and (nn != 1):
        idx_sele = np.arange(0,row,nn,dtype=int)
        # print(idx_sele)

        #print(reshaped_dlnL3.shape)
        # reshaped_dlnL3.shape中，第一维d,第二维是每个tm0对应的个数，第三维是tm0

        # 按照gap = dmin/5来选取
        dlnL2 = dlnL2[:,:,idx_sele]
    #print(reshaped_dlnL3.shape)

    # np.count_nonzero(~np.isnan(reshaped_dlnL3), axis=1))是二维的，原来的第一维没有了
    row_indices, col_indices  = np.where(np.count_nonzero(~np.isnan(dlnL2), axis=1) < 3)
    # row_indices是d,col是tm0
    dlnL2[row_indices,:,col_indices] = np.nan
    dlnL_co = np.nansum(dlnL2,axis=1) # 行数是d,列数是tm0
    # 此时有了一个dlnL,tm0_sam的map
    dlnLmax0 = np.nanmax(dlnL_co)
    if dlnLmax0 == 0: # 就是都小于3段
        d_best0 = np.nan
        tm0_best0 = np.nan
        dlnLmax0 = np.nan
    else:
        idx_d,idx_tm0 = np.unravel_index(np.nanargmax(dlnL_co), dlnL_co.shape)
        d_best0 = d_sam2[idx_d]
        if (nn != 0) and ((nn != 1)):
            tm0_sam = tm_sam[:row][idx_sele] 
            tm0_best0 = tm0_sam[idx_tm0]
        else: 
            tm0_sam = tm_sam[:row] 
            tm0_best0 = tm0_sam[idx_tm0]
    return d_best0,tm0_best0,dlnLmax0

@jit(nopython=True)
def get_phase(t,tm,P):  # 一个周期和tm跑一次
    u = (t-tm)/P + 0.5
    group = np.floor(u) 
    return u-group,group  # 这个就是transit发生在0.5了

@jit(nopython=True)
def get_wX(X_sig_cut,X_trend_cut,inv_df_cut):
    # 输入sig和trend的列向量，1/sigma的一维列向量
    X_cut = np.hstack((X_sig_cut,X_trend_cut))
    wX_cut = X_cut*inv_df_cut 
    return wX_cut 

def get_XwX(X_sig_cut,X_trend_cut,inv_df_cut):
    # 输入sig和trend的列向量，1/sigma的一维列向量
    X_cut = np.hstack((X_sig_cut,X_trend_cut))
    wX_cut = X_cut*inv_df_cut 
    return X_cut,wX_cut 

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

@jit(nopython=True) 
def selection_fixed_window(phase,group,q,qmax,e,trend_order,P,cri0,ar_order):
    # 只要transit里面至少有一个点，外面有trend+1个点 的段，而且这段没有gap, 否则扔掉该段。
    cut_index = np.where((phase>=0.5-qmax*e)&(phase<=0.5+qmax*e))[0] # 所有cut
    group_cut = group[cut_index] 
    phase_cut = phase[cut_index] # 得到所有cut出来的phase_cut和group_cut
    g_uni = np.unique(group_cut) 
    g_uni_final = [] # 一共有几段
    for s in g_uni:
        idx = np.where(group_cut==s)[0]
        phase_cut_seg = phase_cut[idx]  # 每一段的phase_cut_seg  
        in_idx = np.where((phase_cut_seg>=0.5-q/2.0)&(phase_cut_seg<=0.5+q/2.0))[0]
        in_num = len(in_idx)
        if in_num<1: # 不符合条件
            cut_index = np.delete(cut_index,idx)
            group_cut = np.delete(group_cut,idx)
            phase_cut = np.delete(phase_cut,idx)
            continue
        inbool0_left = ((phase_cut_seg>=0.5-qmax*e)&(phase_cut_seg<=0.5-q/2.0))
        left_num = np.sum(inbool0_left)
        cut_num = len(phase_cut_seg)  
        if left_num < trend_order+1+ar_order:
            cut_index = np.delete(cut_index,idx)
            group_cut = np.delete(group_cut,idx)
            phase_cut = np.delete(phase_cut,idx)    
            continue 
        if (cut_num-left_num-in_num) < trend_order+1+ar_order: 
            cut_index = np.delete(cut_index,idx)
            group_cut = np.delete(group_cut,idx)
            phase_cut = np.delete(phase_cut,idx)    
            continue

        d_phase = np.diff(phase_cut_seg) 
        cri = cri0/P 
        if np.max(d_phase) > cri:  # 跑这里的，至少分成了两段
            split_idx = np.where(d_phase > cri)[0] # 至少有一个数
            if len(split_idx)>3: # 有三个及以上个数，就扔掉这一段
                # print('3以上',len(split_idx))
                cut_index = np.delete(cut_index,idx)
                group_cut = np.delete(group_cut,idx)
                phase_cut = np.delete(phase_cut,idx) 
                continue
            elif len(split_idx) == 1:
                # print('1',len(split_idx))
                aa = (split_idx+1)[0]
                a = phase_cut_seg[split_idx]    
                b = phase_cut_seg[split_idx+1]
                if (a<0.5)and(b<0.5): 
                    # 删掉左边的
                    phase_cut_seg = phase_cut_seg[aa:]
                    in_idx = np.where((phase_cut_seg>=0.5-q/2.0)&(phase_cut_seg<=0.5+q/2.0))[0]
                    in_num = len(in_idx)
                    if in_num<1: # 不符合条件
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)
                        continue
                    inbool0_left = ((phase_cut_seg>=0.5-qmax*e)&(phase_cut_seg<=0.5-q/2.0))
                    left_num = np.sum(inbool0_left)
                    cut_num = len(phase_cut_seg) 
                    if left_num < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue 
                    if (cut_num-left_num-in_num) < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue
                    cut_index = np.delete(cut_index,idx[:aa]) 
                    group_cut = np.delete(group_cut,idx[:aa])
                    phase_cut = np.delete(phase_cut,idx[:aa])
                elif (a>0.5)and(b>0.5):
                    # 删掉右边的
                    phase_cut_seg = phase_cut_seg[:aa]  
                    in_idx = np.where((phase_cut_seg>=0.5-q/2.0)&(phase_cut_seg<=0.5+q/2.0))[0]
                    in_num = len(in_idx)
                    if in_num<1: # 不符合条件
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)
                        continue
                    inbool0_left = ((phase_cut_seg>=0.5-qmax*e)&(phase_cut_seg<=0.5-q/2.0))
                    left_num = np.sum(inbool0_left)
                    cut_num = len(phase_cut_seg) 
                    if left_num < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue 
                    if (cut_num-left_num-in_num) < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue 
                    cut_index = np.delete(cut_index,idx[aa:])
                    group_cut = np.delete(group_cut,idx[aa:])
                    phase_cut = np.delete(phase_cut,idx[aa:]) 
                else:
                    # 全去掉
                    cut_index = np.delete(cut_index,idx)
                    group_cut = np.delete(group_cut,idx)
                    phase_cut = np.delete(phase_cut,idx) 
                    continue 
            else: # 有两个 
                # print('2',len(split_idx))
                edge1 = split_idx[0]
                edge2 = split_idx[1]  
                a = phase_cut_seg[edge1]    
                b = phase_cut_seg[edge1+1]
                c = phase_cut_seg[edge2]    
                dd = phase_cut_seg[edge2+1] 
                if dd < 0.5:  
                    # 删掉左边的
                    phase_cut_seg = phase_cut_seg[edge2+1:]
                    in_idx = np.where((phase_cut_seg>=0.5-q/2.0)&(phase_cut_seg<=0.5+q/2.0))[0]
                    in_num = len(in_idx)
                    if in_num<1: # 不符合条件
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)
                        continue
                    inbool0_left = ((phase_cut_seg>=0.5-qmax*e)&(phase_cut_seg<=0.5-q/2.0))
                    left_num = np.sum(inbool0_left)
                    cut_num = len(phase_cut_seg) 
                    if left_num < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue 
                    if (cut_num-left_num-in_num) < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue 
                    cut_index = np.delete(cut_index,idx[:edge2+1]) 
                    group_cut = np.delete(group_cut,idx[:edge2+1])
                    phase_cut = np.delete(phase_cut,idx[:edge2+1]) 
                elif a > 0.5:
                    # 删掉右边的
                    phase_cut_seg = phase_cut_seg[:edge1+1]  
                    in_idx = np.where((phase_cut_seg>=0.5-q/2.0)&(phase_cut_seg<=0.5+q/2.0))[0]
                    in_num = len(in_idx)
                    if in_num<1: # 不符合条件
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)
                        continue
                    inbool0_left = ((phase_cut_seg>=0.5-qmax*e)&(phase_cut_seg<=0.5-q/2.0))
                    left_num = np.sum(inbool0_left)
                    cut_num = len(phase_cut_seg) 
                    if left_num < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue 
                    if (cut_num-left_num-in_num) < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue 
                    cut_index = np.delete(cut_index,idx[edge1+1:])
                    group_cut = np.delete(group_cut,idx[edge1+1:])
                    phase_cut = np.delete(phase_cut,idx[edge1+1:]) 
                elif (b<0.5) and (c>0.5):
                    # 去掉两边的
                    phase_cut_seg = phase_cut_seg[edge1+1:edge2+1]  
                    in_idx = np.where((phase_cut_seg>=0.5-q/2.0)&(phase_cut_seg<=0.5+q/2.0))[0]
                    in_num = len(in_idx)
                    if in_num<1: # 不符合条件
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)
                        continue
                    inbool0_left = ((phase_cut_seg>=0.5-qmax*e)&(phase_cut_seg<=0.5-q/2.0))
                    left_num = np.sum(inbool0_left)
                    cut_num = len(phase_cut_seg) 
                    if left_num < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue 
                    if (cut_num-left_num-in_num) < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue 
                    xx = np.concatenate((idx[:edge1+1],idx[edge2+1:])) 
                    cut_index = np.delete(cut_index,xx)   
                    group_cut = np.delete(group_cut,xx)
                    phase_cut = np.delete(phase_cut,xx) 
                else:
                    # 全去掉
                    cut_index = np.delete(cut_index,idx) 
                    group_cut = np.delete(group_cut,idx) 
                    phase_cut = np.delete(phase_cut,idx)  
                    continue                           
        g_uni_final.append(s)
    return cut_index,group_cut,phase_cut,g_uni_final

@jit(nopython=True) # 必须给加上numba
def selection_varied_window(phase,group,q,e,trend_order,P,cri0,ar_order):
    # 只要transit里面至少有一个点，外面有trend+1个点 的段，而且这段没有gap, 否则扔掉该段。
    cut_index = np.where((phase>=0.5-q*e)&(phase<=0.5+q*e))[0] # 所有cut
    group_cut = group[cut_index] 
    phase_cut = phase[cut_index] # 得到所有cut出来的phase_cut和group_cut
    g_uni = np.unique(group_cut) 
    g_uni_final = [] # 一共有几段
    for s in g_uni:
        idx = np.where(group_cut==s)[0]
        phase_cut_seg = phase_cut[idx]  # 每一段的phase_cut_seg  
        in_idx = np.where((phase_cut_seg>=0.5-q/2.0)&(phase_cut_seg<=0.5+q/2.0))[0]
        in_num = len(in_idx)
        if in_num<1: # 不符合条件
            cut_index = np.delete(cut_index,idx)
            group_cut = np.delete(group_cut,idx)
            phase_cut = np.delete(phase_cut,idx)
            continue
        inbool0_left = ((phase_cut_seg>=0.5-q*e)&(phase_cut_seg<=0.5-q/2.0))
        left_num = np.sum(inbool0_left)
        cut_num = len(phase_cut_seg)  
        if left_num < trend_order+1+ar_order:
            cut_index = np.delete(cut_index,idx)
            group_cut = np.delete(group_cut,idx)
            phase_cut = np.delete(phase_cut,idx)    
            continue 
        if (cut_num-left_num-in_num) < trend_order+1+ar_order:
            cut_index = np.delete(cut_index,idx)
            group_cut = np.delete(group_cut,idx)
            phase_cut = np.delete(phase_cut,idx)    
            continue  

        d_phase = np.diff(phase_cut_seg) 
        cri = cri0/P 
        if np.max(d_phase) > cri:  # 跑这里的，至少分成了两段
            split_idx = np.where(d_phase > cri)[0] # 至少有一个数
            if len(split_idx)>3: # 有三个及以上个数，就扔掉这一段
                cut_index = np.delete(cut_index,idx)
                group_cut = np.delete(group_cut,idx)
                phase_cut = np.delete(phase_cut,idx) 
                continue
            elif len(split_idx) == 1:
                aa = (split_idx+1)[0]
                a = phase_cut_seg[split_idx]    
                b = phase_cut_seg[split_idx+1]
                if (a<0.5)and(b<0.5): 
                    # 删掉左边的
                    phase_cut_seg = phase_cut_seg[aa:]
                    in_idx = np.where((phase_cut_seg>=0.5-q/2.0)&(phase_cut_seg<=0.5+q/2.0))[0]
                    in_num = len(in_idx)
                    if in_num<1: # 不符合条件
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)
                        continue
                    inbool0_left = ((phase_cut_seg>=0.5-q*e)&(phase_cut_seg<=0.5-q/2.0))
                    left_num = np.sum(inbool0_left)
                    cut_num = len(phase_cut_seg)  
                    if left_num < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue 
                    if (cut_num-left_num-in_num) < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue  
                    cut_index = np.delete(cut_index,idx[:aa]) 
                    group_cut = np.delete(group_cut,idx[:aa])
                    phase_cut = np.delete(phase_cut,idx[:aa])
                elif (a>0.5)and(b>0.5):
                    # 删掉右边的
                    phase_cut_seg = phase_cut_seg[:aa]  
                    in_idx = np.where((phase_cut_seg>=0.5-q/2.0)&(phase_cut_seg<=0.5+q/2.0))[0]
                    in_num = len(in_idx)
                    if in_num<1: # 不符合条件
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)
                        continue
                    inbool0_left = ((phase_cut_seg>=0.5-q*e)&(phase_cut_seg<=0.5-q/2.0))
                    left_num = np.sum(inbool0_left)
                    cut_num = len(phase_cut_seg)  
                    if left_num < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue 
                    if (cut_num-left_num-in_num) < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue  
                    cut_index = np.delete(cut_index,idx[aa:])
                    group_cut = np.delete(group_cut,idx[aa:])
                    phase_cut = np.delete(phase_cut,idx[aa:]) 
                else:
                    # 全去掉
                    cut_index = np.delete(cut_index,idx)
                    group_cut = np.delete(group_cut,idx)
                    phase_cut = np.delete(phase_cut,idx) 
                    continue 
            else: # 有两个
                edge1 = split_idx[0]
                edge2 = split_idx[1]  
                a = phase_cut_seg[edge1]    
                b = phase_cut_seg[edge1+1]
                c = phase_cut_seg[edge2]    
                dd = phase_cut_seg[edge2+1] 
                if dd < 0.5:  
                    # 删掉左边的
                    phase_cut_seg = phase_cut_seg[edge2+1:]
                    in_idx = np.where((phase_cut_seg>=0.5-q/2.0)&(phase_cut_seg<=0.5+q/2.0))[0]
                    in_num = len(in_idx)
                    if in_num<1: # 不符合条件
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)
                        continue
                    inbool0_left = ((phase_cut_seg>=0.5-q*e)&(phase_cut_seg<=0.5-q/2.0))
                    left_num = np.sum(inbool0_left)
                    cut_num = len(phase_cut_seg)  
                    if left_num < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue 
                    if (cut_num-left_num-in_num) < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue  
                    cut_index = np.delete(cut_index,idx[:edge2+1]) 
                    group_cut = np.delete(group_cut,idx[:edge2+1])
                    phase_cut = np.delete(phase_cut,idx[:edge2+1]) 
                elif a > 0.5:
                    # 删掉右边的
                    phase_cut_seg = phase_cut_seg[:edge1+1]  
                    in_idx = np.where((phase_cut_seg>=0.5-q/2.0)&(phase_cut_seg<=0.5+q/2.0))[0]
                    in_num = len(in_idx)
                    if in_num<1: # 不符合条件
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)
                        continue
                    inbool0_left = ((phase_cut_seg>=0.5-q*e)&(phase_cut_seg<=0.5-q/2.0))
                    left_num = np.sum(inbool0_left)
                    cut_num = len(phase_cut_seg)  
                    if left_num < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue 
                    if (cut_num-left_num-in_num) < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue  
                    cut_index = np.delete(cut_index,idx[edge1+1:])
                    group_cut = np.delete(group_cut,idx[edge1+1:])
                    phase_cut = np.delete(phase_cut,idx[edge1+1:]) 
                elif (b<0.5) and (c>0.5):
                    # 去掉两边的
                    phase_cut_seg = phase_cut_seg[edge1+1:edge2+1]  
                    in_idx = np.where((phase_cut_seg>=0.5-q/2.0)&(phase_cut_seg<=0.5+q/2.0))[0]
                    in_num = len(in_idx)
                    if in_num<1: # 不符合条件
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)
                        continue
                    inbool0_left = ((phase_cut_seg>=0.5-q*e)&(phase_cut_seg<=0.5-q/2.0))
                    left_num = np.sum(inbool0_left)
                    cut_num = len(phase_cut_seg)  
                    if left_num < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue 
                    if (cut_num-left_num-in_num) < trend_order+1+ar_order:
                        cut_index = np.delete(cut_index,idx)
                        group_cut = np.delete(group_cut,idx)
                        phase_cut = np.delete(phase_cut,idx)    
                        continue  
                    xx = np.concatenate((idx[:edge1+1],idx[edge2+1:])) 
                    cut_index = np.delete(cut_index,xx)   
                    group_cut = np.delete(group_cut,xx)
                    phase_cut = np.delete(phase_cut,xx) 
                else:
                    # 全去掉
                    cut_index = np.delete(cut_index,idx) 
                    group_cut = np.delete(group_cut,idx) 
                    phase_cut = np.delete(phase_cut,idx)  
                    continue            
        g_uni_final.append(s)
    return cut_index,group_cut,phase_cut,g_uni_final
@jit(nopython=True)
def get_X0_trend(phase_cut_len,g_uni_final,group_cut):
    X0 = np.zeros((phase_cut_len,len(g_uni_final)))
    a = 0
    for s in g_uni_final:
        s_idx = np.where(group_cut==s)[0]
        X0[s_idx,a] = 1
        a += 1 
    return X0

# 加一个全局拟合的，在最佳的d_best0,tm_best0下全局拟合
#@jit(nopython=True)
def fitting_global_fixed_window_no_limit_ar(P,d_sam,tm_sam,dlnL,tm_gap,OS_tm,get_X_sig_cut,get_X_trend_cut,cri0,e,t,inv_df,wF,trend_order,sig_order,ar_order,get_X_ar,f_new):     
    # 这个d和tm0都是这个周期下最佳的
    d,tm0,dlnLmax1 = fold_map_tm0d(P,d_sam,tm_sam,dlnL,tm_gap,OS_tm)
    if dlnLmax1 != dlnLmax1: # 不满足三段
        dlnLmax2 = np.nan
    else: 
        q = d/P # 目前只有一个数 
        qmax = dmax/P  

        phase,group = get_phase(t,tm0,P)
        cut_index,group_cut,phase_cut,g_uni_final = selection_fixed_window(phase,group,q,qmax,e,trend_order,P,cri0,ar_order) 

        phase_cut = (2*phase_cut-1)/q
        # 拟合的是phase_cut
        inbool = (phase_cut>=-1)&(phase_cut<=1)
        y_cut = np.where(inbool,phase_cut,1)
        X_sig_cut = get_X_sig_cut(y_cut) 
        X0 = get_X0_trend(len(phase_cut),np.array(g_uni_final),group_cut) 
        X_trend_cut = get_X_trend_cut(phase_cut.reshape(len(phase_cut),1),X0)
        
        X_ar_cut = get_X_ar(cut_index,f_new)  
        X_base_cut = np.hstack((X_trend_cut,X_ar_cut)) 
        
        inv_df_cut = inv_df[cut_index]
        wX_cut = get_wX(X_sig_cut,X_base_cut,inv_df_cut)
        wF_cut = wF[cut_index]
        L = get_L(wX_cut,wF_cut)
        # 求chi2_base 
        chi2_base = get_chi2_base(X_base_cut,inv_df_cut,wF_cut)
        wR = wF_cut-wX_cut@L
        chi2 = wR.T@wR
        dlnLmax2 = 0.5*(chi2_base-chi2)
        # if dlnL_global > dlnLmax: 
        #     dlnLmax = np.copy(dlnL_global)
        #     q_chosen = q
    return dlnLmax1,dlnLmax2,d,tm0  

#@jit(nopython=True)
def fitting_global_fixed_window_limit_ar(P,d_sam,tm_sam,dlnL,tm_gap,OS_tm,get_X_sig_cut,get_X_trend_cut,cri0,e,t,inv_df,wF,trend_order,sig_order,ar_order,get_X_ar,f_new):     
    # 这个d和tm0都是这个周期下最佳的
    d,tm0,dlnLmax1 = fold_map_tm0d(P,d_sam,tm_sam,dlnL,tm_gap,OS_tm)
    if dlnLmax1 != dlnLmax1: # 不满足三段
        dlnLmax2 = np.nan
    else:
        q = d/P # 目前只有一个数 
        qmax = dmax/P  

        phase,group = get_phase(t,tm0,P)
        cut_index,group_cut,phase_cut,g_uni_final = selection_fixed_window(phase,group,q,qmax,e,trend_order,P,cri0,ar_order) 
        # 下面这行命令应该没有用了
        # if len(g_uni_final)<3:  # 我最少设置3段。
        #     print(f'全局拟合的时候不满足要求，周期是{P}天,一共有{len(g_uni_final)}段')
        phase_cut = (2*phase_cut-1)/q
        # 拟合的是phase_cut
        inbool = (phase_cut>=-1)&(phase_cut<=1)
        y_cut = np.where(inbool,phase_cut,1)
        X_sig_cut = get_X_sig_cut(y_cut) 
        X0 = get_X0_trend(len(phase_cut),np.array(g_uni_final),group_cut) 
        X_trend_cut = get_X_trend_cut(phase_cut.reshape(len(phase_cut),1),X0)

        X_ar_cut = get_X_ar(cut_index,f_new)  
        X_base_cut = np.hstack((X_trend_cut,X_ar_cut)) 

        inv_df_cut = inv_df[cut_index]
        wX_cut = get_wX(X_sig_cut,X_base_cut,inv_df_cut)
        wF_cut = wF[cut_index]
        # print(np.sum(inbool))
        L = get_L(wX_cut,wF_cut)

        if sig_order == 2:
            if L[0]<0:
                chi2 = np.nan
            else:
                wR = wF_cut-wX_cut@L
                chi2 = wR.T@wR
        if sig_order == 4: 
            if (L[0]>0) and (L[1]>0): # 这个是ok的
                wR = wF_cut-wX_cut@L 
                chi2 = wR.T@wR
            elif (L[0]>0) and (L[0]/L[1]<-2):  
                wR = wF_cut-wX_cut@L
                chi2 = wR.T@wR 
            elif (L[0]/L[1]>-2) and (L[0]/L[1]<0):
                # 换方程 
                xin = -2*phase_cut**2+phase_cut**4
                xout = -1 
                x2 = np.where(inbool,xin,xout)
                wX_cut = get_wX(x2[:,None],X_base_cut,inv_df_cut)
                L = get_L(wX_cut,wF_cut) # 只有A4 
                if L[0]>0:
                    chi2 = np.nan
                else: 
                    wR = wF_cut-wX_cut@L
                    chi2 = wR.T@wR
            else:
                chi2 = np.nan 

        # 求chi2_base 
        chi2_base = get_chi2_base(X_base_cut,inv_df_cut,wF_cut)
        dlnLmax2 = 0.5*(chi2_base-chi2)
        # if dlnL_global > dlnLmax: 
        #     dlnLmax = np.copy(dlnL_global)
        #     q_chosen = q
    return dlnLmax1,dlnLmax2,d,tm0  

def fitting_global_varied_window_no_limit_ar(P,d_sam,tm_sam,dlnL,tm_gap,OS_tm,get_X_sig_cut,get_X_trend_cut,cri0,e,t,inv_df,wF,trend_order,sig_order,ar_order,get_X_ar,f_new):     
    # 这个d和tm0都是这个周期下最佳的
    d,tm0,dlnLmax1 = fold_map_tm0d(P,d_sam,tm_sam,dlnL,tm_gap,OS_tm)
    if dlnLmax1 != dlnLmax1: # 不满足三段
        dlnLmax2 = np.nan
    else: 
        q = d/P # 目前只有一个数  

        phase,group = get_phase(t,tm0,P)
        cut_index,group_cut,phase_cut,g_uni_final = selection_varied_window(phase,group,q,e,trend_order,P,cri0,ar_order)

        phase_cut = (2*phase_cut-1)/q
        # 拟合的是phase_cut
        inbool = (phase_cut>=-1)&(phase_cut<=1)
        y_cut = np.where(inbool,phase_cut,1)
        X_sig_cut = get_X_sig_cut(y_cut) 
        X0 = get_X0_trend(len(phase_cut),np.array(g_uni_final),group_cut) 
        X_trend_cut = get_X_trend_cut(phase_cut.reshape(len(phase_cut),1),X0)

        X_ar_cut = get_X_ar(cut_index,f_new)  
        X_base_cut = np.hstack((X_trend_cut,X_ar_cut)) 

        inv_df_cut = inv_df[cut_index]
        wX_cut = get_wX(X_sig_cut,X_base_cut,inv_df_cut)
        wF_cut = wF[cut_index]
        L = get_L(wX_cut,wF_cut)
        # 求chi2_base 
        chi2_base = get_chi2_base(X_base_cut,inv_df_cut,wF_cut)
        wR = wF_cut-wX_cut@L
        chi2 = wR.T@wR
        dlnLmax2 = 0.5*(chi2_base-chi2)
        # if dlnL_global > dlnLmax: 
        #     dlnLmax = np.copy(dlnL_global)
        #     q_chosen = q
    return dlnLmax1,dlnLmax2,d,tm0  

#@jit(nopython=True)
def fitting_global_varied_window_limit_ar(P,d_sam,tm_sam,dlnL,tm_gap,OS_tm,get_X_sig_cut,get_X_trend_cut,cri0,e,t,inv_df,wF,trend_order,sig_order,ar_order,get_X_ar,f_new):     
    # 这个d和tm0都是这个周期下最佳的
    d,tm0,dlnLmax1 = fold_map_tm0d(P,d_sam,tm_sam,dlnL,tm_gap,OS_tm)
    if dlnLmax1 != dlnLmax1: # 不满足三段
        dlnLmax2 = np.nan
    else:
        q = d/P # 目前只有一个数 

        phase,group = get_phase(t,tm0,P) 
        cut_index,group_cut,phase_cut,g_uni_final = selection_varied_window(phase,group,q,e,trend_order,P,cri0,ar_order)
        # 下面这行命令应该没有用了
        # if len(g_uni_final)<3:  # 我最少设置3段。
        #     print(f'全局拟合的时候不满足要求，周期是{P}天,一共有{len(g_uni_final)}段')
        phase_cut = (2*phase_cut-1)/q
        # 拟合的是phase_cut
        inbool = (phase_cut>=-1)&(phase_cut<=1)
        y_cut = np.where(inbool,phase_cut,1)
        X_sig_cut = get_X_sig_cut(y_cut) 
        X0 = get_X0_trend(len(phase_cut),np.array(g_uni_final),group_cut) 
        X_trend_cut = get_X_trend_cut(phase_cut.reshape(len(phase_cut),1),X0)

        X_ar_cut = get_X_ar(cut_index,f_new)  
        X_base_cut = np.hstack((X_trend_cut,X_ar_cut)) 

        inv_df_cut = inv_df[cut_index]
        wX_cut = get_wX(X_sig_cut,X_base_cut,inv_df_cut)
        wF_cut = wF[cut_index]
        # print(np.sum(inbool))
        L = get_L(wX_cut,wF_cut)

        if sig_order == 2:
            if L[0]<0:
                chi2 = np.nan
            else:
                wR = wF_cut-wX_cut@L
                chi2 = wR.T@wR
        if sig_order == 4: 
            if (L[0]>0) and (L[1]>0): # 这个是ok的
                wR = wF_cut-wX_cut@L 
                chi2 = wR.T@wR
            elif (L[0]>0) and (L[0]/L[1]<-2):  
                wR = wF_cut-wX_cut@L
                chi2 = wR.T@wR 
            elif (L[0]/L[1]>-2) and (L[0]/L[1]<0):
                # 换方程 
                xin = -2*phase_cut**2+phase_cut**4
                xout = -1 
                x2 = np.where(inbool,xin,xout)
                wX_cut = get_wX(x2[:,None],X_base_cut,inv_df_cut)
                L = get_L(wX_cut,wF_cut) # 只有A4 
                if L[0]>0:
                    chi2 = np.nan
                else: 
                    wR = wF_cut-wX_cut@L
                    chi2 = wR.T@wR
            else:
                chi2 = np.nan 

        # 求chi2_base 
        chi2_base = get_chi2_base(X_base_cut,inv_df_cut,wF_cut)
        dlnLmax2 = 0.5*(chi2_base-chi2)
        # if dlnL_global > dlnLmax: 
        #     dlnLmax = np.copy(dlnL_global)
        #     q_chosen = q
    return dlnLmax1,dlnLmax2,d,tm0 


# 。。。。。。。。。。。。。。。。。。。画图拟合
# 知道了最佳的P,tm0,d
#@jit(nopython=True)
def result_fitting_no_limit_ar(window_mode,P,d,tm0,t,f,e,trend_order,cri0,get_X_sig_cut,get_X_trend_cut,inv_df,wF,sig_order,ar_order,get_X_ar,f_new):
# 出三个图：
# 1相位图：散点[phase_cut_group],[f_cut_group],拟合数据[model_folded_f_cut_group]
# 2去趋势相位图：散点phase_cut_ori_sort,de_f_cut,拟合数据model_folded_de_f_cut
# 3时间图：散点t,f,拟合数据[t_cut_group],[model_f_group]

# in_num,cut_num
    q = d/P  
    qmax = dmax/P
    phase,group = get_phase(t,tm0,P)
    if window_mode == 'fixed': 
        cut_index,group_cut,phase_cut_ori,g_uni_final = selection_fixed_window(phase,group,q,qmax,e,trend_order,P,cri0,ar_order)
    else:
        cut_index,group_cut,phase_cut_ori,g_uni_final = selection_varied_window(phase,group,q,e,trend_order,P,cri0,ar_order)
    phase_cut = (2*phase_cut_ori-1)/q
    # 拟合的是phase_cut
    t_cut = t[cut_index]
    f_cut = f[cut_index]

    inbool = (phase_cut>=-1)&(phase_cut<=1)
    y_cut = np.where(inbool,phase_cut,1)
    X_sig_cut = get_X_sig_cut(y_cut) 
    X0 = get_X0_trend(len(phase_cut),np.array(g_uni_final),group_cut) 
    X_trend_cut = get_X_trend_cut(phase_cut.reshape(len(phase_cut),1),X0)
    
    X_ar_cut = get_X_ar(cut_index,f_new)  
    X_base_cut = np.hstack((X_trend_cut,X_ar_cut)) 
    
    inv_df_cut = inv_df[cut_index]
    X_cut,wX_cut = get_XwX(X_sig_cut,X_base_cut,inv_df_cut)
    wF_cut = wF[cut_index]
    covar = get_covar(wX_cut) 
    L = covar@wX_cut.T@wF_cut
    # 上面得到了t,f,phase_cut_ori

    # 计算model_f
    model_f = X_cut@L 

    # 计算[phase_cut_group],[t_cut_group],[model_f_group]
    phase_cut_group = []
    f_cut_group = []
    model_folded_f_cut_group = []
    t_cut_group = []
    model_f_group = []
    a = 1
    for s in g_uni_final:
        idx = np.where(group_cut==s)
        phase_cut_group.append(phase_cut_ori[idx])
        t_cut_group.append(t_cut[idx])
        model_f_group.append(model_f[idx])
        # 计算[f_cut_group],[model_folded_f_cut_group]
        if a: 
            f0 = f_cut[idx]  
            model_f0 = model_f[idx]
            f0_min = np.min(f0)
            a = 0
        else:
            f0 = f_cut[idx]
            model_f0 = model_f[idx]
            f0_max = np.max(f0)   
            delta_f0 = np.abs(f0_min-f0_max)
            f0 -= delta_f0
            model_f0 -= delta_f0
            f0_min = np.min(f0)
        f_cut_group.append(f0)
        model_folded_f_cut_group.append(model_f0)

    # 计算减掉trend之后的
    nL_trend = (X_base_cut.shape)[1] 
    L_trend = L[-nL_trend:]
    f_trend = X_base_cut@L_trend

    de_f_cut = f_cut-f_trend # 原始数据
    model_folded_de_f_cut = model_f-f_trend # 拟合数据
    # 排序
    idx2 = np.argsort(phase_cut_ori)
    phase_cut_ori_sort = phase_cut_ori[idx2] 
    model_folded_de_f_cut = model_folded_de_f_cut[idx2]
    de_f_cut = de_f_cut[idx2]

    in_num = np.sum(inbool)
    cut_num = len(cut_index)
    # depth = np.max(model_folded_df_cut)
    # 看看几个depth的大小
    nn = int(sig_order/2.0) 
    covar_cut = covar[:nn,:nn]   
    diff_arr = np.array([1]*nn)
    depth = np.sum(L[:nn]) 
    depth_sigma = (diff_arr@covar_cut@diff_arr.T)**0.5
    depth_snr = depth/depth_sigma

    return phase_cut_group,f_cut_group,model_folded_f_cut_group,phase_cut_ori_sort,de_f_cut,model_folded_de_f_cut,t_cut_group,model_f_group,in_num,cut_num,depth,depth_sigma,depth_snr

# 1相位图：散点[phase_cut_group],[f_cut_group],拟合数据[model_folded_f_cut_group]
# 2去趋势相位图：散点phase_cut_ori_sort,de_f_cut,拟合数据model_folded_de_f_cut
# 3时间图：散点t,f,拟合数据[t_cut_group],[model_f_group]

def result_fitting_limit_ar(window_mode,P,d,tm0,t,f,e,trend_order,cri0,get_X_sig_cut,get_X_trend_cut,inv_df,wF,sig_order,ar_order,get_X_ar,f_new):
# 出三个图：
# 1相位图：散点[phase_cut_group],[f_cut_group],拟合数据[model_folded_f_cut_group]
# 2去趋势相位图：散点phase_cut_ori_sort,de_f_cut,拟合数据model_folded_de_f_cut
# 3时间图：散点t,f,拟合数据[t_cut_group],[model_f_group]

# in_num,cut_num
    q = d/P  
    qmax = dmax/P
    phase,group = get_phase(t,tm0,P)
    if window_mode == 'fixed':
        cut_index,group_cut,phase_cut_ori,g_uni_final = selection_fixed_window(phase,group,q,qmax,e,trend_order,P,cri0,ar_order)
    else:
        cut_index,group_cut,phase_cut_ori,g_uni_final = selection_varied_window(phase,group,q,e,trend_order,P,cri0,ar_order)
    phase_cut = (2*phase_cut_ori-1)/q
    # 拟合的是phase_cut
    t_cut = t[cut_index]
    f_cut = f[cut_index]

    inbool = (phase_cut>=-1)&(phase_cut<=1)
    y_cut = np.where(inbool,phase_cut,1)
    X_sig_cut = get_X_sig_cut(y_cut) 
    X0 = get_X0_trend(len(phase_cut),np.array(g_uni_final),group_cut) 
    X_trend_cut = get_X_trend_cut(phase_cut.reshape(len(phase_cut),1),X0)
    
    X_ar_cut = get_X_ar(cut_index,f_new)  
    X_base_cut = np.hstack((X_trend_cut,X_ar_cut))     
    
    inv_df_cut = inv_df[cut_index]
    X_cut,wX_cut = get_XwX(X_sig_cut,X_base_cut,inv_df_cut)
    wF_cut = wF[cut_index]
    covar = get_covar(wX_cut) 
    L = covar@wX_cut.T@wF_cut
    # 上面得到了t,f,phase_cut_ori

    if sig_order == 4:
        if (L[0]/L[1]>-2) and (L[0]*L[1]<0): 
            yyin = -2*phase_cut**2+phase_cut**4
            yyout = -1
            y2 = np.where(inbool,yyin,yyout)
            X_sig_cut2 = y2.reshape(len(y2),1)
            X_cut,wX_cut = get_XwX(X_sig_cut2,X_base_cut,inv_df_cut)
            covar = get_covar(wX_cut) 
            L = covar@wX_cut.T@wF_cut # 只有A4
            depth = -np.copy(L[0]) 
            depth_sigma = np.copy(covar[0,0])**0.5
        else:
            nn = int(sig_order/2.0) 
            covar_cut = covar[:nn,:nn]   
            diff_arr = np.array([1]*nn)
            depth = np.sum(L[:nn]) 
            depth_sigma = (diff_arr@covar_cut@diff_arr.T)**0.5
    else:
        nn = int(sig_order/2.0) 
        covar_cut = covar[:nn,:nn]   
        diff_arr = np.array([1]*nn)
        depth = np.sum(L[:nn]) 
        depth_sigma = (diff_arr@covar_cut@diff_arr.T)**0.5

    # 计算model_f
    model_f = X_cut@L 

    # 计算[phase_cut_group],[t_cut_group],[model_f_group]
    phase_cut_group = []
    f_cut_group = []
    model_folded_f_cut_group = []
    t_cut_group = []
    model_f_group = []
    a = 1
    for s in g_uni_final:
        idx = np.where(group_cut==s)
        phase_cut_group.append(phase_cut_ori[idx])
        t_cut_group.append(t_cut[idx])
        model_f_group.append(model_f[idx])
        # 计算[f_cut_group],[model_folded_f_cut_group]
        if a: 
            f0 = f_cut[idx]  
            model_f0 = model_f[idx]
            f0_min = np.min(f0)
            a = 0
        else:
            f0 = f_cut[idx]
            model_f0 = model_f[idx]
            f0_max = np.max(f0)   
            delta_f0 = np.abs(f0_min-f0_max)
            f0 -= delta_f0
            model_f0 -= delta_f0
            f0_min = np.min(f0)
        f_cut_group.append(f0)
        model_folded_f_cut_group.append(model_f0)

    # 计算减掉trend之后的
    nL_trend = (X_base_cut.shape)[1] 
    L_trend = L[-nL_trend:]
    f_trend = X_base_cut@L_trend

    de_f_cut = f_cut-f_trend # 原始数据
    model_folded_de_f_cut = model_f-f_trend # 拟合数据
    # 排序
    idx2 = np.argsort(phase_cut_ori)
    phase_cut_ori_sort = phase_cut_ori[idx2] 
    model_folded_de_f_cut = model_folded_de_f_cut[idx2]
    de_f_cut = de_f_cut[idx2]
    

    in_num = np.sum(inbool)
    cut_num = len(cut_index)
    # depth = np.max(model_folded_df_cut)
    # 看看几个depth的大小
    # 求depth，depth snr
    depth_snr = depth/depth_sigma
    return phase_cut_group,f_cut_group,model_folded_f_cut_group,phase_cut_ori_sort,de_f_cut,model_folded_de_f_cut,t_cut_group,model_f_group,in_num,cut_num,depth,depth_sigma,depth_snr