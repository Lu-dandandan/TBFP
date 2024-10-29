import numpy as np   
from TBFP704_2_add_ar.constants import gap_cri,min_num_in_one_seg
import matplotlib.pyplot as plt       
from matplotlib import rcParams; rcParams["figure.dpi"] = 250   


def get_segs_weak(t,f,df): 
    # 输入要求，确保时间是有序的
    dt = np.diff(t)
    dt_median = np.median(dt) 
    
    # 按照间隔分段 
    split_indices = np.where(dt > dt_median*gap_cri)[0]+1
    if len(split_indices) > 0:
        t_segs = np.split(t, split_indices)
        f_segs = np.split(f, split_indices)
        df_segs = np.split(df, split_indices)  
    else:
        print('没分成子段，直接用原来的数据')
        t_segs = [t]
        f_segs = [f]
        df_segs = [df]
    
    # 数据少于10个点的段不要
    t_segs2 = [] 
    f_segs2 = [] 
    df_segs2 = []  
    for i in range(len(t_segs)):
        t_seg = t_segs[i]
        f_seg = f_segs[i]
        df_seg = df_segs[i] 
        if len(t_seg) > min_num_in_one_seg:
            t_segs2.append(t_seg)
            f_segs2.append(f_seg)
            df_segs2.append(df_seg) 
        else:
            pass 
            #print('这个段数据少于10个')
    print(f'一共分成了{len(t_segs2)}段')   
    return t_segs2,f_segs2,df_segs2 
 
def get_segs_strong(t,f,df):  
    # 输入要求，确保时间是有序的

    dt = np.diff(t)
    dt_median = np.median(dt) 
    
    # 按照间隔分段
    split_indices = np.where(dt > dt_median*gap_cri)[0]+1
    if len(split_indices) > 0: 
        t_segs = np.split(t, split_indices)
        f_segs = np.split(f, split_indices)
        df_segs = np.split(df, split_indices)  
    else:
        print('没分成子段，直接用原来的数据')
        t_segs = [t]
        f_segs = [f]
        df_segs = [df]
    
    # 数据少于10个点的段不要
    t_segs2 = [] 
    f_segs2 = [] 
    df_segs2 = []  
    for i in range(len(t_segs)):
        t_seg = t_segs[i]
        f_seg = f_segs[i]
        df_seg = df_segs[i] 
        if len(t_seg) > min_num_in_one_seg:
            t_segs2.append(t_seg)
            f_segs2.append(f_seg)
            df_segs2.append(df_seg) 
        else:
            pass 
            #print('这个段数据少于10个')
    print(f'一共分成了{len(t_segs2)}段')   
    
    # 3. 遍历每一段，当该段df有超过5sigma就断开
    t_segs3 = []
    f_segs3 = []
    df_segs3 = [] 
    for i in range(len(t_segs2)):
        t_seg = t_segs2[i]
        f_seg = f_segs2[i]
        df_seg = df_segs2[i]  
        diff_f = np.abs(np.diff(f_seg))
        diff_f_std = np.std(diff_f,ddof=1)  
        diff_f_median = np.median(diff_f) 
        split_indices = np.where(np.abs(diff_f-diff_f_median)>5*diff_f_std)[0]+1
        if len(split_indices) > 0: 
            plt.plot(t_seg,f_seg) 
            print('有新段')
            t_seg0 = np.split(t_seg, split_indices)
            f_seg0 = np.split(f_seg, split_indices)
            df_seg0 = np.split(df_seg, split_indices)
            for j in range(len(t_seg0)): 
                plt.plot(t_seg0[j],f_seg0[j])
                print('j',len(t_seg0[j])) 
            plt.show()
            t_segs3.extend(t_seg0)
            f_segs3.extend(f_seg0) 
            df_segs3.extend(df_seg0) 
        else:
            t_segs3.append(t_seg)
            f_segs3.append(f_seg)
            df_segs3.append(df_seg)
    
    # 4. 少于4个点的段就不要。认为向下的也不是transit
    t_segs4 = []
    f_segs4 = []
    df_segs4 = [] 
    for i in range(len(t_segs3)):
        t_seg = t_segs3[i]
        f_seg = f_segs3[i]
        df_seg = df_segs3[i] 
        if (len(t_seg) > 4): 
            t_segs4.append(t_seg)
            f_segs4.append(f_seg)
            df_segs4.append(df_seg)            
    print(f'一共分成了{len(t_segs4)}段') 
    return t_segs4,f_segs4,df_segs4