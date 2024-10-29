# 处理其他数据还得想一下

# 发现用multiprocessing很难用

import numpy as np
import matplotlib.pyplot as plt       
from matplotlib import rcParams; rcParams["figure.dpi"] = 250  
from tqdm import tqdm 
from TBFP704_2_add_ar.constants import gap_cri,dmin_ori,dmax
from TBFP704_2_add_ar.fun_sig_and_trend import sele_sig,sele_trend,sele_global_trend,sele_ar
from TBFP704_2_add_ar.step_map import sele_fitting_2D
from TBFP704_2_add_ar.step_map_ar import sele_fitting_2D_ar
from TBFP704_2_add_ar.plots import map2d_plot,periodogram_plot0,periodogram_plot1,periodogram_plot2,phase_plot1,phase_plot2,time_plot
from TBFP704_2_add_ar.step_BFP import sele_fitting_global,sele_result_fitting,running_median 
from TBFP704_2_add_ar.step_BFP_ar import sele_fitting_global_ar,sele_result_fitting_ar
from TBFP704_2_add_ar.grid import dp_fun,P_grid
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from functools import partial
from time import time

class TransitBayesFactorPeriodogram:
    def __init__(self,ts,fs,dfs):
        self.ts,self.fs,self.dfs = ts,fs,dfs  
        self.t,self.f,self.df = np.concatenate(self.ts),np.concatenate(self.fs),np.concatenate(self.dfs)
        self.time_span = self.t[-1]-self.t[0] 
        self.dt = np.ediff1d(self.t)   
        self.tm_gap,self.tm_sam,self.d_sam = 0,0,0
        self.dlnL = 0 
        self.e = 0
        self.trend_order,self.sig_order = 0,0
        self.get_X_sig_cut = 0
        self.get_X_trend_cut = 0
        self.get_X_ar = 0
        self.ar_order = 0
        self.limit = 0

        self.Rs = 0
        self.Ms = 0
        self.OS_P = 0
        self.OS_tm = 0

        # # 周期图
        self.P_sam = []
        self.lnBF1 = 0
        self.lnBF2 = 0
        # self.lnBF = 0
        # self.trend_lnBF = 0
        self.Pmin_step1 = 0

        # # 最佳参数
        self.P_best = 0,
        self.d_best = 0,
        self.tm0_best = 0,
        self.lnBFmax = 0,

        self.phase_cut_group = 0,
        self.f_cut_group = 0,
        self.model_folded_f_cut_group = 0,
        self.phase_cut_ori_sort = 0,
        self.de_f_cut = 0,
        self.model_folded_de_f_cut = 0,
        self.t_cut_group = 0,
        self.model_f_group = 0,
        self.in_num,self.cut_num = 0,0,
        self.depth_best,self.depth_sigma_best,self.depth_snr = 0,0,0
        self.cost_time1,self.cost_time2 = 0,0
        self.SR, self.power_raw, self.power, self.SDE_raw, self.SDE = 0,0,0,0,0
    def spectra(self,dlnL, oversampling_factor):
        SR = dlnL/np.nanmax(dlnL)
        SDE_raw = (1 - np.nanmean(SR)) / np.nanstd(SR) 

        # Scale SDE_power from 0 to SDE_raw
        power_raw = SR - np.nanmean(SR)  # shift down to the mean being zero
        scale = SDE_raw / np.nanmax(power_raw)  # scale factor to touch max=SDE_raw
        power_raw = power_raw * scale
        SDE_MEDIAN_KERNEL_SIZE = 30
        # Detrended SDE, named "power"
        kernel = oversampling_factor * SDE_MEDIAN_KERNEL_SIZE
        if kernel % 2 == 0:
            kernel = kernel + 1
        if len(power_raw) > 2 * kernel:
            my_median = running_median(power_raw, kernel)
            power = power_raw - my_median
            # Re-normalize to range between median = 0 and peak = SDE
            # shift down to the mean being zero
            power = power - np.nanmean(power)
            SDE = np.nanmax(power / np.nanstd(power))
            # scale factor to touch max=SDE
            scale = SDE / np.nanmax(power)
            power = power * scale
        else:
            power = power_raw 
            SDE = SDE_raw

        return SR, power_raw, power, SDE_raw, SDE

# 改完了
    def BF_map(self,max_workers,window_mode,limit,win,sig_order,trend_order,ar_order,OS_d,OS_tm,Pmin_step1=10,OS_P=2,Rs=1,Ms=1):
        t_start = time()
        self.Rs = Rs
        self.Ms = Ms
        self.OS_P = OS_P
        self.trend_order = trend_order
        self.sig_order = sig_order
        self.ar_order = ar_order  
        self.Pmin_step1 = Pmin_step1
        self.OS_tm = OS_tm 
        self.limit = limit
        self.window_mode = window_mode

        if win<=1:  
            raise ValueError("输入的win小于1,必须得大于1")
        print(f'一共分成了{len(self.ts)}个拟合段')
        # row = len(self.seg_t)//3+1
        # fig,ax = plt.subplots(row,3,figsize=(15,2.5*row))  
        
        # d_sam
        dtmin = np.min(self.dt)   
        dmin = np.maximum(dmin_ori,(sig_order/2.0+2)*dtmin) # 也可以设置的更小，只不过后面采样也不满足条件，不会计算
        print('撒的最小的d',dmin)
        if dmin >= dmax:
            raise ValueError("撒点的最小值大于最大值，请重新调整撒点")
        print('d_sampling: 撒的最小的d',dmin,'撒的最大的d',dmax)
        self.d_sam = np.logspace(np.log10(dmin),np.log10(dmax),OS_d)
        size_d = len(self.d_sam) 
        print(f'一共撒了{OS_d}个,{self.d_sam}') 

        # tm_sam
        Pmin_ori = win*dmax
        if Pmin_step1<Pmin_ori:
            print('输入的周期小于能探测的最小周期,默认换成win*dmax')
            Pmin_step1 = np.copy(Pmin_ori)  
        else: # 否则就判断输入的周期，对不对  
            if Pmin_step1 > self.time_span/2.0:
                raise ValueError("输入的Pmin比默认最大Pmax大,请重新输入")

        # 确定tm_gap  
        dPmin = dp_fun(Rs,Ms,self.time_span,OS_P,Pmin_step1)
        self.tm_gap = np.minimum(dmin/OS_tm,dPmin) # OS_tm设置成5好了
        print('最小的周期是',Pmin_step1,'tm_gap相当于OS_tm是',dmin/dPmin)      
        self.tm_sam = np.arange(self.t[0],self.t[-1],self.tm_gap)
        print(f'一共撒了{len(self.tm_sam)}个tm_sam') 

        # 下面拟合用到的  
        self.get_X_sig_cut = sele_sig(sig_order) 
        self.get_X_trend_cut = sele_trend(trend_order)
            
        if ar_order == 0:
            fitting_2D = sele_fitting_2D(limit,window_mode)
        else:
            self.get_X_ar = sele_ar(ar_order) 
            fitting_2D = sele_fitting_2D_ar(limit,window_mode)

        self.e = win/2.0
        
        if ar_order == 0:
            if max_workers == 1: 
                pbar = tqdm(total=len(self.ts))
                self.dlnL = fitting_2D(0,self.ts,self.fs,self.dfs,size_d,self.d_sam,self.tm_sam,self.e,
                                        sig_order,trend_order,self.get_X_sig_cut,self.get_X_trend_cut)
                for i in range(1,len(self.ts)): 
                    dlnL_here = fitting_2D(i,self.ts,self.fs,self.dfs,size_d,self.d_sam,self.tm_sam,self.e,
                                            sig_order,trend_order,self.get_X_sig_cut,self.get_X_trend_cut)
                    self.dlnL = np.hstack((self.dlnL,dlnL_here)) 
                    pbar.update(1)
                pbar.close()
                    # j = i//3  
                    # k = i%3
                    # if row != 1:
                    #     map2d_plot0(self.dlnL[:,idx_tm],self.tm_sam[idx_tm],self.d_sam*24,i,j,k,fig,ax)
                    # else:
                    #     map2d_plot01(self.dlnL[:,idx_tm],self.tm_sam[idx_tm],self.d_sam*24,i,j,fig,ax)
                # fig.tight_layout() 
            else: # 多核
                i_arr = np.arange(0,len(self.ts),dtype=int)
                fun_i = partial(fitting_2D,
                                seg_ts=self.ts,
                                seg_fs=self.fs,
                                seg_dfs=self.dfs,
                                size_d=size_d,
                                d_sam=self.d_sam,
                                tm_sam=self.tm_sam,
                                e=self.e,
                                sig_order=sig_order,
                                trend_order=trend_order,
                                get_X_sig_cut=self.get_X_sig_cut,
                                get_X_trend_cut=self.get_X_trend_cut)
                with ProcessPoolExecutor(max_workers=max_workers) as exe:
                    result = list(tqdm(exe.map(fun_i,i_arr),total=len(i_arr))) # 这个保持了原来的顺序
                self.dlnL = np.concatenate(result, axis=1) 
        else:
            if max_workers == 1: 
                pbar = tqdm(total=len(self.ts))
                self.dlnL = fitting_2D(0,self.ts,self.fs,self.dfs,size_d,self.d_sam,self.tm_sam,self.e,
                                        sig_order,trend_order,self.get_X_sig_cut,self.get_X_trend_cut,ar_order,self.get_X_ar)
                for i in range(1,len(self.ts)): 
                    dlnL_here = fitting_2D(i,self.ts,self.fs,self.dfs,size_d,self.d_sam,self.tm_sam,self.e,
                                            sig_order,trend_order,self.get_X_sig_cut,self.get_X_trend_cut,ar_order,self.get_X_ar)
                    self.dlnL = np.hstack((self.dlnL,dlnL_here)) 
                    pbar.update(1)
                pbar.close()
                    # j = i//3  
                    # k = i%3
                    # if row != 1:
                    #     map2d_plot0(self.dlnL[:,idx_tm],self.tm_sam[idx_tm],self.d_sam*24,i,j,k,fig,ax)
                    # else:
                    #     map2d_plot01(self.dlnL[:,idx_tm],self.tm_sam[idx_tm],self.d_sam*24,i,j,fig,ax)
                # fig.tight_layout() 
            else: # 多核
                i_arr = np.arange(0,len(self.ts),dtype=int)
                fun_i = partial(fitting_2D,
                                seg_ts=self.ts,
                                seg_fs=self.fs,
                                seg_dfs=self.dfs,
                                size_d=size_d,
                                d_sam=self.d_sam,
                                tm_sam=self.tm_sam,
                                e=self.e,
                                sig_order=sig_order,
                                trend_order=trend_order,
                                get_X_sig_cut=self.get_X_sig_cut,
                                get_X_trend_cut=self.get_X_trend_cut,
                                ar_order=ar_order,
                                get_X_ar=self.get_X_ar)
                with ProcessPoolExecutor(max_workers=max_workers) as exe:
                    result = list(tqdm(exe.map(fun_i,i_arr),total=len(i_arr))) # 这个保持了原来的顺序
                self.dlnL = np.concatenate(result, axis=1) 

        lnBF_step1 = self.dlnL-(2+sig_order/2.0)/2.0*np.log(len(self.t))
        map2d_plot(lnBF_step1,self.tm_sam,self.d_sam*24,'lnBF')
        t_end = time()
        self.cost_time1 = (t_end-t_start)/60 # 这个是分钟
        print('第一步花费的时间',self.cost_time1,'min')

    def BFP(self,addre,host_name,P_ref=0,max_workers=1,Pmin_step2=10,Pmax_step2=np.nan):
        t_start = time()
        get_global_X_trend_cut = sele_global_trend(self.trend_order)
        f_new = np.concatenate((np.zeros(5),self.f)) 
         
        if self.ar_order == 0:
            fitting_global = sele_fitting_global(self.limit,self.window_mode)
            result_fitting = sele_result_fitting(self.limit)
        else: 
            fitting_global = sele_fitting_global_ar(self.limit,self.window_mode)
            result_fitting = sele_result_fitting_ar(self.limit)             
        
        # 如果
        if Pmin_step2 < self.Pmin_step1:
            print('输入的Pmin小于第一步的Pmin,自动设置成第一步的Pmin')
            Pmin_step2 = np.copy(self.Pmin_step1)   
        if Pmin_step2 > self.time_span/2.0:
            raise ValueError("输入的Pmin比默认最大Pmax大,请重新输入")

        if Pmax_step2 != Pmax_step2: # 如果外界不输入最大周期，就默认是
            Pmax_step2 = self.time_span/2.0 
        else:        
            if Pmax_step2 < Pmin_step2: 
                raise ValueError("输入的Pmax比默认最小min还要小,请重新输入")    
            if Pmax_step2 > self.time_span/2.0:
                print('输入的Pmax大于默认的时间跨度的一般,自动设置最大窗口为跨度一半')
                Pmax_step2 = self.time_span/2.0

        # 构造P_sam
        self.P_sam = P_grid(self.Rs,self.Ms,self.time_span,self.OS_P,Pmin_step2,Pmax_step2)
        self.P_sam = (self.P_sam//self.tm_gap)*self.tm_gap # 重新修正了
        self.P_sam = np.sort(self.P_sam)
        print('P_sam的个数',len(self.P_sam))

        dt_median = np.median(self.dt)
        cri0 = gap_cri*dt_median
        inv_df = (1/self.df)[:,None]
        wF = self.f/self.df

        if self.ar_order == 0:
            if max_workers == 1:
                dlnL_arr1 = []
                dlnL_arr2 = [] # global之后的
                ds = []
                tm0s = []
                pbar1 = tqdm(total=len(self.P_sam))
                for P in self.P_sam: 

                    dlnLmax1,dlnLmax2,d_best0,tm0_best0 = fitting_global(P,self.d_sam,self.tm_sam,self.dlnL,self.tm_gap,self.OS_tm,self.get_X_sig_cut,get_global_X_trend_cut,cri0,self.e,self.t,inv_df,wF,self.trend_order,self.sig_order)     
                    # print('P',P)
                    # print('d',d_best0)
                    # print('tm0',tm0_best0) 
                    dlnL_arr1.append(dlnLmax1) 
                    dlnL_arr2.append(dlnLmax2) 
                    ds.append(d_best0)
                    tm0s.append(tm0_best0)
                    pbar1.update(1)
                pbar1.close()
                dlnL_arr1 = np.array(dlnL_arr1)
                dlnL_arr2 = np.array(dlnL_arr2)
                ds = np.array(ds)
                tm0s = np.array(tm0s)
            else:
                dlnL_arr1 = []
                dlnL_arr2 = []
                ds = []
                tm0s = []
                # P_seg = np.array_split(self.P_sam, max_workers)  # 数组，分成了几份
                fun_P = partial(fitting_global,
                                d_sam=self.d_sam,
                                tm_sam=self.tm_sam,
                                dlnL=self.dlnL,
                                tm_gap=self.tm_gap,
                                OS_tm=self.OS_tm, 
                                get_X_sig_cut=self.get_X_sig_cut,
                                get_X_trend_cut=get_global_X_trend_cut,
                                cri0=cri0,
                                e=self.e,
                                t=self.t,
                                inv_df=inv_df,
                                wF=wF,
                                trend_order=self.trend_order,
                                sig_order=self.sig_order) 
                with ThreadPoolExecutor(max_workers=max_workers) as exe:
                    result = list(tqdm(exe.map(fun_P,self.P_sam),total=len(self.P_sam))) # 这个保持了原来的顺序
                for k in result:
                    # dlnL_arr1 = np.concatenate((dlnL_arr1,k[0]))
                    # dlnL_arr2 = np.concatenate((dlnL_arr2,k[1]))
                    # ds = np.concatenate((ds,k[2]))
                    # tm0s = np.concatenate((tm0s,k[3]))   
                    dlnL_arr1.append(k[0])
                    dlnL_arr2.append(k[1])
                    ds.append(k[2])
                    tm0s.append(k[3])
                dlnL_arr1 = np.array(dlnL_arr1)
                dlnL_arr2 = np.array(dlnL_arr2)
                ds = np.array(ds)
                tm0s = np.array(tm0s)
        else:
            if max_workers == 1:
                dlnL_arr1 = []
                dlnL_arr2 = [] # global之后的
                ds = []
                tm0s = []
                pbar1 = tqdm(total=len(self.P_sam))
                for P in self.P_sam: 

                    dlnLmax1,dlnLmax2,d_best0,tm0_best0 = fitting_global(P,self.d_sam,self.tm_sam,self.dlnL,self.tm_gap,self.OS_tm,self.get_X_sig_cut,get_global_X_trend_cut,cri0,self.e,self.t,inv_df,wF,self.trend_order,self.sig_order,self.ar_order,self.get_X_ar,f_new)     
                    # print('P',P)
                    # print('d',d_best0)
                    # print('tm0',tm0_best0) 
                    dlnL_arr1.append(dlnLmax1) 
                    dlnL_arr2.append(dlnLmax2) 
                    ds.append(d_best0)
                    tm0s.append(tm0_best0)
                    pbar1.update(1)
                pbar1.close()
                dlnL_arr1 = np.array(dlnL_arr1)
                dlnL_arr2 = np.array(dlnL_arr2)
                ds = np.array(ds)
                tm0s = np.array(tm0s)
            else:
                dlnL_arr1 = []
                dlnL_arr2 = []
                ds = []
                tm0s = []
                # P_seg = np.array_split(self.P_sam, max_workers)  # 数组，分成了几份
                fun_P = partial(fitting_global,
                                d_sam=self.d_sam,
                                tm_sam=self.tm_sam,
                                dlnL=self.dlnL,
                                tm_gap=self.tm_gap,
                                OS_tm=self.OS_tm, 
                                get_X_sig_cut=self.get_X_sig_cut,
                                get_X_trend_cut=get_global_X_trend_cut,
                                cri0=cri0,
                                e=self.e,
                                t=self.t,
                                inv_df=inv_df,
                                wF=wF,
                                trend_order=self.trend_order,
                                sig_order=self.sig_order,
                                ar_order=self.ar_order,
                                X_ar=self.get_X_ar,
                                f_new=f_new)  
                with ThreadPoolExecutor(max_workers=max_workers) as exe:
                    result = list(tqdm(exe.map(fun_P,self.P_sam),total=len(self.P_sam))) # 这个保持了原来的顺序
                for k in result:
                    # dlnL_arr1 = np.concatenate((dlnL_arr1,k[0]))
                    # dlnL_arr2 = np.concatenate((dlnL_arr2,k[1]))
                    # ds = np.concatenate((ds,k[2]))
                    # tm0s = np.concatenate((tm0s,k[3]))   
                    dlnL_arr1.append(k[0])
                    dlnL_arr2.append(k[1])
                    ds.append(k[2])
                    tm0s.append(k[3])
                dlnL_arr1 = np.array(dlnL_arr1)
                dlnL_arr2 = np.array(dlnL_arr2)
                ds = np.array(ds)
                tm0s = np.array(tm0s)

            
        self.lnBF1 = dlnL_arr1-(3+self.sig_order/2.0)/2.0*np.log(len(self.t)) # 这个没有全局拟合
        self.lnBF2 = dlnL_arr2-(3+self.sig_order/2.0)/2.0*np.log(len(self.t)) # 这个全局拟合了

        # SDE_MEDIAN_KERNEL_SIZE = 110
        # kernel = self.OS_P * SDE_MEDIAN_KERNEL_SIZE # 这个可以改
        # if kernel % 2 == 0: 
        #     kernel = kernel + 1  
        # self.trend_lnBF = running_median(self.lnBF2, kernel)
        # # 再存一个去掉趋势的数据   
        # self.lnBF = self.lnBF2-self.trend_lnBF  
        
        # 计算SR,power_raw,power
        # SDE,SDE_raw
        if np.nanmin(self.lnBF2)<0:
            aaa = self.lnBF2 - np.nanmin(self.lnBF2)
        else:
            aaa = np.copy(self.lnBF2)
        self.SR, self.power_raw, self.power, self.SDE_raw, self.SDE = self.spectra(aaa, self.OS_P) 


        # 最佳的参数：
        best_idx = np.nanargmax(self.power) 
        self.P_best = self.P_sam[best_idx]
        self.d_best = ds[best_idx] #6.4663/24
        self.tm0_best = tm0s[best_idx]  #206.58711637652596
        # self.lnBFmax = self.lnBF[best_idx] 
        print('P_best2',self.P_best,'day')
        print('d_best2',self.d_best*24,'hours')
        print('tm0_best2',self.tm0_best,'day')
        print('SDE',self.SDE) 

        periodogram_plot0(self.P_sam,self.SR,P_ref,self.P_sam[np.nanargmax(self.SR)],addre,host_name,self.trend_order,self.e*2,self.window_mode,self.ar_order)
        periodogram_plot1(self.SDE_raw,self.P_sam,self.power_raw,P_ref,self.P_sam[np.nanargmax(self.power_raw)],addre,host_name,self.trend_order,self.e*2,self.window_mode,self.ar_order)
        periodogram_plot2(self.SDE,self.P_sam,self.power,P_ref,self.P_sam[np.nanargmax(self.power)],addre,host_name,self.trend_order,self.e*2,self.window_mode,self.ar_order)

        # 全局拟合周期图的结果
        if self.ar_order == 0:
            self.phase_cut_group,self.f_cut_group,self.model_folded_f_cut_group,self.phase_cut_ori_sort,self.de_f_cut,self.model_folded_de_f_cut,self.t_cut_group,self.model_f_group,self.in_num,self.cut_num,self.depth_best,self.depth_sigma_best,self.depth_snr = result_fitting(self.window_mode,self.P_best,self.d_best,self.tm0_best,self.t,self.f,self.e,self.trend_order,cri0,self.get_X_sig_cut,get_global_X_trend_cut,inv_df,wF,self.sig_order)
        else:
            self.phase_cut_group,self.f_cut_group,self.model_folded_f_cut_group,self.phase_cut_ori_sort,self.de_f_cut,self.model_folded_de_f_cut,self.t_cut_group,self.model_f_group,self.in_num,self.cut_num,self.depth_best,self.depth_sigma_best,self.depth_snr = result_fitting(self.window_mode,self.P_best,self.d_best,self.tm0_best,self.t,self.f,self.e,self.trend_order,cri0,self.get_X_sig_cut,get_global_X_trend_cut,inv_df,wF,self.sig_order,self.ar_order,self.get_X_ar,f_new)

        print('cut_num',self.cut_num)
        print('in_num',self.in_num) 
        print(f'depth {self.depth_best*10**6}({self.depth_sigma_best*10**6})ppm')
        print('depth_snr',self.depth_snr)

        # 画三个图
        phase_plot1(self.P_best,self.d_best*24,self.tm0_best,self.phase_cut_group,self.f_cut_group,self.model_folded_f_cut_group,addre,host_name,self.trend_order,self.e*2,self.window_mode,self.ar_order)
        phase_plot2(self.P_best,self.d_best*24,self.tm0_best,self.phase_cut_ori_sort,self.de_f_cut,self.model_folded_de_f_cut,addre,host_name,self.trend_order,self.e*2,self.window_mode,self.ar_order)
        time_plot(self.P_best,self.d_best*24,self.tm0_best,self.trend_order,self.sig_order,self.e*2,self.window_mode,self.t,self.f,self.t_cut_group,self.model_f_group,addre,host_name,self.ar_order) 

        t_end = time()
        self.cost_time2 = (t_end-t_start)/3600 # 这个是小时
        print('第二步花费的时间',self.cost_time2,'h')  
