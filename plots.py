import numpy as np            
import matplotlib.pyplot as plt       
from matplotlib import rcParams; rcParams["figure.dpi"] = 250   

def map2d_plot(z,tm_sam,d_sam_hour,title):
    fig,ax = plt.subplots(figsize=(12,5))
    im = ax.contourf(z,cmap=plt.cm.coolwarm,levels=20)
    ax.set_xticks(np.linspace(0,len(tm_sam)-1,20),['%.0f'%i for i in np.linspace(tm_sam[0],tm_sam[-1],20)])
    range1 = np.linspace(0,len(d_sam_hour)-1,9)[[0,4,6,7,8]]
    range2 = np.logspace(np.log10(d_sam_hour[0]),np.log10(d_sam_hour[-1]),9)[[0,4,6,7,8]]
    ax.set_yticks([i for i in range1],['%.3f'%i for i in range2])
    ax.set_xlabel('tm [day]')
    ax.set_ylabel('d [hour]')
    ax.set_title(f'{title}')
    fig.colorbar(im)
    plt.show()
# def map2d_plot0(z,tm_sam,d_sam_hour,ii,jj,kk,fig,ax):
#     im = ax[jj,kk].contourf(z,cmap=plt.cm.coolwarm,levels=20)
#     ax[jj,kk].set_xticks(np.linspace(0,len(tm_sam)-1,5),['%.0f'%i for i in np.linspace(tm_sam[0],tm_sam[-1],5)])
#     range1 = np.linspace(0,len(d_sam_hour)-1,9)[[0,4,6,7,8]]
#     range2 = np.logspace(np.log10(d_sam_hour[0]),np.log10(d_sam_hour[-1]),9)[[0,4,6,7,8]]
#     ax[jj,kk].set_yticks([i for i in range1],['%.3f'%i for i in range2])
#     ax[jj,kk].set_xlabel('tm [day]')
#     ax[jj,kk].set_ylabel('d [hour]')
#     ax[jj,kk].set_title(f'$\Delta L$,seg{ii}')
#     fig.colorbar(im)
# def map2d_plot01(z,tm_sam,d_sam_hour,ii,jj,fig,ax):
#     im = ax[jj].contourf(z,cmap=plt.cm.coolwarm,levels=20)
#     ax[jj].set_xticks(np.linspace(0,len(tm_sam)-1,5),['%.0f'%i for i in np.linspace(tm_sam[0],tm_sam[-1],5)])
#     range1 = np.linspace(0,len(d_sam_hour)-1,9)[[0,4,6,7,8]]
#     range2 = np.logspace(np.log10(d_sam_hour[0]),np.log10(d_sam_hour[-1]),9)[[0,4,6,7,8]]
#     ax[jj].set_yticks([i for i in range1],['%.3f'%i for i in range2])
#     ax[jj].set_xlabel('tm [day]')
#     ax[jj].set_ylabel('d [hour]')
#     ax[jj].set_title(f'$\Delta L$,seg{ii}')
#     fig.colorbar(im)

# def periodogram_plot(lnBFmax,P_sam,lnBF,P_ref,P_best):
#     fig,ax = plt.subplots() 
#     ax.plot(P_sam, lnBF, c='k', lw=0.5)
#     ax.set_xlabel('Period [days]')
#     ax.set_ylabel('lnBF')   
#     ax.set_xlim(np.min(P_sam), np.max(P_sam))
#     ax.axvline(P_best, alpha=0.4, lw=3)
#     if P_ref:
#         ax.axvline(P_ref, alpha=0.4, lw=3, c='orange')
#     for n in range(2, 20): 
#         ax.axvline(n*P_best, alpha=0.4, lw=1, linestyle="dashed")
#         ax.axvline(P_best/n, alpha=0.4, lw=1, linestyle="dashed")
#     #fig.savefig("all_plots/periodogram.png")
#     ax.set_title('lnBFmax_peak=%.4f, P=%.2fd' % (lnBFmax,P_best))
#     ax.set_xscale('log')
#     plt.show()
    
# def periodogram_plot1(ymax,P_sam,y,y_trend,P_ref,P_best,addre,host_name,trend_order,win,window_mode,ar_order):
#     plt.figure()
#     plt.axvline(P_best, alpha=0.3, lw=2.5)
#     if P_ref:
#         plt.axvline(P_ref, alpha=0.3, lw=2.5, c='orange')
#     for n in range(2, 20): 
#         plt.axvline(n*P_best, alpha=0.3, lw=1, linestyle="dashed")
#         plt.axvline(P_best/n, alpha=0.3, lw=1, linestyle="dashed")
#     plt.plot(P_sam, y, c='k', lw=0.5)
#     plt.plot(P_sam, y_trend, c='r') 
#     plt.xlabel('Period [days]')
#     plt.ylabel('lnBF')   
#     plt.xlim(np.min(P_sam), np.max(P_sam))
#     plt.title('TBFP, lnBF_peak=%.4f, P=%.2fd' % (ymax,P_best))
#     plt.xscale('log') 
#     plt.savefig(f"{addre}/TBFP_periodograms/dmax1.1_Kepler-{host_name}_TBFP_trend{trend_order}_win{win}_window_mode_{window_mode}_ar{ar_order}_periodogram1.png",bbox_inches='tight')
#     plt.show()

# 画SR的
def periodogram_plot0(P_sam,y,P_ref,P_best,addre,host_name,trend_order,win,window_mode,ar_order):
    plt.figure()
    plt.axvline(P_best, alpha=0.3, lw=2.5)
    if P_ref:
        plt.axvline(P_ref, alpha=0.3, lw=2.5, c='orange')
    for n in range(2, 20):  
        plt.axvline(n*P_best, alpha=0.3, lw=1, linestyle="dashed")
        plt.axvline(P_best/n, alpha=0.3, lw=1, linestyle="dashed")
    plt.plot(P_sam, y, c='k', lw=0.5)
    plt.xlabel('Period [days]')
    plt.ylabel('SR')   
    plt.xlim(np.min(P_sam), np.max(P_sam))  
    plt.title('TBFP')
    plt.xscale('log') 
    plt.savefig(f"{addre}/TBFP_periodograms/dmax2_KIC-{host_name}_TBFP_trend{trend_order}_win{win}_window_mode_{window_mode}_ar{ar_order}_periodogram_SR.png",bbox_inches='tight')
    plt.show()
# 画SDE_raw
def periodogram_plot1(SDE_raw,P_sam,y,P_ref,P_best,addre,host_name,trend_order,win,window_mode,ar_order):
    plt.figure()
    plt.axvline(P_best, alpha=0.3, lw=2.5)
    if P_ref:
        plt.axvline(P_ref, alpha=0.3, lw=2.5, c='orange')
    for n in range(2, 20):  
        plt.axvline(n*P_best, alpha=0.3, lw=1, linestyle="dashed")
        plt.axvline(P_best/n, alpha=0.3, lw=1, linestyle="dashed")
    plt.plot(P_sam, y, c='k', lw=0.5)
    plt.xlabel('Period [days]')
    plt.ylabel('SDE_raw')   
    plt.xlim(np.min(P_sam), np.max(P_sam))  
    plt.title('TBFP, SDE_raw=%.4f, P=%.2fd' % (SDE_raw,P_best))
    plt.xscale('log') 
    plt.savefig(f"{addre}/TBFP_periodograms/dmax2_KIC-{host_name}_TBFP_trend{trend_order}_win{win}_window_mode_{window_mode}_ar{ar_order}_periodogram_SDE_raw.png",bbox_inches='tight')
    plt.show()
# 画SDE的
def periodogram_plot2(SDE,P_sam,y,P_ref,P_best,addre,host_name,trend_order,win,window_mode,ar_order):
    plt.figure()
    plt.axvline(P_best, alpha=0.3, lw=2.5)
    if P_ref:
        plt.axvline(P_ref, alpha=0.3, lw=2.5, c='orange')
    for n in range(2, 20):  
        plt.axvline(n*P_best, alpha=0.3, lw=1, linestyle="dashed")
        plt.axvline(P_best/n, alpha=0.3, lw=1, linestyle="dashed")
    plt.plot(P_sam, y, c='k', lw=0.5)
    plt.xlabel('Period [days]')
    plt.ylabel('SDE')   
    plt.xlim(np.min(P_sam), np.max(P_sam))  
    plt.title('TBFP, SDE=%.4f, P=%.2fd' % (SDE,P_best))
    plt.xscale('log') 
    plt.savefig(f"{addre}/TBFP_periodograms/dmax2_KIC-{host_name}_TBFP_trend{trend_order}_win{win}_window_mode_{window_mode}_ar{ar_order}_periodogram_SDE.png",bbox_inches='tight')
    plt.show()


def phase_plot1(P,d,tm0,phase_cut_group,f_cut_group,model_folded_f_cut_group,addre,host_name,trend_order,win,window_mode,ar_order):
    # 输入的d是小时
    # 用到的数据：
    # phase_cut_group,f_cut_group,model_folded_f_cut_group
    plt.figure(figsize=(5,0.5*len(phase_cut_group))) 
    for k in range(len(phase_cut_group)):
        plt.scatter(phase_cut_group[k],f_cut_group[k],c='black',s=10,alpha=0.5)
        plt.plot(phase_cut_group[k], model_folded_f_cut_group[k])
    plt.xlabel('Phase')
    plt.ylabel('Flux')   
    plt.title('TBFP, P=%.2fd, d=%.2fh, tm0=%.3fd' % (P,d,tm0))
    plt.ticklabel_format(useOffset=False)
    plt.savefig(f"{addre}/TBFP_phase_plots/dmax2_KIC-{host_name}_TBFP_trend{trend_order}_win{win}_window_mode_{window_mode}_ar{ar_order}_phase1.png",bbox_inches='tight')
    plt.show()

def phase_plot2(P,d,tm0,phase_cut_ori_sort,de_f_cut,model_folded_de_f_cut,addre,host_name,trend_order,win,window_mode,ar_order):
    # 输入的d是小时
    # 用到的数据：
    # phase_cut_ori_sort,de_f_cut,model_folded_de_f_cut
    plt.figure(figsize=(5,2))  
    plt.scatter(phase_cut_ori_sort,de_f_cut,c='black',s=10,alpha=0.5)
    plt.plot(phase_cut_ori_sort,model_folded_de_f_cut,c='r',lw=1.5)
    plt.xlabel('Phase')
    plt.ylabel('Flux without trend')
    plt.ticklabel_format(useOffset=False)
    plt.title('TBFP, P=%.2fd, d=%.2fh, tm0=%.3fd' % (P,d,tm0))
    plt.savefig(f"{addre}/TBFP_phase_plots/dmax2_KIC-{host_name}_TBFP_trend{trend_order}_win{win}_window_mode_{window_mode}_ar{ar_order}_phase2.png",bbox_inches='tight')
    plt.show()

def time_plot(P,d,tm0,trend_order,sig_order,win,window_mode,t,f,t_cut_group,model_f_group,addre,host_name,ar_order):
    # 输入的d是小时
    # 用到的数据：
    # t_cut_group,model_f_group
    plt.figure()  
    plt.scatter(t,f,c='black',s=1,alpha=0.5,label='ar_order=%.0f \ntrend_order=%.0f \nsig_order=%.0f \nwindow=%.1fdmax \nwindow_mode=%s' % (ar_order,trend_order,sig_order,win,window_mode))
    for k in range(len(t_cut_group)):
        plt.plot(t_cut_group[k], model_f_group[k],lw=1.5,c='r') 
    plt.xlim(np.min(t), np.max(t))
    plt.xlabel('Time [days]')
    plt.ylabel('Flux') 
    plt.legend() 
    plt.title('TBFP, P=%.2fd, d=%.2fh, tm0=%.3fd' % (P,d,tm0))
    plt.ticklabel_format(useOffset=False)
    plt.savefig(f"{addre}/TBFP_time_plots/dmax2_KIC-{host_name}_TBFP_trend{trend_order}_win{win}_window_mode_{window_mode}_ar{ar_order}_time.png",bbox_inches='tight')
    plt.show()