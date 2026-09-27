from analysis_fn import denormalize_data_vectors
from metrics_and_plotting import ModelType
import numpy as np
import matplotlib.pyplot as plt
import glob
plt.rcParams.update({'font.size': 12})
from sklearn.metrics import root_mean_squared_error
import matplotlib

def power_spectra_plots(vel_gt, vel_pred, dx=0.016, filename=None, title=None,
                        starting_point=3):
    """
    Generate power spectra plots for ground truth and predicted velocities.

    Parameters:
    vel_gt (numpy.ndarray): Ground truth velocities with shape (2, H, W).
    vel_pred (numpy.ndarray): Predicted velocities with shape (2, H, W).
    dx (float): Spatial resolution of the velocity fields (Mm).
    filename (str): Name for the saved file.
    title (str): Name for the plot title.
    starting_point (int): Index to start plotting the power spectrum.

    """
    indx_pred=[1,0]
    plt.rcParams['font.size'] = 14

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12,6))
    for j, comp in enumerate([r'$v_x$', r'$v_y$']):
        Nx = vel_gt.shape[1]
        Ny = vel_gt.shape[2]
        f_gt = np.fft.fftshift(np.fft.fft2(vel_gt[j])) * dx**2
        f_pred = np.fft.fftshift(np.fft.fft2(vel_pred[indx_pred[j]])) * dx**2
        psd_gt = np.abs(f_gt)**2 / (Nx * Ny * (dx**2))
        psd_pred = np.abs(f_pred)**2 / (Nx * Ny * (dx**2))

        kx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
        ky = np.fft.fftshift(np.fft.fftfreq(Ny, d=dx))
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)  # Radial frequency grid

        k_max = min(np.max(kx), np.max(ky))
        dk = 1.0 / (max(Nx, Ny) * dx) # Frequency resolution bin
        nbins = int(k_max / dk)

        k_1d = np.linspace(dk, k_max, nbins)
        psd_gt_1d = np.zeros(nbins)
        psd_pred_1d = np.zeros(nbins)

        for i in range(nbins):
            k_bin_min = i * dk
            k_bin_max = (i + 1) * dk
            mask = (K >= k_bin_min) & (K < k_bin_max)
            if np.any(mask):
                psd_gt_1d[i] = np.mean(psd_gt[mask])
                psd_pred_1d[i] = np.mean(psd_pred[mask])

        start = max(0, min(starting_point, len(k_1d)))
        ax[j].loglog(k_1d[start:], psd_gt_1d[start:], color='crimson', lw=2, label='Ground Truth')
        ax[j].loglog(k_1d[start:], psd_pred_1d[start:], color='blue', lw=2, label='Predicted')
        ax[j].set_title("1D Radially Averaged Power Spectrum - {}".format(comp))
        ax[j].set_xlabel("Spatial Frequency $k$ [1/Mm]")
        if j == 0: 
            ax[j].set_ylabel("Power Density $P(k)$")
        # ax[j].grid(True, which="both", ls="--", alpha=0.5)
        ax[j].legend()
    
    fig.tight_layout()
    plt.suptitle(title, y=1.05)
    plt.savefig("PSD_"+filename+".png", dpi=300)
    plt.cla()
    plt.close()

def power_spectra_plots_both_simulations(vel_gt_m, vel_pred_m, vel_gt_c, vel_pred_c, dx=0.016, filename=None, title=None,
                        starting_point=3):
    """
    Generate power spectra plots for ground truth and predicted velocities.

    Parameters:
    vel_gt_m (numpy.ndarray): Ground truth velocities from MURaM with shape (2, H, W).
    vel_pred_m (numpy.ndarray): Predicted velocities from MURaM with shape (2, H, W).
    vel_gt_c (numpy.ndarray): Ground truth velocities from CO5BOLD with shape (2, H, W).
    vel_pred_c (numpy.ndarray): Predicted velocities from CO5BOLD with shape (2, H, W).
    dx (float): Spatial resolution of the velocity fields (Mm).
    filename (str): Name for the saved file.
    title (str): Name for the plot title.
    starting_point (int): Index to start plotting the power spectrum.
    """
    
    plt.rcParams['font.size'] = 12

    vel_gt = [vel_gt_m, vel_gt_c]
    vel_pred = [vel_pred_m, vel_pred_c]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10,5))

    for i, sim in enumerate(['MURaM', 'CO5BOLD']): 
        if sim == 'MURaM':
            Nx = vel_gt_m.shape[1]
            Ny = vel_gt_m.shape[2]
        else:
            Nx = vel_gt_c.shape[1]
            Ny = vel_gt_c.shape[2]
            
        for j, comp in enumerate([r'$v_x$', r'$v_y$']):
            if comp == r'$v_x$':
                line_style = '-'
            else:
                line_style = '--'

            f_gt = np.fft.fftshift(np.fft.fft2(vel_gt[i][j])) * dx**2
            f_pred = np.fft.fftshift(np.fft.fft2(vel_pred[i][j])) * dx**2
            psd_gt = np.abs(f_gt)**2 / (Nx * Ny * (dx**2))
            psd_pred = np.abs(f_pred)**2 / (Nx * Ny * (dx**2))

            kx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
            ky = np.fft.fftshift(np.fft.fftfreq(Ny, d=dx))
            KX, KY = np.meshgrid(kx, ky)
            K = np.sqrt(KX**2 + KY**2)  # Radial frequency grid

            k_max = min(np.max(kx), np.max(ky))
            dk = 1.0 / (max(Nx, Ny) * dx) # Frequency resolution bin
            nbins = int(k_max / dk)

            k_1d = np.linspace(dk, k_max, nbins)
            psd_gt_1d = np.zeros(nbins)
            psd_pred_1d = np.zeros(nbins)

            for k in range(nbins):
                k_bin_min = k * dk
                k_bin_max = (k + 1) * dk
                mask = (K >= k_bin_min) & (K < k_bin_max)
                if np.any(mask):
                    psd_gt_1d[k] = np.mean(psd_gt[mask])
                    psd_pred_1d[k] = np.mean(psd_pred[mask])

            start = max(0, min(starting_point, len(k_1d)))
            ax[i].loglog(k_1d[start:], psd_gt_1d[start:], color='crimson', lw=2, label=f'Ground Truth {comp}', linestyle=line_style)
            ax[i].loglog(k_1d[start:], psd_pred_1d[start:], color='blue', lw=2, label=f'Predicted {comp}', linestyle=line_style)
            ax[i].set_title("{} test".format(sim))
            ax[i].set_xlabel("Spatial Frequency $k$ [1/Mm]")
            if j == 0: 
                ax[j].set_ylabel("Power Density $P(k)$")
            ax[i].legend()
    
    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig("PSD_"+filename+"both_simulations.png", dpi=300)
    plt.cla()
    plt.close()

def comparison_plot(vel_gt, vel_pred, filename, title, pix_scale=0.016):
    plt.rcParams['font.size'] = 17

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 0.05, 1, 0.05],
                          wspace=0.25, hspace=0.2)
    ax = np.empty((2, 2), dtype=object)
    ax[0, 0] = fig.add_subplot(gs[0, 0])
    ax[0, 1] = fig.add_subplot(gs[0, 2], sharex=ax[0, 0], sharey=ax[0, 0])
    ax[1, 0] = fig.add_subplot(gs[1, 0], sharex=ax[0, 0], sharey=ax[0, 0])
    ax[1, 1] = fig.add_subplot(gs[1, 2], sharex=ax[0, 0], sharey=ax[0, 0])
    cax01 = fig.add_subplot(gs[0, 3])
    cax11 = fig.add_subplot(gs[1, 3])

    im = ax[0,0].imshow(vel_gt[0], origin="lower", cmap='bwr', vmin=-6, vmax=6)
    ax[0,0].set_title(r'$v_x$ - ground truth')
    im01 = ax[0,1].imshow(vel_gt[1], origin="lower", cmap='bwr', vmin=-6, vmax=6)
    ax[0,1].set_title(r'$v_y$ - ground truth')
    ax[1,0].imshow(vel_pred[1], origin="lower", cmap='bwr', vmin=-6, vmax=6)
    ax[1,0].set_title(r'$v_x$ - predicted')
    im11 = ax[1,1].imshow(vel_pred[0], origin="lower", cmap='bwr', vmin=-6, vmax=6)
    ax[1,1].set_title(r'$v_y$ - predicted')
    # ax[0,0].set_xlabel('x [Mm]')
    ax[0,0].set_ylabel('y [Mm]')
    #ax[0,1].set_xlabel('x (Mm)')
    #ax[0,1].set_ylabel('y (Mm)')
    ax[1,0].set_xlabel('x [Mm]')
    ax[1,0].set_ylabel('y [Mm]')
    ax[1,1].set_xlabel('x [Mm]')
    fig.colorbar(im01, cax=cax01)
    fig.colorbar(im11, cax=cax11)
    #ax[1,1].set_ylabel('y [Mm]')
    for i in range(2):
        for j in range(2):
            ax[i,j].set_xticks(np.arange(0, ax[i,j].get_xlim()[1], 150))   
            ax[i,j].set_yticks(np.arange(0, ax[i,j].get_ylim()[1], 150))  
            ax[i,j].set_xticklabels(np.round(np.arange(0, 5)*(150*pix_scale),1))   
            ax[i,j].set_yticklabels(np.round(np.arange(0, 5)*(150*pix_scale),1))  
                        
    fig.tight_layout()
    plt.suptitle(title, y=1.05)
    plt.savefig("Comparison_full_CO5bolt_" + filename + ".png", dpi=300)
    plt.cla()
   
def correlation_plot(vel_gt, vel_pred, filename, title):
    """
    Generate correlation plots between ground truth and predicted velocities.

    Parameters:
    vel_gt (numpy.ndarray): Ground truth velocities with shape (2, H, W).
    vel_pred (numpy.ndarray): Predicted velocities with shape (2, H, W).
    filename (str): Name for the plot filename.
    title (str): Title for the plot.

    Returns:
    tuple: Root mean squared error of velocity magnitudes, correlation coefficient and slope for Vx, correlation coefficient and slope for Vy.
    """
    matplotlib.use('agg')
    plt.rcParams['font.size'] = 43
    plt.clf()
    left_center = 0.30  
    right_center = 0.72
    figsize=(30, 17)
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=figsize)
    ax[0].scatter(vel_gt[0].flatten(), vel_pred[1].flatten(), marker='.', alpha=0.05)
    z_x = np.polyfit(vel_gt[0].flatten(), vel_pred[1].flatten(), 1)
    p_x = np.poly1d(z_x)
    lims = [np.min([ax[0].get_xlim(), ax[0].get_ylim()]), np.max([ax[0].get_xlim(), ax[0].get_ylim()]),]
    ax[0].plot(lims, lims, 'r-')
    #ax[0].plot(lims, p_x(lims), 'r-')
    ax[0].set_aspect('equal')
    ax[0].set_xlim(lims)
    ax[0].set_ylim(lims)
    p_corr_x = np.round(np.corrcoef(vel_gt[0].flatten(), vel_pred[1].flatten())[0,1],3)
    #ax[0].text(lims[0]-lims[0]//6,lims[0]-lims[0]//16, 'Correlation coef {}'.format(p_corr_x))
    ax[0].set_xlabel('Ground Truth [km/s]')
    ax[0].set_ylabel('Predicted data [km/s]')
    ax[0].set_title(r'$V_x$ - ground truth vs predicted')

    ax[1].scatter(vel_gt[1].flatten(), vel_pred[0].flatten(), marker='.', alpha=0.05)
    z_y = np.polyfit(vel_gt[1].flatten(), vel_pred[0].flatten(), 1)
    p_y = np.poly1d(z_y)
    lims = [np.min([ax[1].get_xlim(), ax[1].get_ylim()]), np.max([ax[1].get_xlim(), ax[1].get_ylim()]),]
    ax[1].plot(lims, lims, 'r-')
    #ax[1].plot(lims, p_y(lims), 'r-')
    ax[1].set_aspect('equal')
    ax[1].set_xlim(lims)
    ax[1].set_ylim(lims)
    p_corr_y = np.round(np.corrcoef(vel_gt[1].flatten(), vel_pred[0].flatten())[0,1],3)
    #ax[1].text(lims[0]-lims[0]//6,lims[0]-lims[0]//16, 'Correlation coef {}'.format(p_corr_y))
    ax[1].set_xlabel('Ground Truth [km/s]')
    #ax[1].set_ylabel('Predicted data [km/s]')
    ax[1].set_title(r'$V_y$ - ground truth vs predicted')

    v_mag_gt = np.sqrt(vel_gt[0].flatten()**2+vel_gt[1].flatten()**2)
    v_mag_pred = np.sqrt(vel_pred[0].flatten()**2 + vel_pred[1].flatten()**2)   
    rmse = root_mean_squared_error(v_mag_gt, v_mag_pred)

    fig.text(left_center, 0.7, f'$r_x$ = {p_corr_x:.2f}', ha='center', fontsize=60)
    fig.text(right_center, 0.7, f'$r_y$ = {p_corr_y:.2f}', ha='center', fontsize=60)
    
    fig.tight_layout()
    plt.suptitle(title, y=1.05)
    plt.savefig("Correlation_full_CO5bolt_" + filename + ".png", dpi=300)
    plt.cla()

    return rmse, p_corr_x, z_x[0], p_corr_y, z_y[0]

vel_gt_full_muram = np.load("/home/xenoss/dat/models_multi_height/Hybrid_B_vz/tau_1e-1/test/last_4_gt_velocity_denorm.npy")
vel_pd_full_muram = np.load("/home/xenoss/dat/models_multi_height/Hybrid_B_vz/tau_1e-1/test/last_4_pred_velocity_denorm.npy")

vel_gt_full_muram = vel_gt_full_muram - np.mean(vel_gt_full_muram, axis=(1,2), keepdims=True)
vel_pd_full_muram = vel_pd_full_muram - np.mean(vel_pd_full_muram, axis=(1,2), keepdims=True)

list_files_vel_gt=glob.glob("pred_results_v2/vel_gt_*.npy")
list_files_vel_pd=glob.glob("pred_results_v2/vel_pd_*.npy")
list_files_vel_pdh=glob.glob("pred_results_v2/vel_pdh_*.npy")

vel_gt = {}
vel_gt_full = {}
for i, kw in enumerate(['gt_1_', '1e_1', '1e_2', '1e_3', '1e_4']):
    file = [st for st in list_files_vel_gt if kw in st] # 
    arr = np.load(file[0])
    vel_gt_full[kw] = arr[0]
    arr_t = np.transpose(arr, (1,0,2,3))
    ax1 = arr_t.shape[0]
    ax2 = arr_t.shape[1]*arr_t.shape[2]*arr_t.shape[3]
    vel_gt[kw] = arr_t.reshape(ax1, ax2)

vel_pd = {}
vel_pd_full = {}
for i, kw in enumerate(['I', 'Bz_0_1', 'Bz_1', 'vz_0_1', 'vz_1']):
    file = [st for st in list_files_vel_pd if kw in st]
    arr = np.load(file[0])
    vel_pd_full[kw] = arr[0]
    arr_t = np.transpose(arr, (1,0,2,3))
    ax1 = arr_t.shape[0]
    ax2 = arr_t.shape[1]*arr_t.shape[2]*arr_t.shape[3]
    vel_pd[kw] = arr_t.reshape(ax1, ax2)

vel_pdh = {}
vel_pdh_full = {}
for i, kw in enumerate(['Bz_vz_1e0', 'Bz_vz_1e1', 'Bz_vz_1e2', 'Bz_vz_1e3', 'Bz_vz_1e4', 'I_Bz_1', 'I_vz_1']):
    file = [st for st in list_files_vel_pdh if kw in st]
    arr = np.load(file[0])
    vel_pdh_full[kw] = arr[0]
    arr_t = np.transpose(arr, (1,0,2,3))
    ax1 = arr_t.shape[0]
    ax2 = arr_t.shape[1]*arr_t.shape[2]*arr_t.shape[3]
    vel_pdh[kw] = arr_t.reshape(ax1, ax2)
print('vel_gt_full_muram shape: ', vel_gt_full_muram.shape)
print('vel_gt[1e_1] shape: ', vel_gt['1e_1'].shape)
print('vel_pdh[Bz_vz_1e1] shape: ', vel_pdh['Bz_vz_1e1'].shape)
print('vel_pdh_full[Bz_vz_1e1] shape: ', vel_pdh_full['Bz_vz_1e1'].shape)

# power_spectra_plots_both_simulations(vel_gt_m= vel_gt_full_muram, vel_pred_m=vel_pd_full_muram, 
#                                      vel_gt_c= vel_gt_full['1e_1'] - np.mean(vel_gt_full['1e_1'], axis=(1,2), keepdims=True), vel_pred_c= vel_pdh_full['Bz_vz_1e1'] - np.mean(vel_pdh_full['Bz_vz_1e1'], axis=(1,2), keepdims=True),
#                                      dx=0.016, filename='Hybrid_DeepVel_Bz_vz_norm', 
#                                      title=r'1D Radially Averaged Power Spectrum: Hybrid-input DeepVel ($B_z$ $v_z$), $\tau$=$10^{-1}$', starting_point=1)



# I_o_metrics = correlation_plot(vel_gt['gt_1_'], vel_pd['I'], filename='DeepVel_original_I', title='DeepVel original I')
# Bz_o_1_metrics = correlation_plot(vel_gt['gt_1_'], vel_pd['Bz_1'], filename='DeepVel_original_Bz_tau=1', title=r'DeepVel original $B_z$ $\tau$=1')
# Bz_o_0_1_metrics = correlation_plot(vel_gt['1e_1'], vel_pd['Bz_0_1'], filename='DeepVel_original_Bz_tau=1e-1', title=r'DeepVel original $B_z$ $\tau$=$10^{-1}$')
# vz_o_1_metrics = correlation_plot(vel_gt['gt_1_'], vel_pd['vz_1'], filename='DeepVel_original_vz_tau=1', title=r'DeepVel original $v_z$ $\tau$=1')
# vz_o_0_1_metrics = correlation_plot(vel_gt['1e_1'], vel_pd['vz_0_1'], filename='DeepVel_original_vz_tau=1e-1', title=r'DeepVel original $v_z$ $\tau$=$10^{-1}$')

# Bz_vz_hy_1_metrics = correlation_plot(vel_gt['gt_1_'], vel_pdh['Bz_vz_1e0'], filename='DeepVel_hybrid_Bz_vz_tau=1', title=r'DeepVel hybrid $B_z$ $v_z$ $\tau$=1')
# Bz_vz_hy_1e_1_metrics = correlation_plot(vel_gt['1e_1'], vel_pdh['Bz_vz_1e1'], filename='DeepVel_hybrid_Bz_vz_tau=1e-1', title=r'DeepVel hybrid $B_z$ $v_z$ $\tau$=$10^{-1}$')
# Bz_vz_hy_1e_2_metrics = correlation_plot(vel_gt['1e_2'], vel_pdh['Bz_vz_1e2'], filename='DeepVel_hybrid_Bz_vz_tau=1e-2', title=r'DeepVel hybrid $B_z$ $v_z$ $\tau$=$10^{-2}$')
# Bz_vz_hy_1e_3_metrics = correlation_plot(vel_gt['1e_3'], vel_pdh['Bz_vz_1e3'], filename='DeepVel_hybrid_Bz_vz_tau=1e-3', title=r'DeepVel hybrid $B_z$ $v_z$ $\tau$=$10^{-3}$')
# Bz_vz_hy_1e_4_metrics = correlation_plot(vel_gt['1e_4'], vel_pdh['Bz_vz_1e4'], filename='DeepVel_hybrid_Bz_vz_tau=1e-4', title=r'DeepVel hybrid $B_z$ $v_z$ $\tau$=$10^{-4}$')
# I_vz_hy_1_metrics = correlation_plot(vel_gt['gt_1_'], vel_pdh['I_vz_1'], filename='DeepVel_hybrid_I_vz_tau=1', title=r'DeepVel hybrid $I$ $v_z$ $\tau$=1')
# I_Bz_hy_1_metrics = correlation_plot(vel_gt['gt_1_'], vel_pdh['I_Bz_1'], filename='DeepVel_hybrid_I_Bz_tau=1', title=r'DeepVel hybrid $I$ $B_z$ $\tau$=1')

comparison_plot(vel_gt_full['1e_1'], vel_pd_full['vz_0_1'], filename='DeepVel_original_vz_tau=1e-1', title=r'DeepVel original $v_z$, $\tau$=$10^{-1}$')
#power_spectra_plots(vel_gt_full['gt_1_'], vel_pd_full['I'], dx=0.016, filename='DeepVel_original_I', title='DeepVel original I')
# power_spectra_plots(vel_gt_full['gt_1_'], vel_pd_full['Bz_1'], dx=0.016, filename='DeepVel_original_Bz_tau=1', title='DeepVel original $B_z$ $\tau$=1')
# power_spectra_plots(vel_gt_full['1e_1'], vel_pd_full['Bz_0_1'], dx=0.016, filename='DeepVel_original_Bz_tau=1e-1', title='DeepVel original $B_z$ $\tau$=$10^{-1}$')
# power_spectra_plots(vel_gt_full['gt_1_'], vel_pd_full['vz_1'], dx=0.016, filename='DeepVel_original_vz_tau=1', title='DeepVel original $v_z$ $\tau$=1')
# power_spectra_plots(vel_gt_full['1e_1'], vel_pd_full['vz_0_1'], dx=0.016, filename='DeepVel_original_vz_tau=1e-1', title='DeepVel original $v_z$ $\tau$=$10^{-1}$')

# power_spectra_plots(vel_gt_full['gt_1_'], vel_pdh_full['Bz_vz_1e0'], dx=0.016, filename='DeepVel_hybrid_Bz_vz_tau=1', title='DeepVel hybrid $B_z$ $v_z$ $\tau$=1')
# power_spectra_plots(vel_gt_full['1e_1'], vel_pdh_full['Bz_vz_1e1'], dx=0.016, filename='DeepVel_hybrid_Bz_vz_tau=1e-1', title='DeepVel hybrid $B_z$ $v_z$ $\tau$=$10^{-1}$')
# power_spectra_plots(vel_gt_full['1e_2'], vel_pdh_full['Bz_vz_1e2'], dx=0.016, filename='DeepVel_hybrid_Bz_vz_tau=1e-2', title='DeepVel hybrid $B_z$ $v_z$ $\tau$=$10^{-2}$')
# power_spectra_plots(vel_gt_full['1e_3'], vel_pdh_full['Bz_vz_1e3'], dx=0.016, filename='DeepVel_hybrid_Bz_vz_tau=1e-3', title='DeepVel hybrid $B_z$ $v_z$ $\tau$=$10^{-3}$')
# power_spectra_plots(vel_gt_full['1e_4'], vel_pdh_full['Bz_vz_1e4'], dx=0.016, filename='DeepVel_hybrid_Bz_vz_tau=1e-4', title='DeepVel hybrid $B_z$ $v_z$ $\tau$=$10^{-4}$')
# power_spectra_plots(vel_gt_full['gt_1_'], vel_pdh_full['I_vz_1'], dx=0.016, filename='DeepVel_hybrid_I_vz_tau=1', title='DeepVel hybrid $I$ $v_z$ $\tau$=1')
# power_spectra_plots(vel_gt_full['gt_1_'], vel_pdh_full['I_Bz_1'], dx=0.016, filename='DeepVel_hybrid_I_Bz_tau=1', title='DeepVel hybrid $I$ $B_z$ $\tau$=1')

# print("Benchmark code: RMSE v, r_vx, slope vx, r_vy, slope vy")
# print('DeepVel_original_I', I_o_metrics)
# print('DeepVel_original_Bz_tau=1', Bz_o_1_metrics)
# print('DeepVel_original_Bz_tau=1e-1', Bz_o_0_1_metrics)
# print('DeepVel_original_vz_tau=1', vz_o_1_metrics)
# print('DeepVel_original_vz_tau=1e-1', vz_o_0_1_metrics)
# print('----------------------------------------------------------------------')
# print("Hybrid code: RMSE v, r_vx, slope vx, r_vy, slope vy")
# print('DeepVel_hybrid_Bz_vz_tau=1', Bz_vz_hy_1_metrics)
# print('DeepVel_hybrid_Bz_vz_tau=1e-1', Bz_vz_hy_1e_1_metrics)
# print('DeepVel_hybrid_Bz_vz_tau=1e-2', Bz_vz_hy_1e_2_metrics)
# print('DeepVel_hybrid_Bz_vz_tau=1e-3', Bz_vz_hy_1e_3_metrics)
# print('DeepVel_hybrid_Bz_vz_tau=1e-4', Bz_vz_hy_1e_4_metrics)
# print('DeepVel_hybrid_I_vz_tau=1', I_vz_hy_1_metrics)
# print('DeepVel_hybrid_I_Bz_tau=1', I_Bz_hy_1_metrics)

