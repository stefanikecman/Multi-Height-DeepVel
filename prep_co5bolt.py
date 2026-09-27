import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from astropy.io import fits
from skimage.exposure import match_histograms
import glob
import random
import h5py
import sys
from deepvel_torch_benchmarking import DeepVel_run as DeepVel_run_I
from hybrid_deepvel_torch import DeepVel_run as DeepVel_run_hybrid
import os
plt.rcParams.update({'font.size': 16})

sys.path.append('/dat/milic/2D')
import muram
path_muram = '/dat/milic/2D'

#file structure in co5bolt - each file is a snapshot
# keywords ['bolo', 'bz', 'vx', 'vy', 'vz'] with cubes with [H,W] for bolo and [tau,H,W] for the others

def correlation_plot(vel_gt, vel_pred, name):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12,8))
    ax[0].scatter(vel_gt, vel_pred, marker='.', alpha=0.5)
    z = np.polyfit(vel_gt, vel_pred, 1)
    p = np.poly1d(z)   
    lims = [np.min([ax[0].get_xlim(), ax[0].get_ylim()]), np.max([ax[0].get_xlim(), ax[0].get_ylim()]),]
    ax[0].plot(lims, lims, 'k-')
    ax[0].plot(lims, p(lims), 'r--')
    ax[0].set_aspect('equal')
    ax[0].set_xlim(lims)
    ax[0].set_ylim(lims)
    ax[0].text(lims[0]-lims[0]//6,lims[0]-lims[0]//16, 'Correlation coef {}'.format(np.round(np.corrcoef(vel_gt[0].flatten(), vel_pred[1].flatten())[0,1],3)))
    ax[0].set_xlabel('Ground Truth [km/s]')
    ax[0].set_ylabel('Predicted data [km/s]')
    ax[0].set_title('Vx - ground truth vs predicted')
    ax[1].scatter(vel_gt, vel_pred, marker='.', alpha=0.5)
    z = np.polyfit(vel_gt, vel_pred, 1)
    p = np.poly1d(z)
    lims = [np.min([ax[1].get_xlim(), ax[1].get_ylim()]), np.max([ax[1].get_xlim(), ax[1].get_ylim()]),]
    ax[1].plot(lims, lims, 'k-')
    ax[1].plot(lims, p(lims), 'r--')
    ax[1].set_aspect('equal')
    ax[1].set_xlim(lims)
    ax[1].set_ylim(lims)
    ax[1].text(lims[0]-lims[0]//6,lims[0]-lims[0]//16, 'Correlation coef {}'.format(np.round(np.corrcoef(vel_gt[1].flatten(), vel_pred[0].flatten())[0,1],3)))
    ax[1].set_xlabel('Ground Truth [km/s]')
    ax[1].set_ylabel('Predicted data [km/s]')
    ax[1].set_title('Vy - ground truth vs predicted')
    fig.tight_layout()
    plt.suptitle(name, y=1.05)
    plt.savefig("Correlation_full_CO5bolt_"+name+".png", dpi=300)
    plt.cla()

def comparison_plot(vel_gt, vel_pred, name, pix_scale=0.016):
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10,10))
    ax[0,0].imshow(vel_gt[0]/1e5, origin="lower", cmap='seismic', vmin=-6, vmax=6)
    ax[0,0].set_title('Ground Truth - Vx')
    ax[0,1].imshow(vel_gt[1]/1e5, origin="lower", cmap='seismic', vmin=-6, vmax=6)
    ax[0,1].set_title('Ground Truth - Vy')
    ax[1,0].imshow(vel_pred[1], origin="lower", cmap='seismic', vmin=-6, vmax=6)
    ax[1,0].set_title('Predicted data - Vx')
    ax[1,1].imshow(vel_pred[0], origin="lower", cmap='seismic', vmin=-6, vmax=6)
    ax[1,1].set_title('Predicted data - Vy')
    #Add physical spatial scales (ticks base on pixel scale pix per Mm) in x and y axis
    ax[0,0].set_xlabel('X (Mm)')
    ax[0,0].set_ylabel('Y (Mm)')
    ax[0,1].set_xlabel('X (Mm)')
    ax[0,1].set_ylabel('Y (Mm)')
    ax[1,0].set_xlabel('X (Mm)')
    ax[1,0].set_ylabel('Y (Mm)')
    ax[1,1].set_xlabel('X (Mm)')
    ax[1,1].set_ylabel('Y (Mm)')
    #Set the tick labels based on the pixel scale in Mm pér pix
    for i in range(2):
        for j in range(2):
            ax[i,j].set_xticks(np.arange(0, ax[i,j].get_xlim()[1], 150))   
            ax[i,j].set_yticks(np.arange(0, ax[i,j].get_ylim()[1], 150))  
            ax[i,j].set_xticklabels(np.round(np.arange(0, 5)*(150*pix_scale),1))   
            ax[i,j].set_yticklabels(np.round(np.arange(0, 5)*(150*pix_scale),1))  
                        
    fig.tight_layout()
    plt.suptitle(name, y=1.05)
    plt.savefig("Comparison_full_CO5bolt_"+name+".png", dpi=300)
    plt.cla()

def denorm(vel_field, mean_vel, std_vel):
    return (vel_field*std_vel + mean_vel)/1e5

def trans_quantity(q, pix1, pix2, int_matching = False, img_muram=None):
    q_resized = resize(q, (q.shape[1]*(pix1/pix2), q.shape[0]*(pix1/pix2)))
    if int_matching == True:
        q_resized = match_histograms(q_resized, img_muram)
    q_norm = (q_resized - np.mean(q_resized))/np.std(q_resized)
    return q_norm

def predic_ori_model(I_seq, Bz_1_seq, Bz_0_1_seq, vz_1_seq, vz_0_1_seq, 
                     vx_1_seq, vy_1_seq, mean_vx_1, std_vx_1, mean_vy_1, std_vy_1,
                     vx_0_1_seq, vy_0_1_seq, mean_vx_0_1, std_vx_0_1, mean_vy_0_1, std_vy_0_1, 
                     sec_num, path_co5bolt_2D='/scratch/vigeesh/for_saida_superhighcad/'):
    
    vel_gt_1 = np.stack([ (vx_1_seq[1] + vx_1_seq[2]) / 2, (vy_1_seq[1] + vy_1_seq[2]) / 2 ], axis=0)
    vel_gt_0_1 = np.stack([ (vx_0_1_seq[1] + vx_0_1_seq[2]) / 2, (vy_0_1_seq[1] + vy_0_1_seq[2]) / 2 ], axis=0)

    path=os.getcwd()

    deepvel_I = DeepVel_run_I(batch=64, dataset_path=path_co5bolt_2D, network_path=path, in_shape=(4,960,960), out_shape=(2,960,960), cadence=10)
    cp1 = 'checkpoints/checkpoints_CO5BOLD_test/single-input/DeepVel_I_tau_1.0.pt'
    cp2 = 'checkpoints/checkpoints_CO5BOLD_test/single-input/DeepVel_Bz_tau_1.0.pt'
    cp3 = 'checkpoints/checkpoints_CO5BOLD_test/single-input/DeepVel_Bz_tau_1E-1.pt'
    cp4 = 'checkpoints/checkpoints_CO5BOLD_test/single-input/DeepVel_vz_tau_1.0.pt'
    cp5 = 'checkpoints/checkpoints_CO5BOLD_test/single-input/DeepVel_vz_tau_1E-1.pt'

    vel_pred_I = deepvel_I.predict(I_seq, cp1)
    vel_pred_Bz_1 = deepvel_I.predict(Bz_1_seq, cp2)
    vel_pred_Bz_0_1 = deepvel_I.predict(Bz_0_1_seq, cp3)
    vel_pred_vz_1 = deepvel_I.predict(vz_1_seq, cp4)
    vel_pred_vz_0_1 = deepvel_I.predict(vz_0_1_seq, cp5)

    vel_pred_denormed_I = [denorm(vel_pred_I[0], mean_vx_1, std_vx_1), denorm(vel_pred_I[1], mean_vy_1, std_vy_1)]
    vel_pred_denormed_Bz_1 = [denorm(vel_pred_Bz_1[0], mean_vx_1, std_vx_1), denorm(vel_pred_Bz_1[1], mean_vy_1, std_vy_1)]
    vel_pred_denormed_Bz_0_1 = [denorm(vel_pred_Bz_0_1[0], mean_vx_0_1, std_vx_0_1), denorm(vel_pred_Bz_0_1[1], mean_vy_0_1, std_vy_0_1)]
    vel_pred_denormed_vz_1 = [denorm(vel_pred_vz_1[0], mean_vx_1, std_vx_1), denorm(vel_pred_vz_1[1], mean_vy_1, std_vy_1)]
    vel_pred_denormed_vz_0_1 = [denorm(vel_pred_vz_0_1[0], mean_vx_0_1, std_vx_0_1), denorm(vel_pred_vz_0_1[1], mean_vy_0_1, std_vy_0_1)]

    comparison_plot(vel_gt_1, vel_pred_denormed_I, "Intensity_{}".format(sec_num))
    comparison_plot(vel_gt_1, vel_pred_denormed_Bz_1, "Bz_tau_1_{}".format(sec_num))
    comparison_plot(vel_gt_0_1, vel_pred_denormed_Bz_0_1, "Bz_tau_01_{}".format(sec_num))
    comparison_plot(vel_gt_1, vel_pred_denormed_vz_1, "vz_tau_1_{}".format(sec_num))
    comparison_plot(vel_gt_0_1, vel_pred_denormed_vz_0_1, "vz_tau_01_{}".format(sec_num))

    return vel_gt_1, vel_gt_0_1, vel_pred_denormed_I, vel_pred_denormed_Bz_1, vel_pred_denormed_Bz_0_1, vel_pred_denormed_vz_1, vel_pred_denormed_vz_0_1

def predic_hybrid_model(I_seq, Bz_1_seq, Bz_1e_1_seq, Bz_1e_2_seq, Bz_1e_3_seq, Bz_1e_4_seq,
                        vz_1_seq, vz_1e_1_seq, vz_1e_2_seq, vz_1e_3_seq, vz_1e_4_seq, 
                        vx_1_seq, vy_1_seq, mean_vx_1, std_vx_1, mean_vy_1, std_vy_1,
                        vx_1e_1_seq, vy_1e_1_seq, mean_vx_1e_1, std_vx_1e_1, mean_vy_1e_1, std_vy_1e_1,
                        vx_1e_2_seq, vy_1e_2_seq, mean_vx_1e_2, std_vx_1e_2, mean_vy_1e_2, std_vy_1e_2,
                        vx_1e_3_seq, vy_1e_3_seq, mean_vx_1e_3, std_vx_1e_3, mean_vy_1e_3, std_vy_1e_3,
                        vx_1e_4_seq, vy_1e_4_seq, mean_vx_1e_4, std_vx_1e_4, mean_vy_1e_4, std_vy_1e_4,
                        sec_num, path_co5bolt_2D='/scratch/vigeesh/for_saida_superhighcad/'):

    vel_gt_1 = np.stack([ (vx_1_seq[1] + vx_1_seq[2]) / 2, (vy_1_seq[1] + vy_1_seq[2]) / 2 ], axis=0)
    vel_gt_1e_1 = np.stack([ (vx_1e_1_seq[1] + vx_1e_1_seq[2]) / 2, (vy_1e_1_seq[1] + vy_1e_1_seq[2]) / 2 ], axis=0)
    vel_gt_1e_2 = np.stack([ (vx_1e_2_seq[1] + vx_1e_2_seq[2]) / 2, (vy_1e_2_seq[1] + vy_1e_2_seq[2]) / 2 ], axis=0)    
    vel_gt_1e_3 = np.stack([ (vx_1e_3_seq[1] + vx_1e_3_seq[2]) / 2, (vy_1e_3_seq[1] + vy_1e_3_seq[2]) / 2 ], axis=0)
    vel_gt_1e_4 = np.stack([ (vx_1e_4_seq[1] + vx_1e_4_seq[2]) / 2, (vy_1e_4_seq[1] + vy_1e_4_seq[2]) / 2 ], axis=0)

    path=os.getcwd()    

    deepvel_hybrid = DeepVel_run_hybrid(batch=64, dataset_path=path_co5bolt_2D, network_path=path, in_shape=(4,960,960), out_shape=(2,960,960))
    cp1 = 'checkpoints/checkpoints_CO5BOLD_test/hybrid-input/DeepVel_Bz_vz_tau_1.0.pt'
    cp2 = 'checkpoints/checkpoints_CO5BOLD_test/hybrid-input/DeepVel_Bz_vz_tau_1E-1.pt'
    cp3 = 'checkpoints/checkpoints_CO5BOLD_test/hybrid-input/DeepVel_Bz_vz_tau_1E-2.pt'
    cp4 = 'checkpoints/checkpoints_CO5BOLD_test/hybrid-input/DeepVel_Bz_vz_tau_1E-3.pt'
    cp5 = 'checkpoints/checkpoints_CO5BOLD_test/hybrid-input/DeepVel_Bz_vz_tau_1E-4.pt'
    cp6 = 'checkpoints/checkpoints_CO5BOLD_test/hybrid-input/DeepVel_I_Bz_tau_1.0.pt'
    cp7 = 'checkpoints/checkpoints_CO5BOLD_test/hybrid-input/DeepVel_I_vz_tau_1.0.pt'    
    
    vel_pred_hybrid_Bz_vz_1 = deepvel_hybrid.predict(Bz_1_seq, vz_1_seq, cp1)
    vel_pred_hybrid_Bz_vz_1e1 = deepvel_hybrid.predict(Bz_1e_1_seq, vz_1e_1_seq, cp2)
    vel_pred_hybrid_Bz_vz_1e2 = deepvel_hybrid.predict(Bz_1e_2_seq, vz_1e_2_seq, cp3)
    vel_pred_hybrid_Bz_vz_1e3 = deepvel_hybrid.predict(Bz_1e_3_seq, vz_1e_3_seq, cp4)
    vel_pred_hybrid_Bz_vz_1e4 = deepvel_hybrid.predict(Bz_1e_4_seq, vz_1e_4_seq, cp5)
    vel_pred_hybrid_I_Bz_1= deepvel_hybrid.predict(I_seq, Bz_1_seq, cp6)
    vel_pred_hybrid_I_vz_1= deepvel_hybrid.predict(I_seq, vz_1_seq, cp7)
    
    vel_pred_denormed_hybrid_Bz_vz_1 = [denorm(vel_pred_hybrid_Bz_vz_1[0], mean_vx_1, std_vx_1), denorm(vel_pred_hybrid_Bz_vz_1[1], mean_vy_1, std_vy_1)]
    vel_pred_denormed_hybrid_Bz_vz_1e1 = [denorm(vel_pred_hybrid_Bz_vz_1e1[0], mean_vx_1e_1, std_vx_1e_1), denorm(vel_pred_hybrid_Bz_vz_1e1[1], mean_vy_1e_1, std_vy_1e_1)]
    vel_pred_denormed_hybrid_Bz_vz_1e2 = [denorm(vel_pred_hybrid_Bz_vz_1e2[0], mean_vx_1e_2, std_vx_1e_2), denorm(vel_pred_hybrid_Bz_vz_1e2[1], mean_vy_1e_2, std_vy_1e_2)]
    vel_pred_denormed_hybrid_Bz_vz_1e3 = [denorm(vel_pred_hybrid_Bz_vz_1e3[0], mean_vx_1e_3, std_vx_1e_3), denorm(vel_pred_hybrid_Bz_vz_1e3[1], mean_vy_1e_3, std_vy_1e_3)]
    vel_pred_denormed_hybrid_Bz_vz_1e4 = [denorm(vel_pred_hybrid_Bz_vz_1e4[0], mean_vx_1e_4, std_vx_1e_4), denorm(vel_pred_hybrid_Bz_vz_1e4[1], mean_vy_1e_4, std_vy_1e_4)]

    vel_pred_denormed_hybrid_I_vz_1 = [denorm(vel_pred_hybrid_I_vz_1[0], mean_vx_1, std_vx_1), denorm(vel_pred_hybrid_I_vz_1[1], mean_vy_1, std_vy_1)]
    vel_pred_denormed_hybrid_I_Bz_1 = [denorm(vel_pred_hybrid_I_Bz_1[0], mean_vx_1, std_vx_1), denorm(vel_pred_hybrid_I_Bz_1[1], mean_vy_1, std_vy_1)]

    comparison_plot(vel_gt_1, vel_pred_denormed_hybrid_Bz_vz_1, "Hybrid_Bz_vz_tau_1_{}".format(sec_num))
    comparison_plot(vel_gt_1e_1, vel_pred_denormed_hybrid_Bz_vz_1e1, "Hybrid_Bz_vz_tau_1e-1_{}".format(sec_num))
    comparison_plot(vel_gt_1e_2, vel_pred_denormed_hybrid_Bz_vz_1e2, "Hybrid_Bz_vz_tau_1e-2_{}".format(sec_num))
    comparison_plot(vel_gt_1e_3, vel_pred_denormed_hybrid_Bz_vz_1e3, "Hybrid_Bz_vz_tau_1e-3_{}".format(sec_num))
    comparison_plot(vel_gt_1e_4, vel_pred_denormed_hybrid_Bz_vz_1e4, "Hybrid_Bz_vz_tau_1e-4_{}".format(sec_num))
    comparison_plot(vel_gt_1, vel_pred_denormed_hybrid_I_vz_1, "Hybrid_I_vz_tau_1_{}".format(sec_num))
    comparison_plot(vel_gt_1, vel_pred_denormed_hybrid_I_Bz_1, "Hybrid_I_Bz_tau_1_{}".format(sec_num))

    return vel_gt_1, vel_gt_1e_1, vel_gt_1e_2, vel_gt_1e_3, vel_gt_1e_4, vel_pred_denormed_hybrid_Bz_vz_1, vel_pred_denormed_hybrid_Bz_vz_1e1, vel_pred_denormed_hybrid_Bz_vz_1e2, vel_pred_denormed_hybrid_Bz_vz_1e3, vel_pred_denormed_hybrid_Bz_vz_1e4, vel_pred_denormed_hybrid_I_Bz_1, vel_pred_denormed_hybrid_I_vz_1

    
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

#muram intensity matching
muram_iteration_list = np.linspace(50, 18000, 360).round().astype(int)
indx_muram = np.random.choice(muram_iteration_list)
I_muram = muram.MuramIntensity(path_muram, indx_muram)
I_muram = (I_muram - np.mean(I_muram))/np.std(I_muram)

path_co5bolt_2D = '/scratch/vigeesh/for_saida_superhighcad/'
list_files = glob.glob(path_co5bolt_2D+'t*0s.h5') #Cadences of 10 seconds
#full sequence 850 sec aproximate 14 minutes - prediction every 3.5 minutes and add all together in the metrics to improve statistics

pix_size_CO5BOLT = 0.0138
pix_size_muram = 0.022

#grund truth
vel_gt_1_full = []
vel_gt_0_1_full = []
vel_gt_1e_2_full = [] 
vel_gt_1e_3_full = []
vel_gt_1e_4_full = []

#original model
vel_pd_I_full = []
vel_pd_Bz_1_full = []
vel_pd_Bz_0_1_full = []
vel_pd_vz_1_full = []
vel_pd_vz_0_1_full = []

#hybrid model
vel_pdh_Bz_vz_1_full = []
vel_pdh_Bz_vz_1e1_full = []
vel_pdh_Bz_vz_1e2_full = []
vel_pdh_Bz_vz_1e3_full = []
vel_pdh_Bz_vz_1e4_full = []
vel_pdh_I_Bz_1_full = []
vel_pdh_I_vz_1_full = []

for sec in range(4):
    I_seq = []
    Bz_1_seq = []
    vz_1_seq = []
    vx_1_seq = []
    vy_1_seq = []

    Bz_1e_1_seq = []
    vz_1e_1_seq = []
    vx_1e_1_seq = []
    vy_1e_1_seq = []

    Bz_1e_2_seq = []
    vz_1e_2_seq = []
    vx_1e_2_seq = []
    vy_1e_2_seq = []

    Bz_1e_3_seq = []
    vz_1e_3_seq = []
    vx_1e_3_seq = []
    vy_1e_3_seq = []

    Bz_1e_4_seq = []
    vz_1e_4_seq = []
    vx_1e_4_seq = []
    vy_1e_4_seq = []

    for t in range(4):
        cube = h5py.File(list_files[21*sec+t], 'r')
        In = cube['bolo']
        Bz = cube['bz']
        vz = cube['vz']

        Bz_1 = Bz[0,:,:]
        Bz_1e_1 = Bz[1,:,:]
        Bz_1e_2 = Bz[2,:,:]
        Bz_1e_3 = Bz[3,:,:]
        Bz_1e_4 = Bz[4,:,:]

        vz_1 = vz[0,:,:]
        vz_1e_1 = vz[1,:,:]
        vz_1e_2 = vz[2,:,:]
        vz_1e_3 = vz[3,:,:]
        vz_1e_4 = vz[4,:,:]

        I_seq.append(trans_quantity(In, pix_size_CO5BOLT, pix_size_muram, int_matching = True, img_muram=I_muram))
        Bz_1_seq.append(trans_quantity(Bz_1, pix_size_CO5BOLT, pix_size_muram))
        Bz_1e_1_seq.append(trans_quantity(Bz_1e_1, pix_size_CO5BOLT, pix_size_muram))
        Bz_1e_2_seq.append(trans_quantity(Bz_1e_2, pix_size_CO5BOLT, pix_size_muram))
        Bz_1e_3_seq.append(trans_quantity(Bz_1e_3, pix_size_CO5BOLT, pix_size_muram))
        Bz_1e_4_seq.append(trans_quantity(Bz_1e_4, pix_size_CO5BOLT, pix_size_muram))

        vz_1_seq.append(trans_quantity(vz_1, pix_size_CO5BOLT, pix_size_muram))
        vz_1e_1_seq.append(trans_quantity(vz_1e_1, pix_size_CO5BOLT, pix_size_muram))
        vz_1e_2_seq.append(trans_quantity(vz_1e_2, pix_size_CO5BOLT, pix_size_muram))
        vz_1e_3_seq.append(trans_quantity(vz_1e_3, pix_size_CO5BOLT, pix_size_muram))
        vz_1e_4_seq.append(trans_quantity(vz_1e_4, pix_size_CO5BOLT, pix_size_muram))

        vx_1 = cube['vx'][0]
        vy_1 = cube['vy'][0]
        vx = resize(vx_1, (vx_1.shape[1]*(pix_size_CO5BOLT/pix_size_muram), vx_1.shape[0]*(pix_size_CO5BOLT/pix_size_muram)))
        vy = resize(vy_1, (vy_1.shape[1]*(pix_size_CO5BOLT/pix_size_muram), vy_1.shape[0]*(pix_size_CO5BOLT/pix_size_muram)))
        vx_1_seq.append(vx)
        vy_1_seq.append(vy)

        vx_1e_1 = cube['vx'][1]
        vy_1e_1 = cube['vy'][1]

        vx_1e_2 = cube['vx'][2]
        vy_1e_2 = cube['vy'][2]

        vx_1e_3 = cube['vx'][3]
        vy_1e_3 = cube['vy'][3]

        vx_1e_4 = cube['vx'][4]
        vy_1e_4 = cube['vy'][4]

        vx_1e_1 = resize(vx_1e_1, (vx_1e_1.shape[1]*(pix_size_CO5BOLT/pix_size_muram), vx_1e_1.shape[0]*(pix_size_CO5BOLT/pix_size_muram)))
        vy_1e_1 = resize(vy_1e_1, (vy_1e_1.shape[1]*(pix_size_CO5BOLT/pix_size_muram), vy_1e_1.shape[0]*(pix_size_CO5BOLT/pix_size_muram)))
        vx_1e_1_seq.append(vx_1e_1)
        vy_1e_1_seq.append(vy_1e_1)

        vx_1e_2 = resize(vx_1e_2, (vx_1e_2.shape[1]*(pix_size_CO5BOLT/pix_size_muram), vx_1e_2.shape[0]*(pix_size_CO5BOLT/pix_size_muram)))
        vy_1e_2 = resize(vy_1e_2, (vy_1e_2.shape[1]*(pix_size_CO5BOLT/pix_size_muram), vy_1e_2.shape[0]*(pix_size_CO5BOLT/pix_size_muram)))
        vx_1e_2_seq.append(vx_1e_2)
        vy_1e_2_seq.append(vy_1e_2)

        vx_1e_3 = resize(vx_1e_3, (vx_1e_3.shape[1]*(pix_size_CO5BOLT/pix_size_muram), vx_1e_3.shape[0]*(pix_size_CO5BOLT/pix_size_muram)))
        vy_1e_3 = resize(vy_1e_3, (vy_1e_3.shape[1]*(pix_size_CO5BOLT/pix_size_muram), vy_1e_3.shape[0]*(pix_size_CO5BOLT/pix_size_muram)))
        vx_1e_3_seq.append(vx_1e_3)
        vy_1e_3_seq.append(vy_1e_3)

        vx_1e_4 = resize(vx_1e_4, (vx_1e_4.shape[1]*(pix_size_CO5BOLT/pix_size_muram), vx_1e_4.shape[0]*(pix_size_CO5BOLT/pix_size_muram)))
        vy_1e_4 = resize(vy_1e_4, (vy_1e_4.shape[1]*(pix_size_CO5BOLT/pix_size_muram), vy_1e_4.shape[0]*(pix_size_CO5BOLT/pix_size_muram)))
        vx_1e_4_seq.append(vx_1e_4)
        vy_1e_4_seq.append(vy_1e_4)


    I_seq = np.array(I_seq) 
    Bz_1_seq = np.array(Bz_1_seq)
    Bz_1e_1_seq = np.array(Bz_1e_1_seq)
    Bz_1e_2_seq = np.array(Bz_1e_2_seq)
    Bz_1e_3_seq = np.array(Bz_1e_3_seq)
    Bz_1e_4_seq = np.array(Bz_1e_4_seq)

    vz_1_seq = np.array(vz_1_seq)
    vz_1e_1_seq = np.array(vz_1e_1_seq)
    vz_1e_2_seq = np.array(vz_1e_2_seq)
    vz_1e_3_seq = np.array(vz_1e_3_seq)
    vz_1e_4_seq = np.array(vz_1e_4_seq)

    vx_1_seq = np.array(vx_1_seq)
    vy_1_seq = np.array(vy_1_seq)

    vx_1e_1_seq = np.array(vx_1e_1_seq)
    vy_1e_1_seq = np.array(vy_1e_1_seq)
    vx_1e_2_seq = np.array(vx_1e_2_seq)
    vy_1e_2_seq = np.array(vy_1e_2_seq)
    vx_1e_3_seq = np.array(vx_1e_3_seq)
    vy_1e_3_seq = np.array(vy_1e_3_seq)
    vx_1e_4_seq = np.array(vx_1e_4_seq)
    vy_1e_4_seq = np.array(vy_1e_4_seq)

    mean_vx_1 = np.mean(vx_1_seq)
    std_vx_1 = np.std(vx_1_seq)
    mean_vy_1 = np.mean(vy_1_seq)
    std_vy_1 = np.std(vy_1_seq)

    mean_vx_1e_1 = np.mean(vx_1e_1_seq)
    std_vx_1e_1 = np.std(vx_1e_1_seq)
    mean_vy_1e_1 = np.mean(vy_1e_1_seq)
    std_vy_1e_1 = np.std(vy_1e_1_seq)

    mean_vx_1e_2 = np.mean(vx_1e_2_seq)
    std_vx_1e_2 = np.std(vx_1e_2_seq)
    mean_vy_1e_2 = np.mean(vy_1e_2_seq)
    std_vy_1e_2 = np.std(vy_1e_2_seq)

    mean_vx_1e_3 = np.mean(vx_1e_3_seq)
    std_vx_1e_3 = np.std(vx_1e_3_seq)
    mean_vy_1e_3 = np.mean(vy_1e_3_seq)
    std_vy_1e_3 = np.std(vy_1e_3_seq)

    mean_vx_1e_4 = np.mean(vx_1e_4_seq)
    std_vx_1e_4 = np.std(vx_1e_4_seq)
    mean_vy_1e_4 = np.mean(vy_1e_4_seq)
    std_vy_1e_4 = np.std(vy_1e_4_seq)

    vel_gt_1, vel_gt_0_1, vel_pd_I, vel_pd_Bz_1, vel_pd_Bz_0_1, vel_pd_vz_1, vel_pd_vz_0_1 = predic_ori_model(I_seq, Bz_1_seq, Bz_1e_1_seq, vz_1_seq, 
                                                                                                              vz_1e_1_seq, vx_1_seq, vy_1_seq, mean_vx_1, 
                                                                                                              std_vx_1, mean_vy_1, std_vy_1, vx_1e_1_seq, 
                                                                                                              vy_1e_1_seq, mean_vx_1e_1, std_vx_1e_1, 
                                                                                                              mean_vy_1e_1, std_vy_1e_1, sec)

    

    _, _, vel_gt_1e_2, vel_gt_1e_3, vel_gt_1e_4, vel_pdh_Bz_vz_1, vel_pdh_Bz_vz_1e1, vel_pdh_Bz_vz_1e2, vel_pdh_Bz_vz_1e3, vel_pdh_Bz_vz_1e4, vel_pdh_I_Bz_1, vel_pdh_I_vz_1 = predic_hybrid_model(I_seq, Bz_1_seq, Bz_1e_1_seq, Bz_1e_2_seq, 
                                                                                                                                                                                                    Bz_1e_3_seq, Bz_1e_4_seq, vz_1_seq, vz_1e_1_seq, 
                                                                                                                                                                                                    vz_1e_2_seq, vz_1e_3_seq, vz_1e_4_seq,
                                                                                                                                                                                                    vx_1_seq, vy_1_seq, mean_vx_1, std_vx_1, 
                                                                                                                                                                                                    mean_vy_1, std_vy_1, vx_1e_1_seq, vy_1e_1_seq, 
                                                                                                                                                                                                    mean_vx_1e_1, std_vx_1e_1, mean_vy_1e_1, std_vy_1e_1,
                                                                                                                                                                                                    vx_1e_2_seq, vy_1e_2_seq, mean_vx_1e_2, std_vx_1e_2, 
                                                                                                                                                                                                    mean_vy_1e_2, std_vy_1e_2, vx_1e_3_seq, vy_1e_3_seq, 
                                                                                                                                                                                                    mean_vx_1e_3, std_vx_1e_3, mean_vy_1e_3, std_vy_1e_3,
                                                                                                                                                                                                    vx_1e_4_seq, vy_1e_4_seq, mean_vx_1e_4, std_vx_1e_4, 
                                                                                                                                                                                                    mean_vy_1e_4, std_vy_1e_4, sec)

    vel_gt_1_full.append(vel_gt_1/1e5)
    vel_gt_0_1_full.append(vel_gt_0_1/1e5)
    vel_gt_1e_2_full.append(vel_gt_1e_2/1e5)
    vel_gt_1e_3_full.append(vel_gt_1e_3/1e5)
    vel_gt_1e_4_full.append(vel_gt_1e_4/1e5)

    vel_pd_I_full.append(vel_pd_I)
    vel_pd_Bz_1_full.append(vel_pd_Bz_1)
    vel_pd_Bz_0_1_full.append(vel_pd_Bz_0_1)
    vel_pd_vz_1_full.append(vel_pd_vz_1)
    vel_pd_vz_0_1_full.append(vel_pd_vz_0_1)

    vel_pdh_Bz_vz_1_full.append(vel_pdh_Bz_vz_1)
    vel_pdh_Bz_vz_1e1_full.append(vel_pdh_Bz_vz_1e1)
    vel_pdh_Bz_vz_1e2_full.append(vel_pdh_Bz_vz_1e2)
    vel_pdh_Bz_vz_1e3_full.append(vel_pdh_Bz_vz_1e3)
    vel_pdh_Bz_vz_1e4_full.append(vel_pdh_Bz_vz_1e4)
    vel_pdh_I_Bz_1_full.append(vel_pdh_I_Bz_1)
    vel_pdh_I_vz_1_full.append(vel_pdh_I_vz_1)


np.save("vel_gt_1_full.npy", np.array(vel_gt_1_full))
np.save("vel_gt_0_1_full.npy", np.array(vel_gt_0_1_full))
np.save("vel_gt_1e_2_full.npy", np.array(vel_gt_1e_2_full))
np.save("vel_gt_1e_3_full", np.array(vel_gt_1e_3_full))
np.save("vel_gt_1e_4_full", np.array(vel_gt_1e_4_full))

np.save("vel_pd_I_full.npy", np.array(vel_pd_I_full))
np.save("vel_pd_Bz_1_full.npy", np.array(vel_pd_Bz_1_full))
np.save("vel_pd_Bz_0_1_full.npy", np.array(vel_pd_Bz_0_1_full))
np.save("vel_pd_vz_1_full.npy", np.array(vel_pd_vz_1_full))
np.save("vel_pd_vz_0_1_full.npy", np.array(vel_pd_vz_0_1_full))

np.save("vel_pdh_Bz_vz_1_full.npy", np.array(vel_pdh_Bz_vz_1_full))
np.save("vel_pdh_Bz_vz_1e1_full.npy", np.array(vel_pdh_Bz_vz_1e1_full))
np.save("vel_pdh_Bz_vz_1e2_full.npy", np.array(vel_pdh_Bz_vz_1e2_full))
np.save("vel_pdh_Bz_vz_1e3_full.npy", np.array(vel_pdh_Bz_vz_1e3_full))
np.save("vel_pdh_Bz_vz_1e4_full.npy", np.array(vel_pdh_Bz_vz_1e4_full))
np.save("vel_pdh_I_Bz_1_full.npy", np.array(vel_pdh_I_Bz_1_full))
np.save("vel_pdh_I_vz_1_full.npy", np.array(vel_pdh_I_vz_1_full))