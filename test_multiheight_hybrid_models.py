import numpy as np
import torch
import os
from collections import OrderedDict
import csv
from hybrid_deepvel_torch import DeepVel_run as DeepVel_run_hybrid2
import pandas as pd
from analysis_fn import get_all_metrics, get_divergence, get_vorticity
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

main_root = "/home/xenoss/dat/"
data_path = os.path.join(main_root, f'multi_height_data_new/final_dataset_multiheight/train')
dataset_path = "/home/xenoss/dat/multi_height_data_new/final_dataset_multiheight/test/"

tau_levels=[1.0, 1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]
taus = ["1.0", "1e-1", "1e-2","1e-3", "1e-4", "1e-5", "1e-6"]
best_models = OrderedDict()
best_models["1.0"] = "DeepVel_torch_epoch_99_0.06743.pt"
best_models["1e-1"] = "DeepVel_torch_epoch_95_0.03001.pt"
best_models["1e-2"] = "DeepVel_torch_epoch_96_0.04096.pt"
best_models["1e-3"] = "DeepVel_torch_epoch_95_0.11373.pt"
best_models["1e-4"] = "DeepVel_torch_epoch_97_0.18295.pt"
best_models["1e-5"] = "DeepVel_torch_epoch_99_0.23170.pt"
best_models["1e-6"] = "DeepVel_torch_epoch_96_0.26705.pt"

metrics_by_tau = OrderedDict()
csv_dir = os.path.join(main_root, 'models_multi_height/test_metrics/')
os.makedirs(csv_dir, exist_ok=True)

# NOTE THIS IS FOR TESTING ON FULL TEST SETS
for i, tau in enumerate(tau_levels):

    model_path = f"/home/xenoss/dat/models_multi_height/tau_{taus[i]}/checkpoints/"
    params_model = model_path + best_models[taus[i]]

    B_path = dataset_path + f"inputs/B/tau_{taus[i]}/"
    vz_path = dataset_path + f"inputs/vz/tau_{taus[i]}/"
    vel_path = dataset_path + f"labels/tau_{taus[i]}/"
    B_test_data = sorted(os.listdir(B_path))
    vz_test_data = sorted(os.listdir(vz_path))
    vel_test_data = sorted(os.listdir(vel_path))

    deepvel_net = DeepVel_run_hybrid2(root = main_root, in_channels=4, batch = 64, dataset_path = data_path, network_path = model_path, tau=taus[i])

    print(f"Evaluating model for tau = {taus[i]} ...")
    metrics_rows = []
    
    # Lists to aggregate all test samples for the distribution plot
    all_preds_x = []
    all_preds_y = []
    all_gts_x = []
    all_gts_y = []

    for j in range(len(B_test_data)):
        B = torch.from_numpy(np.load(B_path + B_test_data[j]).astype(np.float32))
        vz = torch.from_numpy(np.load(vz_path + vz_test_data[j]).astype(np.float32))
        ground_truth = torch.from_numpy(np.load(vel_path + vel_test_data[j]).astype(np.float32))
        
        prediction = deepvel_net.predict(B, vz, params_model)

        # Collect data for plotting (flattening to 1D)
        all_preds_x.append(prediction[0].numpy().flatten())
        all_preds_y.append(prediction[1].numpy().flatten())
        all_gts_x.append(ground_truth[0].numpy().flatten())
        all_gts_y.append(ground_truth[1].numpy().flatten())

        # Metrics calculation
        divergence_orig = get_divergence(ground_truth[0], ground_truth[1])
        divergence_pred = get_divergence(prediction[0], prediction[1])
        vorticity_orig = get_vorticity(ground_truth[0], ground_truth[1])
        vorticity_pred = get_vorticity(prediction[0], prediction[1])

        mse_l, rmse_l, pearson0, pearson1, slope_x, slope_y = get_all_metrics(ground_truth, prediction).values()
        mse_d, rmse_d, pearson_d, slope_d = get_all_metrics(divergence_orig, divergence_pred, is_divergence=True).values()
        mse_v, rmse_v, pearson_v, slope_v = get_all_metrics(vorticity_orig, vorticity_pred, is_divergence=True).values()

        metrics_rows.append({
            'tau': taus[i], 'sample_index': j, 'mse_vel': mse_l, 'rmse_vel': rmse_l,
            'pearson_vel_x': pearson0, 'pearson_vel_y': pearson1, 'slope_vel_x': slope_x, 'slope_vel_y': slope_y,
            'mse_div': mse_d, 'rmse_div': rmse_d, 'pearson_div': pearson_d, 'slope_div': slope_d,
            'mse_vort': mse_v, 'rmse_vort': rmse_v, 'pearson_vort': pearson_v, 'slope_vort': slope_v
        })

    # --- GENERATE HISTOGRAM & SCATTER PLOT ---
    print(f"Generating distribution plots for tau = {taus[i]}...")
    
    # Concatenate all test samples into single arrays for visualization
    flat_gt_x = np.concatenate(all_gts_x)
    flat_gt_y = np.concatenate(all_gts_y)
    flat_pred_x = np.concatenate(all_preds_x)
    flat_pred_y = np.concatenate(all_preds_y)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex='col', gridspec_kw={'height_ratios': [2, 1]})
    
    # Plotting loop for X and Y components
    plot_data = [('Vx', flat_gt_x, flat_pred_x, 0), ('Vy', flat_gt_y, flat_pred_y, 1)]
    
    for label, gt, pred, col in plot_data:
        # Top Panel: Scatter Plot
        # Using a subset of points (e.g., [::10]) if data is too large for matplotlib to handle quickly
        axes[0, col].scatter(gt[::5], pred[::5], alpha=0.2, s=1, c='teal')
        
        # 1:1 Reference Line
        ax_min = min(gt.min(), pred.min())
        ax_max = max(gt.max(), pred.max())
        axes[0, col].plot([ax_min, ax_max], [ax_min, ax_max], 'r--', lw=1.5, label='1:1 Line')
        
        axes[0, col].set_title(f"Velocity Component: {label} (tau={taus[i]})", fontsize=14)
        axes[0, col].set_ylabel("Predicted", fontsize=12)
        axes[0, col].legend()
        axes[0, col].grid(True, alpha=0.3)

        # Bottom Panel: Histograms
        sns.histplot(gt, color="gray", label="Ground Truth", kde=True, ax=axes[1, col], stat="density", alpha=0.4)
        sns.histplot(pred, color="blue", label="DeepVel Prediction", kde=True, ax=axes[1, col], stat="density", alpha=0.3)
        
        axes[1, col].set_xlabel(f"Velocity Value ({label})", fontsize=12)
        axes[1, col].set_ylabel("Density", fontsize=12)
        axes[1, col].legend()

    plt.tight_layout()
    plot_path = os.path.join(csv_dir, f'distribution_tau_{taus[i]}.png')
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # --- SAVE CSV METRICS ---
    df_tau = pd.DataFrame(metrics_rows)
    avg_series = df_tau.select_dtypes(include=[np.number]).mean()
    avg_row = avg_series.to_dict()
    avg_row['tau'], avg_row['sample_index'] = taus[i], 'AVERAGE'
    df_tau = pd.concat([df_tau, pd.DataFrame([avg_row])], ignore_index=True)
    df_tau.to_csv(os.path.join(csv_dir, f'metrics_tau_{taus[i]}.csv'), index=False)