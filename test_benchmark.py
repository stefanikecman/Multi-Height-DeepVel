import numpy as np
import torch
import torch.nn as nn
import h5py
import os
import random
import torch.optim as optim
import sys
from collections import OrderedDict
from datetime import datetime
import matplotlib.pyplot as plt
import time
import matplotlib
import csv
#from prepare_data import normalize_layerwise
from metrics_and_plotting import plot_predictions, plot_test_full_map, plot_scatter_plot, calculate_correlation, plot_prediction_and_scatter_full_map_vertical
from deepvel_torch_benchmarking import DeepVel_run
import pandas as pd
import seaborn as sb
from scipy.stats import norm
from analysis_fn import get_all_metrics, get_divergence, get_vorticity
from metrics_and_plotting import ModelType

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

main_root = "/home/xenoss/dat/"
data_path = os.path.join(main_root, f'multi_height_data_new/final_dataset_multiheight/train')

metrics_by_tau = OrderedDict()
dataset_path = "/home/xenoss/dat/multi_height_data_new/final_dataset_multiheight/test/last_4_layers/" 
checkpoints = ["/home/xenoss/dat/models_benchmarking/DeepVel_I/checkpoints/DeepVel_torch_epoch_99_0.14751.pt","/home/xenoss/dat/models_benchmarking/DeepVel_Bz/tau_1.0/checkpoints/DeepVel_torch_epoch_93_0.11650.pt", "/home/xenoss/dat/models_benchmarking/DeepVel_Bz/tau_0.1/checkpoints/DeepVel_torch_epoch_99_0.05246.pt"]
taus = ["1.0", "1.0", "1e-1"]
model_names = ["DeepVel_I", "DeepVel_Bz_tau_1.0", "DeepVel_Bz_tau_1e-1"]
model_paths = ["/home/xenoss/dat/models_benchmarking/DeepVel_I/", "/home/xenoss/dat/models_benchmarking/DeepVel_Bz/tau_1.0/", "/home/xenoss/dat/models_benchmarking/DeepVel_Bz/tau_0.1/"]
model_types = [ModelType.DEEPVEL_I, ModelType.DEEPVEL_B, ModelType.DEEPVEL_B]

for i, ckpt in enumerate(checkpoints): # first I, second Bz tau=1.0, third Bz tau=0.1

    if i == 0: 
        train_on_B = False
    else:
        train_on_B = True

    deepvel_net = DeepVel_run(root = main_root, in_channels=4, batch = 64, dataset_path = data_path, network_path = model_paths[i]+"checkpoints/", tau=taus[i], train_on_B=train_on_B)
    params_model = ckpt
    
    metrics = plot_prediction_and_scatter_full_map_vertical(
        deepvel_object=deepvel_net,
        params_model=params_model,
        dataset_path=dataset_path,
        test_save_path=model_paths[i]+"test/",
        plot_intensity=False,
        n_input_channels=4,
        return_metrics=True,
        model_type=model_types[i],
        title="Test for tau = "+str(taus[i]),
        tau=taus[i],
        name = "benchmark_tau_"+str(taus[i])
    )
    if i== 0: # DeepVel I
        print(f"Metrics for DeepVel I: {metrics}")
    else:
        print(f"Metrics for DeepVel Bz, tau = {taus[i]}: {metrics}")

    metrics_by_tau[model_names[i]] = metrics
    

output_dir = os.path.join(main_root, 'models_benchmarking')
os.makedirs(output_dir, exist_ok=True)
df_metrics = pd.DataFrame.from_dict(metrics_by_tau, orient='index')
csv_path = os.path.join(output_dir, 'metrics_benchmarking.csv')
df_metrics.to_csv(csv_path)
print(f"Saved metrics summary to {csv_path}")

############### NOTE TESTING DEEPVEL I ON THE PROJECT'S TESTSET ################
# test_project_root = "/home/xenoss/dat/datasets_5x5_experiments/main_dataset_test/test_4x128x128/"
# inputs = os.listdir(os.path.join(test_project_root, "inputs/"))
# inputs.sort()
# outputs = os.listdir(os.path.join(test_project_root, "labels/"))
# outputs.sort()

# deepvel_net = DeepVel_run(root = main_root, in_channels=4, batch = 64, dataset_path = data_path, network_path = "/home/xenoss/dat/models_benchmarking/DeepVel_I/checkpoints/", tau="1.0", train_on_B=False)
# params_model = "/home/xenoss/dat/models_benchmarking/DeepVel_I/checkpoints/DeepVel_torch_epoch_99_0.14751.pt"

# for i in enumerate(inputs):
#     input_file = os.path.join(test_project_root, "inputs/", inputs[i[0]])
#     output_file = os.path.join(test_project_root, "labels/", outputs[i[0]])

#     input_data = np.load(input_file)
#     output_data = np.load(output_file)
#     input_data = torch.from_numpy(input_data.astype(np.float32))
#     output_data = torch.from_numpy(output_data.astype(np.float32))
#     velocity_pred = deepvel_net.predict(input_data, params_model)

#     divergence_orig = get_divergence(output_data[0], output_data[1])
#     divergence_pred = get_divergence(velocity_pred[0], velocity_pred[1])
#     vorticity_orig = get_vorticity(output_data[0], output_data[1])
#     vorticity_pred = get_vorticity(velocity_pred[0], velocity_pred[1])

#     mse_l, rmse_l, pearson0, pearson1, slope_x, slope_y = get_all_metrics(output_data, velocity_pred).values()
#     mse_d, rmse_d, pearson_d, slope_d = get_all_metrics(divergence_orig, divergence_pred, is_divergence=True).values()
#     mse_v, rmse_v, pearson_v, slope_v = get_all_metrics(vorticity_orig, vorticity_pred, is_divergence=True).values()

#     metrics_by_tau["Test_"+str(i)] = OrderedDict({
#         "mse_l": mse_l,
#         "rmse_l": rmse_l,
#         "pearson0": pearson0,
#         "pearson1": pearson1,
#         "slope_x": slope_x,
#         "slope_y": slope_y,
#         "mse_d": mse_d,
#         "rmse_d": rmse_d,
#         "pearson_d": pearson_d,
#         "slope_d": slope_d,
#         "mse_v": mse_v,
#         "rmse_v": rmse_v,
#         "pearson_v": pearson_v,
#         "slope_v": slope_v
#     })
# df_metrics = pd.DataFrame.from_dict(metrics_by_tau, orient='index')
# csv_path = os.path.join('/home/xenoss/dat/models_benchmarking', 'metrics_deepvel_I.csv')
# df_metrics.to_csv(csv_path)
# print(f"Saved metrics summary to {csv_path}")

