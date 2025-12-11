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
dataset_path = "/home/xenoss/dat/multi_height_data_new/final_dataset_multiheight/test/" 

################## NOTE TESTING BENCHMARK MODELS WITH CADENCE = 30S #####################

# checkpoints = ["/home/xenoss/dat/models_benchmarking/cadence_30/DeepVel_I/checkpoints/DeepVel_torch_epoch_98_0.18945.pt","/home/xenoss/dat/models_benchmarking/cadence_30/DeepVel_Bz/tau_1.0/checkpoints/DeepVel_torch_epoch_99_0.19202.pt", "/home/xenoss/dat/models_benchmarking/cadence_30/DeepVel_Bz/tau_0.1/checkpoints/DeepVel_torch_epoch_97_0.10387.pt"]
# taus = ["1.0", "1.0", "1e-1"]
# model_names = ["DeepVel_I", "DeepVel_Bz_tau_1.0", "DeepVel_Bz_tau_1e-1"]
# model_paths = ["/home/xenoss/dat/models_benchmarking/cadence_30/DeepVel_I/", "/home/xenoss/dat/models_benchmarking/cadence_30/DeepVel_Bz/tau_1.0/", "/home/xenoss/dat/models_benchmarking/cadence_30/DeepVel_Bz/tau_0.1/"]
# model_types = [ModelType.DEEPVEL_I, ModelType.DEEPVEL_B, ModelType.DEEPVEL_B]

# for i, ckpt in enumerate(checkpoints): # first I, second Bz tau=1.0, third Bz tau=0.1

#     if i == 0: 
#         train_on_B = False
#     else:
#         train_on_B = True

#     deepvel_net = DeepVel_run(root = main_root, in_channels=2, batch = 64, dataset_path = data_path, network_path = model_paths[i]+"checkpoints/", tau=taus[i], train_on_B=train_on_B, cadence=30)
#     params_model = ckpt

#     B_path = dataset_path + f"inputs/B/tau_{taus[i]}/"
#     vz_path = dataset_path + f"inputs/vz/tau_{taus[i]}/"
#     vel_path = dataset_path + f"labels/tau_{taus[i]}/"
#     I_test_data = None
#     if i == 0:
#         I_path = dataset_path + f"inputs/I/tau_{taus[i]}/"
#         I_test_data = os.listdir(I_path)
#         I_test_data.sort()

#     B_test_data = os.listdir(B_path)
#     vz_test_data = os.listdir(vz_path)
#     vel_test_data = os.listdir(vel_path)
#     B_test_data.sort()
#     vz_test_data.sort()
#     vel_test_data.sort()

#     metrics_rows = []
#     print(f"Evaluating model for tau = {taus[i]} ...")
#     for j in range(len(B_test_data)):
#         I = torch.from_numpy(np.load(I_path + I_test_data[j]).astype(np.float32)) if I_test_data is not None else None
#         B = torch.from_numpy(np.load(B_path + B_test_data[j]).astype(np.float32))
#         vz = torch.from_numpy(np.load(vz_path + vz_test_data[j]).astype(np.float32))
#         ground_truth = torch.from_numpy(np.load(vel_path + vel_test_data[j]).astype(np.float32))

#         if I is not None:
#             prediction = deepvel_net.predict(I,params_model)
#         else:
#             prediction = deepvel_net.predict(B,params_model)

#         divergence_orig = get_divergence(ground_truth[0], ground_truth[1])
#         divergence_pred = get_divergence(prediction[0], prediction[1])
#         vorticity_orig = get_vorticity(ground_truth[0], ground_truth[1])
#         vorticity_pred = get_vorticity(prediction[0], prediction[1])

#         mse_l, rmse_l, pearson0, pearson1, slope_x, slope_y = get_all_metrics(ground_truth, prediction).values()
#         mse_d, rmse_d, pearson_d, slope_d = get_all_metrics(divergence_orig, divergence_pred, is_divergence=True).values()
#         mse_v, rmse_v, pearson_v, slope_v = get_all_metrics(vorticity_orig, vorticity_pred, is_divergence=True).values()

#         row = {
#             'tau': taus[i],
#             'sample_index': j,
#             'mse_vel': mse_l,
#             'rmse_vel': rmse_l,
#             'pearson_vel_x': pearson0,
#             'pearson_vel_y': pearson1,
#             'slope_vel_x': slope_x,
#             'slope_vel_y': slope_y,
#             'mse_div': mse_d,
#             'rmse_div': rmse_d,
#             'pearson_div': pearson_d,
#             'slope_div': slope_d,
#             'mse_vort': mse_v,
#             'rmse_vort': rmse_v,
#             'pearson_vort': pearson_v,
#             'slope_vort': slope_v
#         }
#         metrics_rows.append(row)

#     csv_dir = os.path.join(main_root, 'models_benchmarking/cadence_30/test_metrics/')

#     df_tau = pd.DataFrame(metrics_rows)
#     avg_series = df_tau.select_dtypes(include=[np.number]).mean()
#     avg_row = avg_series.to_dict()
#     avg_row['tau'] = taus[i]
#     df_tau = pd.concat([df_tau, pd.DataFrame([avg_row])], ignore_index=True)

#     if i == 0:
#         csv_path_tau = os.path.join(csv_dir, f'I_metrics_tau_{taus[i]}.csv')
#     else:
#         csv_path_tau = os.path.join(csv_dir, f'Bz_metrics_tau_{taus[i]}.csv')
#     df_tau.to_csv(csv_path_tau, index=False)
    

#############################################################################

########## NOTE TESTING BENCHMARK MODELS ON FULL TESTSET ################
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

    B_path = dataset_path + f"inputs/B/tau_{taus[i]}/"
    vz_path = dataset_path + f"inputs/vz/tau_{taus[i]}/"
    vel_path = dataset_path + f"labels/tau_{taus[i]}/"
    I_test_data = None
    if i == 0:
        I_path = dataset_path + f"inputs/I/tau_{taus[i]}/"
        I_test_data = os.listdir(I_path)
        I_test_data.sort()

    B_test_data = os.listdir(B_path)
    vz_test_data = os.listdir(vz_path)
    vel_test_data = os.listdir(vel_path)
    B_test_data.sort()
    vz_test_data.sort()
    vel_test_data.sort()

    metrics_rows = []
    print(f"Evaluating model for tau = {taus[i]} ...")
    for j in range(len(B_test_data)):
        I = torch.from_numpy(np.load(I_path + I_test_data[j]).astype(np.float32)) if I_test_data is not None else None
        B = torch.from_numpy(np.load(B_path + B_test_data[j]).astype(np.float32))
        vz = torch.from_numpy(np.load(vz_path + vz_test_data[j]).astype(np.float32))
        ground_truth = torch.from_numpy(np.load(vel_path + vel_test_data[j]).astype(np.float32))

        if I is not None:
            prediction = deepvel_net.predict(I,params_model)
        else:
            prediction = deepvel_net.predict(B,params_model)

        divergence_orig = get_divergence(ground_truth[0], ground_truth[1])
        divergence_pred = get_divergence(prediction[0], prediction[1])
        vorticity_orig = get_vorticity(ground_truth[0], ground_truth[1])
        vorticity_pred = get_vorticity(prediction[0], prediction[1])

        mse_l, rmse_l, pearson0, pearson1, slope_x, slope_y = get_all_metrics(ground_truth, prediction).values()
        mse_d, rmse_d, pearson_d, slope_d = get_all_metrics(divergence_orig, divergence_pred, is_divergence=True).values()
        mse_v, rmse_v, pearson_v, slope_v = get_all_metrics(vorticity_orig, vorticity_pred, is_divergence=True).values()

        row = {
            'tau': taus[i],
            'sample_index': j,
            'mse_vel': mse_l,
            'rmse_vel': rmse_l,
            'pearson_vel_x': pearson0,
            'pearson_vel_y': pearson1,
            'slope_vel_x': slope_x,
            'slope_vel_y': slope_y,
            'mse_div': mse_d,
            'rmse_div': rmse_d,
            'pearson_div': pearson_d,
            'slope_div': slope_d,
            'mse_vort': mse_v,
            'rmse_vort': rmse_v,
            'pearson_vort': pearson_v,
            'slope_vort': slope_v
        }
        metrics_rows.append(row)

    csv_dir = os.path.join(main_root, 'models_benchmarking/test_metrics/')

    df_tau = pd.DataFrame(metrics_rows)
    avg_series = df_tau.select_dtypes(include=[np.number]).mean()
    avg_row = avg_series.to_dict()
    avg_row['tau'] = taus[i]
    avg_row['sample_index'] = 'AVERAGE'
    df_tau = pd.concat([df_tau, pd.DataFrame([avg_row])], ignore_index=True)

    if i == 0:
        csv_path_tau = os.path.join(csv_dir, f'I_metrics_tau_{taus[i]}.csv')
    else:
        csv_path_tau = os.path.join(csv_dir, f'Bz_metrics_tau_{taus[i]}.csv')
    df_tau.to_csv(csv_path_tau, index=False)
    
   

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

