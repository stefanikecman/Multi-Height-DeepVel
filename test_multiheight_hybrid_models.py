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
from hybrid_deepvel_torch_v2 import DeepVel_run as DeepVel_run_hybrid2
import pandas as pd
import seaborn as sb
from scipy.stats import norm
from analysis_fn import get_all_metrics, get_divergence, get_vorticity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

main_root = "/home/xenoss/dat/"
data_path = os.path.join(main_root, f'multi_height_data_new/final_dataset_multiheight/train')

metrics_by_tau = OrderedDict()
dataset_path = "/home/xenoss/dat/multi_height_data_new/final_dataset_multiheight/test/last_4_layers/"

for i, tau in enumerate(tau_levels):

    model_path = f"/home/xenoss/dat/models_multi_height/tau_{taus[i]}/checkpoints/"
    test_save_path = f"/home/xenoss/dat/models_multi_height/tau_{taus[i]}/test/"
    
    deepvel_net = DeepVel_run_hybrid2(root = main_root, in_channels=4, batch = 64, dataset_path = data_path, network_path = model_path, tau=taus[i])
    params_model = model_path + best_models[taus[i]]
    print(f"Evaluating model for tau = {taus[i]} ...")
    metrics = plot_prediction_and_scatter_full_map_vertical(
        deepvel_object=deepvel_net,
        params_model=params_model,
        dataset_path=dataset_path,
        test_save_path=test_save_path,
        plot_intensity=False,
        n_input_channels=4,
        return_metrics=True,
        hybrid2=True,
        title="Test for tau = "+str(taus[i]),
        tau=taus[i],
        name = "hybrid2_tau_"+str(taus[i])
    )
    print(f"Metrics for tau = {taus[i]}: {metrics}")

    metrics_by_tau[taus[i]] = metrics
    

output_dir = os.path.join(main_root, 'models_multi_height')
os.makedirs(output_dir, exist_ok=True)
df_metrics = pd.DataFrame.from_dict(metrics_by_tau, orient='index')
csv_path = os.path.join(output_dir, 'metrics_by_tau.csv')
df_metrics.to_csv(csv_path)
print(f"Saved metrics summary to {csv_path}")
