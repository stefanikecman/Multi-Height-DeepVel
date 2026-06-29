import numpy as np
import torch
import os
from collections import OrderedDict
import csv
from hybrid_deepvel_torch import DeepVel_run 
import pandas as pd
from analysis_fn import denormalize_data_vectors, get_all_metrics, get_divergence, get_vorticity
from metrics_and_plotting import ModelType, plot_prediction_and_scatter_full_map_vertical, plot_train_val_losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# tau_levels=[1.0, 1e-1,1e-2,1e-3,1e-4]
# taus = ["1.0", "1e-1", "1e-2","1e-3", "1e-4"]
# model_types = [ModelType.HYBRID_Bvz, ModelType.HYBRID_Bvz, ModelType.HYBRID_Bvz, ModelType.HYBRID_Bvz, ModelType.HYBRID_Bvz]
# title_names = [r"Hybrid ($B_z$, $v_z$)", r"Hybrid ($B_z$, $v_z$)", r"Hybrid ($B_z$, $v_z$)", r"Hybrid ($B_z$, $v_z$)", r"Hybrid ($B_z$, $v_z$)"]
# model_names = ["Hybrid_B_vz", "Hybrid_B_vz", "Hybrid_B_vz", "Hybrid_B_vz", "Hybrid_B_vz"]
# best_models = OrderedDict()
# best_models["1.0"] = "DeepVel_torch_epoch_97_0.06555.pt"
# best_models["1e-1"] = "DeepVel_torch_epoch_98_0.03048.pt"
# best_models["1e-2"] = "DeepVel_torch_epoch_98_0.04046.pt"
# best_models["1e-3"] = "DeepVel_torch_epoch_98_0.11586.pt"
# best_models["1e-4"] = "DeepVel_torch_epoch_94_0.18115.pt"

mean = OrderedDict()
#1st element corresponds to vx, 2nd to vy
mean["1.0"] = [-2078.321533203125, 3573.025634765625]
mean["1e-1"] = [-1068.5284423828125, 4305.888671875]
mean["1e-2"] = [2579.50732421875, 4552.302734375]
mean["1e-3"] = [2779.994140625, 5230.796875]
mean["1e-4"] = [-164.8361358642578,  5951.0673828125]

std = OrderedDict()
#1st element corresponds to vx, 2nd to vy
std["1.0"] = [232136.484375, 230852.375]
std["1e-1"] = [223613.75, 220306.984375]
std["1e-2"] = [171844.03125, 169797.921875]
std["1e-3"] = [130505.2578125, 126075.28125]
std["1e-4"] = [143405.125, 140374.46875]


tau_levels=[1.0]
taus = ["1.0"]

best_models = OrderedDict()
best_models["1.0"] = "DeepVel_torch_epoch_95_0.06872.pt"
title_names = [r"Hybrid (I, $v_z$)"]
model_names = ["Hybrid_I_vz"]
model_types = [ModelType.HYBRID_Ivz]
main_root = "/home/xenoss/dat/"
data_path = os.path.join(main_root, f'multi_height_data_new/final_dataset_multiheight/train')

# metrics_by_tau = OrderedDict()
in_shape=(4, 1536, 1536)
out_shape = (2, 1536, 1536)
# # NOTE THIS IS FOR TESTING ON FULL TEST SETS
dataset_path = "/home/xenoss/dat/multi_height_data_new/final_dataset_multiheight/test/"
for i, tau in enumerate(tau_levels):

    model_path = f"/home/xenoss/dat/models_multi_height/{model_names[i]}/tau_{taus[i]}/checkpoints/"
    
    if model_types[i] == ModelType.HYBRID_Bvz:
        input1_path = dataset_path + f"inputs/B/tau_{taus[i]}/"

    elif model_types[i] == ModelType.HYBRID_Ivz:
        input1_path = dataset_path + f"inputs/I/tau_{taus[i]}/"

    vz_path = dataset_path + f"inputs/vz/tau_{taus[i]}/"
    vel_path = dataset_path + f"labels/tau_{taus[i]}/"
    input1_test_data = os.listdir(input1_path)
    vz_test_data = os.listdir(vz_path)
    vel_test_data = os.listdir(vel_path)
    input1_test_data.sort()
    vz_test_data.sort()
    vel_test_data.sort()

    deepvel_net = DeepVel_run(root = main_root, in_shape=in_shape, out_shape=out_shape, batch = 64, dataset_path = data_path, network_path = model_path, tau=taus[i], type=model_types[i])
    params_model = model_path + best_models[taus[i]]

    metrics_rows = []
    print(f"Evaluating model for tau = {taus[i]} ...")
    for j in range(len(input1_test_data)):
        input1 = torch.from_numpy(np.load(os.path.join(input1_path, input1_test_data[j])).astype(np.float32))
        vz = torch.from_numpy(np.load(os.path.join(vz_path, vz_test_data[j])).astype(np.float32))
        ground_truth = torch.from_numpy(np.load(os.path.join(vel_path, vel_test_data[j])).astype(np.float32))

        prediction = deepvel_net.predict(input1, vz, params_model)

        #NOTE Denormalize velocities before computing the metrics, velocity unit (km/s)
        ground_truth = denormalize_data_vectors(ground_truth, mean[taus[i]], std[taus[i]])/1e5
        prediction = denormalize_data_vectors(prediction, mean[taus[i]], std[taus[i]])/1e5

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

    csv_dir = os.path.join(main_root, f'models_multi_height/{model_names[i]}/test_metrics/')

    df_tau = pd.DataFrame(metrics_rows)
    avg_series = df_tau.select_dtypes(include=[np.number]).mean()
    avg_row = avg_series.to_dict()
    avg_row['tau'] = taus[i]
    avg_row['sample_index'] = 'AVERAGE'
    df_tau = pd.concat([df_tau, pd.DataFrame([avg_row])], ignore_index=True)

    csv_path_tau = os.path.join(csv_dir, f'metrics_tau_{taus[i]}.csv')
    df_tau.to_csv(csv_path_tau, index=False)


#NOTE THIS IS FOR TESTING ONLY ON LAST 4 LAYERS
# dataset_path = os.path.join(dataset_path, "last_4_layers")
# for i, tau in enumerate(tau_levels):

#     model_path = f"/home/xenoss/dat/models_multi_height/{model_names[i]}/tau_{taus[i]}/checkpoints/"
#     test_save_path = f"/home/xenoss/dat/models_multi_height/{model_names[i]}/tau_{taus[i]}/test/"
    
#     deepvel_net = DeepVel_run(root = main_root, in_shape=in_shape, batch = 64, dataset_path = data_path, network_path = model_path, tau=taus[i], type=model_types[i])
#     params_model = model_path + best_models[taus[i]]
#     print(f"Evaluating model for tau = {taus[i]} ...")
#     plot_prediction_and_scatter_full_map_vertical(deepvel_object = deepvel_net, params_model = params_model, 
#                 dataset_path = dataset_path, test_save_path = test_save_path, 
#                 name = f"{model_names[i]}_{taus[i]}", zoomed_in_size = None, 
#                 plot_intensity = False, n_input_channels = 4, scatter_dim = None, zoom_out = 0, 
#                 write_metrics = False,denormalized=True, mean_vel=mean[taus[i]], std_vel=std[taus[i]], return_metrics = False, model_type = model_types[i], 
#                 title = title_names[i]+r", $\tau$ = " + taus[i], tau = taus[i])
#     # print(f"Metrics for tau = {taus[i]}: {metrics}")

# #     metrics_by_tau[taus[i]] = metrics
    

# output_dir = os.path.join(main_root, 'models_multi_height')
# os.makedirs(output_dir, exist_ok=True)
# df_metrics = pd.DataFrame.from_dict(metrics_by_tau, orient='index')
# csv_path = os.path.join(output_dir, 'metrics_by_tau.csv')
# df_metrics.to_csv(csv_path)
# print(f"Saved metrics summary to {csv_path}")

################## NOTE PLOTTING LOSS CURVES FOR MULTI-HEIGHT HYBRID MODELS #####################
# tau = ["1.0", "1.0", "1e-1", "1e-2", "1e-3", "1e-4"]
# model_types = [ModelType.HYBRID_Ivz, ModelType.HYBRID_Bvz, ModelType.HYBRID_Bvz, ModelType.HYBRID_Bvz, ModelType.HYBRID_Bvz, ModelType.HYBRID_Bvz]
# ckpts_path = os.path.join(main_root, 'models_multi_height')
# plot_train_val_losses(tau_levels=tau, ckpts_path=ckpts_path, model_type=model_types, name='multiheight_training_all_combined', save_path=os.path.join(main_root, 'models_multi_height/'))