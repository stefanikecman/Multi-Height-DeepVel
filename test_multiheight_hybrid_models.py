import numpy as np
import torch
import os
from collections import OrderedDict
import csv
from hybrid_deepvel_torch import DeepVel_run as DeepVel_run_hybrid2
import pandas as pd
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

# NOTE THIS IS FOR TESTING ON FULL TEST SETS
dataset_path = "/home/xenoss/dat/multi_height_data_new/final_dataset_multiheight/test/"
for i, tau in enumerate(tau_levels):

    model_path = f"/home/xenoss/dat/models_multi_height/tau_{taus[i]}/checkpoints/"
    
    B_path = dataset_path + f"inputs/B/tau_{taus[i]}/"
    vz_path = dataset_path + f"inputs/vz/tau_{taus[i]}/"
    vel_path = dataset_path + f"labels/tau_{taus[i]}/"
    B_test_data = os.listdir(B_path)
    vz_test_data = os.listdir(vz_path)
    vel_test_data = os.listdir(vel_path)
    B_test_data.sort()
    vz_test_data.sort()
    vel_test_data.sort()

    deepvel_net = DeepVel_run_hybrid2(root = main_root, in_channels=4, batch = 64, dataset_path = data_path, network_path = model_path, tau=taus[i])
    params_model = model_path + best_models[taus[i]]

    metrics_rows = []
    print(f"Evaluating model for tau = {taus[i]} ...")
    for j in range(len(B_test_data)):
        B = torch.from_numpy(np.load(B_path + B_test_data[j]).astype(np.float32))
        vz = torch.from_numpy(np.load(vz_path + vz_test_data[j]).astype(np.float32))
        ground_truth = torch.from_numpy(np.load(vel_path + vel_test_data[j]).astype(np.float32))

        prediction = deepvel_net.predict(B, vz, params_model)

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

    csv_dir = os.path.join(main_root, 'models_multi_height/test_metrics/')

    df_tau = pd.DataFrame(metrics_rows)
    avg_series = df_tau.select_dtypes(include=[np.number]).mean()
    avg_row = avg_series.to_dict()
    avg_row['tau'] = taus[i]
    avg_row['sample_index'] = 'AVERAGE'
    df_tau = pd.concat([df_tau, pd.DataFrame([avg_row])], ignore_index=True)

    csv_path_tau = os.path.join(csv_dir, f'metrics_tau_{taus[i]}.csv')
    df_tau.to_csv(csv_path_tau, index=False)


#NOTE THIS IS FOR TESTING ONLY ON LAST 4 LAYERS

# for i, tau in enumerate(tau_levels):

#     model_path = f"/home/xenoss/dat/models_multi_height/tau_{taus[i]}/checkpoints/"
#     test_save_path = f"/home/xenoss/dat/models_multi_height/tau_{taus[i]}/test/"
    
#     deepvel_net = DeepVel_run_hybrid2(root = main_root, in_channels=4, batch = 64, dataset_path = data_path, network_path = model_path, tau=taus[i])
#     params_model = model_path + best_models[taus[i]]
#     print(f"Evaluating model for tau = {taus[i]} ...")
#     metrics = plot_prediction_and_scatter_full_map_vertical(
#         deepvel_object=deepvel_net,
#         params_model=params_model,
#         dataset_path=dataset_path,
#         test_save_path=test_save_path,
#         plot_intensity=False,
#         n_input_channels=4,
#         return_metrics=True,
#         hybrid2=True,
#         title="Test for tau = "+str(taus[i]),
#         tau=taus[i],
#         name = "hybrid2_tau_"+str(taus[i])
#     )
#     print(f"Metrics for tau = {taus[i]}: {metrics}")

#     metrics_by_tau[taus[i]] = metrics
    

# output_dir = os.path.join(main_root, 'models_multi_height')
# os.makedirs(output_dir, exist_ok=True)
# df_metrics = pd.DataFrame.from_dict(metrics_by_tau, orient='index')
# csv_path = os.path.join(output_dir, 'metrics_by_tau.csv')
# df_metrics.to_csv(csv_path)
# print(f"Saved metrics summary to {csv_path}")
