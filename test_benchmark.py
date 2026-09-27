import numpy as np
import torch
import os
import csv
from collections import OrderedDict
from deepvel_torch_benchmarking import DeepVel_run
import pandas as pd
from analysis_fn import denormalize_data_vectors, get_all_metrics, get_divergence, get_vorticity
from metrics_and_plotting import ModelType, plot_prediction_and_scatter_full_map_vertical, plot_prediction_and_scatter_full_map_vertical_custom, plot_prediction_and_scatter_full_map_vertical_custom_scatter, plot_train_val_losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

main_root = "/home/xenoss/dat/"
data_path = os.path.join(main_root, f'multi_height_data_new/final_dataset_multiheight/train')

metrics_by_tau = OrderedDict()
dataset_path = "/home/xenoss/dat/multi_height_data_new/final_dataset_multiheight/test/" 

################## NOTE PLOTTING LOSS CURVES FOR BENCHMARK MODELS #####################
tau = ["1.0", "1.0", "0.1", "1.0", "0.1"]
model_types = [ModelType.DEEPVEL_I, ModelType.DEEPVEL_B, ModelType.DEEPVEL_B, ModelType.DEEPVEL_Vz, ModelType.DEEPVEL_Vz]
ckpts_path = os.path.join(main_root, 'models_benchmarking/')
plot_train_val_losses(tau_levels=tau, ckpts_path=ckpts_path, model_type=model_types, name='benchmarking_training_all_combined', save_path=os.path.join(main_root, 'models_benchmarking/'))


#######################################################################################

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
# checkpoints = ["/home/xenoss/dat/models_benchmarking/DeepVel_I/checkpoints/DeepVel_torch_epoch_99_0.14751.pt",
#                "/home/xenoss/dat/models_benchmarking/DeepVel_Bz/tau_1.0/checkpoints/DeepVel_torch_epoch_93_0.11650.pt", 
#                "/home/xenoss/dat/models_benchmarking/DeepVel_Bz/tau_0.1/checkpoints/DeepVel_torch_epoch_99_0.05246.pt",
#                "/home/xenoss/dat/models_benchmarking/DeepVel_vz/tau_1.0/checkpoints/DeepVel_torch_epoch_93_0.07742.pt", 
#                "/home/xenoss/dat/models_benchmarking/DeepVel_vz/tau_0.1/checkpoints/DeepVel_torch_epoch_95_0.03773.pt"]
# taus = ["1.0", "1.0", "1e-1", "1.0", "1e-1"]
# model_names = ["DeepVel_I", "DeepVel_Bz_tau_1.0", "DeepVel_Bz_tau_1e-1", "DeepVel_Vz_tau_1.0", "DeepVel_Vz_tau_1e-1"]
# model_paths = ["/home/xenoss/dat/models_benchmarking/DeepVel_I/", 
#                "/home/xenoss/dat/models_benchmarking/DeepVel_Bz/tau_1.0/", 
#                "/home/xenoss/dat/models_benchmarking/DeepVel_Bz/tau_0.1/", 
#                "/home/xenoss/dat/models_benchmarking/DeepVel_vz/tau_1.0/", 
#                "/home/xenoss/dat/models_benchmarking/DeepVel_vz/tau_0.1/"]
# model_types = [ModelType.DEEPVEL_I, ModelType.DEEPVEL_B, ModelType.DEEPVEL_B, ModelType.DEEPVEL_Vz, ModelType.DEEPVEL_Vz]

# csvs = ["DeepVel_I", "DeepVel_Bz_tau_1.0", "DeepVel_Bz_tau_1e-1", "DeepVel_Vz_tau_1.0", "DeepVel_Vz_tau_1e-1"]

# for i, ckpt in enumerate(checkpoints):

#     deepvel_net = DeepVel_run(root = main_root, in_shape=(4, 1536, 1536), out_shape=(2, 1536, 1536), type=model_types[i], batch = 64, dataset_path = data_path, network_path = model_paths[i]+"checkpoints/", tau=taus[i])
#     params_model = ckpt

#     if model_types[i] == ModelType.DEEPVEL_I:
#         input_path = dataset_path + f"inputs/I/tau_{taus[i]}/"

#     elif model_types[i] == ModelType.DEEPVEL_B:
#         input_path = dataset_path + f"inputs/B/tau_{taus[i]}/"
    
#     elif model_types[i] == ModelType.DEEPVEL_Vz:
#         print("Using Vz as input")
#         input_path = dataset_path + f"inputs/vz/tau_{taus[i]}/"
    
#     input_test_data = os.listdir(input_path)
#     input_test_data.sort()

#     vel_path = dataset_path + f"labels/tau_{taus[i]}/"
#     vel_test_data = os.listdir(vel_path)
#     vel_test_data.sort()

#     metrics_rows = []
#     print(f"Evaluating model for tau = {taus[i]} ...")

#     if taus[i] == "1.0":
#         mean_vel = [-2078.321533203125, 3573.025634765625]
#         std_vel = [232136.484375, 230852.375]

#     elif taus[i] == "1e-1":
#         mean_vel = [-1068.5284423828125, 4305.888671875]
#         std_vel = [223613.75, 220306.984375]
        
#     for j in range(len(input_test_data)):
#         input = torch.from_numpy(np.load(os.path.join(input_path, input_test_data[j])).astype(np.float32))
#         ground_truth = torch.from_numpy(np.load(os.path.join(vel_path, vel_test_data[j])).astype(np.float32))

#         prediction = deepvel_net.predict(input,params_model)

#         #NOTE denormalize results and ground truth and then compute the metrics (in km/s)
      #   ground_truth = denormalize_data_vectors(ground_truth, mean_vel, std_vel)/1e5
      #   prediction = denormalize_data_vectors(prediction, mean_vel, std_vel)/1e5 #1e5 to convert to km/s

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

#     csv_dir = os.path.join(main_root, 'models_benchmarking/test_metrics/')

#     df_tau = pd.DataFrame(metrics_rows)
#     avg_series = df_tau.select_dtypes(include=[np.number]).mean()
#     avg_row = avg_series.to_dict()
#     avg_row['tau'] = taus[i]
#     avg_row['sample_index'] = 'AVERAGE'
#     df_tau = pd.concat([df_tau, pd.DataFrame([avg_row])], ignore_index=True)

#     csv_path_tau = os.path.join(csv_dir, f'{csvs[i]}_metrics_denorm.csv')
#     df_tau.to_csv(csv_path_tau, index=False)


#################################  NOTE TESTING ON LAST 4 LAYERS #############################################

mean = OrderedDict()
# #1st element corresponds to vx, 2nd to vy
mean["1.0"] = [-2078.321533203125, 3573.025634765625]
mean["1e-1"] = [-1068.5284423828125, 4305.888671875]

std = OrderedDict()
# #1st element corresponds to vx, 2nd to vy
std["1.0"] = [232136.484375, 230852.375]
std["1e-1"] = [223613.75, 220306.984375]

checkpoints = [
               #  "/home/xenoss/dat/models_benchmarking/DeepVel_I/checkpoints/DeepVel_torch_epoch_99_0.14751.pt", 
               "/home/xenoss/dat/models_benchmarking/DeepVel_Bz/tau_1.0/checkpoints/DeepVel_torch_epoch_93_0.11650.pt", 
               "/home/xenoss/dat/models_benchmarking/DeepVel_Bz/tau_0.1/checkpoints/DeepVel_torch_epoch_99_0.05246.pt",
               "/home/xenoss/dat/models_benchmarking/DeepVel_vz/tau_1.0/checkpoints/DeepVel_torch_epoch_93_0.07742.pt", 
               "/home/xenoss/dat/models_benchmarking/DeepVel_vz/tau_0.1/checkpoints/DeepVel_torch_epoch_95_0.03773.pt"
            ]
taus = ["1.0", "0.1", "1.0", "0.1"]
tau_int = ["1.0", "1e-1", "1.0", "1e-1"]
# taus = ["1.0"]
# tau_int = ["1.0"]

model_names = [
            # "DeepVel_I", 
               "DeepVel_Bz", "DeepVel_Bz", "DeepVel_vz", "DeepVel_vz"
               ]
title_names = [
            # r"DeepVel ($I$)", 
              r"DeepVel ($B_z$)", r"DeepVel ($B_z$)", r"DeepVel ($v_z$)", r"DeepVel ($v_z$)"
               ]

model_paths = [
            # "/home/xenoss/dat/models_benchmarking/DeepVel_I/", 
               "/home/xenoss/dat/models_benchmarking/DeepVel_Bz/tau_1.0/", 
               "/home/xenoss/dat/models_benchmarking/DeepVel_Bz/tau_0.1/", 
               "/home/xenoss/dat/models_benchmarking/DeepVel_vz/tau_1.0/", 
               "/home/xenoss/dat/models_benchmarking/DeepVel_vz/tau_0.1/"
               ]
model_types = [
               #  ModelType.DEEPVEL_I, 
               ModelType.DEEPVEL_B, ModelType.DEEPVEL_B, ModelType.DEEPVEL_Vz, ModelType.DEEPVEL_Vz
               ]
dataset_path = os.path.join(dataset_path, "last_4_layers")
for i, ckpt in enumerate(checkpoints):

    deepvel_net = DeepVel_run(root = main_root, in_shape=(4, 1536, 1536), out_shape=(2, 1536, 1536), type=model_types[i], batch = 64, dataset_path = data_path, network_path = model_paths[i]+"checkpoints/", tau=tau_int[i])
    params_model = ckpt

    test_save_path = f"/home/xenoss/dat/models_benchmarking/{model_names[i]}/test/" if model_types[i] == ModelType.DEEPVEL_I else f"/home/xenoss/dat/models_benchmarking/{model_names[i]}/tau_{taus[i]}/test/"
    # plot_prediction_and_scatter_full_map_vertical (deepvel_object = deepvel_net, params_model = params_model, 
    #                                                dataset_path = dataset_path, test_save_path = test_save_path, 
    #                                                name = f"{model_names[i]}_{taus[i]}", zoomed_in_size = None, 
    #                                                plot_intensity = False, n_input_channels = 4, scatter_dim = None, zoom_out = 0, 
    #                                                write_metrics = False, denormalized = True, mean_vel=mean[tau_int[i]], std_vel=std[tau_int[i]], return_metrics = False, model_type = model_types[i], 
    #                                                title = title_names[i]+r", $\tau$ = " + taus[i], tau = tau_int[i])

    plot_prediction_and_scatter_full_map_vertical_custom_scatter (deepvel_object = deepvel_net, params_model = params_model, 
                                                    dataset_path = dataset_path, test_save_path = test_save_path, 
                                                    name = f"{model_names[i]}_{taus[i]}", zoomed_in_size = None, 
                                                    n_input_channels = 4, scatter_dim = None, zoom_out = 0, 
                                                    write_metrics = False, denormalized = True, mean_vel=mean[tau_int[i]], 
                                                    std_vel=std[tau_int[i]], return_metrics = False, model_type = model_types[i], 
                                                    #title = title_names[i]+r", $\tau$ = " + taus[i], 
                                                    tau = tau_int[i])

#######################################################################################################################################