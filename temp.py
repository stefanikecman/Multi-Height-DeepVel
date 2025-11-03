import glob
import numpy as np
import muram as muram
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from analysis_fn import get_divergence, get_vorticity


tau_levels=[1.0, 1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]
taus = ["1.0", "1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6"]

for i, tau in enumerate(tau_levels):
    model_path = f"/home/xenoss/dat/models_multi_height/tau_{taus[i]}/checkpoints/*.npy"
    path = glob.glob(model_path)[0]
    with open(path, "rb") as f:
        params_dict = np.load(f, allow_pickle=True).item()
        save_losses = np.load(f, allow_pickle=True)

    #print(save_losses)
    print (save_losses.shape)

    x = save_losses[:,0].tolist()
    train_loss = np.sqrt(save_losses[:,1].tolist())
    val_loss = np.sqrt(save_losses[:,2].tolist())

    plt.figure(figsize=(10, 6))
    plt.rcParams['font.size'] = 17
    plt.plot(x, train_loss, label='train loss')
    plt.plot(x, val_loss, label='val loss')
    #plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (RMSE)')
    plt.title('Training and Validation Loss Curves tau = '+str(tau))
    plt.legend()
    #plt.grid(True)
    plt.savefig(f'/home/xenoss/dat/models_multi_height/tau_{taus[i]}/training_validation_loss_'+str(tau)+'.png')
    # plt.show()

# data_cube_vx = np.load("/home/xenoss/dat/datasets_5x5_experiments/main_dataset_train/stacked_velocities_x_normalized.npy")
# data_cube_vy = np.load("/home/xenoss/dat/datasets_5x5_experiments/main_dataset_train/stacked_velocities_y_normalized.npy")

# divergence = get_divergence(data_cube_vx, data_cube_vy).cpu().numpy()
# vorticity = get_vorticity(data_cube_vx, data_cube_vy).cpu().numpy()

# mean_div = np.mean(divergence)
# std_div = np.std(divergence)
# mean_vor = np.mean(vorticity)
# std_vor = np.std(vorticity)
# print("Divergence:")
# print("mean:", mean_div)
# print("std:", std_div)
# print("Vorticity:")
# print("mean:", mean_vor)
# print("std:", std_vor)

# mse1 = np.load("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/new_loss_experiments/version_9/mse1.npy")
# mse2 = np.load("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/new_loss_experiments/version_9/mse2.npy")
# mse3 = np.load("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/new_loss_experiments/version_9/mse3.npy")

# def average_chunks(arr, chunk_size):
#     return np.mean(arr.reshape(-1, chunk_size), axis=1)

# mse1 = average_chunks(mse1, 862)
# mse2 = average_chunks(mse2, 862)
# mse3 = average_chunks(mse3, 862)

# x = list(range(0, 120))

# plt.figure(figsize=(10, 6))
# plt.rcParams['font.size'] = 17
# plt.plot(x, mse1, label='MSE 1')
# plt.plot(x, mse2/10, label='MSE 2')
# plt.plot(x, mse3/10, label='MSE 3')
# #plt.yscale('log')
# plt.xlabel('Epochs')
# plt.ylabel('Loss (MSE)')
# plt.title('Individual Loss Components ')
# plt.legend()
# #plt.grid(True)
# plt.savefig('new_loss_curves_v9.png')
# plt.show()