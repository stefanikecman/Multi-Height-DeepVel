import os
import shutil
import muram as muram
import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import random
import shutil
import muram as mio

main_path = "/dat/milic/2D/"

def load_data (origin_path, destination_path):
    """
    Load Intensity data from the origin path and save it to the destination path.
    Args:
        origin_path: Path where the original intensity files are located
        destination_path: Path where the intensity files will be copied
    """
    filenames = os.listdir(origin_path)
    intensities = []
    for n in filenames:
        if "I_out" in n:
            intensities.append(n)

    for i in intensities:
        shutil.copy(origin_path + i, destination_path)

def file_naming(name, iter):
    """
    Create a filename with leading zeros based on the iteration number.
    """
    padding = ""
    if iter <10:
        padding = "00000"
    elif iter<100 :
        padding = "0000"
    elif iter<1000:
        padding = "000"
    elif iter<10000:
        padding = "00"
    else: padding = "0"
    return name + "_" + padding + str(iter) 

def create_intensities(main_path, intensity_path_train, intensity_path_test):
    """
    Create intensity files from MURaM simulation data and save them as .npy files.
    Args:
        main_path: Path where the MURaM simulation data is located
        intensity_path: Path where the intensity files will be saved
    """

    for i in range(0, 12000, 50):
        data = muram.MuramIntensity(main_path,i)
        np.save(os.path.join(intensity_path_train, file_naming("intensities", i)+".npy"), data)

    for i in range(12000, 18050, 50):
        data = muram.MuramIntensity(main_path,i)
        np.save(os.path.join(intensity_path_test, file_naming("intensities", i)+".npy"), data)

def create_velocities(main_path, velocity_path_train, velocity_path_test, tau = 1.0): 
    """
    Create velocity files for horizontal velocity components (vx, vy) from MURaM simulation data and save them as .npy files.
    Args:
        main_path: Path where the MURaM simulation data is located
        velocity_path: Path where the velocity files will be saved
        tau: Optical depth level to extract velocities (default is 1.0)
    """

    for i in range(0, 12000, 50):
        data = muram.MuramTauSlice(main_path,i,tau)
        vx = data.vy
        vy = data.vz
        np.save(os.path.join(velocity_path_train, file_naming("vx", i)+".npy"), vx)
        np.save(os.path.join(velocity_path_train, file_naming("vy", i)+".npy"), vy)

    for i in range(12000, 18050, 50):
        data = muram.MuramTauSlice(main_path,i,tau)
        vx = data.vy
        vy = data.vz
        np.save(os.path.join(velocity_path_test, file_naming("vx", i)+".npy"), vx)
        np.save(os.path.join(velocity_path_test, file_naming("vy", i)+".npy"), vy)

def create_B (main_path, B_path_save_train, B_path_save_test, tau = 1.0):
    """
    Create vertical magnetic field files (Bz) from MURaM simulation data and save them as .npy files.
    Args:
        main_path: Path where the MURaM simulation data is located
        B_path_save: Path where the magnetic field files will be saved
        tau: Optical depth level to extract magnetic field (default is 1.0)
    """

    for i in range(0, 12000, 50):
        data = mio.MuramTauSlice(main_path,i,tau)
        B = data.Bx
        np.save(os.path.join(B_path_save_train, file_naming("B", i)+".npy"), B)

    for i in range(12000, 18050, 50):
        data = mio.MuramTauSlice(main_path,i,tau)
        B = data.Bx
        np.save(os.path.join(B_path_save_test, file_naming("B", i)+".npy"), B)

def create_vz (main_path, vz_path_save_train, vz_path_save_test, tau = 1.0):
    """
    Create vertical velocity files (vz) from MURaM simulation data and save them as .npy files.
    Args:
        main_path: Path where the MURaM simulation data is located
        vz_path_save: Path where the vertical velocity files will be saved
        tau: Optical depth level to extract vertical velocity (default is 1.0)
    """

    for i in range(0, 12000, 50):
        data = mio.MuramTauSlice(main_path,i,tau)
        vz = data.vx
        np.save(os.path.join(vz_path_save_train, file_naming("vz", i)+".npy"), vz)

    for i in range(12000, 18050, 50):
        data = mio.MuramTauSlice(main_path,i,tau)
        vz = data.vx
        np.save(os.path.join(vz_path_save_test, file_naming("vz", i)+".npy"), vz)

def create_dataset_N_intensities (dataset_path, int_path, vel_path, last_idx, indices_list):
    """
    Create a dataset, where the input intensity has N channels, from the original intensity data.
    Args:
        dataset_path: Path where the new dataset will be saved
        int_path: Path where the original intensity files are located
        vel_path: Path where the original velocity files are located
        last_idx: Last index of the intensity files to be considered
        indices_list: List of lists, where each sublist contains the relative indices for the N channels
    """
    velocities = os.listdir(vel_path)
    velocities.sort()
    
    count = 0
    for i in range(len(indices_list[0])):
        intensities_concat = []
        for j in range(len(indices_list)):
            idx = indices_list[j][i]
            intensity = muram.MuramIntensity(int_path, idx)
            intensities_concat.append(intensity)
        
        intensities_concat = np.stack(intensities_concat, axis = 0)

        np.save(dataset_path+"inputs/" + file_naming("intensities", count) + ".npy", intensities_concat)
        count +=1
    
    count = 0
    for i in range(len(indices_list[0])):
        vx_list = []
        vy_list = []
        for j in range(len(indices_list)):
            idx = indices_list[j][i]
            vx = np.load(vel_path + file_naming("vx", idx)+ ".npy")
            vy = np.load(vel_path + file_naming("vy", idx)+ ".npy")
            vx_list.append(vx)
            vy_list.append(vy)

        vx_out = np.mean(vx_list, axis=0)
        vy_out = np.mean(vy_list, axis=0)

        velocities_concat = np.stack([vx_out, vy_out], axis = 0)
        np.save(dataset_path + "labels/" + file_naming("velocities", count) + ".npy", velocities_concat)
        count+=1

    
def crop_image_center (original_dir, input_data_name, new_dim, save_dir):
    """
    Crop the center of the image to a new dimension.
    Args:
        original_dir: Directory where the original image is located
        input_data_name: Name of the input data file
        new_dim: New dimension for the cropped image (assumed to be square)
        save_dir: Directory where the cropped image will be saved
    """

    input_data = np.load(original_dir + input_data_name)

    _, h, w = input_data.shape
    cropped_img = []

    y_im_center = int(h/2)
    x_im_center = int(w/2)

    cropped_img = input_data[:, (y_im_center - int(new_dim/2)): (y_im_center + int(new_dim/2)), (x_im_center - int(new_dim/2)): (x_im_center + int(new_dim/2))]

    np.save(save_dir + input_data_name.replace(".npy", "_cropped.npy"), cropped_img)

def crop_image_every_nth (n, original_dir, data_idx, new_dim, save_dir, sample_random = False):
    """
    Crop the image into smaller patches of size new_dim x new_dim, take every n-th patch.
    Args:
        n: Step size for cropping (every n-th patch will be saved)
        original_dir: Directory where the original image is located
        data_idx: Index of the data file to be cropped
        new_dim: Dimension of the cropped patches (assumed to be square)
        save_dir: Directory where the cropped patches will be saved
        sample_random: If True, sample patches randomly instead of uniformly
    """

    input_data = np.load(original_dir + 'inputs/' + 'intensities_' + data_idx + '.npy')
    label_data = np.load(original_dir + 'labels/' + 'velocities_' + data_idx + '.npy')

    _, h, w = input_data.shape
    cropped_img_inputs = []
    cropped_img_labels = []

    for i in range(0, h, new_dim):
        for j in range(0, w, new_dim):
            cropped_img_inputs.append(input_data[:, i: (i + new_dim), j: (j + new_dim)])
            cropped_img_labels.append(label_data[:, i: (i + new_dim), j: (j + new_dim)])
    
    if sample_random == False: # sample uniformly
        for i in range(0, len(cropped_img_inputs), n):
            np.save(save_dir + 'inputs/intensities_' + data_idx + '_cropped_' + str (i) + ".npy", cropped_img_inputs[i])
            np.save(save_dir + 'labels/velocities_' + data_idx + '_cropped_' + str (i) + ".npy", cropped_img_labels[i])
    
    else: #sample randomly
        to_save = int(len(cropped_img_inputs) / n)
        indices_to_save = random.sample(range(0, len(cropped_img_inputs)), to_save)

        for i in indices_to_save:
            np.save(save_dir + 'inputs/intensities_' + data_idx + '_cropped_' + str (i) + ".npy", cropped_img_inputs[i])
            np.save(save_dir + 'labels/velocities_' + data_idx + '_cropped_' + str (i) + ".npy", cropped_img_labels[i])



def normalize_layerwise (input_data):
    '''
    Normalizes each datapoint separately, layers all together

    Args:
        input_data: Input data to be normalized (2-channel tensor or numpy array)
    '''
    if (torch.is_tensor(input_data)):
        mean = torch.mean(input_data)
        std = torch.std(input_data)
        norm_input_data = input_data.detach().clone()

    else:
        mean = np.mean(input_data)
        std = np.std(input_data)

        norm_input_data = input_data.copy()

    norm_input_data = (input_data - mean) / std

    return norm_input_data

def normalize_all_outputs (output_path, save_path):
    '''
    Normalizes outputs (velocities) all together
    Args:
        output_path: Path where the original velocity files are located
        save_path: Path where the normalized velocity files will be saved
    '''
    velocity_names = os.listdir(output_path)
    velocity_names.sort()

    velocities = [np.load(output_path + v) for v in velocity_names]

    mean = np.mean(velocities)
    std = np.std(velocities)

    for v in velocity_names:
        velocity = np.load(output_path + v)
        velocity = (velocity - mean)/std
        np.save(save_path + v, velocity)
    
    return mean, std

def denormalize_data (data, mean, std):
    '''
    Denormalizes data using the provided mean and standard deviation.
    Args:
        data: Normalized data to be denormalized (tensor or numpy array)
        mean: Mean used for normalization
        std: Standard deviation used for normalization
    '''
    return data*std + mean

def create_multiheight_datacubes(main_save_path = "/home/xenoss/dat/multi_height_data_new"):
    """
    Creates multi-height data by stacking B, vz, and velocity components for different optical depth levels.
    """
    
    B_save_path = main_save_path + "/B/"
    vz_save_path = main_save_path + "/vz/"
    vel_save_path = main_save_path + "/out_velocities/"

    tau_levels=[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]
    taus = ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6"]

    # tau_levels=[1.0]
    # taus = ["1.0"]

    for i, t in enumerate(tau_levels):
        print(f"Processing tau = {t}")

        print(f"Creating B files at: {os.path.join(main_path, B_save_path+f'train/individual/tau_{taus[i]}/')}")
        create_B(main_path, os.path.join(B_save_path+f'train/individual/tau_{taus[i]}/'), os.path.join(B_save_path+f'test/individual/tau_{taus[i]}/'), tau = t)

        print(f"Creating velocity files at: {os.path.join(main_path, vel_save_path+f'train/individual/tau_{taus[i]}/')}")
        create_velocities(main_path, os.path.join(vel_save_path+f'train/individual/tau_{taus[i]}/'), os.path.join(vel_save_path+f'test/individual/tau_{taus[i]}/'), tau = t)

        print(f"Creating vz files at: {os.path.join(main_path, vz_save_path+f'train/individual/tau_{taus[i]}/')}")
        create_vz(main_path, os.path.join(vz_save_path+f'train/individual/tau_{taus[i]}/'), os.path.join(vz_save_path+f'test/individual/tau_{taus[i]}/'), tau = t)

    for i, t in enumerate(tau_levels):
        print(f"Processing tau = {t}")
        individual_B_path_train = os.path.join(B_save_path,f'train/individual/tau_{taus[i]}/')
        individual_vz_path_train = os.path.join(vz_save_path,f'train/individual/tau_{taus[i]}/')
        individual_vel_path_train = os.path.join(vel_save_path,f'train/individual/tau_{taus[i]}/')

        B_list_train = os.listdir(individual_B_path_train)
        vz_list_train = os.listdir(individual_vz_path_train)
        vel_list_train = os.listdir(individual_vel_path_train)

        B_list_train.sort()
        vz_list_train.sort()
        vel_list_train.sort()

        B_train = [np.load(os.path.join(individual_B_path_train, B)) for B in B_list_train]
        vz_train = [np.load(os.path.join(individual_vz_path_train, vz)) for vz in vz_list_train]
        vel_x_train = [np.load(os.path.join(individual_vel_path_train, vel)) for vel in vel_list_train if "vx" in vel]
        vel_y_train = [np.load(os.path.join(individual_vel_path_train, vel)) for vel in vel_list_train if "vy" in vel]

        stacked_B_train = np.stack(B_train, axis=0)
        stacked_vz_train = np.stack(vz_train, axis=0)
        stacked_vel_x_train = np.stack(vel_x_train, axis=0)
        stacked_vel_y_train = np.stack(vel_y_train, axis=0)

        individual_B_path_test = os.path.join(B_save_path,f'test/individual/tau_{taus[i]}/')
        individual_vz_path_test = os.path.join(vz_save_path,f'test/individual/tau_{taus[i]}/')
        individual_vel_path_test = os.path.join(vel_save_path,f'test/individual/tau_{taus[i]}/')

        B_list_test = os.listdir(individual_B_path_test)
        B_list_test.sort()
        vz_list_test = os.listdir(individual_vz_path_test)
        vz_list_test.sort()
        vel_list_test = os.listdir(individual_vel_path_test)
        vel_list_test.sort()

        B_test = [np.load(os.path.join(individual_B_path_test, B)) for B in B_list_test]
        vz_test = [np.load(os.path.join(individual_vz_path_test, vz)) for vz in vz_list_test]
        vel_x_test = [np.load(os.path.join(individual_vel_path_test, vel)) for vel in vel_list_test if "vx" in vel]
        vel_y_test = [np.load(os.path.join(individual_vel_path_test, vel)) for vel in vel_list_test if "vy" in vel]

        stacked_B_test = np.stack(B_test, axis=0)
        stacked_vz_test = np.stack(vz_test, axis=0)
        stacked_vel_x_test = np.stack(vel_x_test, axis=0)
        stacked_vel_y_test = np.stack(vel_y_test, axis=0)

        mu_B_train = np.mean(stacked_B_train)
        std_B_train = np.std(stacked_B_train)

        mu_B_test = np.mean(stacked_B_test)
        std_B_test = np.std(stacked_B_test)

        mu_vz_train = np.mean(stacked_vz_train)
        std_vz_train = np.std(stacked_vz_train)

        mu_vz_test = np.mean(stacked_vz_test)
        std_vz_test = np.std(stacked_vz_test)

        mu_vel_x_train = np.mean(stacked_vel_x_train)
        std_vel_x_train = np.std(stacked_vel_x_train)

        mu_vel_y_train = np.mean(stacked_vel_y_train)
        std_vel_y_train = np.std(stacked_vel_y_train)

        mu_vel_x_test = np.mean(stacked_vel_x_test)
        std_vel_x_test = np.std(stacked_vel_x_test)

        mu_vel_y_test = np.mean(stacked_vel_y_test)
        std_vel_y_test = np.std(stacked_vel_y_test)

        stacked_B_train = (stacked_B_train-mu_B_train)/std_B_train
        stacked_B_test = (stacked_B_test-mu_B_test)/std_B_test

        stacked_vz_train = (stacked_vz_train-mu_vz_train)/std_vz_train
        stacked_vz_test = (stacked_vz_test-mu_vz_test)/std_vz_test

        stacked_vel_x_train = (stacked_vel_x_train-mu_vel_x_train)/std_vel_x_train
        stacked_vel_x_test = (stacked_vel_x_test-mu_vel_x_test)/std_vel_x_test

        stacked_vel_y_train = (stacked_vel_y_train-mu_vel_y_train)/std_vel_y_train
        stacked_vel_y_test = (stacked_vel_y_test-mu_vel_y_test)/std_vel_y_test

        with open("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/important_stats/multiheight_data_stats.txt", "a") as f:
            f.write(f"Tau level: {taus[i]}\n")
            f.write(f"B mean (train): {mu_B_train}, B std (train): {std_B_train}\n")
            f.write(f"B mean (test): {mu_B_test}, B std (test): {std_B_test}\n")
            f.write(f"vz mean (train): {mu_vz_train}, vz std (train): {std_vz_train}\n")
            f.write(f"vz mean (test): {mu_vz_test}, vz std (test): {std_vz_test}\n")
            f.write(f"vel_x mean (train): {mu_vel_x_train}, vel_x std (train): {std_vel_x_train}\n")
            f.write(f"vel_x mean (test): {mu_vel_x_test}, vel_x std (test): {std_vel_x_test}\n")
            f.write(f"vel_y mean (train): {mu_vel_y_train}, vel_y std (train): {std_vel_y_train}\n")
            f.write(f"vel_y mean (test): {mu_vel_y_test}, vel_y std (test): {std_vel_y_test}\n")
            f.write("--------------------------------------------------\n")

        np.save(os.path.join(B_save_path,f'train/stacked/tau_{taus[i]}/stacked_B_train_denorm.npy'), stacked_B_train)
        np.save(os.path.join(vz_save_path,f'train/stacked/tau_{taus[i]}/stacked_vz_train_denorm.npy'), stacked_vz_train)
        np.save(os.path.join(vel_save_path,f'train/stacked/tau_{taus[i]}/stacked_vel_x_train_denorm.npy'), stacked_vel_x_train)
        np.save(os.path.join(vel_save_path,f'train/stacked/tau_{taus[i]}/stacked_vel_y_train_denorm.npy'), stacked_vel_y_train)

        np.save(os.path.join(B_save_path,f'test/stacked/tau_{taus[i]}/stacked_B_test_denorm.npy'), stacked_B_test)
        np.save(os.path.join(vz_save_path,f'test/stacked/tau_{taus[i]}/stacked_vz_test_denorm.npy'), stacked_vz_test)
        np.save(os.path.join(vel_save_path,f'test/stacked/tau_{taus[i]}/stacked_vel_x_test_denorm.npy'), stacked_vel_x_test)
        np.save(os.path.join(vel_save_path,f'test/stacked/tau_{taus[i]}/stacked_vel_y_test_denorm.npy'), stacked_vel_y_test)

        np.save(os.path.join(B_save_path,f'train/stacked/tau_{taus[i]}/stacked_B_train_norm.npy'), stacked_B_train)
        np.save(os.path.join(vz_save_path,f'train/stacked/tau_{taus[i]}/stacked_vz_train_norm.npy'), stacked_vz_train)
        np.save(os.path.join(vel_save_path,f'train/stacked/tau_{taus[i]}/stacked_vel_x_train_norm.npy'), stacked_vel_x_train)
        np.save(os.path.join(vel_save_path,f'train/stacked/tau_{taus[i]}/stacked_vel_y_train_norm.npy'), stacked_vel_y_train)

        np.save(os.path.join(B_save_path,f'test/stacked/tau_{taus[i]}/stacked_B_test_norm.npy'), stacked_B_test)
        np.save(os.path.join(vz_save_path,f'test/stacked/tau_{taus[i]}/stacked_vz_test_norm.npy'), stacked_vz_test)
        np.save(os.path.join(vel_save_path,f'test/stacked/tau_{taus[i]}/stacked_vel_x_test_norm.npy'), stacked_vel_x_test)
        np.save(os.path.join(vel_save_path,f'test/stacked/tau_{taus[i]}/stacked_vel_y_test_norm.npy'), stacked_vel_y_test)


def generate_random_centers(final_shape, full_shape, n_samples, txt_path = '/home/xenoss/data/kecman_project/DeepVel_3D_velocity/important_stats/multiheight_centers.txt'):
    """
    Generates random centers for data extraction and saves them to a text file.
    Args:
        final_shape: Shape of the data cube (depth, height, width) after cropping
        full_shape: Shape of the full data cube (depth, height, width) before cropping
        n_samples: Number of random centers to generate
        txt_path: Path to save the generated centers
    """

    channels, height, width = final_shape
    half_c = channels // 2
    half_h = height // 2
    half_w = width // 2
    full_channels, full_height, full_width = full_shape

    centers = set()
    for _ in range(n_samples):
        c = random.randint(half_c, full_channels - half_c)
        h = random.randint(half_h, full_height - half_h)
        w = random.randint(half_w, full_width - half_w)
        centers.add((c, h, w))

    with open(txt_path, 'w') as f:
        for c, h, w in centers:
            f.write(f"{c},{h},{w}\n")
    return centers

def load_centers_from_txt(txt_path):

    centers = []
    with open(txt_path, 'r') as f:
        for line in f:
            c, h, w = map(int, line.strip().split(','))
            centers.append((c, h, w))
    return centers

def create_multiheight_trainset(main_save_path = "/home/xenoss/dat/multi_height_data_new"):
    """
    Creates a multi-height trainset by extracting data cubes centered at specified coordinates.
    """
    
    B_load_path = main_save_path + "/B/train/stacked/"
    vz_load_path = main_save_path + "/vz/train/stacked/"
    vel_load_path = main_save_path + "/out_velocities/train/stacked/"

    final_save_path = main_save_path + "/final_dataset_multiheight/train"

    tau_levels=[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]
    taus = ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6"]

    # tau_levels=[1.0]
    # taus = ["1.0"]

    final_shape = (4, 128, 128)  # (depth, height, width)
    full_shape = (240, 1536, 1536)  # (depth, height, width)
    n_samples = 70000

    #centers = generate_random_centers(final_shape, full_shape, n_samples)
    centers = load_centers_from_txt('/home/xenoss/data/kecman_project/DeepVel_3D_velocity/important_stats/multiheight_centers.txt')

    for tau_i, t in enumerate(tau_levels):
        print(f"Processing tau = {t}")

        B_data = np.load(os.path.join(B_load_path+f'tau_{taus[tau_i]}/stacked_B_train_norm.npy'))
        vz_data = np.load(os.path.join(vz_load_path+f'tau_{taus[tau_i]}/stacked_vz_train_norm.npy'))
        vel_x_data = np.load(os.path.join(vel_load_path+f'tau_{taus[tau_i]}/stacked_vel_x_train_norm.npy'))
        vel_y_data = np.load(os.path.join(vel_load_path+f'tau_{taus[tau_i]}/stacked_vel_y_train_norm.npy'))

        B_save_path = final_save_path + f"/inputs/B/tau_{taus[tau_i]}/"
        vz_save_path = final_save_path + f"/inputs/vz/tau_{taus[tau_i]}/"
        vel_save_path = final_save_path + f"/labels/tau_{taus[tau_i]}/"

        for i, (cc, ch, cw) in enumerate(centers):
            if i % 1000 == 0:
                print(f"Processing sample {i}")
            half_c = final_shape[0] // 2
            half_h = final_shape[1] // 2
            half_w = final_shape[2] // 2

            B_crop = B_data[cc-half_c:cc+half_c, ch-half_h:ch+half_h, cw-half_w:cw+half_w]
            vz_crop = vz_data[cc-half_c:cc+half_c, ch-half_h:ch+half_h, cw-half_w:cw+half_w]
            vel_x_crop = vel_x_data[cc-half_c:cc+half_c, ch-half_h:ch+half_h, cw-half_w:cw+half_w]
            vel_y_crop = vel_y_data[cc-half_c:cc+half_c, ch-half_h:ch+half_h, cw-half_w:cw+half_w]

            vel_crop = np.stack([ (vel_x_crop[1] + vel_x_crop[2]) / 2, (vel_y_crop[1] + vel_y_crop[2]) / 2 ], axis=0)
            if B_crop.shape != final_shape or vz_crop.shape != final_shape or vel_crop.shape != (2, final_shape[1], final_shape[2]):
                print(f"Skipping sample at center ({cc}, {ch}, {cw}) due to shape mismatch.")
                continue
            np.save(os.path.join(B_save_path, f'B_{cc}_{ch}_{cw}.npy'), B_crop)
            np.save(os.path.join(vz_save_path, f'vz_{cc}_{ch}_{cw}.npy'), vz_crop)
            np.save(os.path.join(vel_save_path, f'velocities_{cc}_{ch}_{cw}.npy'), vel_crop)


def create_multiheight_testset_last4(main_save_path = "/home/xenoss/dat/multi_height_data_new"):
    
    B_load_path = main_save_path + "/B/test/stacked/"
    vz_load_path = main_save_path + "/vz/test/stacked/"
    vel_load_path = main_save_path + "/out_velocities/test/stacked/"

    final_save_path = main_save_path + "/final_dataset_multiheight/test/"

    tau_levels=[1, 1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]
    taus = ["1.0", "1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6"]

    for tau_i, t in enumerate(tau_levels):
        print(f"Processing tau = {t}")

        # B_data = np.load(os.path.join(B_load_path+f'tau_{taus[tau_i]}/stacked_B_test_norm.npy'))[-4:, :, :]
        # vz_data = np.load(os.path.join(vz_load_path+f'tau_{taus[tau_i]}/stacked_vz_test_norm.npy'))[-4:, :, :]
        vel_x_data = np.load(os.path.join(vel_load_path+f'tau_{taus[tau_i]}/stacked_vel_x_test_norm.npy'))[-3:-1, :, :]
        vel_y_data = np.load(os.path.join(vel_load_path+f'tau_{taus[tau_i]}/stacked_vel_y_test_norm.npy'))[-3:-1, :, :]

        out_vel = np.stack(((vel_x_data[0] + vel_x_data[1]) / 2, (vel_y_data[0] + vel_y_data[1]) / 2), axis=0)

        # B_data = np.load(os.path.join(B_load_path+f'tau_{taus[tau_i]}/stacked_B_test_norm.npy'))[-7:-3, :, :]
        # vz_data = np.load(os.path.join(vz_load_path+f'tau_{taus[tau_i]}/stacked_vz_test_norm.npy'))[-7:-3, :, :]
        # vel_x_data = np.load(os.path.join(vel_load_path+f'tau_{taus[tau_i]}/stacked_vel_x_test_norm.npy'))[-6:-4, :, :]
        # vel_y_data = np.load(os.path.join(vel_load_path+f'tau_{taus[tau_i]}/stacked_vel_y_test_norm.npy'))[-6:-4, :, :]

        # out_vel = (vel_x_data + vel_y_data) / 2

        # np.save(final_save_path+f"B_tau_{taus[tau_i]}.npy", B_data)
        # np.save(final_save_path+f"vz_tau_{taus[tau_i]}.npy", vz_data)
        np.save(final_save_path+f"velocities_tau_{taus[tau_i]}.npy", out_vel)

def create_multiheight_full_testset(main_save_path = "/home/xenoss/dat/multi_height_data_new"):
    
    B_load_path = main_save_path + "/B/test/stacked/"
    vz_load_path = main_save_path + "/vz/test/stacked/"
    vel_load_path = main_save_path + "/out_velocities/test/stacked/"

    final_save_path = main_save_path + "/final_dataset_multiheight/test/"
    B_save_path = final_save_path + "/inputs/B/"
    vz_save_path = final_save_path + "/inputs/vz/"
    vel_save_path = final_save_path + "/labels/"
    intensity_save_path = final_save_path + "/inputs/I/tau_1.0/"

    tau_levels=[1, 1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]
    taus = ["1.0", "1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6"]

    for tau_i, t in enumerate(tau_levels):
        print(f"Processing tau = {t}")

        B_data = np.load(os.path.join(B_load_path+f'tau_{taus[tau_i]}/stacked_B_test_norm.npy'))[-61:, :, :]
        vz_data = np.load(os.path.join(vz_load_path+f'tau_{taus[tau_i]}/stacked_vz_test_norm.npy'))[-61:, :, :]
        vel_x_data = np.load(os.path.join(vel_load_path+f'tau_{taus[tau_i]}/stacked_vel_x_test_norm.npy'))[-61:, :, :]
        vel_y_data = np.load(os.path.join(vel_load_path+f'tau_{taus[tau_i]}/stacked_vel_y_test_norm.npy'))[-61:, :, :]

        for i in range(58): #58 = data.shape[0] - 4 + 1, 4 is our test data n channels
            B_test = B_data[i:i+4, :, :]
            vz_test = vz_data[i:i+4, :, :]
            vel_x_test = vel_x_data[i+1:i+3, :, :]
            vel_y_test = vel_y_data[i+1:i+3, :, :]
            out_vel_test = np.stack(((vel_x_test[0] + vel_x_test[1]) / 2, (vel_y_test[0] + vel_y_test[1]) / 2), axis=0)

            np.save(B_save_path+f"tau_{taus[tau_i]}/B_{i}.npy", B_test)
            np.save(vz_save_path+f"tau_{taus[tau_i]}/vz_{i}.npy", vz_test)
            np.save(vel_save_path+f"tau_{taus[tau_i]}/velocities_{i}.npy", out_vel_test)

    # Intensity needs to be created individually
    intensity_cube = np.load('/home/xenoss/dat/multi_height_data_new/I/test/stacked/stacked_intensities_test_norm.npy')[-61:, :, :]
    for i in range(58):
        intensity_test = intensity_cube[i:i+4, :, :]
        np.save(intensity_save_path+f'intensity_{i}.npy', intensity_test)

def create_intensity_cubes():
    main_path = '/home/xenoss/dat/intensities/'
    create_intensities(main_path, '/home/xenoss/dat/multi_height_data_new/I/train/individual/', '/home/xenoss/dat/multi_height_data_new/I/test/individual/')

    intensities_train = os.listdir('/home/xenoss/dat/multi_height_data_new/I/train/individual/')
    intensities_train.sort()
    mu_intensity_train = np.mean([np.load('/home/xenoss/dat/multi_height_data_new/I/train/individual/' + i) for i in intensities_train])
    std_intensity_train = np.std([np.load('/home/xenoss/dat/multi_height_data_new/I/train/individual/' + i) for i in intensities_train])

    intensities_test = os.listdir('/home/xenoss/dat/multi_height_data_new/I/test/individual/')
    intensities_test.sort()
    mu_intensity_test = np.mean([np.load('/home/xenoss/dat/multi_height_data_new/I/test/individual/' + i) for i in intensities_test])
    std_intensity_test = np.std([np.load('/home/xenoss/dat/multi_height_data_new/I/test/individual/' + i) for i in intensities_test])

    with open("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/important_stats/multiheight_data_stats.txt", "a") as f:
        f.write(f"Intensity mean (train): {mu_intensity_train}, Intensity std (train): {std_intensity_train}\n")
        f.write(f"Intensity mean (test): {mu_intensity_test}, Intensity std (test): {std_intensity_test}\n")
        f.write("--------------------------------------------------\n")

    intensities_train_stacked = np.stack([np.load('/home/xenoss/dat/multi_height_data_new/I/train/individual/' + i) for i in intensities_train], axis=0)
    intensities_test_stacked = np.stack([np.load('/home/xenoss/dat/multi_height_data_new/I/test/individual/' + i) for i in intensities_test], axis=0)

    np.save('/home/xenoss/dat/multi_height_data_new/I/train/stacked/stacked_intensities_train.npy', intensities_train_stacked)
    np.save('/home/xenoss/dat/multi_height_data_new/I/test/stacked/stacked_intensities_test.npy', intensities_test_stacked)

    intensities_train_stacked = (intensities_train_stacked - mu_intensity_train) / std_intensity_train
    intensities_test_stacked = (intensities_test_stacked - mu_intensity_test) / std_intensity_test

    np.save('/home/xenoss/dat/multi_height_data_new/I/train/stacked/stacked_intensities_train_norm.npy', intensities_train_stacked)
    np.save('/home/xenoss/dat/multi_height_data_new/I/test/stacked/stacked_intensities_test_norm.npy', intensities_test_stacked)

def create_benchmarking_trainset(main_save_path = "/home/xenoss/dat/multi_height_data_new"):
    centers = load_centers_from_txt('/home/xenoss/data/kecman_project/DeepVel_3D_velocity/important_stats/multiheight_centers.txt')
    final_shape = (4, 128, 128)  # (depth, height, width)

    intensity_data = np.load(os.path.join('/home/xenoss/dat/multi_height_data_new/I/train/stacked/stacked_intensities_train_norm.npy'))
    final_save_path = main_save_path + "/final_dataset_multiheight/train"
    intensity_save_path = final_save_path + f"/inputs/I/tau_1.0/"

    for i, (cc, ch, cw) in enumerate(centers):
            if i % 1000 == 0:
                print(f"Processing sample {i}")
            half_c = final_shape[0] // 2
            half_h = final_shape[1] // 2
            half_w = final_shape[2] // 2

            I_crop = intensity_data[cc-half_c:cc+half_c, ch-half_h:ch+half_h, cw-half_w:cw+half_w]

            if I_crop.shape != final_shape:
                print(f"Skipping sample at center ({cc}, {ch}, {cw}) due to shape mismatch.")
                continue

            np.save(os.path.join(intensity_save_path, f'intensity_{cc}_{ch}_{cw}.npy'), I_crop)

def create_benchmarking_testset(main_save_path = "/home/xenoss/dat/multi_height_data_new"):

    I_load_path = main_save_path + "/I/test/stacked/stacked_intensities_test_norm.npy"
    final_save_path = main_save_path + "/final_dataset_multiheight/test/"

    I_data = np.load(I_load_path)[-4:, :, :]

    np.save(final_save_path+f"intensities_tau_1.0.npy", I_data)
    
if (__name__ == '__main__'):
    
    main_root = "/dat/xenoss/"
    ############ NOTE CREATING FULL MULTIHEIGHT TESTSETS ##############
    # create_multiheight_full_testset()

    #############################################################################
    ############ NOTE INTENSITY GENERATION FOR BENCHMARK COMPARISON ##############
    # create_benchmarking_testset()
    # create_intensity_cubes()
    # print("Intensities for DeepVel benchmarking created.")

    # create_benchmarking_trainset()
    # print("Benchmarking trainset created.")
    #############################################################################

    ############ NOTE MULTI-HEIGHT TEST DATA GENERATION ##############
    # create_multiheight_testset()
    # print("Multi-height test set created.")
    ##################################################################

    ############ NOTE MULTI-HEIGHT DATA DENERATION ##############
    # main_save_path = "/home/xenoss/dat/multi_height_data_new"
    # create_multiheight_datacubes()
    # print("Multi-height data cubes created.")
    # create_multiheight_trainset(main_save_path)
    #########################################################################################################