from metrics_and_plotting import ModelType
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import Dataset, DataLoader, random_split
import os
import torch.optim as optim
import sys
from collections import OrderedDict
from datetime import datetime
import time
from torchinfo import summary


class dataset_deepVel(Dataset): 
    """
    Dataset class for DeepVel, modified to take two inputs: B and vz, I and vz or I and B.
    """ 
    def __init__(self, dataset_path, in_channels = 4, test_idx = None, test = False, tau = None, type = ModelType.HYBRID_Bvz):
        
        self.in_channels = in_channels
        self.type = type

        self.velocities_dir = os.path.join(dataset_path, "labels") if tau is None else os.path.join(dataset_path, f"labels/tau_{tau}")
        velocities = os.listdir(self.velocities_dir)
        velocities.sort()        
        self.velocities = velocities

        self.n_train_datapoints = len(self.velocities)
        self.test = test
        self.test_idx = test_idx
        
        if type == ModelType.HYBRID_Bvz:
            self.magnetic_fields_dir = os.path.join(dataset_path, "inputs/B") if tau is None else os.path.join(dataset_path, f"inputs/B/tau_{tau}")
            magnetic_fields = os.listdir(self.magnetic_fields_dir)
            magnetic_fields.sort()
            self.magnetic_fields = magnetic_fields

            self.vz_dir = os.path.join(dataset_path, "inputs/vz") if tau is None else os.path.join(dataset_path, f"inputs/vz/tau_{tau}")
            vz = os.listdir(self.vz_dir)
            vz.sort()
            self.vz = vz

        elif type == ModelType.HYBRID_Ivz:
            self.intensities_dir = os.path.join(dataset_path, "inputs/I") if tau is None else os.path.join(dataset_path, f"inputs/I/tau_{tau}")
            intensities = os.listdir(self.intensities_dir)
            intensities.sort()
            self.intensities = intensities

            self.vz_dir = os.path.join(dataset_path, "inputs/vz") if tau is None else os.path.join(dataset_path, f"inputs/vz/tau_{tau}")
            vz = os.listdir(self.vz_dir)
            vz.sort()
            self.vz = vz

        elif type == ModelType.HYBRID_IB:
            self.intensities_dir = os.path.join(dataset_path, "inputs/I") if tau is None else os.path.join(dataset_path, f"inputs/I/tau_{tau}")
            intensities = os.listdir(self.intensities_dir)
            intensities.sort()
            self.intensities = intensities

            self.magnetic_fields_dir = os.path.join(dataset_path, "inputs/B") if tau is None else os.path.join(dataset_path, f"inputs/B/tau_{tau}")
            magnetic_fields = os.listdir(self.magnetic_fields_dir)
            magnetic_fields.sort()
            self.magnetic_fields = magnetic_fields

    def __len__(self):

        if self.test and self.test_idx:
            return self.n_test_datapoints
            
        else:
            return self.n_train_datapoints

    def check_indices (self, idx, test=False):
        """
        Check if the input magnetic field and vz files correspond to the velocity files.
        Args:
            idx: Index of the data point to check
            test: Boolean indicating if in test mode
        Raises:
            ValueError: If the input and label data do not correspond
        """

        if test == True:
            idx = self.validation_idx[idx]

        velocity_index = self.velocities[idx].replace('velocities_', '')

        if self.type == ModelType.HYBRID_Bvz:
            mag_index = self.magnetic_fields[idx].replace('B_', '')
            vz_index = self.vz[idx].replace('vz_', '')
            if vz_index != velocity_index:
                raise ValueError("Input (vz) - label data not corresponding: ", self.vz[idx] + " "+ self.velocities[idx])
            if mag_index != velocity_index:
                raise ValueError("Input (B) - label data not corresponding: ", self.magnetic_fields[idx] + " "+ self.velocities[idx])
        
        elif self.type == ModelType.HYBRID_Ivz:
            intensity_index = self.intensities[idx].replace('intensity_', '')
            vz_index = self.vz[idx].replace('vz_', '')
            if vz_index != velocity_index:
                raise ValueError("Input (vz) - label data not corresponding: ", self.vz[idx] + " "+ self.velocities[idx])
            if intensity_index != velocity_index:
                raise ValueError("Input (I) - label data not corresponding: ", self.intensities[idx] + " "+ self.velocities[idx])
            
        elif self.type == ModelType.HYBRID_IB:
            intensity_index = self.intensities[idx].replace('intensity_', '')
            B_index = self.magnetic_fields[idx].replace('B_', '')
            if B_index != velocity_index:
                raise ValueError("Input (B) - label data not corresponding: ", self.magnetic_fields[idx] + " "+ self.velocities[idx])
            if intensity_index != velocity_index:
                raise ValueError("Input (I) - label data not corresponding: ", self.intensities[idx] + " "+ self.velocities[idx])
            

    def __getitem__(self, idx):
        """
        Get a data point from the dataset.
        Args:
            idx: Index of the data point to retrieve
        Returns:
            Tuple of (magnetic field, vz, velocity) tensors if type is HYBRID_Bvz
            or (intensity, vz, velocity) tensors if type is HYBRID_Ivz
            or (intensity, magnetic field, velocity) tensors if type is HYBRID_IB
        Raises:
            ValueError: If the input and label data do not correspond
        """

        if self.test:
            idx = self.validation_idx[idx]

        if self.type == ModelType.HYBRID_Ivz:
            I = np.load(os.path.join(self.intensities_dir, self.intensities[idx]))
            vz = np.load(os.path.join(self.vz_dir, self.vz[idx]))
            vel = np.load(os.path.join(self.velocities_dir, self.velocities[idx]))

            self.check_indices(idx, self.test)

            return I, vz, vel
        
        elif self.type == ModelType.HYBRID_Bvz:
            B = np.load(os.path.join(self.magnetic_fields_dir, self.magnetic_fields[idx]))
            vz = np.load(os.path.join(self.vz_dir, self.vz[idx]))
            vel = np.load(os.path.join(self.velocities_dir, self.velocities[idx]))

            self.check_indices(idx, self.test)

            return B, vz, vel
        
        elif self.type == ModelType.HYBRID_IB:
            I = np.load(os.path.join(self.intensities_dir, self.intensities[idx]))
            B = np.load(os.path.join(self.magnetic_fields_dir, self.magnetic_fields[idx]))
            vel = np.load(os.path.join(self.velocities_dir, self.velocities[idx]))

            self.check_indices(idx, self.test)

            return I, B, vel

class ResidualBlock(nn.Module):
    """
    Residual block definition
    """
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()
                        )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out
    
class DeepVel_net(nn.Module):
    """
        Model definition for DeepVel, modified to take magnetic field and vz as inputs or intensity and vz as inputs.
    """
    def __init__(self, in_channels, out_channels, n_filters=32, blocks=20):
        super(DeepVel_net, self).__init__()
        self.n_filters = n_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = blocks
        self.conv_in1_1 = nn.Sequential(nn.Conv2d(self.in_channels, self.n_filters, kernel_size = 3, stride=1, padding=1),
                                       nn.BatchNorm2d(self.n_filters),
                                       nn.ReLU()
                                       )
        self.conv_in2_1 = nn.Sequential(nn.Conv2d(self.in_channels, self.n_filters, kernel_size = 3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.n_filters),
                                        nn.ReLU()
                                        )
        
        self.residual2 = self.make_Reslayer(self.n_filters, self.blocks)
        self.residual3 = self.make_Reslayer(self.n_filters, self.blocks)

        self.conv_in1_2 = nn.Sequential(nn.Conv2d(self.n_filters, self.n_filters, kernel_size = 3, stride=1, padding=1), nn.BatchNorm2d(self.n_filters))
        self.conv_in2_2 = nn.Sequential(nn.Conv2d(self.n_filters, self.n_filters, kernel_size = 3, stride=1, padding=1), nn.BatchNorm2d(self.n_filters))

        self.conv3 = nn.Conv2d(2*self.n_filters, self.out_channels, kernel_size = 1, stride=1, padding=0)

    def make_Reslayer(self, out_channels, num_blocks, stride=1):
            """
            Create a sequence of residual blocks.
            Args:
                out_channels: Number of output channels for each block
                num_blocks: Number of residual blocks to create
                stride: Stride for the first block
            Returns:
                nn.Sequential containing the residual blocks
            """
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(ResidualBlock(self.n_filters, out_channels, stride))
                self.n_filters = out_channels
            return nn.Sequential(*layers)

    def forward(self, in1, in2):
        """
        Forward pass through the network.
        """
        out_in1 = self.conv_in1_1(in1)
        res_in1 = out_in1

        out_in2 = self.conv_in2_1(in2)
        res_in2 = out_in2

        out_in1 = self.residual2(out_in1)
        out_in1 = self.conv_in1_2(out_in1)
        out_in1 += res_in1

        out_in2 = self.residual3(out_in2)
        out_in2 = self.conv_in2_2(out_in2)
        out_in2 += res_in2

        feat_ensemble = torch.cat((out_in1, out_in2), dim=1)
        out = self.conv3(feat_ensemble)

        return out


class DeepVel_run(object):
    """
    Class for running the DeepVel model, modified to take two inputs.
    """
    def __init__(self, batch, dataset_path, network_path, in_shape= (4, 128, 128), out_shape= (2, 128, 128), root = None, tau = None, type = ModelType.HYBRID_Bvz):
 
        self.root = root
        self.n_filters = 32     
        self.batch_size = batch
        self.n_conv_layers = 20
        self.tau = tau
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.lr = 1e-4
        self.type = type
        self.network_path = network_path
        self.model = DeepVel_net(in_channels=self.in_shape[0], out_channels=self.out_shape[0], n_filters=self.n_filters, blocks=self.n_conv_layers).to(device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        if root:
            self.dataset = dataset_deepVel(dataset_path, in_channels=self.in_shape[0], test_idx=None, test=False, tau = self.tau, type=self.type)
            self.train_len = int(0.8*len(self.dataset))
            self.test_len = len(self.dataset) - self.train_len

            self.trainset, self.testset = random_split(self.dataset, [self.train_len, self.test_len])

            self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle = True, num_workers = 4, pin_memory=True)
            self.testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle = False, num_workers = 4, pin_memory=True)

            self.summary = summary(self.model, input_data = (torch.randn(self.batch_size, self.in_shape[0], self.in_shape[1], self.in_shape[2]), torch.randn(self.batch_size, self.in_shape[0], self.in_shape[1], self.in_shape[2])), device=device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
  

    def train(self, epochs):

        """
        Used to train DeepVel

        Args:
            epochs : number of epochs in the training

        """

        min_loss = torch.tensor(float('inf'))
        loss_list = []
        save_losses = []

        scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.96)

        with open(os.path.join(self.network_path.replace("checkpoints", ""), "architecture.txt"), "w") as f:
            print("Writing architecture to file...", os.path.join(self.network_path.replace("checkpoints", ""), "architecture.txt"))
            f.write(str(self.summary))

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for batch_i, (in1, in2, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                outputs = self.model(in1.to(device), in2.to(device))

                loss_tuple = self.criterion(outputs, y.to(device))
                if isinstance(loss_tuple, tuple):
                    loss, mse1, mse2, mse3 = loss_tuple
                else:
                    loss = loss_tuple

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                loss_list.append(loss.cpu().detach().numpy())
        
                sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f)]"
                % (
                    epoch,
                    epochs,
                    batch_i,
                    len(self.trainloader),
                    loss.cpu().detach().numpy(),
                    np.mean(loss_list),
                    )
                )
            scheduler.step()

            val_loss_list = []
            val_acc_list = []

            self.model.eval()
            for batch_i, (in1, in2, y) in enumerate(self.testloader):
                with torch.no_grad():    
                    output = self.model(in1.to(device), in2.to(device))

                val_loss_tuple = self.criterion(output, y.to(device))
                if isinstance(val_loss_tuple, tuple):
                    val_loss = val_loss_tuple[0]
                else:
                    val_loss = val_loss_tuple

                val_loss_list.append(val_loss.cpu().detach().numpy())
        

            print(' Epoch {} - Loss : {:.5f} - Validation loss : {:.5f}'.format(epoch, 
                                                                                np.mean(loss_list), 
                                                                                np.mean(val_loss_list)))
            save_losses.append([epoch, np.mean(loss_list), np.mean(val_loss_list)])
        
            compare_loss = np.mean(val_loss_list)
            is_best = compare_loss < min_loss
            print(min_loss, compare_loss)
            if is_best == True:
                print("Best_model")      
                min_loss = min(compare_loss, min_loss)
                save_path = os.path.join(self.network_path, 'DeepVel_torch_epoch_{}_{:.5f}.pt'.format(epoch,np.mean(val_loss_list)))
                model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
                torch.save(model_state, save_path)
                # torch.save(self.model.state_dict(), os.path.join(self.network_path, 'DeepVel_torch_epoch_{}_{:.5f}.pt'.format(epoch,np.mean(val_loss_list))))
            
        print('Finished Training')

        dt = datetime.now()
        date_time = dt.strftime("%m_%d_%Y_%H_%M_%S")
        dict = OrderedDict()
        dict['root'] = self.root
        dict['dataset_length'] = len(self.trainset)+len(self.testset)
        dict['channels'] = self.n_filters
        dict['Number_epochs'] = epochs
        dict['Bach_size'] = self.batch_size
        dict['Optimizer_lr'] = self.lr

        with open(os.path.join(self.network_path, 'deepvel_torch_train_params_{}.npy'.format(date_time)), 'wb') as f:
        
            np.save(f, dict)
            np.save(f, save_losses)
    
    def to_tensor(self, x):
            """
            Convert input to tensor and add batch dimension if necessary.
            """
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x.astype(np.float32))
            if x.dim() == 3:
                x = x.unsqueeze(0)
            return x
         
    def predict(self, in1, in2, saved_model):
        """
        Predict using a DeepVel saved model with two inputs: B, vz, I, vz or I, B.

        Args:
            in1 : Tensor or np.ndarray
                First input, I or B, shape [C, H, W] or [1, C, H, W]
            in2 : Tensor or np.ndarray
                Second input, vz, shape [C, H, W] or [1, C, H, W]
            saved_model: str
                Path to trained model (.pt file)
        Returns:
            output : Tensor
                Predicted velocity field, shape [out_channels, H, W]
        Raises: 
            ValueError: If input dimensions are incorrect
        """

        model_weights = torch.load(saved_model, map_location=torch.device(device))
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model_weights)
        else:
            self.model.load_state_dict(model_weights)
        self.model.eval()

        in1 = self.to_tensor(in1)
        in2 = self.to_tensor(in2)

        start = time.time()
        with torch.no_grad():
            output = self.model(in1.to(device), in2.to(device))
        end = time.time()
        print("Prediction took {0} seconds...".format(end - start))

        output = output.squeeze(0)
        return output
       
if (__name__ == '__main__'):

    #main_root = "/dat/xenoss/"

    ######### NOTE training multiheight models ########
    main_root = "/home/xenoss/dat/"
    tau_levels=[1e-1,1e-2,1e-3,1e-4]
    taus = ["1e-1", "1e-2", "1e-3", "1e-4"]

    for i, tau in enumerate(tau_levels):
        print(f"Training for tau = {tau}")
        network_path = os.path.join(main_root, f'models_multi_height/Hybrid_I_B/tau_{taus[i]}')
        data_path = os.path.join(main_root, f'multi_height_data_new/final_dataset_multiheight/train')
        deepvel_net = DeepVel_run(root = main_root, in_shape=(4, 128, 128), out_shape=(2, 128, 128), batch = 64, dataset_path = data_path, network_path = os.path.join(network_path, 'checkpoints'), tau=taus[i], type=ModelType.HYBRID_IB)
        deepvel_net.train(100)
