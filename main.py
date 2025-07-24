import pandas as pd
import numpy as np
import torch
import time 
import torch.nn.functional as F
import torch.nn as nn
import argparse 

from Functions import Unsupervisedlearning
from Functions import CreateModel_ShapeEncoding
from Functions import CreateModel_NonlinearMapping
from Functions import CreateModel_StressDecoding
from Functions import ComputeError_peak
from Functions import ComputeError
from Functions import ComputeVonMisesStress_all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#********************************************************

ShapeDataFile = pd.read_csv("ShapeData.csv", skiprows=3)
ShapeDataFile = ShapeDataFile.to_numpy()

StressDataFile = pd.read_csv("StressData.csv")
StressDataFile = StressDataFile.to_numpy()

#********************************************************

rng = np.random.RandomState(0)

# Create IndexList as an array from 0 to 728
IndexList = np.arange(0, 729, 1)

# Initialize empty lists for various metrics
S11MAE, S11NMAE = [], []
S22MAE, S22NMAE = [], []
S12MAE, S12NMAE = [], []
VMMAE, VMNMAE = [], []
S11AE, S11APE = [], []
S22AE, S22APE = [], []
S12AE, S12APE = [], []
VMAE, VMAPE = [], []

# Initialize lists for train and test indices
IndexList_test = []
IndexList_train = []

#********************************************************
def main_loop(iterations):

    for k in range(iterations):
        # Shuffle the indices
        rng.shuffle(IndexList)
        
        # Split into training and testing sets
        idx_train = IndexList[:656]  # First 656 indices for training
        idx_test = IndexList[656:]   # Remaining 73 indices for testing

        # Store indices in lists
        IndexList_train.append(idx_train)
        IndexList_test.append(idx_test)

        # Select training and testing data
        ShapeData_train = ShapeDataFile[:, idx_train]
        ShapeData_test = ShapeDataFile[:, idx_test]
        start = time.perf_counter()

        Unsupervisedlearning (ShapeDataFile, StressDataFile, idx_train, idx_test)

        loaded_data = np.load("data.npz")

        # Access each array like this:
        Meanshape = loaded_data["Meanshape"]
        Proj = loaded_data["Proj"]
        ShapeCode_train = loaded_data["ShapeCode_train"]
        Y= loaded_data["Y2n_train"]
        L= loaded_data["L2"]
        W1= loaded_data["W1"]
        W2= loaded_data["W2"]
        S_t= loaded_data["Stress_test"]

        for n in range(ShapeData_train.shape[1]):
            ShapeData_train[:, n] = ShapeData_train[:, n] - Meanshape

        for n in range(ShapeData_test.shape[1]):
            ShapeData_test[:, n] = ShapeData_test[:, n] - Meanshape

        # Shape Encoding **********************************************************************************

        ShapeEncoder= CreateModel_ShapeEncoding(5000, Proj)

        X_tensor = torch.tensor(ShapeData_train.T, dtype=torch.float32).to(device)  # Transpose if original is (features, samples)
        X_t_tensor = torch.tensor(ShapeData_test.T, dtype=torch.float32).to(device)
        # Run the model for prediction
        with torch.no_grad():  # Ensure no gradients are computed
            X = ShapeEncoder(X_tensor)  # Convert back to NumPy  # Should be (num_samples, 3)
            X_t = ShapeEncoder(X_t_tensor)

        X=X.cpu().numpy()
        X_t=X_t.cpu().numpy()

        end = time.perf_counter()

        print(k, "Time taken for Shape encoding: ", end-start)
        
        
        #Nonlinear mapping **********************************************************************************
        start = time.perf_counter()
        Y = np.squeeze(Y)
        # Convert X, Y to PyTorch tensors and move them to GPU
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

        # Create the model and move it to GPU
        NMapper = CreateModel_NonlinearMapping(X.shape, Y.shape)

        # Define loss function and optimizer (like Keras' compile step)
        criterion = nn.MSELoss()  # Mean Squared Error
        optimizer = torch.optim.Adamax(NMapper.parameters(), lr=0.002)  # Adamax optimizer

        # Training Loop (equivalent to .fit())
        epochs = 5000
        batch_size = 100
        dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch_X, batch_Y in dataloader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)  # Move batch data to GPU
                
                optimizer.zero_grad()  # Reset gradients
                Y_pred = NMapper(batch_X)  # Forward pass
                loss = criterion(Y_pred, batch_Y)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

        # Testing (equivalent to .predict())
        X_t_tensor = torch.tensor(X_t, dtype=torch.float32).to(device)
        with torch.no_grad():  # No gradient needed during inference
            Yp = NMapper(X_t_tensor)

        # Move predictions back to CPU for NumPy conversion
        Yp = Yp.cpu().numpy()

        for n in range(64):
            Yp[:,n]=Yp[:,n] * L[n]

        Ypp = np.zeros((idx_test.size, 64, 1, 1)) 
        # Assign values from Yp to Ypp
        for n in range(idx_test.size):
            Ypp[n, :, 0, 0] = Yp[n, :]

        end = time.perf_counter()
        print(k, "Time taken for nonlinear mapping: ", end-start)
        print("Ypp shape", Ypp.shape)
        
        #Shape decoding **********************************************************************************
        start = time.perf_counter()

        Ypp_tensor = torch.tensor(Ypp, dtype=torch.float32).to(device)
        # Create the model and move it to GPU
        StressDecoder = CreateModel_StressDecoding(W1, W2)

        # Perform inference (equivalent to `predict(Ypp)`)
        with torch.no_grad():  # No need for gradients during inference
            Spp = StressDecoder(Ypp_tensor)

        # Move output to CPU and convert to NumPy
        Spp = Spp.cpu().numpy()

        # Initialize Sp with zeros instead of resizing an empty array
        Sp = np.zeros((15000, idx_test.size))

        # Loop through test indices
        for n in range(idx_test.size):
            # Extract and reshape each stress component
            tempS11 = Spp[n, 0, :, :].reshape((5000), order='C')  # Column-major reshaping
            tempS22 = Spp[n, 1, :, :].reshape((5000), order='C')
            tempS12 = Spp[n, 2, :, :].reshape((5000), order='C')

            # Assign values to the correct slices in Sp
            Sp[0:5000, n] = tempS11
            Sp[5000:10000, n] = tempS22
            Sp[10000:15000, n] = tempS12

        # Print shape for verification
        end = time.perf_counter()
        print(k, "Time taken for stress decoding: ", end-start)

        #Error computing **********************************************************************************

        # Compare ground-truth S and predicted Sp
        S11MAE_k, S11NMAE_k = ComputeError(S_t[0:5000, :], Sp[0:5000, :])
        S11MAE.append(S11MAE_k)
        S11NMAE.append(S11NMAE_k)

        S22MAE_k, S22NMAE_k = ComputeError(S_t[5000:10000, :], Sp[5000:10000, :])
        S22MAE.append(S22MAE_k)
        S22NMAE.append(S22NMAE_k)

        S12MAE_k, S12NMAE_k = ComputeError(S_t[10000:15000, :], Sp[10000:15000, :])
        S12MAE.append(S12MAE_k)
        S12NMAE.append(S12NMAE_k)

        # Peak stress error
        S11AE_k, S11APE_k = ComputeError_peak(S_t[0:5000, :], Sp[0:5000, :])
        S11AE.append(S11AE_k)
        S11APE.append(S11APE_k)

        S22AE_k, S22APE_k = ComputeError_peak(S_t[5000:10000, :], Sp[5000:10000, :])
        S22AE.append(S22AE_k)
        S22APE.append(S22APE_k)

        S12AE_k, S12APE_k = ComputeError_peak(S_t[10000:15000, :], Sp[10000:15000, :])
        S12AE.append(S12AE_k)
        S12APE.append(S12APE_k)

        VM_t=ComputeVonMisesStress_all(S_t)
        VMp=ComputeVonMisesStress_all(Sp)
        VMMAE_k, VMNMAE_k= ComputeError(VM_t, VMp)
        VMMAE.append(VMMAE_k)
        VMNMAE.append(VMNMAE_k)
        #peak stress error
        VMAE_k, VMAPE_k= ComputeError_peak(VM_t, VMp)
        VMAE.append(VMAE_k)
        VMAPE.append(VMAPE_k)

        # Print results with better formatting
        print(f"{k}")  # Ensuring consistent float format
        print(f"VM      → Mean: {np.mean(VMMAE):.6f}, Std: {np.std(VMMAE):.6f} | "
            f"Norm Mean: {np.mean(VMNMAE):.6f}, Norm Std: {np.std(VMNMAE):.6f}")
        print(f"S11     → Mean: {np.mean(S11MAE):.6f}, Std: {np.std(S11MAE):.6f} | "
            f"Norm Mean: {np.mean(S11NMAE):.6f}, Norm Std: {np.std(S11NMAE):.6f}")
        print(f"S22     → Mean: {np.mean(S22MAE):.6f}, Std: {np.std(S22MAE):.6f} | "
            f"Norm Mean: {np.mean(S22NMAE):.6f}, Norm Std: {np.std(S22NMAE):.6f}")
        print(f"S12     → Mean: {np.mean(S12MAE):.6f}, Std: {np.std(S12MAE):.6f} | "
            f"Norm Mean: {np.mean(S12NMAE):.6f}, Norm Std: {np.std(S12NMAE):.6f}")

        print(f"VMpeak  → Mean: {np.mean(VMAE):.6f}, Std: {np.std(VMAE):.6f} | "
            f"Norm Mean: {np.mean(VMAPE):.6f}, Norm Std: {np.std(VMAPE):.6f}")
        print(f"S11peak → Mean: {np.mean(S11AE):.6f}, Std: {np.std(S11AE):.6f} | "
            f"Norm Mean: {np.mean(S11APE):.6f}, Norm Std: {np.std(S11APE):.6f}")
        print(f"S22peak → Mean: {np.mean(S22AE):.6f}, Std: {np.std(S22AE):.6f} | "
            f"Norm Mean: {np.mean(S22APE):.6f}, Norm Std: {np.std(S22APE):.6f}")
        print(f"S12peak → Mean: {np.mean(S12AE):.6f}, Std: {np.std(S12AE):.6f} | "
            f"Norm Mean: {np.mean(S12APE):.6f}, Norm Std: {np.std(S12APE):.6f}")
    #********************************************************





    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A deep learning approach to estimate stress distribution: a fast and accurate surrogate of finite-element analysis")
    parser.add_argument("--i", type=int, default=2, help="number of iterations to run with different test/train indices")

    args = parser.parse_args()

    main_loop( iterations=args.i)