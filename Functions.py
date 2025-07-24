# here the libraries come
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#**********************************************

def im2patch(I, patchSize, stride):

    L1, L2 = patchSize  # Extracts patch dimensions
    S1, S2 = stride     # Extracts stride values
    # Get the dimensions of I
    L1max, L2max, L3max = I.shape
    # List to collect patches
    patch_list = []
    for i in range(0, L1max, S1):
        for j in range(0, L2max, S2):
            # Compute the ending indices for the patch.
            # In MATLAB: idx_1 = i + L1 - 1, idx_2 = j + L2 - 1. In Python, since i is 0-indexed,
            # the patch extends from i to i+L1 (exclusive) and j to j+L2.
            idx_1 = i + L1
            idx_2 = j + L2
            
            # Check if the patch fits entirely inside I.
            if idx_1 <= L1max and idx_2 <= L2max:
                # Extract the patch. This gives an array of shape (L1, L2, L3max)
                patch = I[i:idx_1, j:idx_2, :]
                patch_list.append(patch)

    # Optionally, stack patches into a single 4D NumPy array.
    # In MATLAB, patches are stored in patch(:,:,:,counter), i.e. the fourth dimension indexes patches.
    # We stack along a new last axis.
    if patch_list:
        patches = np.stack(patch_list, axis=-1)
    else:
        patches = np.empty((L1, L2, L3max, 0))

    return patches


def Unsupervisedlearning (ShapeDataFile, StressDataFile, IdxList_train, IdxList_test):

    data= StressDataFile
    # Extract unique dimensions
    x_vals = np.unique(data[:, 0]).astype(int)
    y_vals = np.unique(data[:, 1]).astype(int)
    z_vals = np.unique(data[:, 2]).astype(int)

    # Determine the shape of the 3D array
    X_size, Y_size, Z_size = len(x_vals), len(y_vals), len(z_vals)

    # Create an empty 3D array
    array_3d = np.full((X_size, Y_size, Z_size), np.nan)  # Use NaN to indicate missing values if any

    # Fill the 3D array
    for row in data:
        x, y, z, value = row
        x_idx = np.where(x_vals == int(x))[0][0]
        y_idx = np.where(y_vals == int(y))[0][0]
        z_idx = np.where(z_vals == int(z))[0][0]
        array_3d[x_idx, y_idx, z_idx] = value

    StressDataFile = array_3d
    
    ShapeData_train = np.asarray(ShapeDataFile[:, IdxList_train])
    #IdxList_train should be a list (defined horizontally)
    ShapeData_test = np.asarray(ShapeDataFile[:, IdxList_test])
    #IdxList_test should be a list (defined horizontally) 

    Meanshape= np.mean(ShapeData_train, axis=1)
    X = np.zeros_like(ShapeData_train)
    for i in range(ShapeData_train.shape[1]):
        X[:,i] = ShapeData_train[:,i] - Meanshape
    X = X / np.sqrt(len(IdxList_train))
    U, S, Vt = np.linalg.svd(X, full_matrices=False)         # Full_matrice might need to change to true if needed

    PC = []  # Initialize an empty list
    PC_count = 3

    for k in range(PC_count):  # Loop from 0 to PC_count-1 (Python is 0-based)
        PC.append(U[:, k])  # Append the k-th column of U
    
    Proj = np.zeros((U.shape[0], PC_count))  # Initialize a zero matrix with the correct shape
    for k in range(PC_count):
        Proj[:, k] = U[:, k] / S[k]  # Element-wise division by Lambda[k]
 
    ShapeCode_train = np.zeros((PC_count, len(IdxList_train)))
    ShapeError_train = np.zeros((1, len(IdxList_train)))

    for k in range(len(IdxList_train)):
        temp = ShapeData_train[:, k] - Meanshape
        c = np.zeros(PC_count)
        for n in range(PC_count):
            c[n] = np.sum(PC[n] * temp) / S[n]

            # Step 4: Store result in ShapeCode_train
        ShapeCode_train[:, k] = c

    ShapeCode_test = np.zeros((PC_count, len(IdxList_test)))
    ShapeError_test = np.zeros((1, len(IdxList_test)))
      
    for k in range(len(IdxList_test)):
        temp = ShapeData_test[:, k] - Meanshape
        c = np.zeros(PC_count)
        for n in range(PC_count):
            c[n] = np.sum(PC[n] * temp) / S[n]

            # Step 4: Store result in ShapeCode_train
        ShapeCode_test[:, k] = c

    # start of stress decoding phase

    StressData_train = StressDataFile[:, :, IdxList_train]
    StressData_test = StressDataFile[:, :, IdxList_test]
    # Reshape the specific slices of StressData_train (Note: 0-based indexing in Python)
    S11_train = np.reshape(StressData_train[0, :, :], (5000, len(IdxList_train)))
    S22_train = np.reshape(StressData_train[1, :, :], (5000, len(IdxList_train)))
    S12_train = np.reshape(StressData_train[3, :, :], (5000, len(IdxList_train)))

    # Initialize Sdata_train as a 4D zero array with shape (50, 100, 3, len(IdxList_train))
    Sdata_train = np.zeros((50, 100, 3, len(IdxList_train)))

    for k in range(len(IdxList_train)):  # Looping over the length of IdxList_train
    # Reshaping the respective columns of S11_train, S22_train, and S12_train
        Sdata_train[:, :, 0, k] = np.reshape(S11_train[:, k], (50, 100), order='C')
        Sdata_train[:, :, 1, k] = np.reshape(S22_train[:, k ], (50, 100), order='C')
        Sdata_train[:, :, 2, k] = np.reshape(S12_train[:, k], (50, 100), order='C')


    # Reshape the specific slices of StressData_train (Note: 0-based indexing in Python)
    S11_test = np.reshape(StressData_test[0, :, :], (5000, len(IdxList_test)))
    S22_test = np.reshape(StressData_test[1, :, :], (5000, len(IdxList_test)))
    S12_test = np.reshape(StressData_test[3, :, :], (5000, len(IdxList_test)))

    # Initialize Sdata_train as a 4D zero array with shape (50, 100, 3, len(IdxList_train))
    Sdata_test = np.zeros((50, 100, 3, len(IdxList_test)))

    for k in range(len(IdxList_test)):  # Looping over the length of IdxList_train
    # Reshaping the respective columns of S11_train, S22_train, and S12_train
        Sdata_test[:, :, 0, k] = np.reshape(S11_test[:, k], (50, 100), order='C')
        Sdata_test[:, :, 1, k] = np.reshape(S22_test[:, k], (50, 100), order='C')
        Sdata_test[:, :, 2, k] = np.reshape(S12_test[:, k], (50, 100), order='C')

    data1_list = []

    for k in range(len(IdxList_train)):
        # Extract patches from the k-th image
        tempPatch = im2patch(Sdata_train[:, :, :, k], [50, 100], [50, 100])
        
        # tempPatch is assumed to be a 4D array with shape (patch_height, patch_width, channels, num_patches)
        for n in range(tempPatch.shape[3]):
            temp = tempPatch[:, :, :, n]
            # Flatten the patch into a column vector
            data1_list.append(temp.flatten())

    # Stack all patch vectors as columns to form a 2D array.
    Data1 = np.column_stack(data1_list)
    # Convert Data1 to single precision (float32 in NumPy)
    Data1 = Data1.astype(np.float32)
    # Compute C1 as (Data1 * Data1.T) divided by the number of rows in Data1.
    C1 = (Data1 @ Data1.T) / Data1.shape[0]

    P1, L1, Vt1 = np.linalg.svd(C1, full_matrices=False) 
    L1=np.sqrt(L1)

    Ps1 = P1[:, :6400]

    # Pre-allocate the output array.
    # The resulting shape will be (10, 20, 3, 256)
    W1 = np.empty((50, 100, 3, Ps1.shape[1]), dtype=Ps1.dtype)

    # Loop over each column and reshape it.
    for k in range(Ps1.shape[1]):
        W1[:, :, :, k] = Ps1[:, k].reshape((50, 100, 3))

    # Convert Sdata_train to a torch tensor and change dimension order from (H, W, C, N) to (N, C, H, W)
    Sdata_train_torch = torch.from_numpy(Sdata_train).permute(3, 2, 0, 1).float()
    print("Sdata_train_torch shape", Sdata_train_torch.shape)
    Sdata_test_torch = torch.from_numpy(Sdata_test).permute(3, 2, 0, 1).float()

    # Convert W1 to a torch tensor and permute dimensions from (fH, fW, C, F) to (F, C, fH, fW)
    W1_torch = torch.from_numpy(W1).permute(3, 2, 0, 1).float()

    # Create a zero bias for each filter (F filters)
    bias = torch.zeros(W1_torch.shape[0])

    # Set the stride (as provided)
    stride = (50, 100)

    # Perform the convolution using PyTorch's conv2d
    # This performs: output[n, f, i, j] = sum_{c, p, q} Sdata_train_torch[n, c, i*p, j*q] * W1_torch[f, c, p, q] + bias[f]
    Y1_torch = F.conv2d(Sdata_train_torch, W1_torch, bias=bias, stride=stride)
    Y1_t_torch = F.conv2d(Sdata_test_torch, W1_torch, bias=bias, stride=stride)

    # If desired, convert the result back to a NumPy array.
    # Note: The output Y1_torch has shape (N, F, H_out, W_out).
    Y1 = Y1_torch.detach().numpy()
    print("Y1 shape", Y1.shape)

    data2_list = []
    for k in range(Y1.shape[0]):
        temp = Y1[k, :, :, :]
        data2_list.append(temp.flatten())

    Data2 = np.column_stack(data2_list)
    # Convert Data1 to single precision (float32 in NumPy)
    Data2 = Data2.astype(np.float32)
    C2 = (Data2 @ Data2.T) / Data2.shape[0]
    P2, L2, Vt2 = np.linalg.svd(C2, full_matrices=False) 
    L2=np.sqrt(L2)

    Ps2 = P2[:, :64]
    W2_torch = np.empty((Ps2.shape[1], Y1.shape[1], Y1.shape[2], Y1.shape[3]), dtype=Ps2.dtype)
    for k in range(Ps2.shape[1]):
        W2_torch[k, :, :, :] = Ps2[:, k].reshape((Y1.shape[1], Y1.shape[2], Y1.shape[3]))

    #Y1_torch= torch.from_numpy(Y1).float()
    W2_torch= torch.from_numpy(W2_torch).float()

    bias = torch.zeros(W2_torch.shape[0])
    stride = (1, 1)

    Y2_torch = F.conv2d(Y1_torch, W2_torch, bias=bias, stride=stride)
    Y2_t_torch = F.conv2d(Y1_t_torch, W2_torch, bias=bias, stride=stride)

    Y2 = Y2_torch.detach().numpy()
    Y2_t = Y2_t_torch.detach().numpy()
    # Optionally, print the shape of the output

    Y2n_train = np.zeros_like(Y2)
    for k in range(64):
        Y2n_train[:, k, :, :] = Y2[:, k, :, :] / L2[k]
    print("Y2n shape", Y2n_train.shape)
    Y2n_test = np.zeros_like(Y2_t)
    for k in range(64):
        Y2n_test[:, k, :, :] = Y2_t[:, k, :, :] / L2[k]

    Stress_train = np.vstack([S11_train, S22_train, S12_train])
    Stress_test = np.vstack([S11_test, S22_test, S12_test])

    W1= W1_torch.detach().numpy()
    W2= W2_torch.detach().numpy()
    print("W1 shape", W1.shape)
    print("W2 shape", W2.shape)

    #return Meanshape, Proj, ShapeCode_train, ShapeCode_test, Stress_train, Stress_test, Y2n_train, Y2n_test, L2, W1, W2
    #Meanshape, Proj, ShapeCode_train, ShapeCode_test, Stress_train, Stress_test, Y2n_train, Y2n_test, L2, W1, W2= Unsupervisedlearning (ShapeDataFile, StressDataFile,IdxList_train, IdxList_test)

    # Dictionary of arrays to save

    data_dict = {
        "Meanshape": Meanshape,
        "Proj": Proj,
        "ShapeCode_train": ShapeCode_train,
        "ShapeCode_test": ShapeCode_test,
        "Stress_train": Stress_train,
        "Stress_test": Stress_test,
        "Y2n_train": Y2n_train,
        "Y2n_test": Y2n_test,
        "L2": L2,
        "W1": W1,
        "W2": W2
    }

    # Save as a compressed numpy file
    np.savez_compressed("data.npz", **data_dict)


#**********************************************

def ComputeVonMisesStress_all(S_all):
    N = S_all.shape[1]  # Get the number of columns
    VM_all = np.zeros((5000, N))  # Properly initialize array

    S11_all = S_all[0:5000, :]
    S22_all = S_all[5000:10000, :]
    S12_all = S_all[10000:15000, :]

    for n in range(N):  # No need for (0, N) since range(N) is equivalent
        for k in range(5000):  # No need for (0, 5000)
            S11 = S11_all[k, n]
            S22 = S22_all[k, n]
            S12 = S12_all[k, n]
            
            VM = S11**2 + S22**2 - S11*S22 + 3*S12**2  # Corrected power notation
            VM = np.sqrt(VM)
            VM_all[k, n] = VM  # Assign computed VM stress

    return VM_all  # Properly indented
    
#**********************************************

def ComputeError(A, B):
    MAE = np.zeros(A.shape[1])  # Initialize arrays correctly
    NMAE = np.zeros(A.shape[1])

    for n in range(A.shape[1]):  # No need for (0, A.shape[1])
        a = A[:, n]
        b = B[:, n]

        c = np.abs(a - b)  # Compute absolute error
        a_abs = np.abs(a)

        a_max = np.max(a_abs[301:4700])  # Extract the max over the range
        MAE[n] = np.mean(c)
        NMAE[n] = MAE[n] / a_max if a_max != 0 else 0  # Avoid division by zero

    return MAE, NMAE  # Properly indented

#**********************************************

def ComputeError_peak(A, B):
    AE = np.zeros(A.shape[1])  # Initialize arrays correctly
    APE = np.zeros(A.shape[1])

    for n in range(A.shape[1]):  # No need for (0, A.shape[1])
        a = A[:, n]
        b = B[:, n]

        a_abs = np.abs(a)
        b_abs = np.abs(b)

        a_max = np.max(a_abs[301:4700])  # Extract max over range
        b_max = np.max(b_abs[301:4700])

        AE[n] = np.abs(a_max - b_max)
        APE[n] = AE[n] / a_max if a_max != 0 else 0  # Avoid division by zero

    return AE, APE  # Properly indented

#**********************************************

def CreateModel_ShapeEncoding(NodeCount, Proj):
    model = nn.Sequential(
        nn.Linear(NodeCount * 3, 3)  # Fully connected layer
    )

    # Initialize weights with Proj (Ensure shape matches: Proj should be [3, NodeCount * 3])
    with torch.no_grad():
        model[0].weight.data = torch.tensor(Proj.T, dtype=torch.float32)  # Transpose for PyTorch shape convention
        model[0].bias.data.zero_()  # Set bias to zero

    return model.to(device)

#**********************************************

def CreateModel_NonlinearMapping(Xshape, Yshape):
    model = nn.Sequential(
        nn.Linear(Xshape[1], 128),  # Input layer
        nn.Softplus(),  # Softplus activation

        nn.Linear(128, 128),  # Hidden layer
        nn.Softplus(),  # Softplus activation

        nn.Linear(128, Yshape[1])  # Output layer (Linear activation by default)
    )

    # Initialize weights with normal distribution
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0.0, std=0.05)  # Similar to Keras 'normal'
            nn.init.zeros_(layer.bias)  # Bias initialized to zero

    return model.to(device)

#**********************************************

def CreateModel_StressDecoding(W1_in, W2_in):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_channels=64, out_channels=6400, kernel_size=(1, 1), stride=(1, 1)),  # First layer
        nn.ConvTranspose2d(in_channels=6400, out_channels=3, kernel_size=(50, 100), stride=(50, 100))  # Second layer
    )

    # Set pre-trained weights manually
    with torch.no_grad():
        model[0].weight.data = torch.tensor(W2_in, dtype=torch.float32)  # Shape: (256, 64, 5, 5)
        model[1].weight.data = torch.tensor(W1_in, dtype=torch.float32)  # Shape: (3, 256, 10, 20)

        # Set bias to zero (if needed)
        model[0].bias.data.zero_()
        model[1].bias.data.zero_()

    return model.to(device)

#**********************************************