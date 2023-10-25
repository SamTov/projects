#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Quantum simulation libraries
from qutip import (
    basis, 
    expect, 
    mesolve, 
    qeye, 
    sigmax, 
    sigmay, 
    sigmaz, 
    tensor,
    )
import qutip
from qutip.measurement import measure

# Machine learning libraries
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Plotting libraries
import matplotlib.pyplot as plt
from rich.progress import track

# Linalg libraries
import numpy as np
import h5py as hf

# Data source
import yfinance as yf


# In[2]:


# set the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# # Load target data

# In[3]:


msft = yf.Ticker("MSFT")

full_data = msft.history(period="max")


# In[4]:


full_data


# In[5]:


data = full_data["Close"].to_numpy()


# In[6]:


data.shape


# # Run Simulation

# In[7]:


# Set the system parameters
N = 5

# initial state
state_list = [basis(2, 1)] + [basis(2, 0)] * (N - 1)
psi0 = tensor(state_list)

# Interaction coefficients
Jx = 2.0 * np.pi * np.ones(N)
Jy = 2.0 * np.pi * np.ones(N)
Jz = 2.0 * np.pi * np.ones(N)


# In[8]:


def compute_hamiltonian(t, args):
    """
    Compute the Hamiltonian at time t.
    
    Parameters
    ----------
    t : float
        Current time.
    args : dict
        System parameters in the H computation.
    """
    sx_list = args["sx_list"]
    sy_list = args["sy_list"]
    sz_list = args["sz_list"]
    Jx = args["Jx"]
    Jy = args["Jy"]
    Jz = args["Jz"]
    driving_field = args["driving"]

    N = args["N"]
    
    # Hamiltonian - Energy splitting terms
    H = 0
    for i in range(N):        
        H -= driving_field[t] * sz_list[i]

    # Interaction terms
    for n in range(N - 1):
        H += -0.5 * Jx[n] * sx_list[n] * sx_list[n + 1]
        H += -0.5 * Jy[n] * sy_list[n] * sy_list[n + 1]
        H += -0.5 * Jz[n] * sz_list[n] * sz_list[n + 1]
    
    return H
    


# In[9]:


# Setup operators for individual qubits
sx_list, sy_list, sz_list = [], [], []
for i in range(N):
    op_list = [qeye(2)] * N
    op_list[i] = sigmax()
    sx_list.append(tensor(op_list))
    op_list[i] = sigmay()
    sy_list.append(tensor(op_list))
    op_list[i] = sigmaz()
    sz_list.append(tensor(op_list))
    
args = {
    "sx_list": sx_list,
    "sy_list": sy_list,
    "sz_list": sz_list,
    "Jx": Jx,
    "Jy": Jy,
    "Jz": Jz,
    "N": N,
    "driving": data
}


# In[10]:


# Observables for state description.
state_dimension = 50

measurements = []

for _ in range(state_dimension):
    seed = np.random.randint(641)
    measurements.append(
        qutip.rand_herm(
            32, 
            0.5, 
            dims=[[2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]
        )
    )


# In[11]:


# signal_length = data.shape[0]
# repeats = 5  # Number of times to measure a single operator
# # Equilibration run
# hamiltonian = compute_hamiltonian(0, args)
# times = np.linspace(0, 1, 5)
# new_state = mesolve(hamiltonian, psi0, times, [], [], args).states[-1]

# observations = []
# for t in range(signal_length):
#     hamiltonian = compute_hamiltonian(t, args)
#     times = np.linspace(0, 1, 5)
#     time_observations = []
#     for operator in measurements:
#         for _ in range(repeats):
#             new_state = mesolve(hamiltonian, new_state, times, [], [], args).states[-1]
#             value = expect(new_state * new_state.dag(), operator)
# #             value = measure(new_state * new_state.dag(), operator)
#             time_observations.append(value)
#     observations.append(time_observations)
    
# observations = np.array(observations)


# # Fit Model

# In[4]:


full_ds = np.load("dataset.npy", allow_pickle=True)


# In[5]:


class NeuralNetwork(nn.Module):
    
    def __init__(self, state_dimension: int, output_dimension: int):
        """
        Build the network.
        
        Parameters
        ----------
        state_dimension : int
                Dimension of the state representation.
                This is the input to the layer.
        output_dimension : int
                Dimension of the output being predicted.
        """
        super().__init__()
        
        self.linear_stack = nn.Sequential(
            nn.Linear(state_dimension, 500),
            nn.ReLU(),
            nn.Linear(500, output_dimension),
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        As we are doing reservoir computing, this is 
        simply a linear layer.
        """
        return self.linear_stack(x)


# In[6]:


class ReservoirDataset(Dataset):
    """
    Custom dataset for the training.
    """
    def __init__(
        self, 
        state_data: np.ndarray, 
        function_data: np.ndarray,
        prediction_length: int
    ):
        """
        Constructor for the dataset.

        Parameters
        ----------
        state_data : np.ndarray
                State description data.
        function_data : np.ndarray
                Function data being fit.
                This will be the target data.
        prediction_length : int
                How far into the future you will predict.
        """
        self.state_data = torch.Tensor(state_data).to(device)
        self.function_data = torch.Tensor(function_data).to(device)
        
        self.norm_factor = 1 # max(abs(function_data.flatten()))
    
    def __len__(self):
        """
        Length of the dataset.
        """
        return int(
            len(self.function_data)
        )
    
    def __getitem__(self, idx: int):
        """
        Collect an item from the dataset.
        
        Parameters
        ----------
        idx : int
                Index of the state to take.
        """
        state = self.state_data[idx]
        target = self.function_data[idx] / self.norm_factor
        
        return state, target


# In[7]:


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            
    return loss


# In[8]:


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")
    
    return test_loss


# In[9]:


def train_model(input_data, target_data, model = None):
    """
    Train a model on the current data.
    """
    dataset = ReservoirDataset(
        state_data=input_data,
        function_data=target_data,
        prediction_length=1
    )
    
    if model is None:
        model = NeuralNetwork(
            state_dimension=250, 
            output_dimension=1
        ).to(device)

        model = model.type(torch.float32)

    # Use MSE loss
    loss_fn = nn.MSELoss()

    # Use ADAM optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # Create the loader
    loader = DataLoader(dataset, batch_size=50, shuffle=False)
    
    # Train the network
    epochs = 500
    train_losses = []

    for t in track(range(epochs)):
        loss = train(loader, model, loss_fn, optimizer)
        train_losses.append(loss)

    train_losses = [item.cpu().detach().numpy() for item in train_losses]
    
    return train_losses, model


# ## Backtest the strategy

# In[10]:


pre_train_inputs = full_ds[:1000, :250]
pre_train_targets = full_ds[1:1000 + 1, 250:]

start_loss, model = train_model(pre_train_inputs, pre_train_targets)

predictions = []
targets = []
losses = []

for month in range(len(full_ds[1000:])):
    # Get the real target value
    targets.append(full_ds[int(1000 + month), 250:])
    
    # Get the model prediction
    predictions.append(
        model(
            torch.Tensor(full_ds[int(1000 + month - 1), :250]).to(device)
        ).cpu().detach().numpy()
    )
    
    # Update the model
    train_inputs = full_ds[:int(1000 + month - 1), :250]
    train_targets = full_ds[1:int(1000 + month), 250:]
    loss, model = train_model(train_inputs, train_targets)
    losses.append(loss[-1])
    
    print(f"Prediction: {predictions[-1]}")
    print(f"Truth: {targets[-1]}")


# In[ ]:




