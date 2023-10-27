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
from qutip.measurement import measure, measure_observable, measure_povm, measurement_statistics_observable

# Machine learning libraries
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Plotting libraries
import matplotlib.pyplot as plt

# Linalg libraries
import numpy as np
import h5py as hf

# Helpers
from rich.progress import track


# Set the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# # Load target data

with hf.File("trajectory.hdf5", "r") as db:
    trajectory = db["Wanted_Positions"][:]

# # Run Simulation

# Set the system parameters
N = 5

# initial state
state_list = [basis(2, 1)] + [basis(2, 0)] * (N - 1)
psi0 = tensor(state_list)

# Interaction coefficients
Jx = 2.0 * np.pi * np.ones(N)
Jy = 2.0 * np.pi * np.ones(N)
Jz = 2.0 * np.pi * np.ones(N)

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
        H -= driving_field[t][0] * sz_list[i]
        H -= driving_field[t][1] * sz_list[i]
        H -= driving_field[t][2] * sz_list[i]

    # Interaction terms
    for n in range(N - 1):
        H += -0.5 * Jx[n] * sx_list[n] * sx_list[n + 1]
        H += -0.5 * Jy[n] * sy_list[n] * sy_list[n + 1]
        H += -0.5 * Jz[n] * sz_list[n] * sz_list[n + 1]
    
    return H

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
    "driving": trajectory
}

# Observables for state description.
state_dimension = 50

measurements = []

for _ in range(state_dimension):
    seed = np.random.randint(641)
    measurements.append(
        qutip.rand_herm(
            32, 
            dims=[[2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]
        )
    )

# Perform the simulation

signal_length = trajectory.shape[0]
repeats = 5  # Number of times to measure a single operator

# Equilibration run
hamiltonian = compute_hamiltonian(0, args)
times = np.linspace(0, 1, 5)
new_state = mesolve(hamiltonian, psi0, times, [], [], args).states[-1]

observations = []
for t in track(range(signal_length), description="Running Simulation"):
    hamiltonian = compute_hamiltonian(t, args)
    times = np.linspace(0, 1, 5)
    time_observations = []
    for operator in measurements:
        for _ in range(repeats):
            new_state = mesolve(hamiltonian, new_state, times, [], [], args).states[-1]
            eigs, states, probs = measurement_statistics_observable(new_state * new_state.dag(), operator)
            i = np.random.choice(range(len(eigs)), p=np.real(probs) / np.real(probs).sum())
            new_state = states[i]
            value = eigs[i]
            time_observations.append(value)
    observations.append(time_observations)
    
observations = np.array(observations)
np.save("observables.npy", observations)

# full_ds = np.hstack((observations, target_trajectory))


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
        self.state_data = state_data
        self.function_data = function_data
        
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


# In[ ]:


# Compile the neural network.
model = NeuralNetwork(
    state_dimension=250, 
    output_dimension=3
).to(device)

model = model.type(torch.float64)

# Use MSE loss
loss_fn = nn.MSELoss()

# Use ADAM optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Short model summary for sanity check.
print(model)


# In[ ]:


training_data = ReservoirDataset(
    state_data=train_ds[:, :250],
    function_data=train_ds[:, 250:],
    prediction_length=5
)
test_data = ReservoirDataset(
    state_data=test_ds[:, :250],
    function_data=test_ds[:, 250:],
    prediction_length=5
)


# In[ ]:


training_loader = DataLoader(training_data, batch_size=10, shuffle=False)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)


# In[ ]:


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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    return loss


# In[ ]:


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


# In[ ]:


epochs = 3000
train_losses = []
test_losses = []

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss = train(training_loader, model, loss_fn, optimizer)
    train_losses.append(loss)
    loss = test(test_loader, model, loss_fn)
    test_losses.append(loss)
print("Done!")
train_losses = [item.cpu().detach().numpy() for item in train_losses]


# In[ ]:


plt.plot(test_losses)
plt.yscale("log")


# In[ ]:


predictions = []
for point in observations:
    X = torch.Tensor(point).to(device).double()
    predictions.append(model(X).cpu().detach().numpy())

predictions = np.array(predictions)


# In[ ]:


plt.plot(trajectory[:, 0], trajectory[:, 1])
plt.plot(predictions[:, 0], predictions[:, 1])


# In[ ]:


plt.plot(target_trajectory[:, 0], target_trajectory[:, 1])
plt.plot(predictions[:, 0], predictions[:, 1])


# In[ ]:




