#!/usr/bin/env python
# coding: utf-8

# # Reservoir Optimizer
# 
# A notebook to choose optimal parameters for selected reservoirs. The following parameters must be optimized:
# 
# * Coupling Strength
# * Relaxation Time
# * State Dimension
# * Measurement Points

# In[ ]:


# Quantum simulation libraries
from qutip import (
    basis, 
    mesolve, 
    qeye, 
    sigmax, 
    sigmay, 
    sigmaz, 
    tensor,
    )
import qutip

# Machine learning libraries
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Plotting libraries
import matplotlib.pyplot as plt
from rich.progress import track
import seaborn as sns

# Linalg libraries
import numpy as np
from scipy.stats import pearsonr

# Helper libraries
from dataclasses import dataclass
import pathlib


# In[ ]:


# set the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")


# In[ ]:


pathlib.Path("simulation_states").mkdir(exist_ok=True)
pathlib.Path("reservoir_measurements").mkdir(exist_ok=True)
pathlib.Path("fit_results").mkdir(exist_ok=True)


# ## Reservoir Simulation

# In[ ]:


@dataclass
class ExperimentParameters:
    """
    Helper class for the simulation parameters.
    """
    length: int
    coupling: list
    driving_field: np.ndarray
    relaxation_time: int
    state_dimension: int

@dataclass
class SimulationState:
    """ 
    Helper class for the simulation state.
    """
    number_of_spins: int
    quantum_state: list
    spin_list: list
    coupling_list: list


# In[ ]:


def get_simulation_state(parameters: ExperimentParameters):
    """
    Returns the initial state of the simulation.
    """
    # Get the initial wavefunction
    number_of_spins = parameters.length
    initial_state = []
    for i in range(number_of_spins):
        initial_state.append(
            basis(2, np.random.randint(0, 2))
        )

    # Setup operators for individual qubits
    sx_list, sy_list, sz_list = [], [], []
    for i in range(number_of_spins):
        op_list = [qeye(2)] * number_of_spins
        op_list[i] = sigmax()
        sx_list.append(tensor(op_list))
        op_list[i] = sigmay()
        sy_list.append(tensor(op_list))
        op_list[i] = sigmaz()
        sz_list.append(tensor(op_list))

    # Setup the operators for the coupling
    Jx = parameters.coupling * np.ones(number_of_spins)
    Jy = parameters.coupling * np.ones(number_of_spins)
    Jz = parameters.coupling * np.ones(number_of_spins)    

    return SimulationState(
        number_of_spins=number_of_spins,
        quantum_state=tensor(initial_state),
        spin_list=[sx_list, sy_list, sz_list],
        coupling_list=[Jx, Jy, Jz],
    )


# In[ ]:


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

    # Magnetic field terms to top row
    H = 0
    for i in [0, 2, 3]:        
        H -= driving_field[t] * sz_list[i]

    
    # Interaction terms
    for n in range(N - 1):
        H += -0.5 * Jx[n] * sx_list[n] * sx_list[n + 1]
        H += -0.5 * Jy[n] * sy_list[n] * sy_list[n + 1]
        H += -0.5 * Jz[n] * sz_list[n] * sz_list[n + 1]

    return H


# In[ ]:


def run_simulation(parameters: ExperimentParameters):
    """Run the Simulation"""
    simulation_state = get_simulation_state(parameters)

    args = {
        "sx_list": simulation_state.spin_list[0],
        "sy_list": simulation_state.spin_list[1],
        "sz_list": simulation_state.spin_list[2],
        "Jx": simulation_state.coupling_list[0],
        "Jy": simulation_state.coupling_list[1],
        "Jz": simulation_state.coupling_list[2],
        "N": simulation_state.number_of_spins,
        "driving": parameters.driving_field
    }

    signal_length = parameters.driving_field.shape[0]

    # Equilibration run
    hamiltonian = compute_hamiltonian(0, args)
    times = np.linspace(0, parameters.relaxation_time, parameters.relaxation_time)
    new_state = mesolve(hamiltonian, simulation_state.quantum_state, times, [], [], args).states[-1]
    states = []
    for t in track(range(signal_length), description="Running Simulation"):
        hamiltonian = compute_hamiltonian(t, args)
        new_state = mesolve(hamiltonian, new_state, times, [], [], args).states[-1]
        states.append(new_state)

    qutip.qsave(states, f"simulation_states/{parameters.coupling}_{parameters.relaxation_time}")

    return states


# ## Reservoir Measurement

# In[ ]:


def generate_gue_matrix(size, dims):
    # Generate a random complex matrix with Gaussian-distributed entries
    real_part = np.random.normal(scale=1/np.sqrt(2), size=(size, size))
    imag_part = np.random.normal(scale=1/np.sqrt(2), size=(size, size))
    random_matrix = real_part + 1j * imag_part

    # Symmetrize the matrix to make it Hermitian
    hermitian_matrix = (random_matrix + random_matrix.conj().T) / 2

    return qutip.Qobj(hermitian_matrix, dims=dims)


# In[ ]:


def perform_system_measurement(
        states: list, parameters: ExperimentParameters
    ):
    """
    Perform the measurements of the system.
    """
    measurements = []
    tensor_structure = [[2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]
    size = np.prod(tensor_structure[0])

    for _ in range(parameters.state_dimension):
        gue_matrix = generate_gue_matrix(size, tensor_structure)

        measurements.append(gue_matrix)

    # Compute observables
    observations = np.zeros((np.shape(states)[0], parameters.state_dimension))

    for t, state in enumerate(states):
        for o, operator in enumerate(measurements):
            measure_state = state
            observations[t][o] = qutip.expect(measure_state * measure_state.dag(), operator)

    np.save(
        f"reservoir_measurements/{parameters.coupling}_{parameters.state_dimension}_{parameters.relaxation_time}.npy", 
        observations
    )

    return observations


# ## Readout Layer Fitting

# In[ ]:


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
            nn.Linear(state_dimension, output_dimension),
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        As we are doing reservoir computing, this is 
        simply a linear layer.
        """
        return self.linear_stack(x)


# In[ ]:


class ReservoirDataset(Dataset):
    """
    Custom dataset for the training.
    """
    def __init__(
        self, 
        state_data: np.ndarray, 
        prediction_length: int,
        function_data: np.ndarray,
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
        self.state_data = torch.Tensor(state_data[:-prediction_length]).to(device)

        self.function_data = torch.Tensor(
            function_data[prediction_length:].reshape(-1, 1)
        ).to(device)
        
        self.norm_factor = 1 # max(abs(function_data.flatten()))

    def split_data(self, train_size: float):
        """
        Split the data into training and validation sets.
        
        Parameters
        ----------
        train_size : float
                Size of the training set.
        """
        train_size = int(train_size * len(self))
        val_size = len(self) - train_size
        return torch.utils.data.random_split(self, [train_size, val_size])
    
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


def train(dataloader, model, loss_fn, optimizer):
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


# In[ ]:


def test(dataloader, model, loss_fn) -> np.ndarray:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    
    return test_loss


# In[ ]:


def train_model(dataset, test_ds, parameters: ExperimentParameters):
    """
    Train a model on the current data.
    """    
    model = NeuralNetwork(
        state_dimension=parameters.state_dimension,
        output_dimension=1
    ).to(device)

    model = model.type(torch.float32)

    # Use MSE loss
    loss_fn = nn.MSELoss()

    # Use ADAM optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Create the loader
    loader = DataLoader(dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=50, shuffle=True)
    
    # Train the network
    epochs = 300
    train_losses = []
    test_losses = []

    for t in track(range(epochs)):
        loss = train(loader, model, loss_fn, optimizer)
        train_losses.append(loss)
        loss = test(test_loader, model, loss_fn)
        test_losses.append(loss)

    train_losses = [item.cpu().detach().numpy() for item in train_losses]
    
    return train_losses, test_losses, model


# In[ ]:


def fit_readout_layer(reservoir_representations: np.ndarray, parameters: ExperimentParameters):
    """
    Fit the readout layer
    """
    # observables = np.load(f"{strength}_observations.npy", allow_pickle=True)
    train_ds = ReservoirDataset(
        state_data=reservoir_representations[:700, :],
        function_data=parameters.driving_field[:700],
        prediction_length=1
    )

    test_ds = ReservoirDataset(
        state_data=reservoir_representations[700:, :],
        function_data=parameters.driving_field[700:],
        prediction_length=1
    )

    train_losses, test_losses, model = train_model(train_ds, test_ds, model=None)

    test_predictions = []
    test_targets = []
    for item in test_ds:
        state, target = item
        test_predictions.append(model(state).cpu().detach().numpy())
        test_targets.append(target.cpu().detach().numpy())

    train_predictions = []
    train_targets = []
    for item in train_ds:
        state, target = item
        train_predictions.append(model(state).cpu().detach().numpy())
        train_targets.append(target.cpu().detach().numpy())

    
    np.save(
        f"fit_results/{parameters.coupling}_{parameters.state_dimension}_{parameters.relaxation_time}.npy",
        [train_losses, test_losses, train_predictions, test_predictions]
        )


# ## Experiment

# In[ ]:


driving_field = np.sin(2 * np.pi * np.linspace(0, 5, 5000))


# In[ ]:


parameters = ExperimentParameters(
    coupling=0.1,
    driving_field=driving_field,
    state_dimension=2,
    relaxation_time=5
)


# In[ ]:


reservoir_states = run_simulation(parameters)
reservoir_measurements = perform_system_measurement(reservoir_states, parameters)
fit_readout_layer(reservoir_measurements, parameters)

