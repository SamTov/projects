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
from rich import print
from argparse import ArgumentParser


# set the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")


# # Load target data


class BFieldGenerator:
    def __init__(
        self,  amplitude: float, frequency: float, resolution: int = 0.01, length: int = 1000
        ):
        """
        Create the BField generator.

        Parameters
        ----------
        amplitude : float
            Amplitude of the magnetic field.
        frequency : float
            Frequency of the cosine wave.
        resoltion : float
            Distance between measured points.
        length : int
            Amount of time to run it for.
        """
        self.amplitude = amplitude
        self.frequency = frequency

        self.counter = 0

        self.measured_field = []
        self.times = []

    def __call__(self, t: float):
        """
        Get the next value of the magnetic field.

        Parameters
        ----------
        t : float
            Current time, but this is ignored.
        """
        b_field = self.amplitude * np.cos(self.frequency * t)
        self.measured_field.append(b_field)
        self.times.append(t)
        self.counter += 1

        return b_field


# # Run Simulation


@dataclass
class SimulationParameters:
    """
    Helper class for the simulation parameters.
    """
    length: int
    coupling: list

@dataclass
class SimulationState:
    """ 
    Helper class for the simulation state.
    """
    number_of_spins: int
    quantum_state: list
    spin_list: list
    coupling_list: list


def get_simulation_state(parameters: SimulationParameters):
    """
    Returns the initial state of the simulation.
    """
    # Get the initial wavefunction
    number_of_spins = parameters.length

    initial_state = [basis(2, 1)] + [basis(2, 0)] * (number_of_spins - 1)

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
    for i in [0, 1, 3]:        
        H -= driving_field(t) * sz_list[i]

    
    # Interaction terms
    for n in range(N - 1):
        H += -0.5 * Jx[n] * sx_list[n] * sx_list[n + 1]
        H += -0.5 * Jy[n] * sy_list[n] * sy_list[n + 1]
        H += -0.5 * Jz[n] * sz_list[n] * sz_list[n + 1]

    return H


# # Fit Model


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
            nn.Linear(state_dimension, 128),
            nn.ReLU(),
            nn.Linear(128, output_dimension)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        As we are doing reservoir computing, this is 
        simply a linear layer.
        """
        # return self.readout_layer(x)
        return self.linear_stack(x)


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
            loss = loss.item()
            
    return loss


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


def train_model(dataset, test_ds, observable_size: int):
    """
    Train a model on the current data.
    """    
    model = NeuralNetwork(
        state_dimension=observable_size,
        output_dimension=1
    ).to(device)

    model = model.type(torch.float32)

    # Use MSE loss
    loss_fn = nn.MSELoss()

    # Use ADAM optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Create the loader
    loader = DataLoader(dataset, batch_size=250, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=250, shuffle=False)
    
    # Train the network
    epochs = 750
    train_losses = []
    test_losses = []

    for t in track(range(epochs)):
        loss = train(loader, model, loss_fn, optimizer)
        train_losses.append(loss)
        loss = test(test_loader, model, loss_fn)
        test_losses.append(loss)

    train_losses = [item.cpu().detach().numpy() for item in train_losses]
    
    return train_losses, test_losses, model


def run_simulation(coupling_strength: float):
    """
    Helper function for running the simulations.

    Parameters
    ----------
    coupling_strength : float
        Coupling strength of the system.
    """
    # for strength in np.linspace(1, 10, 10, dtype=int):
    simulation_parameters = SimulationParameters(
        length=5,
        coupling=coupling_strength * np.pi,
    )
    simulation_state = get_simulation_state(simulation_parameters)

    field_generator = BFieldGenerator(1.0, 0.1 * np.pi, length=20)
    args = {
        "sx_list": simulation_state.spin_list[0],
        "sy_list": simulation_state.spin_list[1],
        "sz_list": simulation_state.spin_list[2],
        "Jx": simulation_state.coupling_list[0],
        "Jy": simulation_state.coupling_list[1],
        "Jz": simulation_state.coupling_list[2],
        "N": simulation_state.number_of_spins,
        "driving": field_generator
    }

    times = np.linspace(0, 500, 15000)
    results = mesolve(compute_hamiltonian, simulation_state.quantum_state, times, [], [], args)

    fit_generator = BFieldGenerator(1.0, 0.1 * np.pi, length=20)
    fit_field = fit_generator(times)[:-1]

    return results, fit_field


def compute_state_description(states: list, size: int):
    """
    Compute the state description from the states.
    """
    state_dimension = size

    measurements = []

    for _ in range(state_dimension):
        non_gue_matrix = qutip.rand_herm(
            32, 
            0.9, 
            dims=[[2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]
        )
        measurements.append(non_gue_matrix)

    observations = np.zeros((np.shape(states)[0], state_dimension))

    for t, state in enumerate(states):
        for o, operator in enumerate(measurements):
            measure_state = state
            observations[t][o] = qutip.expect(measure_state * measure_state.dag(), operator)

    return observations


def train_and_measure(observables, field_data, prediction_length):
    """
    Helper function for fitting the observables.
    """
    split_offset=7000

    train_ds = ReservoirDataset(
        state_data=observables[:split_offset, :],
        function_data=field_data[:split_offset],
        prediction_length=prediction_length
    )

    test_ds = ReservoirDataset(
        state_data=observables[split_offset:, :],
        function_data=field_data[split_offset:],
        prediction_length=prediction_length
    )

    train_losses, test_losses, model = train_model(train_ds, test_ds, observable_size=observables.shape[1])

    test_predictions = []
    test_targets = []
    for item in test_ds:
        state, target = item
        test_predictions.append(model(state).cpu().detach().numpy())
        test_targets.append(target.cpu().detach().numpy())

    test_predictions = np.array(test_predictions).reshape(-1)
    test_targets = np.array(test_targets).reshape(-1)

    train_predictions = []
    train_targets = []
    for item in train_ds:
        state, target = item
        train_predictions.append(model(state).cpu().detach().numpy())
        train_targets.append(target.cpu().detach().numpy())

    train_predictions = np.array(train_predictions).reshape(-1)
    train_targets = np.array(train_targets).reshape(-1)

    return train_losses, test_losses, train_predictions, train_targets, test_predictions, test_targets, pearsonr(test_predictions, test_targets)[0], pearsonr(train_predictions, train_targets)[0]


@dataclass
class Measurement:
    coupling_strength: float
    state_size: int
    prediction_length: int

    train_losses: list
    test_losses: list
    train_predictions: np.ndarray
    train_targets: np.ndarray
    test_predictions: np.ndarray
    test_targets: np.ndarray
    test_pearson: float
    train_pearson: float


# Create argparser to take coupling strength, state size and prediction length
parser = ArgumentParser()
parser.add_argument("--coupling", type=float, default=1.0)
parser.add_argument("--state_size", type=int, default=100)
parser.add_argument("--prediction_length", type=int, default=100)

args = parser.parse_args()  # Parse the arguments

# Run the simulation
coupling_strength = args.coupling
state_size = args.state_size
prediction_length = args.prediction_length


# Run simulation and save results
try:
    results = qutip.qload(f"/work/stovey/sine_scan/simulation_states/{coupling_strength}")
    field_data = np.load(
        "/work/stovey/sine_scan/simulation_states/{coupling_strength}_field.npy", allow_pickle=True
    )
    print(f"Loading Spin Chain with {coupling_strength} coupling strength.")
except FileNotFoundError:
    print(f"Simulating Spin Chain with {coupling_strength} coupling strength.")
    results, field_data = run_simulation(coupling_strength)
    qutip.qsave(results, f"/work/stovey/sine_scan/simulation_states/{coupling_strength}")
    np.save(f"/work/stovey/sine_scan/simulation_states/{coupling_strength}_field.npy", field_data)

# Extract states
states = results.states[1:]

# Compute observable representation and save
try:
    observables = np.load(
        f"/work/stovey/sine_scan/reservoir_measurements/{coupling_strength}_{state_size}.npy", allow_pickle=True    
    )
    print(f"Loading state description with {state_size} elements.")
except FileNotFoundError:
    print(f"Computing state description with {state_size} elements.")
    observables = compute_state_description(states, state_size)
    np.save(f"/work/stovey/sine_scan/reservoir_measurements/{coupling_strength}_{state_size}.npy", observables)

# Train and save results
print(f"Fitting readout for a prediction length {prediction_length} steps.")
(
    train_losses, 
    test_losses, 
    train_predictions,
    train_targets, 
    test_predictions, 
    test_targets, 
    test_pearson, 
    train_pearson
) = train_and_measure(
    observables, field_data, prediction_length
    )

measurement = Measurement(
    coupling_strength=coupling_strength,
    state_size=state_size,
    prediction_length=prediction_length,
    train_losses=train_losses,
    test_losses=test_losses,
    train_predictions=train_predictions,
    train_targets=train_targets,
    test_predictions=test_predictions,
    test_targets=test_targets,
    test_pearson=test_pearson,
    train_pearson=train_pearson,
)
np.save(f"/work/stovey/sine_scan/fit_results/{coupling_strength}_{state_size}_{prediction_length}.npy", measurement)
