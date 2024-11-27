
# NN Libraries
import jax
import jax.numpy as np
import neural_tangents as nt
import optax


def get_ntk_function(apply_fn):
    empirical_ntk = nt.empirical_ntk_fn(
        f=apply_fn, 
        trace_axes=(-1,),
        implementation=nt.NtkImplementation.JACOBIAN_CONTRACTION
    )
    
    return jax.jit(empirical_ntk)


def compute_entropy(matrix: np.ndarray):
    """
    Compute the entropy observable.
    """
    values, vectors = np.linalg.eigh(matrix)
    values /= values.sum()
    values = np.clip(values, 1e-8, None)
    
    return (-1 * values * np.log(values)).sum()


def compute_trace(matrix: np.ndarray):
    """
    Compute the trace observable
    """
    return np.trace(matrix)

def compute_loss(state, dataset):
    """ Compute loss on a dataset."""
    logits = state.apply_fn({'params': state.params}, dataset['inputs']
    )

    loss = optax.softmax_cross_entropy(logits, dataset["targets"]).mean()

    return loss

def compute_accuracy(state, dataset):
    logits = jax.nn.softmax(
        state.apply_fn({'params': state.params}, dataset['inputs'])
    )