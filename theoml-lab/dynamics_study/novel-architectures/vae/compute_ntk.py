import jax.numpy as np
import jax
from jax import vmap, jit, grad, jacfwd, jacrev


def get_ntk_fn(apply_fn):
    """
    Create and return an NTK function for a given apply_fn.
    """
    # @jit
    def ntk_fn(params, x1, x2):
        """
        Compute the NTK for a given apply_fn and input data.
        """
        jacobian_fun = jit(jacrev(apply_fn, argnums=0))
        jac1 = jacobian_fun(params, x1)
        print(jac1)
        jac2 = jacobian_fun(params, x2)
        return np.tensordot(jac1, jac2, axes=([0, 2], [0, 2]))
    
    return ntk_fn

def full_ntk_matrix(params, data, ntk_fn, batch_size=10):
    """ Compute the full NTK matrix for all data points """
    num_points = data.shape[0]
    ntk_matrix = np.zeros((num_points, num_points))
    
    for i in range(0, num_points, batch_size):
        for j in range(0, num_points, batch_size):
            batch1 = data[i:i + batch_size]
            batch2 = data[j:j + batch_size]
            ntk_block = ntk_fn(params, batch1, batch2)
            ntk_matrix = ntk_matrix.at[i:i + batch_size, j:j + batch_size].set(ntk_block)
    
    return ntk_matrix
