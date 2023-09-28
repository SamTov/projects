"""
Module for building networks.
"""
from functools import partial

import flax.linen as nn
import jax

# ############## #
# Dense Networks #
# ############## #

# Layer Templates
class DenseLayer(nn.Module):
    """
    Template for a dense layer.
    """

    width: int
    activation: callable
    use_bias: bool

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.width, use_bias=self.use_bias)(x)
        return self.activation(x)


class DenseNetwork(nn.Module):
    """
    Template for a dense network.
    """

    # Hidden layer parameters
    width: int
    depth: int
    activation: callable
    use_bias: bool

    # Output parameters
    output_size: int

    def setup(self):
        """
        Create the network architecture.
        """
        self.hidden_layers = [
            DenseLayer(self.width, self.activation, self.use_bias) for _ in range(self.depth)
        ]

    @nn.compact
    def __call__(self, x):
        """
        Call the network.
        """
        for item in self.hidden_layers:
            x = item(x)

        # Output layer at the end.
        x = nn.Dense(self.output_size, use_bias=self.use_bias)(x)

        if self.depth == 0:
            return nn.sigmoid(x)
        else:
            return x


# Create a network
def build_dense_network(
    width: int,
    depth: int,
    output_dimension: int,
    use_bias: bool,
    activation: callable = nn.relu,
) -> nn.Module:
    """
    Create a dense neural network of uniform architecture.

    Parameters
    ----------
    width : int
        Width of the network.
    depth : int
        Number of layers in the network.
    activation : callable (default = ReLu)
        Activation function to use.
    output_dimension : int
        Output dimension of the network.
    use_bias : bool
        Use bias.

    Returns
    -------
    network : nn.Module
        A flax neural network.
    """
    return DenseNetwork(width, depth, activation, use_bias, output_dimension)


# ############# #
# Conv Networks #
# ############# #

# Layer Templates
class ConvLayer(nn.Module):
    """
    Template for a conv layer.
    """

    width: int
    kernel: int
    stride: int
    window: int
    activation: callable
    pool_op: callable

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.width, kernel_size=(self.kernel, self.kernel))(x)
        x = self.activation(x)
        x = self.pool_op(
            x,
            window_shape=(self.window, self.window),
            strides=(self.stride, self.stride),
        )

        return x


class ConvNetwork(nn.Module):
    """
    Template for a conv network.
    """

    # Hidden layer parameters
    width: int
    depth: int
    kernel: int
    stride: int
    window: int
    activation: callable
    pool_op: callable

    # Output parameters
    output_size: int

    def setup(self):
        """
        Create the network architecture.
        """
        self.hidden_layers = [
            ConvLayer(
                self.width,
                self.kernel,
                self.stride,
                self.window,
                self.activation,
                self.pool_op,
            )
            for _ in range(self.depth)
        ]

    @nn.compact
    def __call__(self, x):
        """
        Call the network.
        """
        for item in self.hidden_layers:
            x = item(x)

        # Output layer at the end.
        return nn.Dense(self.output_size)(x)


# Create a network
def build_conv_network(
    width: int,
    depth: int,
    kernel_size: int,
    strides: int,
    window_size: int,
    output_dimension: int,
    activation: callable = nn.relu,
    pool_op: callable = nn.max_pool,
) -> nn.Module:
    """
    Create a convolutional neural network of uniform architecture.

    Parameters
    ----------
    width : int
        Width of the network.
    depth : int
        Number of layers in the network.
    activation : callable (default = ReLu)
        Activation function to use.
    output_dimension : int
        Output dimension of the network.
    kernel_size : int
        Size of the kernel to use.
    strides : int
        Number of strides in the network pooling.
    window_size : int
        Size of the pooling window.
    pool_op : callable (default = max pooling)
        Pooling operation to apply.

    Returns
    -------
    network : nn.Module
        A flax neural network.
    """
    return ConvNetwork(
        width,
        depth,
        kernel_size,
        strides,
        window_size,
        activation,
        pool_op,
        output_dimension,
    )
