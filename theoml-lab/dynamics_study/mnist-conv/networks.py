"""
Helper functions for creating networks.
"""
# Neural network libraries
import flax.linen as nn

# Helper libraries
from dataclasses import dataclass


# Perceptron
class Perceptron(nn.Module):
    """
    Perceptron generator.
    """
    # Attributes
    width: int
    activation: callable = nn.relu
    bias: bool = True

    # Methods
    @nn.compact
    def __call__(self, x):
        """
        Forward pass.
        """
        # Dense layer
        x = nn.Dense(self.width, use_bias=self.bias)(x)
        x = self.activation(x)

        return nn.Dense(2)(x)


# Dense network
class DenseNetwork(nn.Module):
    """
    Dense network generator.
    """
    # Attributes
    width: int
    depth: int
    activation: callable = nn.relu
    bias: bool = True

    # Methods
    @nn.compact
    def __call__(self, x):
        """
        Forward pass.
        """
        # Flatten
        x = x.reshape((x.shape[0], -1))
        
        # Dense layers
        for _ in range(self.depth):
            x = nn.Dense(self.width, use_bias=self.bias)(x)
            x = self.activation(x)

        return nn.softmax(nn.Dense(10)(x))
    

@dataclass
class PoolOp:
    """
    Helper class for pooling.
    """
    pool_op: callable
    window_shape: tuple
    strides: tuple
    
# Convolutional network
class ConvolutionalNetwork(nn.Module):
    """
    Convolutional network generator.
    """
    # Attributes
    width: int
    depth: int
    activation: callable = nn.relu
    pool: PoolOp = PoolOp(nn.avg_pool, (2, 2), (2, 2))
    bias: bool = True

    # Methods
    @nn.compact
    def __call__(self, x):
        """
        Forward pass.
        """
        # Convolutional layers
        for _ in range(self.depth):
            x = nn.Conv(self.width, kernel_size=self.pool.window_shape, use_bias=self.bias)(x)
            x = self.activation(x)
            x = self.pool.pool_op(
                x, 
                window_shape=self.pool.window_shape, 
                strides=self.pool.strides
            )
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = self.activation(x)
        
        return nn.Dense(10)(x)