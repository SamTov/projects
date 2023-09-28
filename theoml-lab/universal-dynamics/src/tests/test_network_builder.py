"""
Module to test the network builder.
"""
import flax.linen as nn

from network_builder import *


class TestDenseNetwork:
    """
    Test the dense networks.
    """

    def test_layer_module(self):
        """
        Test that layers are built correctly.
        """
        my_layer = DenseLayer(10, nn.relu)

    def test_network(self):
        """
        Build a full network.
        """
        my_network = build_dense_network(12, 5, 2)


class TestConvNetwork:
    """
    Test the conv networks.
    """

    def test_layer_module(self):
        """
        Test that layers are built correctly.
        """
        my_layer = ConvLayer(10, 3, 2, 3, nn.relu, nn.avg_pool)

    def test_network(self):
        """
        Build a full network.
        """
        my_network = build_conv_network(12, 5, 3, 2, 3, 10)
