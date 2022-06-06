import numpy as np
from tensorflow import keras
import tensorflow as tf
from network_utils import build_network


class BaselineNetwork(keras.Model):
    """
    keras.Model implementing a Baseline network.

    In policy gradient methods, the BaselineNetwork predicts the expected future returns
    from a given environment state.  This is used to normalize the current action, and
    allows the policy network to focus on learning the difference between each action,
    instead of both the differences between each action and the future returns based on
    the current state.

    Args:
        network: underlying neural network to use as the BaseLine.
    """

    def __init__(self, network, **kwargs):
        super().__init__(**kwargs)
        self.network = network

    def call(self, observations):
        """
        Args:
            observations: tf.Tensor of shape [batch size, dim(observation space)]
        Returns:
            output: tf.Tensor of shape [batch size]
        """
        output = tf.squeeze(self.network(observations), axis=-1)
        return output

    def calculate_advantage(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size]
                all discounted future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]

        Evaluate the baseline and use the result to compute the advantages.
        Put the advantages in a variable called "advantages" (which will be
        returned).
        """
        predictions = self.predict(observations)
        advantages = returns - predictions
        return advantages
