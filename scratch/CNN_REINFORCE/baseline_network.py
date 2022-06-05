import numpy as np
from tensorflow import keras
import tensorflow as tf
from network_utils import build_network

class BaselineNetwork(keras.Model):
    """
    Class for implementing Baseline network
    """

    def __init__(self, env, config):
        """
        Create self.network using build_mlp, and create self.optimizer to
        optimize its parameters.
        You should find some values in the config, such as the number of layers,
        the size of the layers, and the learning rate.
        """
        super().__init__()
        self.config = config
        self.env = env
        self.lr = self.config.learning_rate

        self.network = build_network(
            self.env.observation_space.shape,
            1, name="baseline"
        )
        self.optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)

    def call(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            output: torch.Tensor of shape [batch size]

        Run the network forward and then squeeze the result so that it's
        1-dimensional. Put the squeezed result in a variable called "output"
        (which will be returned).

        Note:
        A nn.Module's forward method will be invoked if you
        call it like a function, e.g. self(x) will call self.forward(x).
        When implementing other methods, you should use this instead of
        directly referencing the network (so that the shape is correct).
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

        Note:
        The arguments and return value are numpy arrays. The np2torch function
        converts numpy arrays to torch tensors. You will have to convert the
        network output back to numpy, which can be done via the numpy() method.
        """
        predictions = self.predict(observations).numpy()
        advantages = returns - predictions
        return advantages
