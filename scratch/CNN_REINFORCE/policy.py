import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_probability as tfp
from general import get_logger

class BasePolicy(keras.Model):
    def action_distribution(self, observations):
        """
        Args:
            observations: array-like of shape [batch_size, *]
        Returns:
            distribution: a tensorflow_probability.distributions.Distribution.
        """
        raise NotImplementedError

    def act(self, observations):
        """
        Args:
            observations: np.array of shape [batch_size, *]
        Returns:
            sampled_actions: np.array of shape [batch_size, num_actions]
        """
        distribution = self.action_distribution(observations)
        return distribution.sample().numpy()


class CategoricalPolicy(BasePolicy):
    def __init__(self, network):
        super().__init__(self)
        self.network = network

    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: torch.distributions.Categorical where the logits
                are computed by self.network

        See https://pytorch.org/docs/stable/distributions.html#categorical
        """
        predictions = self.network(observations)
        distribution = tfp.distributions.categorical.Categorical(probs=None, logits=predictions)
        return distribution


class GaussianPolicy(BasePolicy):
    def __init__(self, network, action_dim):
        """
        After the basic initialization, you should create a nn.Parameter of
        shape [dim(action space)] and assign it to self.log_stddev.
        A reasonable initial value for log_stddev is 0 (corresponding to an
        initial stddev of 1), but you are welcome to try different values.
        """
        super().__init__(self)
        self.network = network
        self.log_stddev = self.add_weight(
            shape=(action_dim,),
            initializer='zeros',
            trainable=True
        )

    def stddev(self):
        """
        Returns:
            stddev: torch.Tensor of shape [dim(action space)]

        The return value contains the standard deviations for each dimension
        of the policy's actions. It can be computed from self.log_stddev
        """
        stddev = tf.exp(self.log_stddev)
        return stddev

    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: distribution representing the Gaussian action
                distribution based on the locs predicted by self.network and the
                standard deviations of `self.stddev`.
        """
        locs = self.network(observations)
        stddevs = self.stddev()
        return tfp.distributions.MultivariateNormalTriL(
            loc=locs,
            scale_tril=tf.linalg.diag(stddevs)
        )
