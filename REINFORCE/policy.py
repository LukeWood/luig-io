import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_probability as tfp

class BasePolicy:
    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: instance of a subclass of torch.distributions.Distribution

        See https://pytorch.org/docs/stable/distributions.html#distribution

        This is an abstract method and must be overridden by subclasses.
        It will return an object representing the policy's conditional
        distribution(s) given the observations. The distribution will have a
        batch shape matching that of observations, to allow for a different
        distribution for each observation in the batch.
        """
        raise NotImplementedError

    def act(self, observations):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            sampled_actions: np.array of shape [batch size, *shape of action]

        Call self.action_distribution to get the distribution over actions,
        then sample from that distribution. You will have to convert the
        actions to a numpy array, via numpy(). Put the result in a variable
        called sampled_actions (which will be returned).
        """

        distribution = self.action_distribution(observations)
        sampled_actions = tfp.Sample(distribution, shape=()).numpy()
        return sampled_actions


class CategoricalPolicy(Bkeras.Model):
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


class GaussianPolicy(keras.Model):
    def __init__(self, network, action_dim):
        """
        After the basic initialization, you should create a nn.Parameter of
        shape [dim(action space)] and assign it to self.log_stddev.
        A reasonable initial value for log_stddev is 0 (corresponding to an
        initial stddev of 1), but you are welcome to try different values.
        """
        super().__init__(self)
        self.network = network
        self.log_stddev = self.add_weight(shape=(action_dim,), initializer='zeros', trainable=True)

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
            distribution: an instance of a subclass of
                torch.distributions.Distribution representing a diagonal
                Gaussian distribution whose mean (loc) is computed by
                self.network and standard deviation (scale) is self.stddev()

        Note: PyTorch doesn't have a diagonal Gaussian built in, but you can
            fashion one out of
            (a) torch.distributions.MultivariateNormal
            or
            (b) A combination of torch.distributions.Normal
                             and torch.distributions.Independent
        """
        locs = self.network(observations)
        stddevs = self.stddev()
        distribution = tfp.distributions.MultivariateNormal(loc=locs, scale_tril=tf.linalg.diag(stddevs))
        return distribution
