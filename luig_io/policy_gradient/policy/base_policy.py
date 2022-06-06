from tensorflow import keras


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
