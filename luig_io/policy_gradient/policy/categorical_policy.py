import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from luig_io.policy_gradient.policy import BasePolicy


class CategoricalPolicy(BasePolicy):
    def __init__(self, network, lowest_log=-50.0):
        super().__init__(self)
        self.network = network
        # np.log(0.0000000000000000000001)==-50
        self.lowest_log = lowest_log

    def action_distribution(self, observations):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            distribution: tensorflow_probability.distributions.Categorical where the
                logits are computed by self.network
        """
        predictions = self.network(observations)
        distribution = tfp.distributions.categorical.Categorical(
            logits=predictions, allow_nan_stats=True, validate_args=True
        )
        return distribution

    def train_step(self, data):
        """train_step runs via `model.fit()`.

        It accepts x in the form of observations, and y in the form of a tuple of
        the actions and advantages
        """
        observations, (actions, advantages) = data
        with tf.GradientTape() as tape:
            log_probs = self.action_distribution(observations).log_prob(actions)

            if self.lowest_log is not None:
                log_probs = tf.math.maximum(log_probs, self.lowest_log)

            loss = log_probs * advantages
            loss = -tf.math.reduce_mean(loss, axis=-1)
            # Make sure to add regularization losses
            loss += sum(self.network.losses)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {"loss": loss}
