# REINFORCE (aka Vanilla Policy Gradient)
[The REINFORCE algorithm](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf), or Vanilla Policy Gradient, is a Policy Gradient method (duh).

The algorithm has poor sample efficiency, but comes with some nice mathematical guarantees regarding convergence.
My policy gradient implementation consists of the following components:

- [CategoricalPolicy with a custom `train_step()`](../luig_io/policy_gradient/policy/categorical_policy.py)
- [Custom trainer](policy_gradient.py)
- Video logging

My implementation relies on `tensorflow_probability`, `Keras`, and `gym`.

## Running the code
First you'll need to setup the code by following the quickstart in the [root README](https://github.com/lukewood/luig-io).

Then you can run:
```bash
python main.py
```

## Challenges

I faced two massive challenges in implementing the REINFORCE algorithm.
This section serves to document both of these in the hope that future implementations do
not become stuck on the same issues.

### tfp.distributions.Distribtion.sample([nan, ...]).sample() returns invalid result

[I opened a GitHub issue on TensorFlow probability.](https://github.com/tensorflow/probability/issues/1571)

My first issue in implementing the algorithm came when I received an error from my gym environment stating that the agent selected an invalid action.
This was a symptom of the fact that my policy gradient was exploding loss was exploding, ultimately causing my model to have `NaN` values in it.

While this wasn't the root cause, it did take me quite a bit of debugging time to locate how the model was producing invalid results.

A full report of the root cause is available in the next section.

### Loss Explosion

During my training, my loss would sky rocket in the negative direction.  The following
code was responsible for my loss function:

```
def train_step(self, data):
    """train_step runs via `model.fit()`.

    It accepts x in the form of observations, and y in the form of a tuple of
    the actions and advantages
    """
    observations, (actions, advantages) = data
    with tf.GradientTape() as tape:
        log_probs = self.action_distribution(observations).log_prob(actions)
        loss = log_probs * advantages
        loss = -tf.math.reduce_sum(loss, axis=-1)
        # Make sure to add regularization losses
        loss += sum(self.network.losses)

    grads = tape.gradient(loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

    return {"loss": loss}
```

When adding `tf.print(log_probs)` to the line below the variable assignment I noticed
that some of the values trended towards `-infinity`.

*Warning this is just my own reasoning!*

I reckon that this is due to the following circumstances:

- a negative reward consistently in the `advantages` for one of the actions
- the model begins to perform gradient descent on `-(log_probs * advantages)`, which causes that specific actions' probability to converge to 0
- `lim(log(x)), x->0` is `-infinity`, so the model begins to over-optimize for this parameter, as this allows the negative loss to sky-rocket
- this causes the parameters in every point in the model to eventually become NaN or infinity

*Some solutions to this:*

I reasoned that by limiting the `log_prob` term to an absolute minimum, such as `-50` the model will still be able to learn to produce an infintesmally small probability for these actions.  Keep in mind that:

```
np.log(0.0000000000000000000001)
> -50
```

Yep, so the model will still be able to learn to give probabilities of `0.0000000000000000000001` to some actions even with this constraint.

I implemented this with the following:

```
log_probs = self.action_distribution(observations).log_prob(actions)
log_probs = tf.math.maximum(log_probs, -50)
```
