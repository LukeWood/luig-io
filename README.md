![luig-io](media/title.png)

![mario](media/resized_mario.png)

luig-io is a collection of deep reinforcement learning algorithms
implemented using the Keras library.
My implementations focus on readability, following Keras best practices, and producing
idiomatic abstractions.

Due to the challenging nature of Reinforcement Learning, `luig-io` implementations tend to
rely on low level Keras constructs, such as `train_step()` overriding.

The algorithms are geared at
solving the [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)
environment.

## Quickstart

This section shows how to begin working with the codebase.

First, you will need to install the `luig_io` library:

```bash
python setup.py develop
```

To verify that the installation worked, lets run the `policy_gradient` algorithm:

```bash
python policy_gradient/main.py
```

If your environment is properly configured, you will begin to see log statements.

More information about each algorithm and it's corresponding setup
can be found each directories' corresponding README.

## Project Structure

The project is structured into a [core library, `luig_io`](luig_io/) and algorithms.
The `luig_io` package contains helpers and share utilities.
These include simple models, such as a simple 3 layer: `SimpleCNN`, [gym wrappers](luig_io/wrappers) to implement `FrameStack`, `GrayScale`, and `Resize` operations.
These are used across all environments.

The [`luig_io`](luig_io/) package also contains the agents and their components.

The entrypoints contain code to run the actual models.  This includes helpers, environment loading and wrapping to prepare for the algorithm, training code, sample collection code, and anything else needed to run an algorithm.

## Algorithms
The repo contains the following algorithms

- [REINFORCE (aka vanilla Policy Gradient)](policy_gradient/)
- More in progress!

## Contributing

Contributions are welcome.  Send them over via a PR.

## Thanks for Checking out Luig-IO!

If any code in this repository is useful in any of your research, please consider citing the codebase:

```
@misc{wood2022luigio,
  title={Luig-IO},
  author={Wood, Luke},
  year={2022},
  howpublished={\url{https://github.com/lukewood/luig-io}},
}
```
