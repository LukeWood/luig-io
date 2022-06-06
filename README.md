![luig-io](media/title.png)

luig-io is a collection of deep reinforcement learning algorithms
implemented using the Keras library.  The algorithms are geared at
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


## Algorithms
The repo contains the following algorithms

- [REINFORCE (aka vanilla Policy Gradient)](policy_gradient/)
- More in progress!

## Project Structure

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
