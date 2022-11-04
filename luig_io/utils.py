import random

import numpy as np
import tensorflow as tf


def set_seed(seed):
    """set all seeds to the value of `seed`.

    Args:
        seed: integer to be used as the seed.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)
