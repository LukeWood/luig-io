# -*- coding: UTF-8 -*-

import argparse
import random

import gym
import gym_super_mario_bros
import numpy as np
import tensorflow as tf
from config import get_config
from helpers import get_env

from policy_gradient import PolicyGradient

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)

if __name__ == "__main__":
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)
    # train model
    config = get_config(args.seed)
    env = get_env(config)
    model = PolicyGradient(env, config, args.seed)
    model.run()
