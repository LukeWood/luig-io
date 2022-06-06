# -*- coding: UTF-8 -*-

import argparse
import numpy as np
import gym
from policy_gradient import PolicyGradient
from config import get_config
import random
import gym_super_mario_bros
from helpers import get_env
import tensorflow as tf

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
    print("action space", env.action_space)
    print("action_space", env.action_space.n)
    model = PolicyGradient(env, config, args.seed)
    model.run()
