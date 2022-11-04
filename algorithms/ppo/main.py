import os
import sys

import gym_super_mario_bros
from absl import flags
from nes_py.wrappers import JoypadSpace

import luig_io
from luig_io.wrappers import FrameStack
from luig_io.wrappers import GrayScale
from luig_io.wrappers import ResizeFrame

flags.DEFINE_integer("seed", 0, "The random seed to use.")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

luig_io.set_seed(FLAGS.seed)


def get_env(config):
    env = gym_super_mario_bros.make("SuperMarioBros-v3")
    env = JoypadSpace(env, config.actions)
    env = ResizeFrame(env, (84, 84))
    env = GrayScale(env)
    env = FrameStack(env, num_stack=4, axis=-1)
    return env


get_env()
