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
flags.DEFINE_string("env", "SuperMarioBros-1-1-v4")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

luig_io.set_seed(FLAGS.seed)
luig_io.get_env(FLAGS.env)
