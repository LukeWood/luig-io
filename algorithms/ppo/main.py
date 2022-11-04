from absl import flags
import sys
import os
import luig_io

flags.DEFINE_integer("seed", 0, "The random seed to use.")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

luig_io.set_seed(FLAGS.seed)
