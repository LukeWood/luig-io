import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from luig_io.wrappers import ResizeFrame, GrayScale, FrameStack


def get_env(config):
    env = gym_super_mario_bros.make(config.env_name)
    env = JoypadSpace(env, config.actions)
    env = ResizeFrame(env, (84, 84))
    env = GrayScale(env)
    env = FrameStack(env, num_stack=4, axis=-1)
    return env
