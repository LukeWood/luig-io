import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from luig_io.wrappers import ResizeFrame, GrayScale, FrameStack

def get_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = ResizeFrame(env, (84, 84))
    env = GrayScale(env)
    env = FrameStack(env, num_stack=4, axis=-1)
    return env
