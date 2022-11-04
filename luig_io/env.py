import gym_super_mario_bros
import gym_super_mario_bros.actions
from nes_py.wrappers import JoypadSpace

from luig_io.wrappers import FrameStack
from luig_io.wrappers import GrayScale
from luig_io.wrappers import ResizeFrame

actions_mapping = {
    "right": gym_super_mario_bros.actions.RIGHT_ONLY,
    "simple": gym_super_mario_bros.actions.SIMPLE_MOVEMENT,
    "complex": gym_super_mario_bros.actions.COMPLEX_MOVEMENT,
}


def get_env(env_name, actions="simple"):
    """get_env produces a pre-configured gym mario bros environment.

    Args:
        env_name: valid `gym_super_mario_bros` name, defaults to
            'SuperMarioBrosRandomStages-v3'.
        actions: one of 'right', 'simple', 'complex'.
    """
    if actions not in actions_mapping:
        raise ValueError(
            f"`actions` must be one of [{' '.join(actions_mapping.keys())}]"
        )
    actions = actions_mapping[actions]
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, actions)
    env = ResizeFrame(env, (84, 84))
    env = GrayScale(env)
    env = FrameStack(env, num_stack=4, axis=-1)
    return env
