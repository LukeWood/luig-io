import cv2
import gym
import numpy as np


class GrayScale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = env.observation_space.shape[:-1]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=shape + (1,), dtype=np.float32
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return gray
