import cv2
import numpy as np
import gym

class ResizeFrame(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=shape +(3,), dtype=np.float32)

    def observation(self, obs):
        if obs.size == 240 * 256 * 3:
            img = np.reshape(obs, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        resized_screen = cv2.resize(img, self.shape, interpolation=cv2.INTER_AREA)
        return resized_screen.astype(np.float32)/255.0
