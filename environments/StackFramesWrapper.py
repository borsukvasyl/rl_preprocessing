from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import numpy as np
import gym


class StackFramesWrapper(gym.ObservationWrapper):
    """
    For atari games with RGB observations
    """
    def __init__(self, env, stack_size=4):
        super().__init__(env)

        self.stack_size = stack_size
        self.prev_frames = deque([None for _ in range(stack_size)], maxlen=4)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.reset_observation(observation)

    def reset_observation(self, observation):
        for i in range(self.stack_size):
            self.prev_frames.append(observation)
        return np.dstack(self.prev_frames)

    def observation(self, observation):
        self.prev_frames.append(observation)
        return np.dstack(self.prev_frames)
