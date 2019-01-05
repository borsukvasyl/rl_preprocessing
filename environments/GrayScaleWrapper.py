from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np


class GreyScaleWrapper(gym.ObservationWrapper):
    """
    For atari games with RGB observations
    """
    def observation(self, observation):
        return rgb2gray(observation)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
