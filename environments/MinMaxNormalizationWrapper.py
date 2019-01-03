from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym


class MinMaxNormalizationWrapper(gym.ObservationWrapper):
    BIG_INT = 10e+20

    def __init__(self, env):
        super().__init__(env)

        self.min_obs, self.max_obs = self.env.observation_space.low, self.env.observation_space.high
        # remove infinite numbers
        self.min_obs[self.min_obs < -self.BIG_INT] = 0
        self.max_obs[self.max_obs > self.BIG_INT] = 1

    def observation(self, observation):
        return (observation - self.min_obs) / (self.max_obs - self.min_obs)
