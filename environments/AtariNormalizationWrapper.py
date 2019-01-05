from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import cv2


class AtariNormalizationWrapper(gym.ObservationWrapper):
    """
    For atari games with RGB observations
    """
    def observation(self, observation):
        cropped_frame = observation[8:-12, 4:-12]
        normalized_frame = cropped_frame / 255.0
        preprocessed_frame = cv2.resize(normalized_frame, (110, 84))
        return preprocessed_frame
