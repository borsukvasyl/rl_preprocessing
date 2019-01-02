from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf
import random

from deeprl.agents import QAgent
from deeprl.callbacks import Tensorboard, Saver
from deeprl.trainers.qlearning import DoubleDQNTrainer, DQNConfig
from deeprl.utils import record_video
from base_models.simple_models import SimpleDQNetwork


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


def main():
    env = gym.make("CartPole-v0")
    # env = MinMaxNormalizationWrapper(env)

    a_size = env.action_space.n
    s_size = env.observation_space.shape[0]
    print("Action space size: {}".format(a_size))
    print("State space size: {}".format(s_size))

    sess = tf.Session()
    model = SimpleDQNetwork("main", sess, s_size=s_size, a_size=a_size)
    agent = QAgent(model)
    config = DQNConfig()
    trainer = DoubleDQNTrainer(config, agent, env)
    trainer.callbacks.append(Tensorboard(sess, ["r_total"]))
    trainer.callbacks.append(Saver(model, step=20))

    sess.run(tf.global_variables_initializer())

    trainer.train(300)

    record_video(agent, env)


if __name__ == "__main__":
    RANDOM_SEED = 40
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)

    main()
