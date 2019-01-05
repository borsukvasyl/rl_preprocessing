from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tensorflow as tf
import argparse

from deeprl.agents import QAgent
from deeprl.callbacks import BaseCallback, Tensorboard, Saver
from deeprl.trainers.qlearning import DoubleDQNTrainer, DQNConfig
from deeprl.utils import record_video

from models import SimpleConvLstmDQNetwork
from environments import MinMaxNormalizationWrapper
from train import train_model


class LstmResetter(BaseCallback):
    def __init__(self, models):
        self.models = models

    def on_episode_begin(self, episode, logs=None, **kwargs):
        for model in self.models:
            model.reset()


def main(args):
    n_episodes = 4000

    env = gym.make("SpaceInvaders-v0")
    if args.use_norm:
        env = MinMaxNormalizationWrapper(env)

    a_size = env.action_space.n
    s_size = env.observation_space.shape
    print("Action space size: {}".format(a_size))
    print("State space size: {}".format(s_size))

    checkpoint_path = args.dir + "/model_chkp"
    video_path = args.dir + "/video"

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)

    model = SimpleConvLstmDQNetwork("main", sess, s_size=s_size, a_size=a_size)
    agent = QAgent(model)

    config = DQNConfig()
    config.experience_sampler = ["ordered", "on_sample_clear"]
    config.experience_size = None
    config.min_sample_size = 20
    trainer = DoubleDQNTrainer(config, agent, env)
    trainer.callbacks.append(Tensorboard(sess, ["r_total"], log_dir=checkpoint_path))
    trainer.callbacks.append(Saver(model, step=20, filename=checkpoint_path + "/model"))
    trainer.callbacks.append(LstmResetter([model, trainer.target_model]))

    train_model(sess, trainer, n_episodes, checkpoint_path=checkpoint_path)

    model.reset()
    record_video(agent, env, filename=video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_norm", help="whether to use min-max normalization",
                        action="store_true")
    parser.add_argument("--dir", help="directory",
                        type=str, default="temp")
    args = parser.parse_args()

    main(args)
