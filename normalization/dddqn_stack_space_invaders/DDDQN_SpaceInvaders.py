from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tensorflow as tf
import argparse

from deeprl.agents import QAgent
from deeprl.callbacks import Tensorboard, Saver
from deeprl.trainers.qlearning import DoubleDQNTrainer, DQNConfig
from deeprl.utils import record_video

from models import SimpleConvDQNetwork
from environments import GreyScaleWrapper, AtariNormalizationWrapper, StackFramesWrapper
from train import train_model


def main(args):
    n_episodes = 100

    env = gym.make("SpaceInvaders-v0")
    if args.use_norm:
        env = GreyScaleWrapper(env)
    env = AtariNormalizationWrapper(env)
    env = StackFramesWrapper(env)

    a_size = env.action_space.n
    s_size = (84, 110, 4) if args.use_norm else (84, 110, 12)
    print("Action space size: {}".format(a_size))
    print("State space size: {}".format(s_size))

    checkpoint_path = args.dir + "/model_chkp"
    video_path = args.dir + "/video"

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.Session(config=config)

    model = SimpleConvDQNetwork("main", sess, s_size=s_size, a_size=a_size)
    agent = QAgent(model)

    config = DQNConfig()
    trainer = DoubleDQNTrainer(config, agent, env)
    trainer.callbacks.append(Tensorboard(sess, ["r_total"], log_dir=checkpoint_path))
    trainer.callbacks.append(Saver(model, step=20, filename=checkpoint_path + "/model"))

    train_model(sess, trainer, n_episodes, checkpoint_path=checkpoint_path)

    record_video(agent, env, filename=video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_norm", help="whether to use min-max normalization",
                        action="store_true")
    parser.add_argument("--dir", help="directory",
                        type=str, default="temp")
    args = parser.parse_args()

    main(args)
