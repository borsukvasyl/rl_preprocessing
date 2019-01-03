from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def train_model(sess, trainer, n_episodes, checkpoint_path="model_chkp", log=True):
    sess.run(tf.global_variables_initializer())

    # restore model from the last checkpoint
    start_episode = 0
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        if log: print("Loading: {}".format(checkpoint.model_checkpoint_path))
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, trainer.agent.model.name))
        saver.restore(sess, checkpoint.model_checkpoint_path)
        start_episode = int(checkpoint.model_checkpoint_path.split("-")[-1])

    trainer.train(n_episodes, start_episode=start_episode)
