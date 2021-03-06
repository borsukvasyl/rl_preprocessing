from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from deeprl.models.qlearning import BaseDuelingDQN


class SimpleConvDQNetwork(BaseDuelingDQN):
    def build(self, s_size, a_size):
        self.s_size = s_size
        self.a_size = a_size

        self.states = tf.placeholder(shape=[None, *self.s_size], dtype=tf.float32)
        self.q_target = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.uint8)

        conv1 = tf.layers.conv2d(inputs=self.states, filters=32, kernel_size=[6, 6], strides=3, activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[4, 4], strides=2, activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=96, kernel_size=[4, 4], strides=2, activation=tf.nn.relu)
        flat = tf.layers.flatten(conv3)
        print("Flatten size:", flat.get_shape())

        dense1 = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)
        dense2 = tf.layers.dense(inputs=dense1, units=256, activation=tf.nn.relu)

        self.value = tf.layers.dense(inputs=dense2, units=1)
        self.advantage = tf.layers.dense(inputs=dense2, units=self.a_size)
        self.q_value = self.calculate_q_value(self.value, self.advantage)

        self.loss = self.calculate_loss(self.q_value, self.q_target, self.actions, self.a_size)
        trainer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.optimize = trainer.minimize(self.loss)

    def get_q(self, states):
        return self.session.run(self.q_value, feed_dict={self.states: states})

    def train(self, batch_states, batch_actions, q_target):
        loss, _ = self.session.run([self.loss, self.optimize], feed_dict={
            self.states: batch_states,
            self.actions: batch_actions,
            self.q_target: q_target
        })
        return loss
