from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from deeprl.models.qlearning import BaseDuelingDQN


class SimpleConvLstmDQNetwork(BaseDuelingDQN):
    def build(self, s_size, a_size):
        self.s_size = s_size
        self.a_size = a_size

        self.states = tf.placeholder(shape=[None, *self.s_size], dtype=tf.float32)
        self.q_target = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.uint8)

        self.lstm_state0 = tf.placeholder(tf.float32, shape=[1, 256])
        self.lstm_state1 = tf.placeholder(tf.float32, shape=[1, 256])
        self.lstm_state = tf.contrib.rnn.LSTMStateTuple(self.lstm_state0, self.lstm_state1)
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

        conv1 = tf.layers.conv2d(inputs=self.states, filters=8, kernel_size=[3, 3], activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=16, kernel_size=[3, 3], activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)
        conv3 = tf.layers.conv2d(inputs=pool2, filters=24, kernel_size=[3, 3], activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3, 3], strides=2)
        flat = tf.layers.flatten(pool3)

        dense1 = tf.layers.dense(inputs=flat, units=256, activation=tf.nn.relu)

        step_size = tf.shape(dense1)[:1]
        lstm_input_reshaped = tf.reshape(dense1, [1, -1, 256])
        lstm_outputs, self.base_lstm_state = tf.nn.dynamic_rnn(self.lstm_cell,
                                                               lstm_input_reshaped,
                                                               initial_state=self.lstm_state,
                                                               sequence_length=step_size,
                                                               time_major=False)
        lstm_outputs = tf.reshape(lstm_outputs, [-1, 256])

        self.value = tf.layers.dense(inputs=lstm_outputs, units=1)
        self.advantage = tf.layers.dense(inputs=lstm_outputs, units=self.a_size)
        self.q_value = self.calculate_q_value(self.value, self.advantage)

        self.loss = self.calculate_loss(self.q_value, self.q_target, self.actions, self.a_size)
        trainer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.optimize = trainer.minimize(self.loss)

    def get_q(self, states):
        q_value, self.base_lstm_state_out = self.session.run([self.q_value, self.base_lstm_state], feed_dict={
            self.states: states,
            self.lstm_state0: self.base_lstm_state_out[0],
            self.lstm_state1: self.base_lstm_state_out[1]
        })
        return q_value

    def reset(self):
        self.base_lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 256]), np.zeros([1, 256]))

    def train(self, batch_states, batch_actions, q_target):
        loss, _ = self.session.run([self.loss, self.optimize], feed_dict={
            self.states: batch_states,
            self.actions: batch_actions,
            self.q_target: q_target,
            self.lstm_state0: np.zeros([1, 256]),
            self.lstm_state1: np.zeros([1, 256])
        })
        return loss
