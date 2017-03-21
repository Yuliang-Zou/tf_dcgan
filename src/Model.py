# Define DCGAN model 
# Author: Yuliang Zou
#         ylzou@umich.edu

import tensorflow as tf
import numpy as np

try:
	import ipdb
except:
	import pdb as ipdb


class DCGAN(object):
	def __init__(self, config):
		self.batch_num = config['batch_num']
		self.img_size = config['img_size']
		self.noise_dim = config['noise_dim']
		self.is_training = config['is_training']
		self.base_lr = config['base_lr']
		self.beta1 = config['beta1']
		self.reuse = False

		self.img = tf.placeholder(tf.float32, 
			[self.batch_num, self.img_size, self.img_size, 3])
		self.noise = tf.placeholder(tf.float32,
			[self.batch_num, 1, 1, self.noise_dim])

		self.add_loss()
		self.add_optim()

	def add_gen(self):
		# Deconv1
		with tf.variable_scope('deconv1') as scope:
			w_deconv1 = tf.get_variable('weights', [4, 4, 1024, self.noise_dim],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.02))

			z_deconv1 = tf.nn.conv2d_transpose(self.noise, w_deconv1,
				[self.batch_num, 4, 4, 1024],
				strides=[1,1,1,1], padding='VALID', name='z')

			bn_deconv1 = tf.contrib.layers.batch_norm(z_deconv1, 
				is_training=self.is_training, scale=True, center=True,
				reuse=False)

			a_deconv1 = tf.nn.relu(bn_deconv1)

		# Deconv2
		with tf.variable_scope('deconv2') as scope:
			w_deconv2 = tf.get_variable('weights', [4, 4, 512, 1024],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.02))

			z_deconv2 = tf.nn.conv2d_transpose(a_deconv1, w_deconv2,
				[self.batch_num, 8, 8, 512],
				strides=[1,2,2,1], padding='SAME', name='z')

			bn_deconv2 = tf.contrib.layers.batch_norm(z_deconv2, 
				is_training=self.is_training, scale=True, center=True,
				reuse=False)

			a_deconv2 = tf.nn.relu(bn_deconv2)

		# Deconv3
		with tf.variable_scope('deconv3') as scope:
			w_deconv3 = tf.get_variable('weights', [4, 4, 256, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.02))

			z_deconv3 = tf.nn.conv2d_transpose(a_deconv2, w_deconv3,
				[self.batch_num, 16, 16, 256],
				strides=[1,2,2,1], padding='SAME', name='z')

			bn_deconv3 = tf.contrib.layers.batch_norm(z_deconv3, 
				is_training=self.is_training, scale=True, center=True,
				reuse=False)

			a_deconv3 = tf.nn.relu(bn_deconv3)

		# Deconv4
		with tf.variable_scope('deconv4') as scope:
			w_deconv4 = tf.get_variable('weights', [4, 4, 128, 256],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.02))

			z_deconv4 = tf.nn.conv2d_transpose(a_deconv3, w_deconv4,
				[self.batch_num, 32, 32, 128],
				strides=[1,2,2,1], padding='SAME', name='z')

			bn_deconv4 = tf.contrib.layers.batch_norm(z_deconv4, 
				is_training=self.is_training, scale=True, center=True,
				reuse=False)

			a_deconv4 = tf.nn.relu(bn_deconv4)

		# Deconv5
		with tf.variable_scope('deconv5') as scope:
			w_deconv5 = tf.get_variable('weights', [4, 4, 3, 128],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.02))

			z_deconv5 = tf.nn.conv2d_transpose(a_deconv4, w_deconv5,
				[self.batch_num, self.img_size, self.img_size, 3],
				strides=[1,2,2,1], padding='SAME', name='z')

			a_deconv5 = tf.tanh(z_deconv5)

		self.gen_img = a_deconv5

		return a_deconv5

	def add_disc(self, input):
		# Conv1
		with tf.variable_scope('conv1', reuse=self.reuse) as scope:

			w_conv1 = tf.get_variable('weights', [4, 4, 3, 64],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.02))

			z_conv1 = tf.nn.conv2d(input, w_conv1, strides=[1,2,2,1],
				padding='SAME')

			# Leaky ReLU
			# credit: https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/V6aeBw4nlaE
			a_conv1 = tf.maximum(0.2 * z_conv1, z_conv1)

		# Conv2
		with tf.variable_scope('conv2', reuse=self.reuse) as scope:
			w_conv2 = tf.get_variable('weights', [4, 4, 64, 128],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.02))

			z_conv2 = tf.nn.conv2d(a_conv1, w_conv2, strides=[1,2,2,1],
				padding='SAME')

			bn_conv2 = tf.contrib.layers.batch_norm(z_conv2, 
				is_training=self.is_training, scale=True, center=True,
				reuse=False)

			a_conv2 = tf.maximum(0.2 * bn_conv2, bn_conv2)

		# Conv3
		with tf.variable_scope('conv3', reuse=self.reuse) as scope:
			w_conv3 = tf.get_variable('weights', [4, 4, 128, 256],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.02))

			z_conv3 = tf.nn.conv2d(a_conv2, w_conv3, strides=[1,2,2,1],
				padding='SAME')

			bn_conv3 = tf.contrib.layers.batch_norm(z_conv3, 
				is_training=self.is_training, scale=True, center=True,
				reuse=False)

			a_conv3 = tf.maximum(0.2 * bn_conv3, bn_conv3)

		# Conv4
		with tf.variable_scope('conv4', reuse=self.reuse) as scope:
			w_conv4 = tf.get_variable('weights', [4, 4, 256, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.02))

			z_conv4 = tf.nn.conv2d(a_conv3, w_conv4, strides=[1,2,2,1],
				padding='SAME')

			bn_conv4 = tf.contrib.layers.batch_norm(z_conv4, 
				is_training=self.is_training, scale=True, center=True,
				reuse=False)

			a_conv4 = tf.maximum(0.2 * bn_conv4, bn_conv4)

		# Conv5
		with tf.variable_scope('conv5', reuse=self.reuse) as scope:
			w_conv5 = tf.get_variable('weights', [4, 4, 512, 1],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.02))

			z_conv5 = tf.nn.conv2d(a_conv4, w_conv5, strides=[1,2,2,1],
				padding='VALID')

			a_conv5 = tf.sigmoid(z_conv5)

		self.reuse = True
		return a_conv5


	def add_loss(self):
		fake = self.add_gen()
		fake_pred = self.add_disc(fake)
		one = tf.ones_like(fake_pred)

		# maximize log(D(x)) + log(1 - D(G(z)))
		loss_fake = tf.reshape(-tf.log(one - fake_pred), (self.batch_num, 1))
		loss_fake = tf.reduce_mean(loss_fake)

		real_pred = self.add_disc(self.img)
		loss_real = tf.reshape(-tf.log(real_pred), (self.batch_num, 1))
		loss_real = tf.reduce_mean(loss_real)

		disc_loss = loss_fake + loss_real

		# maximize log(D(G(z)))
		gen_loss = tf.reshape(-tf.log(fake_pred), (self.batch_num, 1))
		gen_loss = tf.reduce_mean(gen_loss)

		self.disc_loss = disc_loss
		self.gen_loss = gen_loss

		self.avg_real = tf.reduce_mean(real_pred)
		self.avg_fake = tf.reduce_mean(fake_pred)

	def add_optim(self):
		self.vars_gen = []
		for scope in ['deconv1', 'deconv2', 'deconv3', 'deconv4', 'deconv5']:
			self.vars_gen += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

		self.vars_disc = []
		for scope in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
			self.vars_disc += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

		self.train_disc_op = tf.train.AdamOptimizer(self.base_lr, 
			beta1=self.beta1).minimize(self.disc_loss,
			var_list=self.vars_disc)

		self.train_gen_op = tf.train.AdamOptimizer(self.base_lr, 
			beta1=self.beta1).minimize(self.gen_loss,
			var_list=self.vars_gen)


if __name__ == '__main__':
	config = {
	'batch_num': 128,
	'img_size': 64,
	'noise_dim': 100,
	'is_training': True,
	'base_lr': 0.0002,
	'beta1': 0.5
	}

	model = DCGAN(config)