# Dataloader for celebA dataset
# Author: Yuliang Zou
#         ylzou@umich.edu

import tensorflow as tf
import numpy as np
import cv2
from os.path import join

try:
	import ipdb
except:
	import pdb as ipdb

"""
The dataloader for celebA dataset. (Single processing version)
"""
class Dataloader(object):
	def __init__(self, config):
		self.root = '../data/img_align_celeba/'
		self.batch_num = config['batch_num']
		self.img_size = config['img_size']

		img_set = join(self.root, 'img_set.txt')
		with open(img_set) as f:
			self.img_list = f.read().rstrip().split('\n')

		self.num_images = len(self.img_list)
		self.iter_per_epoch = self.num_images / self.batch_num + 1

		self.temp_pointer = 0
		self.epoch = 0
		self.iter_in_epoch = 0

		# self._shuffle()

	def _shuffle(self):
		self.img_list = np.random.permutation(self.img_list)
		self.temp_pointer = 0

	def _img_at(self, i):
		return join(self.root, self.img_list[i][:-1])

	def get_epoch(self):
		return self.epoch

	def get_iter_in_epoch(self):
		return self.iter_in_epoch

	def get_iter_per_epoch(self):
		return self.iter_per_epoch

	def get_next_minibatch(self):
		img_blobs = []
		shuffle_flag = False
		for i in xrange(self.batch_num):
			img = cv2.imread(self._img_at(self.temp_pointer))[:,:,::-1] / 128. - 1
			img_re = cv2.resize(img, (self.img_size, self.img_size))
			img_blobs.append(img_re)
			self.temp_pointer += 1

			if self.temp_pointer >= self.num_images:
				self.epoch += 1
				self.iter_in_epoch = 0
				self._shuffle()
				shuffle_flag = True

		if shuffle_flag:
			self._shuffle()
		else:
			self.iter_in_epoch += 1

		return np.array(img_blobs)


if __name__ == '__main__':
	config = {
	'batch_num': 128,
	'img_size': 32,
	'noise_dim': 100,
	'is_training': True,
	'base_lr': 0.0002,
	'beta1': 0.5
	}

	dataloader = Dataloader(config)
	blob = dataloader.get_next_minibatch()