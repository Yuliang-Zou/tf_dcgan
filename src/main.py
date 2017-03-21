# Main function to train and visualize
# Author: Yuliang Zou
#         ylzou@umich.edu

import tensorflow as tf
import numpy as np
import cv2
from Model import DCGAN
from Dataloader import Dataloader

try:
	import ipdb
except:
	import pdb as ipdb

config = {
'batch_num': 128,
'img_size': 64,
'noise_dim': 100,
'is_training': True,
'base_lr': 0.0002,
'beta1': 0.5,
'epoch': 100
}

model = DCGAN(config)
dataloader = Dataloader(config)

init = tf.initialize_all_variables()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
	session.run(init)
	saver = tf.train.Saver()

	niter = dataloader.get_iter_per_epoch() * config['epoch']
	for i in xrange(niter):
		img_blob = dataloader.get_next_minibatch()
		noise_blob = np.random.randn(config['batch_num'], 1, 1, config['noise_dim'])

		# Disc loss
		feed_dict = {model.img:img_blob, model.noise:noise_blob}
		_, disc_loss, Dx, Dz1 = session.run([model.train_disc_op, model.disc_loss, model.avg_real, model.avg_fake],
			feed_dict=feed_dict)

		# Gen loss
		feed_dict = {model.noise:noise_blob}
		_, gen_loss, Dz2 = session.run([model.train_gen_op, model.gen_loss, model.avg_fake],
			feed_dict=feed_dict)

		print('Epoch:[%d/%d], Iter:[%d/%d]. Disc: %.4f, Gen: %.4f. D(x): %.4f, D(G(z)): %.4f/%.4f'
			% (dataloader.get_epoch(), config['epoch'], 
				dataloader.get_iter_in_epoch()-1, 
				dataloader.get_iter_per_epoch(), 
				disc_loss, gen_loss, Dx, Dz1, Dz2))

		if dataloader.get_epoch() > 0 and dataloader.get_epoch() % 10 == 0:
			saver.save(session, '../model/dcgan_epoch_' + str(dataloader.get_epoch()) + '.ckpt')

