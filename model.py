# this file is to save the GAN model
# operations would be saved as functions, otherwise to be class methods
"""This version of the model is all hard coded layer, which TODO to be parametric.
This model is an original GAN CNN version, only replicates the whole portfolio without 
controlling styles or other functionality."""
"""All tf.layers functions are deprecated, needs to redefine in future version"""

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.layers import conv2d,conv2d_transpose,flatten,batch_normalization


class GAN_cnn:
	def __init__(self,num_noise = 100,learning_rate = 1e-4):
		"""model structure
		generator has 5 layers map to 128 by 128
		discriminator map from 128 by 128 through 3 layers"""

		self.num_noise = num_noise
		self.learning_rate = learning_rate

		self.loss()
		self.save = self.saver()

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	"""
	useful functions

		fully_connected(
		inputs,
		num_outputs,
		activation_fn=tf.nn.relu)

		conv2d_transpose(
		inputs,
		filters,
		kernel_size,
		strides=(1, 1),
		padding='valid',
		data_format='channels_last',
		activation=None,
		use_bias=True)

		conv2d(
		inputs,
		filters,
		kernel_size,
		strides=(1, 1),
		padding='valid',
		data_format='channels_last',
		dilation_rate=(1, 1),
		activation=None,
		use_bias=True)

		flatten(inputs)

		batch_normalization(inputs)

		tf.nn.softmax_cross_entropy_with_logits_v2(
		labels,
		logits)
	"""


	def create_generator(self,z,reuse=None):
		# hard code all layer params

		# experiment if placeholder and input work as the same
		# self.noise = tf.placeholder(tf.float32, shape=(None,num_noise))

		with tf.variable_scope("gen",reuse=reuse):

			x = fully_connected(z,8*8*512,activation_fn=tf.nn.elu)
			# x = batch_normalization(x)

			x = tf.reshape(x,shape = (-1,8,8,512))
			x = conv2d_transpose(
				x,
				512,
				(5,5),
				strides=(2, 2),
				padding='same',
				data_format='channels_last',
				activation=tf.nn.elu,
				use_bias=True)
			# assert tf.shape(x) == tf.convert_to_tensor((1,8,8,512),dtype=tf.int32), "{}".format(tf.shape(x))
			x = conv2d_transpose(
				x,
				256,
				(5,5),
				strides=(2, 2),
				padding='same',
				data_format='channels_last',
				activation=tf.nn.elu,
				use_bias=True)
			# assert tf.shape(x) == (16,16,256)
			x = conv2d_transpose(
				x,
				128,
				(5,5),
				strides=(2, 2),
				padding='same',
				data_format='channels_last',
				activation=tf.nn.elu,
				use_bias=True)
			# assert tf.shape(x) == (32,32,128)
			x = conv2d_transpose(
				x,
				64,
				(5,5),
				strides=(2, 2),
				padding='same',
				data_format='channels_last',
				activation=tf.nn.elu,
				use_bias=True)
			# assert tf.shape(x) == (64,64,64)
			x = conv2d_transpose(
				x,
				1,
				(5,5),
				strides=(2, 2),
				padding='same',
				data_format='channels_last',
				activation=tf.sigmoid,
				use_bias=True)
			# assert tf.shape(x) == (128,128,1)
			
			# output is a matrix with -1 to 1
			return x



	def create_discriminator(self,img,reuse=None):
		# hard code all layer params
		# self.img = tf.placeholder(tf.float32, shape=(None,128,128,1)) # (128,128,1)
		# assert self.img

		with tf.variable_scope("dis",reuse=reuse):
			x = conv2d(
				img,
				64,
				(5,5),
				strides=(2, 2),
				padding='same',
				data_format='channels_last',
				dilation_rate=(1, 1),
				activation=tf.nn.elu,
				use_bias=True)
			# assert tf.shape(x) == (64,64,1)
			x = conv2d(
				x,
				128,
				(5,5),
				strides=(2, 2),
				padding='same',
				data_format='channels_last',
				dilation_rate=(1, 1),
				activation=tf.nn.elu,
				use_bias=True)
			x = conv2d(
				x,
				128,
				(5,5),
				strides=(1, 1),
				padding='same',
				data_format='channels_last',
				dilation_rate=(1, 1),
				activation=tf.nn.elu,
				use_bias=True)

			x = flatten(x)
			d_logits = fully_connected(x,1,activation_fn=None)
			d_prob = tf.sigmoid(d_logits)

			return d_prob, d_logits

	def loss(self):

		# import a img, and generate an img
		self.img = tf.placeholder(tf.float32,shape=(None,128,128,1))
		self.noise = tf.placeholder(tf.float32, shape=(None,self.num_noise))

		fake = self.create_generator(self.noise)

		# use discriminator 
		d_prob_real, d_logits_real = self.create_discriminator(self.img)
		fake_img = tf.reshape(fake,shape=(-1,128,128,1))
		d_prob_fake, d_logits_fake = self.create_discriminator(fake_img,reuse=True)


		# real img should approach to 1
		# fake img should approach to 0
		# in discriminator function
		# sum the loss up
		d_real_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
							labels=tf.ones_like(d_logits_real), 
							logits=d_logits_real)
		d_fake_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
							labels=tf.zeros_like(d_logits_fake), 
							logits=d_logits_fake)

		d_real_loss_reduced = tf.reduce_mean(d_real_loss)
		d_fake_loss_reduced = tf.reduce_mean(d_fake_loss)
		self.d_loss_reduced = d_real_loss_reduced+d_fake_loss_reduced

		g_fake_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
							labels=tf.ones_like(d_logits_fake), 
							logits=d_logits_fake)
		self.g_loss_reduced = tf.reduce_mean(g_fake_loss)

		# another way to writing this as paper is to 
		# gradient ascend of d_loss part
		# and then gradient decend of the g_loss part
		# -tf.reduce_mean(tf.log(d_prob_real) + tf.log(1. - d_prob_fake))
		# tf.reduce_mean(tf.log(d_prob_fake))

		# control which part of variables to train
		tvars=tf.trainable_variables()
		d_vars=[var for var in tvars if 'dis' in var.name]
		g_vars=[var for var in tvars if 'gen' in var.name]

		self.d_trainer=tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss_reduced,
			var_list=d_vars)
		self.g_trainer=tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss_reduced,
			var_list=g_vars)

		# self.losses = {
		#   'discriminator_loss': self.d_loss_reduced,
		#   'generator_loss': self.g_loss_reduced,
		#   'total_loss': self.d_trainer+self.g_trainer
		# }

	def train_single_step(self, img, noise):
		# feed in a single step
		# first update Discriminator
		# second update generator

		_,d_loss = self.sess.run([self.d_trainer,self.d_loss_reduced],
			feed_dict={self.img:img,self.noise:noise})
		_,g_loss = self.sess.run([self.g_trainer,self.g_loss_reduced],
			feed_dict={self.noise:noise})
		losses = {
			'discriminator_loss': d_loss,
			'generator_loss': g_loss,
			'total_loss': d_loss+g_loss
		}
		return losses


	def generate_a_img(self,noise):

		generated_img = self.sess.run(self.create_generator(self.noise,reuse=True),
			feed_dict={self.noise:noise})
		return generated_img

	def saver(self):
		return tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=5)

