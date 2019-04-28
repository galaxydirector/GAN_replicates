# this file is to save the GAN model
# operations would be saved as functions, otherwise to be class methods
"""This version of the model is all hard coded layer, which TODO to be parametric.
This model is an original GAN CNN version, only replicates the whole portfolio without 
controlling styles or other functionality."""
"""All tf.layers functions are deprecated, needs to redefine in future version"""
"""
ToDo in this version:
Rewrite all loss functions
"""
import numpy as np
from itertools import count
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.layers import conv2d,conv2d_transpose,flatten,batch_normalization,dropout


class GAN_cnn:
	def __init__(self, logdir, loss = 'WGAN',num_noise = 64,learning_rate = 1e-4):
		"""model structure
		generator has 5 layers map to 128 by 128
		discriminator map from 128 by 128 through 3 layers"""

		self.num_noise = num_noise
		self.learning_rate = learning_rate
		self.step = count()

		self.create_inputs()

		if loss == 'WGAN':
			self.W_loss()


		self.log_writer_init(logdir)
		self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=5)

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def create_generator(self,z,rate,is_training,reuse=None):
		# hard code all layer params

		upsample_power = 5 # 2**5=32
		momentum = 0.99

		with tf.variable_scope("generator",reuse=reuse):
			# Since "mode collapse" issue, feature of FC needs to be small
			# each layer come with dropout and batch_norm to improve the performance
			x = fully_connected(z,4*4*1,activation_fn=tf.nn.leaky_relu)
			# x = dropout(x, rate = rate,training=is_training)
			x = dropout(x, rate = rate)
			x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)

			x = tf.reshape(x,shape = (-1,4,4,1))
			for _ in range(upsample_power):
				x = conv2d_transpose(
					x,
					128,
					(5,5),
					strides=(2, 2),
					padding='same',
					data_format='channels_last',
					activation=tf.nn.leaky_relu,
					use_bias=True)
				# x = dropout(x, rate = rate,training=is_training)
				x = dropout(x, rate = rate)
				x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)

			x = conv2d_transpose(
				x,
				1,
				(5,5),
				strides=(1, 1),
				padding='same',
				data_format='channels_last',
				activation=tf.sigmoid,
				use_bias=True)
			# assert tf.shape(x) == (128,128,1)
			
			# output is a matrix with -1 to 1
			return x

	def create_discriminator(self,img,rate,reuse=None):
		# hard code all layer params

		# 1. small kernels would lead to loss become 0 quickly
		# try a very large one
		# 2. More filters help the performance
		# especially a lot more filters in generator

		with tf.variable_scope("discriminator",reuse=reuse):
			x = conv2d(
				img,
				128,
				(5,5),
				strides=(2, 2),
				padding='same',
				data_format='channels_last',
				dilation_rate=(1, 1),
				activation=tf.nn.leaky_relu,
				use_bias=True)
			x = dropout(x, rate)
			# assert tf.shape(x) == (64,64,1)
			x = conv2d(
				x,
				128,
				(5,5),
				strides=(2, 2),
				padding='same',
				data_format='channels_last',
				dilation_rate=(1, 1),
				activation=tf.nn.leaky_relu,
				use_bias=True)
			x = dropout(x, rate)
			x = conv2d(
				x,
				128,
				(5,5),
				strides=(1, 1),
				padding='same',
				data_format='channels_last',
				dilation_rate=(1, 1),
				activation=tf.nn.leaky_relu,
				use_bias=True)
			x = dropout(x, rate)

			x = flatten(x)
			x = fully_connected(x,512,activation_fn=tf.nn.leaky_relu)
			d_logits = fully_connected(x,1,activation_fn=None)
			d_prob = tf.sigmoid(d_logits)

			return d_prob, d_logits

	def create_inputs(self):
		self.img = tf.placeholder(tf.float32,shape=(None,128,128,1))
		self.noise = tf.placeholder(tf.float32, shape=(None,self.num_noise))
		self.rate = tf.placeholder(dtype=tf.float32, name='rate')
		self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

	def loss_original_paper(self):
		# training flow
		# generate and train pipeline
		fake = self.create_generator(self.noise,self.rate,self.is_training)
		self.d_prob_real, d_logits_real = self.create_discriminator(self.img,self.rate)
		fake_img = tf.reshape(fake,shape=(-1,128,128,1))
		self.d_prob_fake, d_logits_fake = self.create_discriminator(fake_img,self.rate,reuse=True)
		
		with tf.name_scope('loss'):
	
	def W_loss(self):
		# training flow
		# generate and train pipeline
		fake = self.create_generator(self.noise,self.rate,self.is_training)
		self.d_prob_real, d_logits_real = self.create_discriminator(self.img,self.rate)
		fake_img = tf.reshape(fake,shape=(-1,128,128,1))
		self.d_prob_fake, d_logits_fake = self.create_discriminator(fake_img,self.rate,reuse=True)

		with tf.name_scope('loss'):

	def W_loss(self):
		# training flow
		# generate and train pipeline
		fake = self.create_generator(self.noise,self.rate,self.is_training)
		self.d_prob_real, d_logits_real = self.create_discriminator(self.img,self.rate)
		fake_img = tf.reshape(fake,shape=(-1,128,128,1))
		self.d_prob_fake, d_logits_fake = self.create_discriminator(fake_img,self.rate,reuse=True)

		with tf.name_scope('loss'):


	def loss_original(self):

		# generate and train pipeline
		fake = self.create_generator(self.noise,self.rate,self.is_training)
		self.d_prob_real, d_logits_real = self.create_discriminator(self.img,self.rate)
		fake_img = tf.reshape(fake,shape=(-1,128,128,1))
		self.d_prob_fake, d_logits_fake = self.create_discriminator(fake_img,self.rate,reuse=True)

		"""
		Trick for training:
		1. soft lables: random of [0.9,1] for 1 and [0,0.1] for 0
		2. flip real image loss at the beginning of training, e.g. 5% flipped
		"""

		# real img should approach to 1
		# fake img should approach to 0
		# in discriminator function
		# sum the loss up
		# z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
		# logits is before transforming into sigmoid
		with tf.name_scope('loss'):
			ran = tf.random.uniform([1],minval=0,maxval=1)
			self.d_real_loss = tf.cond(ran[0]<0.95, 
				lambda : tf.nn.sigmoid_cross_entropy_with_logits(
								labels=tf.ones_like(d_logits_real)*0.9, 
								logits=d_logits_real),  
				lambda :tf.nn.sigmoid_cross_entropy_with_logits(
								labels=tf.zeros_like(d_logits_real)*0.9, 
								logits=d_logits_real))
			# self.d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
			#                   labels=tf.ones_like(d_logits_real)*0.9, 
			#                   logits=d_logits_real)
			self.d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
								labels=tf.zeros_like(d_logits_fake), 
								logits=d_logits_fake)

			d_real_loss_reduced = tf.reduce_mean(self.d_real_loss)
			d_fake_loss_reduced = tf.reduce_mean(self.d_fake_loss)
			self.d_loss_reduced = 0.5*(d_real_loss_reduced+d_fake_loss_reduced)

			g_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
								labels=tf.ones_like(d_logits_fake), 
								logits=d_logits_fake)
			self.g_loss_reduced = tf.reduce_mean(g_fake_loss)


			"""
			Another way to writing this as paper is to 
			gradient ascend of d_loss part
			and then gradient decend of the g_loss part
			-tf.reduce_mean(tf.log(d_prob_real) + tf.log(1. - d_prob_fake))
			tf.reduce_mean(tf.log(d_prob_fake))
			eps = 1e-8
			self.d_loss_reduced = -tf.reduce_mean(tf.log(d_prob_real+eps)+tf.log(1.0-d_prob_fake+eps))
			self.g_loss_reduced = -tf.reduce_mean(tf.log(d_prob_fake+eps))
			"""

			# control which part of variables to train
			# tvars=tf.trainable_variables()
			# d_vars=[var for var in tvars if var.name.startswith("discriminator")]
			# g_vars=[var for var in tvars if var.name.startswith("generator")]
			g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="generator")
			d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="discriminator")

			d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), d_vars)
			g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), g_vars)

			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch norm ops
			with tf.control_dependencies(update_ops):
				self.d_trainer=tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss_reduced,
					var_list=d_vars)
				self.g_trainer=tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss_reduced,
					var_list=g_vars)

		# tf.summary.scalar("d_real_loss", self.d_real_loss)
		# tf.summary.scalar("d_fake_loss", self.d_fake_loss)
		tf.summary.scalar('d_loss', self.d_loss_reduced)
		tf.summary.scalar('g_loss', self.g_loss_reduced)
	
	def log_writer_init(self,logdir):
		self.writer = tf.summary.FileWriter(logdir)
		self.writer.add_graph(tf.get_default_graph())
		run_metadata = tf.RunMetadata()
		self.summaries = tf.summary.merge_all()

	def binary_cross_entropy(self,x, z):
		# use default cross entropy, which equation has been posted on tf site
		# to better process if close or smaller than 0
		eps = 1e-12
		return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))

	def train_single_step(self, img, noise):
		"""Balance between g_ls and d_ls borrowed from 
		https://towardsdatascience.com/
		implementing-a-generative-adversarial-network-gan-dcgan-to-draw-human-faces-8291616904a"""
		train_d = True
		train_g = True
		rate_train = 0.6 # 0.5 
		
		d_real_ls, d_fake_ls, g_ls, d_ls,summary = self.sess.run([self.d_prob_real, 
			self.d_prob_fake, 
			self.g_loss_reduced, 
			self.d_loss_reduced,
			self.summaries], feed_dict={self.img: img, self.noise: noise, 
									self.rate: rate_train, self.is_training:True})
		
		self.writer.add_summary(summary, next(self.step))

		d_real_ls = np.mean(d_real_ls)
		d_fake_ls = np.mean(d_fake_ls)
		g_ls = g_ls
		d_ls = d_ls
		
		if g_ls * 1.5 < d_ls:
			train_g = False
			pass
		if d_ls * 2 < g_ls:
			train_d = False
			pass
		
		d_loss = -1
		g_loss = -1

		if train_d:
			_,d_loss, = self.sess.run([self.d_trainer,self.d_loss_reduced], feed_dict={self.noise: noise, self.img: img, self.rate: rate_train, self.is_training:True})
			# self.writer.add_summary(summary, step)
			
		if train_g:
			_,g_loss = self.sess.run([self.g_trainer,self.g_loss_reduced], feed_dict={self.noise: noise, self.rate:rate_train, self.is_training:True})
			# self.writer.add_summary(summary, step)

		losses = {
			'discriminator_loss': d_loss,
			'generator_loss': g_loss
		}
		return losses

	def generate_a_img(self,noise):
		# generated_img = self.sess.run(self.create_generator(self.noise,rate=self.rate,is_training=False,reuse=True),
		# 	feed_dict={self.noise:noise,self.rate:0})
		generated_img = self.sess.run(self.create_generator(self.noise,rate=0,is_training=False,reuse=True),
			feed_dict={self.noise:noise})
		return generated_img

	def train_single_step_original(self, img, noise):
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

	# def saver(self):
	# 	return tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=5)


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
