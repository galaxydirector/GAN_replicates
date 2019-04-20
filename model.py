# this file is to save the GAN model
# operations would be saved as functions, otherwise to be class methods
"""This version of the model is all hard coded layer, which TODO to be parametric.
This model is an original GAN CNN version, only replicates the whole portfolio without 
controlling styles or other functionality."""
"""All tf.layers functions are deprecated, needs to redefine in future version"""

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.layers import Conv2D,Conv2DTranspose,flatten,batch_normalization

# train ops
sess = tf.Session()
sess.run(tf.global_variables_initializer())
GAN_cnn(sess,)

optimizer = 





class GAN_cnn:
	def __init__(self,sess,num_noise,learning_rate,batch_size):
		"""model structure
		generator has 5 layers map to 128 by 128
		discriminator map from 128 by 128 through 3 layers"""
		self.sess = sess
		self.num_noise = num_noise
		self.learning_rate = learning_rate
		self.batch_size = batch_size




	"""
	useful functions

		fully_connected(
	    inputs,
	    num_outputs,
	    activation_fn=tf.nn.relu)

		Conv2DTranspose(
	    inputs,
	    filters,
	    kernel_size,
	    strides=(1, 1),
	    padding='valid',
	    data_format='channels_last',
	    activation=None,
	    use_bias=True)

		Conv2D(
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


	def create_generator(self,z):
		# hard code all layer params

		# experiment if placeholder and input work as the same
		# self.noise = tf.placeholder(tf.float32, shape=(None,num_noise))

		x = fully_connected(z,8*8*512,activation_fn=tf.nn.elu)
		# x = batch_normalization(x)

		x = tf.reshape(x,shape = (-1,8,8,512))
		x = Conv2DTranspose(
		    x,
		    filters=512,
		    kernel_size=(5,5),
		    strides=(2, 2),
		    padding='same',
		    data_format='channels_last',
		    activation=tf.nn.elu,
		    use_bias=True)
		assert tf.shape(x) == (None,8,8,256)
		x = Conv2DTranspose(
		    x,
		    filters=256,
		    kernel_size=(5,5),
		    strides=(2, 2),
		    padding='same',
		    data_format='channels_last',
		    activation=tf.nn.elu,
		    use_bias=True)
		assert tf.shape(x) == (16,16,256)
		x = Conv2DTranspose(
		    x,
		    filters=128,
		    kernel_size=(5,5),
		    strides=(2, 2),
		    padding='same',
		    data_format='channels_last',
		    activation=tf.nn.elu,
		    use_bias=True)
		assert tf.shape(x) == (32,32,128)
		x = Conv2DTranspose(
		    x,
		    filters=64,
		    kernel_size=(5,5),
		    strides=(2, 2),
		    padding='same',
		    data_format='channels_last',
		    activation=tf.nn.elu,
		    use_bias=True)
		assert tf.shape(x) == (64,64,64)
		x = Conv2DTranspose(
		    x,
		    filters=1,
		    kernel_size=(5,5),
		    strides=(2, 2),
		    padding='same',
		    data_format='channels_last',
		    activation=tf.sigmoid,
		    use_bias=True)
		assert tf.shape(x) == (128,128,1)
		
		# output is a matrix with -1 to 1
		return x



	def create_discriminator(self,img):
		# hard code all layer params
		# self.img = tf.placeholder(tf.float32, shape=(None,128,128,1)) # (128,128,1)
		# assert self.img

		x = Conv2D(
		    img,
		    filters = 64,
		    kernel_siz=(5,5),
		    strides=(2, 2),
		    padding='same',
		    data_format='channels_last',
		    dilation_rate=(1, 1),
		    activation=tf.nn.elu,
		    use_bias=True)
		# assert tf.shape(x) == (64,64,1)
		x = Conv2D(
		    x,
		    filters = 128,
		    kernel_size=(5,5),
		    strides=(2, 2),
		    padding='same',
		    data_format='channels_last',
		    dilation_rate=(1, 1),
		    activation=tf.nn.elu,
		    use_bias=True)
		x = Conv2D(
		    x,
		    filters = 128,
		    kernel_size =(5,5),
		    strides=(1, 1),
		    padding='same',
		    data_format='channels_last',
		    dilation_rate=(1, 1),
		    activation=tf.nn.elu,
		    use_bias=True)

		x = faltten(x)
		d_logits = fully_connected(x,1,activation_fn=None)
		d_prob = tf.sigmoid(d_logits)

		return d_prob, d_logits


	def loss(self,):

		# import a img, and generate an img
		self.img = tf.placeholder(tf.float32,shape=(None,128,128,1))
		self.noise = tf.placeholder(tf.float32, shape=(None,num_noise))

		fake = self.create_generator(self.noise)

		# use discriminator 
		d_prob_real, d_logits_real = self.create_discriminator(self.img)
		fake_img = tf.reshape(fake,shape=(-1,128,128,1))
		d_prob_fake, d_logits_fake = self.create_discriminator(fake_img)


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
		d_loss_reduced = d_real_loss_reduced+d_fake_loss_reduced

		g_fake_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
							labels=tf.ones_like(d_logits_fake), 
							logits=d_logits_fake)
		g_loss_reduced = tf.reduce_mean(g_fake_loss)

		# another way to writing this as paper is to 
		# gradient ascend of d_loss part
		# and then gradient decend of the g_loss part
		# -tf.reduce_mean(tf.log(d_prob_real) + tf.log(1. - d_prob_fake))
		# tf.reduce_mean(tf.log(d_prob_fake))

		d_trainer=tf.train.AdamOptimizer(self.learning_rate).minimize(d_loss_reduced)
		g_trainer=tf.train.AdamOptimizer(self.learning_rate).minimize(g_loss_reduced)

	def generator_loss(self,):



	def train_single_step(self, img, noise):
		# feed in a single step
		# first update Discriminator
		# second update generator
		_, loss = self.sess.run([train_ops],feed_dict= {self.noise:noise,self.img:img})


	




	def generate_a_img(self,):
		self.sess.run()



	def create_model(self):


	def loss(self):

