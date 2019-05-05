# import sys
# sys.settrace
# use to train
"""
This file controls the operation to train a model
The pipeline starts with 
1.feed in data
2.perprocess it
3.feed 'em into the model, set up training params
4.and save them properly, record all the training data properly and possibly display 'em
"""
import sys
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

import os
from glob import glob
import os.path as path
from datetime import datetime

import tensorflow as tf
from model import GAN_cnn,W_GAN

data_root=path.expanduser('/home/aitrading/Desktop/VAESelfies/output/yanci_only3/')
np_files=glob(path.join(data_root,'*.npy'))
# num_sample = len(np_files)
num_sample=5000

checkpoint_dir = None
model_name = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
logdir = os.path.join(path.expanduser('/home/aitrading/Desktop/models'),model_name)
img_path_root = os.path.join(path.expanduser('/home/aitrading/Desktop/img_results'),model_name)
if not os.path.exists(img_path_root):
    os.makedirs(img_path_root)


def create_data_batch(batch_size=1):
	file_list = np_files
	np.random.shuffle(np_files)
	n = len(file_list)
	i = 0
	while True:
		temp = []
		for _ in range(batch_size):
			i+=1
			if i>=n:
				i=0
				np.random.shuffle(file_list)
			temp.append(np.load(file_list[i]).reshape(128,128,1))
		yield np.array(temp)

def save(saver, sess, logdir, epoch):
	model_name = 'model.ckpt'
	checkpoint_path = os.path.join(logdir, model_name)
	print('Storing checkpoint to {} ...'.format(logdir))
	sys.stdout.flush()

	if not os.path.exists(logdir):
		os.makedirs(logdir)

	saver.save(sess, checkpoint_path, global_step=epoch)
	print(' Done.')

def load(checkpoint_dir,saver):
	print(" [*] Reading checkpoints...")

	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(self.sess, ckpt.model_checkpoint_path)
		return True
	else:
		return False

def save_img(data,path_root,epoch):
	plt.imshow(data.reshape(128,128))
	plt.title("epoch {} sample".format(epoch))

	path_ = os.path.join(path_root,"epoch_{}".format(epoch))
	plt.savefig(path.expanduser(path_))


def trainer(model_object, ckpt = checkpoint_dir, logdir = logdir, learning_rate=1e-4, 
			batch_size=32, num_epoch=331, log_step=20, num_noise = 64):
	"""Operations:
	1. Set up the model
	2. start the training 
	3. save the models while train
	4. generate sample img"""

	model = model_object(logdir=logdir ,num_noise = num_noise, learning_rate = learning_rate)
	data_feed = create_data_batch(batch_size)

	if checkpoint_dir and load(ckpt,model.saver):
		print("An existing model has been restored path : /n {}".format(ckpt))
	else:
		print("initialize a new training")

	for epoch in range(num_epoch):
		start_time = time.time()
		for _ in tqdm(range(num_sample // batch_size)):
			# Get a batch and noise, train it
			batch = next(data_feed)

			z = np.random.uniform(-1,1,size=(batch_size,num_noise))
			losses = model.train_single_step(img=batch,noise=z)

			# show loss while training
			# log_str = ''
			# for k, v in losses.items():
			# 	log_str += '{}: {:.3f}  '.format(k, v)
			# tqdm.write(log_str)
		end_time = time.time()
		

		# save a sample graph of each epoch
		rand_z = np.random.uniform(-1,1,size=(1,num_noise))
		test_img = model.generate_a_img(rand_z)
		test_img = np.array(test_img)*255 # convert it from [-1,1] to [0,255]

		save_img(test_img,img_path_root,epoch)

		log_str = '[Epoch {}] '.format(epoch)
		for k, v in losses.items():
			log_str += '{}: {:.3f}  '.format(k, v)
		log_str += '({:.3f} sec/epoch)'.format(end_time - start_time)
		print(log_str)

		if epoch % log_step == 0:
			save(model.saver, model.sess, logdir, epoch)

	print('Done!')
	return model

if __name__ == '__main__':
	trainer(W_GAN)