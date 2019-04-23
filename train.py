# use to train
"""
This file controls the operation to train a model
The pipeline starts with 
1.feed in data
2.perprocess it
3.feed 'em into the model, set up training params
4.and save them properly, record all the training data properly and possibly display 'em
"""
import numpy as np
from tqdm import tqdm

import os
from glob import glob
import os.path as path

from model import GAN_cnn

data_root=path.expanduser('/home/aitrading/Desktop/GLTransform/output/npy128/')
np_files=glob(path.join(data_root,'*.npy'))


logdir = path.expanduser('/home/aitrading/Desktop/models')
img_path_root = path.expanduser('/home/aitrading/Desktop/img_results')

def create_data_batch(batch_size=1):
	n=len(np_files)
	i=0
	while True:
		temp = []
		
		for j in range(batch_size):
			i+=1
			if i>=n:
				return
			temp.append(np.load(np_files[i]).reshape(128,128,1))
		
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

def save_img(data,path_root,epoch):
	plt.imshow(data.reshape(128,128))
	plt.title("epoch {} sample".format(epoch))

	path = os.path.join(path_root,"epoch_{}".format(epoch))
	plt.savefig(path.expanduser(path))

def trainer(model_object, learning_rate=1e-4, 
			batch_size=32, num_epoch=31, log_step=5, num_noise = 100):
	"""Operations:
	1. Set up the model
	2. start the training 
	3. save the models while train
	4. generate sample img"""

	model = model_object(num_noise = num_noise, learning_rate = learning_rate)

	for epoch in range(num_epoch):
		start_time = time.time()
		for _ in tqdm(range(num_sample // batch_size)):
			# Get a batch and noise
			batch = next(create_data_batch(batch_size))
			z = np.random.uniform(-1,1,size=(batch_size,100))

			# Execute the forward and backward pass 
			# Report computed losses
			losses = model.train_single_step(img=batch,noise=z)
		end_time = time.time()
		
		# save a sample graph of each epoch
		rand_z = np.random.uniform(-1,1,size=(1,100))
		test_img = model.generate_a_img(rand_z)
		save_img(test_img,img_path_root,epoch)

		if epoch % log_step == 0:
			log_str = '[Epoch {}] '.format(epoch)
			for k, v in losses.items():
				log_str += '{}: {:.3f}  '.format(k, v)
			log_str += '({:.3f} sec/epoch)'.format(end_time - start_time)
			print(log_str)
			save(model.save, model.sess, logdir, epoch)

	print('Done!')
	return model

if __name__ == '__main__':
	trainer(GAN_cnn)