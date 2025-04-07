import numpy as np
from pickle import dump
from keras.backend import concatenate
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.distribute.optimizer_combinations import adam_optimizer_v1_fn
from keras.models import Model
from tensorflow.keras.layers import SeparableConv2D, SeparableConv1D
from tqdm import tqdm
import tensorflow as tf
from keras.layers import Dense, GlobalAveragePooling2D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation, Dropout, \
	LayerNormalization, Conv1D
from keras.layers import Conv2D, Concatenate
import time
from keras.regularizers import l2
import pandas as pd
import simon_mrcp as si

bs = 10000
wdir = "./save_final/simon_9r_4plus5/"


def cyclic_lr(num_epochs, high_lr, low_lr):
	res = lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)
	return (res)


def make_checkpoint(datei):
	res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
	return (res)

def convert_to_binary(l):
	n = len(l)
	k = 16 * n
	X = np.zeros((k, len(l[0])), dtype=np.uint8)
	for i in range(k):
		index = i // 16
		offset = 16 - 1 - i % 16
		X[i] = (l[index] >> offset) & 1
	X = X.transpose()
	return (X)


# make residual tower of convolutional blocks
def make_resnet(group_size=2, num_blocks=2, num_filters=32, num_outputs=1, d1=32, d2=64, word_size=16, ks=7, depth=10,
                reg_param=0.000002, final_activation='sigmoid', num_groups=4):
	# Input and preprocessing layers
	inp = Input(shape=(group_size * 4 * num_blocks * word_size,))  # (-1, 2 * 2 * 2 * 16)
	rs = Reshape((group_size, 4 * num_blocks, word_size))(inp)  # (-1, 2, 4, 16)
	perm = Permute((1, 3, 2))(rs)  # (-1, 2, 16, 4)
	# add a single residual layer that will expand the data to num_filters channels
	# this is a bit-sliced layer
	conv0 = Conv1D(num_filters, kernel_size=1, strides=1, padding='same', kernel_regularizer=l2(reg_param))(perm);
	conv0 = BatchNormalization()(conv0)
	conv0 = Activation('relu')(conv0)
	
	# add residual blocks
	shortcut = conv0
	for i in range(depth):
		conv1 = Conv1D(num_filters, kernel_size=ks, strides=1, padding='same', kernel_regularizer=l2(reg_param))(
			shortcut)
		conv1 = BatchNormalization()(conv1)
		conv1 = Activation('relu')(conv1)
		
		conv1 = Conv1D(num_filters, kernel_size=ks, strides=1, padding='same', kernel_regularizer=l2(reg_param))(conv1)
		conv1 = BatchNormalization()(conv1)
		conv1 = Activation('relu')(conv1)
		
		shortcut = Add()([shortcut, conv1])
	
	# add prediction head
	flat1 = Flatten()(shortcut)
	dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(flat1)
	dense1 = BatchNormalization()(dense1)
	dense1 = Activation('relu')(dense1)
	dense1 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
	dense1 = BatchNormalization()(dense1)
	dense1 = Activation('relu')(dense1)
	out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense1)
	model = Model(inputs=inp, outputs=out)
	
	# compute the number of network parameters
	model.summary()
	return (model)


def train_simon_distinguisher(num_epochs, num_rounds=7, diff=(0x0000, 0x0040), group_size=2, depth=1):
	# create the network
	net = make_resnet(group_size=group_size, depth=depth, reg_param=0.000002)
	net.compile(optimizer='adam', loss='mse', metrics=['acc'])
	# generate training and validation data
	X, Y = si.make_dataset_with_group_size(n=group_size * 10 ** 7, nr=num_rounds, first_step=3,
	                                       group_size=group_size, diff=diff)
	X_eval, Y_eval = si.make_dataset_with_group_size(n=group_size * 10 ** 6, nr=num_rounds, first_step=3,
	                                                 group_size=group_size, diff=diff)

	check = make_checkpoint(
		wdir + 'best' + str(num_rounds) + 'depth' + str(depth) + 'groupsize' + str(group_size) + '.h5')
	# create learnrate schedule
	lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
	# train and evaluate
	h = net.fit(X, Y, epochs=num_epochs, batch_size=bs, shuffle=True,
	            validation_data=(X_eval, Y_eval), callbacks=[lr, check])
	np.save(wdir + 'h' + str(num_rounds) + 'r_depth' + str(depth) + '_group' + str(group_size) + 'acc.npy',
	        h.history['acc'])
	np.save(wdir + 'h' + str(num_rounds) + 'r_depth' + str(depth) + '_group' + str(group_size) + 'val_acc.npy',
	        h.history['val_acc'])
	np.save(wdir + 'h' + str(num_rounds) + 'r_depth' + str(depth) + '_group' + str(group_size) + 'loss.npy',
	        h.history['loss'])
	np.save(wdir + 'h' + str(num_rounds) + 'r_depth' + str(depth) + '_group' + str(group_size) + 'val_loss.npy',
	        h.history['val_loss'])
	dump(h.history, open(wdir + 'hist' + str(num_rounds) + 'r_depth_ks3' + str(depth) + '.p', 'wb'))
	
	print("Best validation accuracy: ", np.max(h.history['val_acc']))
	
	# save model
	# net.save('./Rscp/' + str(num_rounds) + '_' + str(depth) + '_rscp_distinguisher.h5')
	net.save(wdir + str(num_rounds) + '_' + str(depth) + '_' + str(
		group_size) + '_rscp_distinguisher.h5')
	# eval.evaluate(net,X_eval, Y_eval)
	return (net, h)

