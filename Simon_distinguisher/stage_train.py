
from keras.models import load_model, model_from_json


from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# import keras
from pickle import dump
import numpy as np
import pandas as pd
import simon_mrcp as si
# import os

import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

bs = 10000
wdir = './save_final/simon_9r_4plus5/'


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

# wdir = './good_trained_nets/'
# 不断修改学习率

def cyclic_lr(num_epochs, high_lr, low_lr):
	def res(i): return low_lr + ((num_epochs - 1) - i %
								 num_epochs) / (num_epochs - 1) * (high_lr - low_lr)

	return (res)


def make_checkpoint(datei):
	res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
	return (res)


def first_stage(num_rounds=8, group_size=8):



	net = load_model(wdir + '9_10_8_rscp_distinguisher.h5')
	net_json = net.to_json()

	net_first = model_from_json(net_json)
	# net_first.compile(optimizer=Adam(learning_rate = 10**-4), loss='mse', metrics=['acc'])
	net_first.compile(optimizer='adam', loss='mse', metrics=['acc'])
	net_first.load_weights(wdir + '9_10_8_rscp_distinguisher.h5')

	X, Y = si.make_dataset_with_group_size(n=group_size * 10 ** 7, nr=num_rounds, first_step=3,
										   group_size=group_size, diff=(0x0100,0x0040))
	X_eval, Y_eval = si.make_dataset_with_group_size(n=group_size * 10 ** 6, nr=num_rounds, first_step=3,
													 group_size=group_size, diff=(0x0100,0x0040))

	# X, Y = sp.make_train_data(n, nr=num_rounds-2, pairs=pairs,diff=(0x8100,0x8102))
	# X_eval,Y_eval = sp.make_train_data(test_n, nr=num_rounds-2, pairs=pairs,diff=(0x8100,0x8102))

	check = make_checkpoint(
		'./save_final/simon_10r_4plus6/' + 'first_best' + str(num_rounds) +  'groupsize' + str(group_size) + '.h5');
	lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
	net_first.fit(X, Y, epochs=20, batch_size=bs,
				  validation_data=(X_eval, Y_eval), callbacks=[lr, check])

	net_first.save(
		'./save_final/simon_10r_4plus6/net_first_'+str(num_rounds)+'_10_'+str(group_size)+'_rscp_distinguisher.h5')


def second_stage(num_rounds=8, group_size=8):
	# n=10**8

	X, Y = si.make_dataset_with_group_size(n=group_size * 10 ** 7, nr=num_rounds, first_step=3,
										   group_size=group_size, diff=(0x0,0x0040))
	X_eval, Y_eval = si.make_dataset_with_group_size(n=group_size * 10 ** 6, nr=num_rounds, first_step=3,
													 group_size=group_size, diff=(0x0,0x0040))

	net = load_model(
		'./save_final/simon_10r_4plus6/net_first_'+str(num_rounds)+'_10_'+str(group_size)+'_rscp_distinguisher.h5')
	net_json = net.to_json()

	net_second = model_from_json(net_json)
	net_second.compile(optimizer=Adam(learning_rate=10 ** -4), loss='mse', metrics=['acc'])
	# net_second.compile(optimizer='adam', loss='mse', metrics=['acc'])
	net_second.load_weights(
		'./save_final/simon_10r_4plus6/net_first_'+str(num_rounds)+'_10_'+str(group_size)+'_rscp_distinguisher.h5')

	check = make_checkpoint(
		'./save_final/simon_10r_4plus6/' + 'second_best' + str(num_rounds) + 'groupsize' + str(group_size) + '.h5')
	lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
	net_second.fit(X, Y, epochs=10, batch_size=bs,
				   validation_data=(X_eval, Y_eval), callbacks=[check])

	net_second.save(
		'./save_final/simon_10r_4plus6/net_second_'+str(num_rounds)+'_10_'+str(group_size)+'_rscp_distinguisher.h5')



def stage_train(num_rounds=8, group_size=8):
	# n=10**8


	X, Y = si.make_dataset_with_group_size(n=group_size * 10 ** 7, nr=num_rounds, first_step=3,
										   group_size=group_size, diff=(0x0,0x0040))
	X_eval, Y_eval = si.make_dataset_with_group_size(n=group_size * 10 ** 6, nr=num_rounds, first_step=3,
													 group_size=group_size, diff=(0x0,0x0040))

	net = load_model(
		'./save_final/simon_10r_4plus6/net_second_'+str(num_rounds)+'_10_'+str(group_size)+'_rscp_distinguisher.h5')
	net_json = net.to_json()

	net_third = model_from_json(net_json)
	# net_third.compile(optimizer=Adam(learning_ rate=10 ** -5), loss='mse', metrics=['acc'])
	net_third.compile(optimizer=Adam(learning_rate=10 ** -5), loss='mse', metrics=['acc'])
	# net_third.compile(optimizer='adam', loss='mse', metrics=['acc'])
	net_third.load_weights(
		'./save_final/simon_10r_4plus6/net_second_'+str(num_rounds)+'_10_'+str(group_size)+'_rscp_distinguisher.h5')
	check = make_checkpoint(
		'./save_final/simon_10r_4plus6/' + 'third_best' + str(num_rounds) +  'groupsize' + str(group_size)  + '.h5')
	lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
	net_third.fit(X, Y, epochs=10, batch_size=bs,
				  validation_data=(X_eval, Y_eval), callbacks=[check])

	net_third.save('./save_final/simon_10r_4plus6/' +  str(num_rounds) + '_10_' + str(group_size) + '_rscp_distinguisher.h5')


if __name__ == "__main__":
	first_stage(num_rounds=10, group_size=8)
	second_stage(num_rounds=10, group_size=8)
	stage_train(num_rounds=10, group_size=8)
