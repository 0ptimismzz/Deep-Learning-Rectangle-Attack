
from keras.models import load_model, model_from_json


from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# import keras
from pickle import dump

import speck_mrcp as sp
# import os

import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

bs = 5000
wdir = './change_paramens/'


# wdir = './good_trained_nets/'
# 不断修改学习率

def cyclic_lr(num_epochs, high_lr, low_lr):
    def res(i): return low_lr + ((num_epochs - 1) - i %
                                 num_epochs) / (num_epochs - 1) * (high_lr - low_lr)

    return (res)


def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return (res)


def first_stage(n, num_rounds=8, group_size=8):
    eval_n = int(n / 10)


    net = load_model(wdir + '7_2_10_mrcp_ci_distinguisher.h5')
    net_json = net.to_json()

    net_first = model_from_json(net_json)
    # net_first.compile(optimizer=Adam(learning_rate = 10**-4), loss='mse', metrics=['acc'])
    net_first.compile(optimizer='adam', loss='mse', metrics=['acc'])
    net_first.load_weights(wdir + '7_2_10_mrcp_ci_distinguisher.h5')
    X, Y = sp.make_dataset_with_group_size(n, nr=num_rounds-3, first_step=1, group_size=group_size, diff=(0x8000, 0x840a))
    X_eval, Y_eval = sp.make_dataset_with_group_size(eval_n, nr=num_rounds-3, first_step=1, group_size=group_size, diff=(0x8000, 0x840a))


    # X, Y = sp.make_train_data(n, nr=num_rounds-2, pairs=pairs,diff=(0x8100,0x8102))
    # X_eval,Y_eval = sp.make_train_data(test_n, nr=num_rounds-2, pairs=pairs,diff=(0x8100,0x8102))

    check = make_checkpoint(
        './change_paramens/stage_train/' + 'first_best' + str(num_rounds) +  'groupsize' + str(group_size) + '.h5');
    lr = LearningRateScheduler(cyclic_lr(10, 0.0035, 0.00022))
    net_first.fit(X, Y, epochs=20, batch_size=bs,
                  validation_data=(X_eval, Y_eval), callbacks=[lr, check])

    net_first.save(
	    './change_paramens/stage_train/net_first_'+str(num_rounds)+'_'+str(group_size)+'_10_mrcp_ci_distinguisher.h5')


def second_stage(n, num_rounds=8, group_size=8):
    # n=10**8
    eval_n = int(n / 10)
    X, Y = sp.make_dataset_with_group_size(n, nr=num_rounds, group_size=group_size, diff=(0x0040, 0))
    X_eval, Y_eval = sp.make_dataset_with_group_size(eval_n, nr=num_rounds, group_size=group_size, diff=(0x0040, 0))

    net = load_model(
	    './change_paramens/stage_train/net_first_'+str(num_rounds) +'_'+str(group_size)+'_10_mrcp_ci_distinguisher.h5')
    net_json = net.to_json()

    net_second = model_from_json(net_json)
    net_second.compile(optimizer=Adam(learning_rate=10 ** -4), loss='mse', metrics=['acc'])
    # net_second.compile(optimizer='adam', loss='mse', metrics=['acc'])
    net_second.load_weights(
	    './change_paramens/stage_train/net_first_'+str(num_rounds) +'_'+str(group_size)+'_10_mrcp_ci_distinguisher.h5')

    check = make_checkpoint(
        './change_paramens/stage_train/' + 'second_best' + str(num_rounds) + 'groupsize' + str(group_size) + '.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.0035, 0.00022))
    net_second.fit(X, Y, epochs=10, batch_size=bs,
                   validation_data=(X_eval, Y_eval), callbacks=[check])

    net_second.save(
        './change_paramens/stage_train/net_second_'+str(num_rounds) +'_'+str(group_size)+'_10_mrcp_ci_distinguisher.h5')



def stage_train(n, num_rounds=8, group_size=8):
    # n=10**8
    eval_n = int(n / 10)

    X, Y = sp.make_dataset_with_group_size(n, nr=num_rounds, group_size=group_size, diff=(0x0040, 0))
    X_eval, Y_eval = sp.make_dataset_with_group_size(eval_n, nr=num_rounds, group_size=group_size, diff=(0x0040, 0))

    net = load_model(
	    './change_paramens/stage_train/net_second_'+str(num_rounds) +'_'+str(group_size)+'_10_mrcp_ci_distinguisher.h5')
    net_json = net.to_json()

    net_third = model_from_json(net_json)
    # net_third.compile(optimizer=Adam(learning_rate=10 ** -5), loss='mse', metrics=['acc'])
    net_third.compile(optimizer=Adam(learning_rate=10 ** -5), loss='mse', metrics=['acc'])
    # net_third.compile(optimizer='adam', loss='mse', metrics=['acc'])
    net_third.load_weights(
	    './change_paramens/stage_train/net_second_'+str(num_rounds) +'_'+str(group_size)+'_10_mrcp_ci_distinguisher.h5')
    check = make_checkpoint(
        './change_paramens/stage_train/' + 'third_best' + str(num_rounds) +  'groupsize' + str(group_size)  + '.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.0035, 0.00022))
    net_third.fit(X, Y, epochs=10, batch_size=bs,
                  validation_data=(X_eval, Y_eval), callbacks=[check])

    net_third.save('./change_paramens/' +  str(num_rounds) + '_' + str(group_size) + '_10_mrcp_ci_distinguisher.h5')


if __name__ == "__main__":
    # (0040,0000)->(8000,8000)->(8100,8102)->(8000,840a)->(850a,9520)
    first_stage(n=2*10 ** 7, num_rounds=8, group_size=2)
    second_stage(n=2*10 ** 7, num_rounds=8, group_size=2)
    stage_train(n=2*10 ** 7, num_rounds=8, group_size=2)
