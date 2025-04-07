import sys
import speck_mrcp as sp
import numpy as np

from keras.models import load_model
import tensorflow as tf
from os import urandom
from math import sqrt, log, log2
from time import time
from math import log2
import multiprocessing as mp
import gc

WORD_SIZE = sp.WORD_SIZE()


def key_average(ctdata0l, ctdata0r, ctdata1l, ctdata1r, ctdata2l, ctdata2r, ctdata3l, ctdata3r, ks_nr):
    # print("############")
    rsubkeys = np.arange(0, 2 ** WORD_SIZE, dtype=np.uint16)
    keys = rsubkeys ^ ks_nr

    # num_key = len(keys)

    # keys = np.repeat(keys, pairs)

    # ctdata0l = np.tile(ctdata0l,num_key)
    # ctdata0r = np.tile(ctdata0r,num_key)
    # ctdata1l = np.tile(ctdata1l,num_key)
    # ctdata1r = np.tile(ctdata1r,num_key)
    # ctdata2l = np.tile(ctdata2l, num_key)
    # ctdata2r = np.tile(ctdata2r, num_key)
    # ctdata3l = np.tile(ctdata3l, num_key)
    # ctdata3r = np.tile(ctdata3r, num_key)

    ctdata0l, ctdata0r = sp.dec_one_round((ctdata0l, ctdata0r), keys)
    ctdata1l, ctdata1r = sp.dec_one_round((ctdata1l, ctdata1r), keys)
    ctdata2l, ctdata2r = sp.dec_one_round((ctdata2l, ctdata2r), keys)
    ctdata3l, ctdata3r = sp.dec_one_round((ctdata3l, ctdata3r), keys)

    X = sp.convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r, ctdata2l, ctdata2r, ctdata3l, ctdata3r])

    # X = X.reshape(num_key,pairs,16*8)
    return X


def predict(X, net, bs):
    Z = net.predict(X, batch_size=bs)

    return Z


def wrong_key_decryption(net, bs, n=3000, diff=(0x0040, 0x0), nr=7):
    # 生成需要测试的明文和密文
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1);
    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16);
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16);
    plain1l = plain0l ^ diff[0];
    plain1r = plain0r ^ diff[1];
    plain2l = np.frombuffer(urandom(2 * n), dtype=np.uint16);
    plain2r = np.frombuffer(urandom(2 * n), dtype=np.uint16);
    plain3l = plain2l ^ diff[0];
    plain3r = plain2r ^ diff[1]
    keys_list = sp.expand_key(keys, nr + 1);
    # if (nr % 2 == 0):
    #   x = nr // 2
    # else:
    #   x = (nr // 2) + 1
    x = 3
    kr = []
    for i in range(x):
        kr.append(keys_list[i])

    ct0l, ct0r = sp.encrypt((plain0l, plain0r), kr)
    ct1l, ct1r = sp.encrypt((plain1l, plain1r), kr)
    ct2l, ct2r = sp.encrypt((plain2l, plain2r), kr)
    ct3l, ct3r = sp.encrypt((plain3l, plain3r), kr)

    joined_elements0l = []
    joined_elements0r = []
    joined_elements1l = []
    joined_elements1r = []
    joined_elements2l = []
    joined_elements2r = []
    joined_elements3l = []
    joined_elements3r = []
    ks = []
    # r = 0
    # ks = np.array([])
    for i in range(n):
        if ((ct0l[i] ^ ct1l[i] == ct2l[i] ^ ct3l[i]) and (ct0r[i] ^ ct1r[i] == ct2r[i] ^ ct3r[i])):

            joined_elements0l.append(ct0l[i])
            joined_elements0r.append(ct0r[i])
            joined_elements1l.append(ct1l[i])
            joined_elements1r.append(ct1r[i])
            joined_elements2l.append(ct2l[i])
            joined_elements2r.append(ct2r[i])
            joined_elements3l.append(ct3l[i])
            joined_elements3r.append(ct3r[i])
            k = []
            for j in range(nr + 1):
                k.append(keys_list[j][i])
            ks.append(k)

    joined_elements0l = np.array(joined_elements0l, dtype=np.uint16)
    joined_elements0r = np.array(joined_elements0r, dtype=np.uint16)
    joined_elements1l = np.array(joined_elements1l, dtype=np.uint16)
    joined_elements1r = np.array(joined_elements1r, dtype=np.uint16)
    joined_elements2l = np.array(joined_elements2l, dtype=np.uint16)
    joined_elements2r = np.array(joined_elements2r, dtype=np.uint16)
    joined_elements3l = np.array(joined_elements3l, dtype=np.uint16)
    joined_elements3r = np.array(joined_elements3r, dtype=np.uint16)
    kw = []
    for i in range(nr + 1):
        kq = []
        for j in range(len(ks)):
            kq.append(ks[j][i])
        kw.append(kq)
    # print(ks)
    # print(kw)
    # print(len(ks), len(kw))
    kt = []
    for i in range(x, nr + 1):
        kt.append(kw[i])
    kt = np.array(kt, dtype=np.uint16)
    # print(kt)
    ct0l, ct0r = sp.encrypt((joined_elements0l, joined_elements0r), kt)
    ct1l, ct1r = sp.encrypt((joined_elements1l, joined_elements1r), kt)
    ct2l, ct2r = sp.encrypt((joined_elements2l, joined_elements2r), kt)
    ct3l, ct3r = sp.encrypt((joined_elements3l, joined_elements3r), kt)
    print(len(ct0l))
    nr_key = kt[len(kt) - 1]

    # print("keys_list[nr] shape",keys_list[nr].shape)
    #
    # slices = 20
    #
    # ct0l = ct0l.reshape(slices, -1)
    # ct0r = ct0r.reshape(slices, -1)
    # ct1l = ct1l.reshape(slices, -1)
    # ct1r = ct1r.reshape(slices, -1)
    # ct2l = ct2l.reshape(slices, -1)
    # ct2r = ct2r.reshape(slices, -1)
    # ct3l = ct3l.reshape(slices, -1)
    # ct3r = ct3r.reshape(slices, -1)
    #
    # nr_key = np.copy(ks[nr])
    # nr_key = nr_key.reshape(slices,-1)
    #
    # # 开始执行多进程
    # process_number = mp.cpu_count()-4
    #
    # Z = []
    # for i in range(slices):
    #
    #     pool = mp.Pool(process_number)
    #     X = pool.starmap(key_average, [
    #                          (pairs, ct0l[i][j*pairs:(j+1)*pairs], ct0r[i][j*pairs:(j+1)*pairs], \
    #          ct1l[i][j*pairs:(j+1)*pairs], ct1r[i][j*pairs:(j+1)*pairs], nr_key[i][j*pairs],) for j in range(int(n/slices))])
    #     print("multiple processing end ......")
    #
    #     pool.close()
    #     pool.join()
    #
    #     X = np.array(X).flatten()
    #
    #     num_keys=2**16
    #     # X = X.reshape(n*num_keys,pairs*96)
    #     X = X.reshape(int(n/slices)*num_keys,pairs*96)
    #     Z.append(predict(X, net,bs))
    #     del X
    #     gc.collect()
    Z = []
    # for i in range(slices):
    #     X = [key_average(pairs, ct0l[i][j * pairs:(j + 1) * pairs], ct0r[i][j * pairs:(j + 1) * pairs], \
    #                      ct1l[i][j * pairs:(j + 1) * pairs], ct1r[i][j * pairs:(j + 1) * pairs], ct2l[i][j * pairs:(j + 1) * pairs], ct2r[i][j * pairs:(j + 1) * pairs],ct3l[i][j * pairs:(j + 1) * pairs], ct3r[i][j * pairs:(j + 1) * pairs], nr_key[i][j * pairs], )
    #          for j in range(int(n / slices))]
    #
    #     print("processing end ......")
    for i in range(len(ct0l)):
        # 从一维数组allkeys中随机选取，生成随机样本，样本长度为num_keys=1000
        # 依次将密文对X的每一列作为测试集，训练神经网络，得到分数值
        X = key_average(ct0l[i], ct0r[i], ct1l[i], ct1r[i], ct2l[i], ct2r[i], ct3l[i], ct3r[i], nr_key[i])
        # X = np.array(X).flatten()
        #
        # num_keys = 2 ** 16
        # X = X.reshape(int(n / slices) * num_keys, pairs * 128)
        Z.append(predict(X, net, bs))
        # del X
        # gc.collect()
    # Z = []
    # X = key_average(pairs=4, ctdata0l=ct0l, ctdata0r=ct0r, ctdata1l=ct1l,ctdata1r=ct1r, ctdata2l=ct2l, ctdata2r=ct2r, ctdata3l=ct3l,ctdata3r=ct3r,ks_nr = nr_key)
    # X = np.array(X).flatten()
    # num_keys=2**16
    # X = X.reshape(n*num_keys,pairs*128)
    # Z.append(predict(X, net,bs))
    Z = np.array(Z).flatten()
    Z = Z.reshape(len(ct0l), -1)
    mean = np.mean(Z, axis=0)
    std = np.std(Z, axis=0)

    print("mean shape", mean.shape)
    print("std shape", std.shape)

    return mean, std


if __name__ == "__main__":

    bs = 5000
    num = 10**5
    # 读取模型网络参数
    path = "./saved_model/rscp/"
    wdir = "./mean_std_rscp/"
    # net6 = load_model(path + "6_1_10_rscp_distinguisher.h5")
    # m6, s6 = wrong_key_decryption(net=net6, bs=bs, n=num, diff=(0x0040, 0x0), nr=6)
    # np.save(wdir + "data_wrong_key_mean_6r_" + "1g_rscp" + ".npy", m6)
    # np.save(wdir + "data_wrong_key_std_6r_" + "1g_rscp"  + ".npy", s6)
    net7 = load_model(path + "8_1_10_rscp_distinguisher.h5")
    m7, s7 = wrong_key_decryption(net=net7, bs=bs, n=num, diff=(0x0040, 0x0), nr=8)
    np.save(wdir + "data_wrong_key_mean_8r_" + "1g_rscp" + ".npy", m7)
    np.save(wdir + "data_wrong_key_std_8r_" + "1g_rscp" + ".npy", s7)

