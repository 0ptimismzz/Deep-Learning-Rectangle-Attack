import numpy as np
from os import urandom
import pandas as pd
import csv
from collections import Counter
from os import urandom
import random
import time

block_size = 32
beta_shift = 2
alpha_shift = 7
word_size = block_size >> 1
mod_mask = (2 ** word_size) - 1
mod_mask_sub = (2 ** word_size)
key_size = 64


# key = random.getrandbits(64) & ((2 ** key_size) - 1)

def WORD_SIZE():
	return (16);


def ALPHA():
	return (7);


def BETA():
	return (2);


MASK_VAL = 2 ** WORD_SIZE() - 1;


def shuffle_together(l):
	state = np.random.get_state();
	for x in l:
		np.random.set_state(state);
		np.random.shuffle(x);


def rol(x, k):
	return (((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)));


def ror(x, k):
	return ((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL));


def enc_one_round(p, k):
	c0, c1 = p[0], p[1];
	c0 = ror(c0, ALPHA());
	c0 = (c0 + c1) & MASK_VAL;
	c0 = c0 ^ k;
	c1 = rol(c1, BETA());
	c1 = c1 ^ c0;
	return (c0, c1);


def dec_one_round(c, k):
	c0, c1 = c[0], c[1];
	c1 = c1 ^ c0;
	c1 = ror(c1, BETA());
	c0 = c0 ^ k;
	c0 = (c0 - c1) & MASK_VAL;
	c0 = rol(c0, ALPHA());
	return (c0, c1);


def expand_key(k, t):
	ks = [0 for i in range(t)]
	ks[0] = k[len(k) - 1]
	l = list(reversed(k[:len(k) - 1]))
	for i in range(t - 1):
		l[i % 3], ks[i + 1] = enc_one_round((l[i % 3], ks[i]), i)
	return (ks)


def encrypt(p, ks):
	x, y = p[0], p[1];
	for k in ks:
		x, y = enc_one_round((x, y), k);
	return (x, y);


def decrypt(c, ks):
	x, y = c[0], c[1];
	for k in reversed(ks):
		x, y = dec_one_round((x, y), k);
	return (x, y);


def check_testvector():
	key = (0x1918, 0x1110, 0x0908, 0x0100)
	pt = (0x6574, 0x694c)
	ks = expand_key(key, 22)
	ct = encrypt(pt, ks)
	if (ct == (0xa868, 0x42f2)):
		print("Testvector verified.")
		return (True);
	else:
		print("Testvector not verified.")
		return (False);


def encrypt_key(x, y, k):
	rs_x = ((x << (word_size - alpha_shift)) + (x >> alpha_shift)) & mod_mask
	add_sxy = (rs_x + y) & mod_mask
	new_x = add_sxy ^ k
	ls_y = ((y >> (word_size - beta_shift)) + (y << beta_shift)) & mod_mask
	new_y = new_x ^ ls_y
	return new_x, new_y


def key_schedule(key):
	key_schedule = [key & mod_mask]
	l_schedule = [(key >> (x * word_size)) & mod_mask for x in
	              range(1, key_size // word_size)]
	
	for x in range(22):
		new_l_k = encrypt_key(l_schedule[x], key_schedule[x], x)
		l_schedule.append(new_l_k[0])
		key_schedule.append(new_l_k[1])
	return key_schedule


# def check_testvector():
#     key = (0x1918, 0x1110, 0x0908, 0x0100)
#     pt = (0x6574, 0x694c)
#     ks = expand_key(key, 22)
#     print(ks)
#     ct = encrypt(pt, ks)
#     print(ct)
#     ct0 = dec_one_round(ct, ks[21])
#     print(ks[21])
#     print(ct0)
#     k1 = expand_key(key, 21)
#     print(k1)
#     c1 = encrypt(key, k1)
#     print(c1)
#     if (ct == (0xa868, 0x42f2)):
#         print("Testvector verified.")
#         return (True);
#     else:
#         print("Testvector not verified.")
#         return (False);
#
# check_testvector()
# convert_to_binary takes as input an array of ciphertext pairs
# where the first row of the array contains the lefthand side of the ciphertexts,
# the second row contains the righthand side of the ciphertexts,
# the third row contains the lefthand side of the second ciphertexts,
# and so on
# it returns an array of bit vectors containing the same data
def convert_to_binary(arr):
	X = np.zeros((8 * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
	for i in range(8 * WORD_SIZE()):
		index = i // WORD_SIZE()
		offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
		X[i] = (arr[index] >> offset) & 1
	X = X.transpose()
	return (X)


# takes a text file that contains encrypted block0, block1, true diff prob, real or random
# data samples are line separated, the above items whitespace-separated
# returns train data, ground truth, optimal ddt prediction
def readcsv(datei):
	data = np.genfromtxt(datei, delimiter=' ', converters={x: lambda s: int(s, 16) for x in range(2)});
	X0 = [data[i][0] for i in range(len(data))]
	X1 = [data[i][1] for i in range(len(data))]
	Y = [data[i][3] for i in range(len(data))]
	Z = [data[i][2] for i in range(len(data))]
	ct0a = [X0[i] >> 16 for i in range(len(data))]
	ct1a = [X0[i] & MASK_VAL for i in range(len(data))]
	ct0b = [X1[i] >> 16 for i in range(len(data))]
	ct1b = [X1[i] & MASK_VAL for i in range(len(data))]
	ct0a = np.array(ct0a, dtype=np.uint16)
	ct1a = np.array(ct1a, dtype=np.uint16)
	ct0b = np.array(ct0b, dtype=np.uint16)
	ct1b = np.array(ct1b, dtype=np.uint16)
	
	# X = [[X0[i] >> 16, X0[i] & 0xffff, X1[i] >> 16, X1[i] & 0xffff] for i in range(len(data))];
	X = convert_to_binary([ct0a, ct1a, ct0b, ct1b])
	Y = np.array(Y, dtype=np.uint8)
	Z = np.array(Z)
	return (X, Y, Z)


# baseline training data generator
def make_train_data(n, nr, first_step=3, random_switch=1, diff=(0x0040, 0)):
	# keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1);
	if random_switch == 0:
		keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
		plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
		plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
		plain1l = plain0l ^ diff[0]
		plain1r = plain0r ^ diff[1]
		plain2l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
		plain2r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
		plain3l = plain2l ^ diff[0]
		plain3r = plain2r ^ diff[1]
		keys_list = expand_key(keys, nr)
		
		# print(keys_list)
		# if (nr % 2 == 0):
		#   x = nr // 2
		# else:
		#   x = (nr // 2) + 1
		x = first_step
		kr = []
		for i in range(x):
			kr.append(keys_list[i])
		
		ct0l, ct0r = encrypt((plain0l, plain0r), kr)
		ct1l, ct1r = encrypt((plain1l, plain1r), kr)
		ct2l, ct2r = encrypt((plain2l, plain2r), kr)
		ct3l, ct3r = encrypt((plain3l, plain3r), kr)
		
		joined_elements0l = []
		joined_elements0r = []
		joined_elements1l = []
		joined_elements1r = []
		joined_elements2l = []
		joined_elements2r = []
		joined_elements3l = []
		joined_elements3r = []
		ks = []
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
				for j in range(nr):
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
		for i in range(nr):
			kq = []
			for j in range(len(ks)):
				kq.append(ks[j][i])
			kw.append(kq)
		# print(ks)
		# print(kw)
		# print(len(ks), len(kw))
		kt = []
		for i in range(x, nr):
			kt.append(kw[i])
		kt = np.array(kt, dtype=np.uint16)
		
		ck0l, ck0r = encrypt((joined_elements0l, joined_elements0r), kt)
		ck1l, ck1r = encrypt((joined_elements1l, joined_elements1r), kt)
		ck2l, ck2r = encrypt((joined_elements2l, joined_elements2r), kt)
		ck3l, ck3r = encrypt((joined_elements3l, joined_elements3r), kt)
		
		elements0l = []
		elements0r = []
		elements1l = []
		elements1r = []
		elements2l = []
		elements2r = []
		elements3l = []
		elements3r = []
		
		for i in range(len(ck0l)):
			elements0l.append(ck0l[i])
			elements0r.append(ck0r[i])
			elements1l.append(ck1l[i])
			elements1r.append(ck1r[i])
			elements2l.append(ck2l[i])
			elements2r.append(ck2r[i])
			elements3l.append(ck3l[i])
			elements3r.append(ck3r[i])
		
		return elements0l, elements0r, elements1l, elements1r, elements2l, elements2r, elements3l, elements3r
	if random_switch == 1:
		p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
		p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
		p1l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
		p1r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
		p2l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
		p2r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
		p3l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
		p3r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
		keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
		ks = expand_key(keys, nr)
		c0l, c0r = encrypt((p0l, p0r), ks)
		c1l, c1r = encrypt((p1l, p1r), ks)
		c2l, c2r = encrypt((p2l, p2r), ks)
		c3l, c3r = encrypt((p3l, p3r), ks)
		X = convert_to_binary([c0l, c0r, c1l, c1r, c2l, c2r, c3l, c3r])  #
		return X


def circulate_data(num, nr, first_step=3, diff=(0x0040, 0)):
	left0 = []
	right0 = []
	left1 = []
	right1 = []
	left2 = []
	right2 = []
	left3 = []
	right3 = []
	while True:
		l_0, r_0, l_1, r_1, l_2, r_2, l_3, r_3 = make_train_data(10 ** 7, nr, first_step, 0, diff)
		left0 = left0 + l_0
		right0 = right0 + r_0
		left1 = left1 + l_1
		right1 = right1 + r_1
		left2 = left2 + l_2
		right2 = right2 + r_2
		left3 = left3 + l_3
		right3 = right3 + r_3
		tn = len(left0)
		if tn > num:
			left0 = left0[:num]
			right0 = right0[:num]
			left1 = left1[:num]
			right1 = right1[:num]
			left2 = left2[:num]
			right2 = right2[:num]
			left3 = left3[:num]
			right3 = right3[:num]
			break
	print(len(left0))
	left0 = np.array(left0, dtype=np.uint16)
	right0 = np.array(right0, dtype=np.uint16)
	left1 = np.array(left1, dtype=np.uint16)
	right1 = np.array(right1, dtype=np.uint16)
	left2 = np.array(left2, dtype=np.uint16)
	right2 = np.array(right2, dtype=np.uint16)
	left3 = np.array(left3, dtype=np.uint16)
	right3 = np.array(right3, dtype=np.uint16)
	X = convert_to_binary([left0, right0, left1, right1, left2, right2, left3, right3])  # 16*4
	return X


def make_dataset_with_group_size(n, nr, first_step=3, group_size=1, diff=(0x0040, 0)):  # n = 10**8
	num = n // 2
	assert num % group_size == 0
	X_p = circulate_data(num, nr, first_step, diff)  # 固定差分
	X_n = make_train_data(num, nr, first_step, 1, diff)  # 随机差分
	Y_p = [1 for i in range(num // group_size)]
	Y_n = [0 for i in range(num // group_size)]
	X = np.concatenate((X_p, X_n), axis=0).reshape(n // group_size, -1)
	print(X.shape)
	Y = np.concatenate((Y_p, Y_n))
	shuffle_indices = np.random.permutation(len(X))
	shuffled_X = X[shuffle_indices]
	shuffled_Y = Y[shuffle_indices]
	return shuffled_X, shuffled_Y

# start_time = time.time()
# l_0, r_0, l_1, r_1, l_2, r_2, l_3, r_3 = make_train_data(10 ** 7, 8, 4, 0, diff=(0x0040, 0))
# end_time = time.time()
# l0, r0, l1, r1, l2, r2, l3, r3 = make_train_data(10 ** 7, 8, 3, 0, diff=(0x0040, 0))
# print(len(l_0))
# print(len(l0))
# print(f"代码运行时间: {end_time - start_time:.4f} 秒")
