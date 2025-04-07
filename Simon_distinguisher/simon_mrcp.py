# L_{i+1}:=((L_{i}>>alpha)+(mod) R_{i})+k, R_{i+1}:=(R_{i} << beta)+L_{i+1}
import numpy as np
from os import urandom
from collections import deque
from time import time

def WORD_SIZE():
	return (16)

MASK_VAL = 2 ** WORD_SIZE() - 1

def rol(x, s):
	return (((x << s) & MASK_VAL) | (x >> (WORD_SIZE() - s)))

def enc_one_round(p, k):
	c0, c1 = p[0], p[1]
	# print("c0 shape",c0)
	ls_1_x = ((c0 >> (WORD_SIZE() - 1)) + (c0 << 1)) & MASK_VAL
	ls_8_x = ((c0 >> (WORD_SIZE() - 8)) + (c0 << 8)) & MASK_VAL
	ls_2_x = ((c0 >> (WORD_SIZE() - 2)) + (c0 << 2)) & MASK_VAL
	
	# XOR Chain
	xor_1 = (ls_1_x & ls_8_x) ^ c1
	xor_2 = xor_1 ^ ls_2_x
	# print("xor_2 = ",xor_2)
	new_c0 = k ^ xor_2
	return (new_c0, c0)

def dec_one_round(c, k):
	c0, c1 = c[0], c[1]
	ls_1_c1 = ((c1 >> (WORD_SIZE() - 1)) + (c1 << 1)) & MASK_VAL
	ls_8_c1 = ((c1 >> (WORD_SIZE() - 8)) + (c1 << 8)) & MASK_VAL
	ls_2_c1 = ((c1 >> (WORD_SIZE() - 2)) + (c1 << 2)) & MASK_VAL
	
	# Inverse XOR Chain
	xor_1 = k ^ c0
	xor_2 = xor_1 ^ ls_2_c1
	new_c0 = (ls_1_c1 & ls_8_c1) ^ xor_2
	
	return (c1, new_c0)

def expand_key(k, nr):
	# k = k & ((2 ** 64) - 1)
	zseq = 0b01100111000011010100100010111110110011100001101010010001011111
	ks = []
	k = np.transpose(k)
	for i in range(len(k)):
		ks.append([])
	# k_init = [[((k >> (WORD_SIZE() * (3 - i))) & MASK_VAL) for i in range(4)]]
	# print("k_init = ", k_init)
	for i in range(len(k)):
		k_init = k[i] & MASK_VAL
		k_reg = deque(k_init)
		rc = MASK_VAL ^ 3
		for x in range(nr):
			rs_3 = ((k_reg[0] << (WORD_SIZE() - 3)) + (k_reg[0] >> 3)) & MASK_VAL
			rs_3 = rs_3 ^ k_reg[2]
			rs_1 = ((rs_3 << (WORD_SIZE() - 1)) + (rs_3 >> 1)) & MASK_VAL
			c_z = ((zseq >> (x % 62)) & 1) ^ rc
			new_k = c_z ^ rs_1 ^ rs_3 ^ k_reg[3]
			ks[i].append(k_reg.pop())
			k_reg.appendleft(new_k)
	ks = np.array(ks, dtype=np.uint16)
	ks = np.transpose(ks)
	
	return ks

def encrypt(p, ks):
	x, y = p[0], p[1]
	for k in ks:
		# print("k  shape",k.shape)
		# print("k shape",k.shape)
		x, y = enc_one_round((x, y), k)
	return (x, y)

def decrypt(c, ks):
	x, y = c[0], c[1]
	for k in reversed(ks):
		x, y = dec_one_round((x, y), k)
	return (x, y)

def convert_to_binary(l):
	n = len(l)
	k = WORD_SIZE() * n
	X = np.zeros((k, len(l[0])), dtype=np.uint8)
	for i in range(k):
		index = i // WORD_SIZE()
		offset = WORD_SIZE() - 1 - i % WORD_SIZE()
		X[i] = (l[index] >> offset) & 1
	X = X.transpose()
	return (X)

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
	
		x = 3
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


def make_dataset_with_group_size(n, nr, first_step=3, group_size=2, diff=(0x0040, 0)):  # n = 10**8
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


# start_time = time()
# l_0, r_0, l_1, r_1, l_2, r_2, l_3, r_3 = make_train_data(10**7, 11, 3, 0, (0x0000,0x0040))
# print(len(l_0))
# end_time = time()
# print(f"代码运行时间: {end_time - start_time:.4f} 秒")