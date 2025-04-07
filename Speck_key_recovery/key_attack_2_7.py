
import speck_mrcp as sp
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
from os import urandom
from math import sqrt, log, log2
from time import time
from math import log2

# WORD_SIZE=16
WORD_SIZE = sp.WORD_SIZE();
wdir = "./mean_std_rscp/"
path = "./saved_model/rscp/"
neutral13 = [22, 21, 20, 14, 15,  7, 23, 30,  0, 24,  8, 31,  1];


net6 = load_model(path + "6_1_10_rscp_distinguisher.h5")
net7 = load_model(path + "7_1_10_rscp_distinguisher.h5")

m7 = np.load(wdir + "data_wrong_key_mean_7r_1g_rscp.npy");
s7 = np.load(wdir + "data_wrong_key_std_7r_1g_rscp.npy"); s7 = 1.0/s7;
m6 = np.load(wdir + "data_wrong_key_mean_6r_1g_rscp.npy");
s6 = np.load(wdir + "data_wrong_key_std_6r_1g_rscp.npy"); s6 = 1.0/s6;

#binarize a given ciphertext sample
#ciphertext is given as a sequence of arrays
#each array entry contains one word of ciphertext for all ciphertexts given


'''
对给定的密文样本进行二值化
密文以数组序列的形式给出
对于给定的所有密文，每个数组条目都包含一个密文单词
'''

# convert_to_binary函数与speck.py文件中定义的函数一样
# 用来生成神经网络的输入数据（二进制化）
def convert_to_binary(l):
  n = len(l);
  # WORD_SIZE=16
  k = WORD_SIZE * n;
  X = np.zeros((k, len(l[0])),dtype=np.uint8);
  # WORD_SIZE=16
  for i in range(k):
    index = i // WORD_SIZE;
    offset = WORD_SIZE - 1 - i%WORD_SIZE;
    X[i] = (l[index] >> offset) & 1;
  # 将X进行转置
  X = X.transpose();
  return(X);

# 计算数组v中每个元素含有1的个数
def hw(v):
  # 数组res的形状与v一致，数据类型为无符号8位整型
  res = np.zeros(v.shape,dtype=np.uint8);
  for i in range(16):
    res = res + ((v >> i) & 1)
  return(res);

low_weight = np.array(range(2**WORD_SIZE), dtype=np.uint16);
# 数组low_weight中每个元素含有1的个数不超过2
low_weight = low_weight[hw(low_weight) <= 2];

#make a plaintext structure
# 创建明文结构
#takes as input a sequence of plaintexts, a desired plaintext input difference, and a set of neutral bits
# 以一个明文序列、一个所需的明文输入差分和一组中性位作为输入
def make_structure(pt0l, pt0r, pt2l, pt2r,diff=(0x211,0xa04),neutral_bits = [20,21,22,14,15]):
  p0l = np.copy(pt0l); p0r = np.copy(pt0r);
  p2l = np.copy(pt2l);
  p2r = np.copy(pt2r);
  # 改变数组形状：将数组p0和p1均变为1列，行数任意
  p0l = p0l.reshape(-1,1); p0r = p0r.reshape(-1,1);
  p2l = p2l.reshape(-1, 1);
  p2r = p2r.reshape(-1, 1);
  for i in neutral_bits:
    d = 1 << i; d0 = d >> 16; d1 = d & 0xffff
    # 两数组拼接，axis=1表示对应行的数组进行拼接
    p0l = np.concatenate([p0l, p0l ^ d0], axis=1);
    p0r = np.concatenate([p0r, p0r ^ d1], axis=1);
    p2l = np.concatenate([p2l, p2l ^ d0], axis=1);
    p2r = np.concatenate([p2r, p2r ^ d1], axis=1);
  p1l = p0l ^ diff[0]; p1r = p0r ^ diff[1];
  p3l = p2l ^ diff[0];
  p3r = p2r ^ diff[1];
  return(p0l, p0r, p1l, p1r, p2l, p2r, p3l, p3r);
# p0,p1,p0b,p1b都是100行32列的数组，得到的100个明文结构是由12800个选择的明文组成

#generate a Speck key, return expanded key
# 生成一个Speck密钥，返回扩展密钥
def gen_key(nr):
  # 随机生成密钥，数据类型为无符号8位整型
  key = np.frombuffer(urandom(8),dtype=np.uint16);
  # 将密钥key扩展至nr轮
  ks = sp.expand_key(key, nr);
  return(ks);

def gen_plain(n):
  # 随机生成明文对(pt0,pt1)，数据类型为无符号16位整型
  pt0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  pt0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  pt2l = np.frombuffer(urandom(2 * n), dtype=np.uint16);
  pt2r = np.frombuffer(urandom(2 * n), dtype=np.uint16);
  return(pt0l, pt0r, pt2l, pt2r)

def gen_challenge(n, nr, diff=(0x211, 0xa04), neutral_bits = [20,21,22,14,15,23], keyschedule='real'):
  # 随机生成明文对(pt0,pt1)，数据类型为无符号16位整型
  pt0l, pt0r, pt2l, pt2r = gen_plain(n);
  # 创建明文结构
  p0l, p0r, p1l, p1r, p2l, p2r, p3l, p3r = make_structure(pt0l, pt0r, pt2l, pt2r, diff=diff, neutral_bits=neutral_bits);
  # 分别对明文对：(pt0a,pt1a)、(pt0b,pt1b)进行一轮解密
  p0l, p0r = sp.dec_one_round((p0l, p0r),0);
  p1l, p1r = sp.dec_one_round((p1l, p1r),0);
  p2l, p2r = sp.dec_one_round((p2l, p2r),0);
  p3l, p3r = sp.dec_one_round((p3l, p3r),0);
  # 将密钥扩展至nr轮
  keys_list = gen_key(nr);
  if (keyschedule == 'free'): keys_list = np.frombuffer(urandom(2*nr),dtype=np.uint16);
  # 明文对pt0a和pt1a进行nr轮加密，得到密文对ct0a和ct1a
  x = nr // 2
  kr = []
  for i in range(x):
    kr.append(keys_list[i])
  # kr = np.array(kr, dtype=np.uint16)
  ct0l, ct0r = sp.encrypt((p0l, p0r), kr)
  ct1l, ct1r = sp.encrypt((p1l, p1r), kr)
  ct2l, ct2r = sp.encrypt((p2l, p2r), kr)
  ct3l, ct3r = sp.encrypt((p3l, p3r), kr)
  joined_element0l = []
  joined_element0r = []
  joined_element1l = []
  joined_element1r = []
  joined_element2l = []
  joined_element2r = []
  joined_element3l = []
  joined_element3r = []

  joined_elements0l = []
  joined_elements0r = []
  joined_elements1l = []
  joined_elements1r = []
  joined_elements2l = []
  joined_elements2r = []
  joined_elements3l = []
  joined_elements3r = []
  n_blocks = len(ct0l)
  n_rt = 2 ** len(neutral_bits)

  for i in range(n_blocks):
    for j in range(n_rt):
      if ((ct0l[i][j] ^ ct1l[i][j] == ct2l[i][j] ^ ct3l[i][j]) and (ct0r[i][j] ^ ct1r[i][j] == ct2r[i][j] ^ ct3r[i][j])):
        joined_element0l.append(ct0l[i][j])
        joined_element0r.append(ct0r[i][j])
        joined_element1l.append(ct1l[i][j])
        joined_element1r.append(ct1r[i][j])
        joined_element2l.append(ct2l[i][j])
        joined_element2r.append(ct2r[i][j])
        joined_element3l.append(ct3l[i][j])
        joined_element3r.append(ct3r[i][j])

    joined_elements0l.append(joined_element0l)
    joined_elements0r.append(joined_element0r)
    joined_elements1l.append(joined_element1l)
    joined_elements1r.append(joined_element1r)
    joined_elements2l.append(joined_element2l)
    joined_elements2r.append(joined_element2r)
    joined_elements3l.append(joined_element3l)
    joined_elements3r.append(joined_element3r)

  joined_elements0l = np.array(joined_elements0l, dtype=np.uint16)
  joined_elements0r = np.array(joined_elements0r, dtype=np.uint16)
  joined_elements1l = np.array(joined_elements1l, dtype=np.uint16)
  joined_elements1r = np.array(joined_elements1r, dtype=np.uint16)
  joined_elements2l = np.array(joined_elements2l, dtype=np.uint16)
  joined_elements2r = np.array(joined_elements2r, dtype=np.uint16)
  joined_elements3l = np.array(joined_elements3l, dtype=np.uint16)
  joined_elements3r = np.array(joined_elements3r, dtype=np.uint16)
  kt = []
  for i in range(x, nr):
    kt.append(keys_list[i])
  kt = np.array(kt, dtype=np.uint16)
  # print(kt)
  ck0l, ck0r = sp.encrypt((joined_elements0l, joined_elements0r), kt)
  ck1l, ck1r = sp.encrypt((joined_elements1l, joined_elements1r), kt)
  ck2l, ck2r = sp.encrypt((joined_elements2l, joined_elements2r), kt)
  ck3l, ck3r = sp.encrypt((joined_elements3l, joined_elements3r), kt)
  print(ck0l.shape)
  return([ck0l, ck0r, ck1l, ck1r, ck2l, ck2r, ck3l, ck3r], keys_list);

# 统计明文对差分等于(0x0040,0x0)的个数
def find_good(cts, key, nr=3, target_diff = (0x0040,0x0)):
  # 解密运算
  pt0l, pt0r = sp.decrypt((cts[0], cts[1]), key[nr:]);
  pt1l, pt1r = sp.decrypt((cts[2], cts[3]), key[nr:]);
  pt2l, pt2r = sp.decrypt((cts[4], cts[5]), key[nr:]);
  pt3l, pt3r = sp.decrypt((cts[6], cts[7]), key[nr:]);
  # 获得明文对输入差分
  diff0 = pt0l ^ pt1l;
  diff1 = pt0r ^ pt1r;
  diff2 = pt2l ^ pt3l;
  diff3 = pt2r ^ pt3r;
  d0 = (diff0 == target_diff[0]);
  d1 = (diff1 == target_diff[1]);
  d2 = (diff2 == target_diff[0]);
  d3 = (diff3 == target_diff[1]);
  d = d0 * d1 * d2 * d3;
  # 对数组d的每行数据求和得到数组v  axis=0:对列求和  axis=1:对行求和（如果是一维数组，axis=0即为对所有元素求和）
  v = np.sum(d,axis=1);
  return(v);

#having a good key candidate, exhaustively explore all keys with hamming distance less than two of this key
# 有一个好的候选密钥，穷尽地探索所有密钥的汉明距离小于该密钥的两个密钥
def verifier_search(cts, best_guess, use_n = 64, net = net6):
  #print(best_guess);
  # ck1与best_guess[0]汉明距离不超过2
  ck1 = best_guess[0] ^ low_weight;
  # ck2与best_guess[1]汉明距离不超过2
  ck2 = best_guess[1] ^ low_weight;
  n = len(ck1);
  # np.repeat():重复数组中的元素，当axis=None时，数组就会变成一个行向量
  ck1 = np.repeat(ck1, n); keys1 = np.copy(ck1);
  # np.tile():将数组ck2当成元素，把数组ck2进行扩展
  ck2 = np.tile(ck2, n); keys2 = np.copy(ck2);
  ck1 = np.repeat(ck1, use_n);
  ck2 = np.repeat(ck2, use_n);
  # 将数组cts降维，分别降为一维数组后切片处理得到数组ct0a,ct1a,ct0b,ct1b
  ct0l = np.tile(cts[0][0:use_n], n*n);
  ct0r = np.tile(cts[1][0:use_n], n*n);
  ct1l = np.tile(cts[2][0:use_n], n*n);
  ct1r = np.tile(cts[3][0:use_n], n*n);
  ct2l = np.tile(cts[4][0:use_n], n * n);
  ct2r = np.tile(cts[5][0:use_n], n * n);
  ct3l = np.tile(cts[6][0:use_n], n * n);
  ct3r = np.tile(cts[7][0:use_n], n * n);
  # 一轮解密运算
  pt0l, pt0r = sp.dec_one_round((ct0l, ct0r), ck1);
  pt1l, pt1r = sp.dec_one_round((ct1l, ct1r), ck1);
  pt2l, pt2r = sp.dec_one_round((ct2l, ct2r), ck1);
  pt3l, pt3r = sp.dec_one_round((ct3l, ct3r), ck1);
  pt0l, pt0r = sp.dec_one_round((pt0l, pt0r), ck2);
  pt1l, pt1r = sp.dec_one_round((pt1l, pt1r), ck2);
  pt2l, pt2r = sp.dec_one_round((pt2l, pt2r), ck2);
  pt3l, pt3r = sp.dec_one_round((pt3l, pt3r), ck2);
  # 生成神经网络的输入数据X
  X = sp.convert_to_binary([pt0l, pt0r, pt1l, pt1r, pt2l, pt2r, pt3l, pt3r]);
  # 训练网络，得到分数值
  Z = net.predict(X, batch_size=10000);
  Z = Z / (1 - Z);
  # 返回以2为底的对数Z的值组成的数组
  Z = np.log2(Z);
  # 改变数组Z的形状：列数为use_n，行数任意
  Z = Z.reshape(-1, use_n);
  # 对数组Z的各行求算术平均值  axis=1:对各行求算术平均值
  v = np.mean(Z, axis=1) * len(cts[0]);
  # 返回数组v中最大值的索引
  m = np.argmax(v); val = v[m];
  key1 = keys1[m]; key2 = keys2[m];
  return(key1, key2, val);


#test wrong-key decryption
# 测试错误密钥解密
def wrong_key_decryption(n, diff=(0x0040,0x0), nr=7, net = net7):
  means = np.zeros(2**16); sig = np.zeros(2**16);
  for i in range(2**16):
    # 生成随机密钥，数据类型为无符号16位整型   二维数组分为4行，不知列数
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
    # 将密钥keys扩展至nr+1轮
    ks = sp.expand_key(keys, nr+1); #ks[nr-1] = 17123;
    # 生成明文对(pt0a,pt1a)，数据类型为无符号16位整型
    pt0a = np.frombuffer(urandom(2*n),dtype=np.uint16);
    pt1a = np.frombuffer(urandom(2*n),dtype=np.uint16);
    # pt0a和pt0b是输入差为0x0040/0000的明文对
    # pt1a和pt1b是两部分完全相同的明文对
    pt0b, pt1b = pt0a ^ diff[0], pt1a ^ diff[1];
    # 明文对pt0a和pt1a进行nr+1轮加密，得到密文对ct0a和ct1a
    ct0a, ct1a = sp.encrypt((pt0a, pt1a), ks);
    # 明文对pt0b和pt1b进行nr+1轮加密，得到密文对ct0b和ct1b
    ct0b, ct1b = sp.encrypt((pt0b, pt1b), ks);
    # i与数组ks的最后一个元素做异或运算
    rsubkeys = i ^ ks[nr];
  #rsubkeys = rdiff ^ 0;
    # 一轮解密运算
    c0a, c1a = sp.dec_one_round((ct0a, ct1a),rsubkeys);
    c0b, c1b = sp.dec_one_round((ct0b, ct1b),rsubkeys);
    # 生成训练神经网络的输入数据X
    X = sp.convert_to_binary([c0a, c1a, c0b, c1b]);
    # 训练神经网络，得到分数值
    Z = net.predict(X,batch_size=10000);
    # 将数组Z平铺，变为一维数组
    Z = Z.flatten();
    # 获得数组Z的算术平均值
    means[i] = np.mean(Z);
    # 获得数组Z的标准偏差
    sig[i] = np.std(Z);
  return(means, sig);

#here, we use some symmetries of the wrong key performance profile
#by performing the optimization step only on the 14 lowest bits and randomizing the others
#on CPU, this only gives a very minor speedup, but it is quite useful if a strong GPU is available
#In effect, this is a simple partial mitigation of the fact that we are running single-threaded numpy code here


'''
这里，我们使用了一些错误的密钥性能配置文件的对称性
通过只对最低的14位执行优化步骤，并将其他的随机化
在CPU上，这只能提供一个非常小的加速，但如果有一个强大的GPU可用，这是非常有用的
实际上，这是对我们运行单线程numpy代码这一事实的一个简单的部分缓解
'''

tmp_br = np.arange(2**14, dtype=np.uint16);
tmp_br = np.repeat(tmp_br, 32).reshape(-1,32);

def bayesian_rank_kr(cand, emp_mean, m=m7, s=s7):
  global tmp_br;
  n = len(cand);
  # tmp_br.shape[1]:tmp_br数组形状的第二维度
  if (tmp_br.shape[1] != n):
      tmp_br = np.arange(2**14, dtype=np.uint16);
      tmp_br = np.repeat(tmp_br, n).reshape(-1,n);
  tmp = tmp_br ^ cand;
  v = (emp_mean - m[tmp]) * s[tmp];
  v = v.reshape(-1, n);
  # 返回矩阵v的范数 axis=1表示按行向量处理，求多个行向量的范数，不保持矩阵的二维特性
  scores = np.linalg.norm(v, axis=1);
  return(scores);

def bayesian_key_recovery(cts, net=net7, m = m7, s = s7, num_cand = 32, num_iter=5, seed = None):
  n = len(cts[0]);
  # 从给定的一维数组生成一个随机样本，num_cand:样本数量 replace:用来设置是否可以取相同元素，True表示可以取相同数字，False表示不可以取相同数字
  keys = np.random.choice(2**(WORD_SIZE-2),num_cand,replace=False); scores = 0; best = 0;
  if (not seed is None):
    keys = np.copy(seed);
  # np.tile():将数组当成元素进行扩展，扩展次数为num_cand
  ct0l, ct0r, ct1l, ct1r = np.tile(cts[0],num_cand), np.tile(cts[1], num_cand), np.tile(cts[2], num_cand), np.tile(cts[3], num_cand);
  ct2l, ct2r, ct3l, ct3r = np.tile(cts[4],num_cand), np.tile(cts[5], num_cand), np.tile(cts[6], num_cand), np.tile(cts[7], num_cand);
  scores = np.zeros(2**(WORD_SIZE-2));
  used = np.zeros(2**(WORD_SIZE-2));
  all_keys = np.zeros(num_cand * num_iter,dtype=np.uint16);
  all_v = np.zeros(num_cand * num_iter);
  for i in range(num_iter):
    k = np.repeat(keys, n);
    # 一轮解密运算
    c0l, c0r = sp.dec_one_round((ct0l, ct0r),k); c1l, c1r = sp.dec_one_round((ct1l, ct1r),k);
    c2l, c2r = sp.dec_one_round((ct2l, ct2r), k);
    c3l, c3r = sp.dec_one_round((ct3l, ct3r), k);
    # 生成训练神经网络的输入数据X
    X = sp.convert_to_binary([c0l, c0r, c1l, c1r, c2l, c2r, c3l, c3r]);
    # 训练神经网络，得到分数值
    Z = net.predict(X,batch_size=10000);
    # 改变数组Z的形状：行数为num_cand，列数任意
    Z = Z.reshape(num_cand, -1);
    # 对数组Z的各行求算术平均值  axis=1:对各行求算术平均值
    means = np.mean(Z, axis=1);
    # np.sum():对数组Z的每行数据求和得到数组v  axis=0:对列求和  axis=1:对行求和（如果是一维数组，axis=0即为对所有元素求和）
    Z = Z/(1-Z); Z = np.log2(Z); v =np.sum(Z, axis=1); all_v[i * num_cand:(i+1)*num_cand] = v;
    all_keys[i * num_cand:(i+1)*num_cand] = np.copy(keys);
    # 返回范数
    scores = bayesian_rank_kr(keys, means, m=m, s=s);
    # 划分重组数组,返回的是重组后数据的索引数组，可以很快地找出第k大的数的位置，以及大于k（排在k后面）和小于k（排在k前面）的数的位置 num_cand：要分区的元素索引
    tmp = np.argpartition(scores+used, num_cand)
    keys = tmp[0:num_cand];
    r = np.random.randint(0,4,num_cand,dtype=np.uint16); r = r << 14; keys = keys ^ r;
  return(all_keys, scores, all_v);

def run_bayes(cts,it=1, cutoff1=10, cutoff2=10, net=net7, net_help=net6, m_main=m7, m_help=m6, s_main=s7, s_help=s6, verify_breadth=None):
  n = len(cts[0]);
  if (verify_breadth is None): verify_breadth=len(cts[0][0]);
  # 返回n的平方根
  alpha = sqrt(n);
  best_val = -100.0; best_key = (0,0); best_pod = 0; bp = 0; bv = -100.0;
  # 从给定的一维数组生成一个随机样本  replace:用来设置是否可以取相同元素，True表示可以取相同数字，False表示不可以取相同数字
  keys = np.random.choice(2**WORD_SIZE, 32, replace=False);
  # 返回指定形状和类型的新数组，数组是一维数组，由n个元素组成，填充值为-10或eps
  eps = 0.001; local_best = np.full(n,-10); num_visits = np.full(n,eps);
  guess_count = np.zeros(2**16,dtype=np.uint16);
  for j in range(it):
      # 优先分数
      priority = local_best + alpha * np.sqrt(log2(j+1) / num_visits); i = np.argmax(priority);
      # 访问次数+1
      num_visits[i] = num_visits[i] + 1;
      if (best_val > cutoff2):
        improvement = (verify_breadth > 0);
        while improvement:
          k1, k2, val = verifier_search([cts[0][best_pod], cts[1][best_pod], cts[2][best_pod], cts[3][best_pod], cts[4][best_pod], cts[5][best_pod], cts[6][best_pod], cts[7][best_pod]], best_key,net=net_help, use_n = verify_breadth);
          improvement = (val > best_val);
          if (improvement):
            best_key = (k1, k2); best_val = val;
        return(best_key, j);
      keys, scores, v = bayesian_key_recovery([cts[0][i], cts[1][i], cts[2][i], cts[3][i], cts[4][i], cts[5][i], cts[6][i], cts[7][i]], num_cand=32, num_iter=5,net=net, m=m_main, s=s_main);
      # 返回数组v的最大值
      vtmp = np.max(v);
      if (vtmp > local_best[i]): local_best[i] = vtmp;
      if (vtmp > bv):
        bv = vtmp; bp = i;
      if (vtmp > cutoff1):
        l2 = [i for i in range(len(keys)) if v[i] > cutoff1];
        for i2 in l2:
          # 一轮解密运算
          c0l, c0r = sp.dec_one_round((cts[0][i],cts[1][i]),keys[i2]);
          c1l, c1r = sp.dec_one_round((cts[2][i],cts[3][i]),keys[i2]);
          c2l, c2r = sp.dec_one_round((cts[4][i], cts[5][i]), keys[i2]);
          c3l, c3r = sp.dec_one_round((cts[6][i], cts[7][i]), keys[i2]);
          keys2,scores2,v2 = bayesian_key_recovery([c0l, c0r, c1l, c1r, c2l, c2r, c3l, c3r],num_cand=32, num_iter=5, m=m6,s=s6,net=net_help);
          vtmp2 = np.max(v2);
          if (vtmp2 > best_val):
            best_val = vtmp2; best_key = (keys[i2], keys2[np.argmax(v2)]); best_pod=i;
  improvement = (verify_breadth > 0);
  while improvement:
    k1, k2, val = verifier_search([cts[0][best_pod], cts[1][best_pod], cts[2][best_pod], cts[3][best_pod], cts[4][best_pod], cts[5][best_pod], cts[6][best_pod], cts[7][best_pod]], best_key, net=net_help, use_n = verify_breadth);
    improvement = (val > best_val);
    if (improvement):
      best_key = (k1, k2); best_val = val;
  return(best_key, it);

def run(n, nr=11, num_structures=40000, it=500, cutoff1=7, cutoff2=10, neutral_bits=[20,21,22,14,15,23], keyschedule='real',net=net7, net_help=net6, m_main=m7, s_main=s7,  m_help=m6, s_help=s6, verify_breadth=None):
  print("Checking Speck32/64 implementation.");
  if (not sp.check_testvector()):
    print("Error. Aborting.");
    return(0);
  arr1 = np.zeros(n, dtype=np.uint16); arr2 = np.zeros(n, dtype=np.uint16);
  # 返回自epoch以来的当前时间(以秒为单位)，如果系统时钟提供，可能会出现一秒的分数
  t0 = time();
  data = 0; av=0.0; good = np.zeros(n, dtype=np.uint8);
  zkey = np.zeros(nr,dtype=np.uint16);
  for i in range(n):
    print("Test:",i);
    ct, key = gen_challenge(num_structures,nr, neutral_bits=neutral_bits, keyschedule=keyschedule);
    g = find_good(ct, key); g = np.max(g); good[i] = g;
    guess, num_used = run_bayes(ct,it=it, cutoff1=cutoff1, cutoff2=cutoff2, net=net, net_help=net_help, m_main=m_main, s_main=s_main, m_help=m_help, s_help=s_help, verify_breadth=verify_breadth);
    num_used = min(num_structures, num_used); data = data + 2 * (2 ** len(neutral_bits)) * num_used;
    # 异或运算
    arr1[i] = guess[0] ^ key[nr-1]; arr2[i] = guess[1] ^ key[nr-2];
    print("Difference between real key and key guess: ", hex(arr1[i]), hex(arr2[i]));
  # 返回自epoch以来的当前时间(以秒为单位)，如果系统时钟提供，可能会出现一秒的分数
  t1 = time();
  print("Done.");
  d1 = [hex(x) for x in arr1]; d2 = [hex(x) for x in arr2];
  print("Differences between guessed and last key:", d1);
  print("Differences between guessed and second-to-last key:", d2);
  print("Wall time per attack (average in seconds):", (t1 - t0)/n);
  print("Data blocks used (average, log2): ", log2(data) - log2(n));
  print("succ_rate1:")
  succ1 = np.sum(arr1==0)/len(arr1)
  print(arr1.shape)
  print(succ1)
  succ2 = np.sum(arr2==0)/len(arr2)
  print("succ_rate2:")
  print(arr2.shape)
  print(succ2)
  # arr1:猜测密钥和最后一个密钥的区别  arr2:猜测密钥和倒数第二个密钥的区别  good:正确密钥的个数
  return(arr1, arr2, good);

arr1, arr2, good = run(50);
folder="./key_recovery_attack/2_7/"
# 写文件 'wb'：以二进制写模式打开文件
np.save(folder + 'run_sols1.npy', arr1)
np.save(folder + 'run_sols2.npy', arr2)
np.save(folder + 'run_good.npy', good)
# np.save(open('run_sols1.npy','wb'),arr1);
# np.save(open('run_sols2.npy','wb'),arr2);
# np.save(open('run_good.npy','wb'),good);

#arr1, arr2, good = test(20, nr=12, num_structures=500, it=2000, cutoff1=20.0, cutoff2=500, neutral_bits=neutral13,keyschedule='free',net=net8, net_help=net7, m_main=m8, s_main=s8, m_help=m7, s_help=s7, verify_breadth=128);
# 写文件
# 12轮Speck密钥恢复攻击 对最后两轮子密钥的密钥猜测
# np.save(open('run_sols1_12r.npy', 'wb'), arr1);
# np.save(open('run_sols2_12r.npy', 'wb'), arr2);
# np.save(open('run_good_12r.npy', 'wb'), good);


# 读文件
# b = np.load('./data_9r_attack/data_'+str(i)+'.npy')
# print(b)
# print(b.shape)
# npy文件保存了numpy数组的结构