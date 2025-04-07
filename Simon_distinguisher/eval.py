from six import reraise

import simon_mrcp as si
import numpy as np
from keras.models import load_model
import pandas as pd

def evaluate(net, X, Y):
	Z = net.predict(X, batch_size=5000).flatten()
	Zbin = (Z >= 0.5)
	diff = Y - Z;
	mse = np.mean(diff * diff)
	n = len(Z);
	n0 = np.sum(Y == 0);
	n1 = np.sum(Y == 1)
	acc = np.sum(Zbin == Y) / n
	tpr = np.sum(Zbin[Y == 1]) / n1
	tnr = np.sum(Zbin[Y == 0] == 0) / n0
	mreal = np.median(Z[Y == 1])
	high_random = np.sum(Z[Y == 0] > mreal) / n0
	print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr, "MSE:", mse)
	print("Percentage of random pairs with score higher than median of real pairs:", 100 * high_random)


wdir = './save_final/simon_9r_4plus5/'
group_size = [1,2,4,8]


for i in group_size:
	net= load_model(wdir+'9_10_' + str(i) + '_rscp_distinguisher.h5')
	X9, Y9 = si.make_dataset_with_group_size(n=i * 10 ** 7, nr=9, first_step=3,
	                                       group_size=i, diff=(0x0000, 0x0040))
	evaluate(net, X9, Y9)
