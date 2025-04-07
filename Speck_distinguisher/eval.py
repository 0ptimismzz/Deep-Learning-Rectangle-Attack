import speck_mrcp as sp
import numpy as np
from keras.models import load_model



def evaluate(net,X,Y):
    Z = net.predict(X,batch_size=5000).flatten()
    Zbin = (Z >= 0.5)
    diff = Y - Z; mse = np.mean(diff*diff)
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1)
    acc = np.sum(Zbin == Y) / n
    tpr = np.sum(Zbin[Y==1]) / n1
    tnr = np.sum(Zbin[Y==0] == 0) / n0
    mreal = np.median(Z[Y==1])
    high_random = np.sum(Z[Y==0] > mreal) / n0
    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr, "MSE:", mse)
    print("Percentage of random pairs with score higher than median of real pairs:", 100*high_random)


wdir = './change_paramens/'
# wdir = './saved_model/CConv1D_rscp/'

# net5= load_model(wdir+'model_'+str(5)+'r_depth'+str(5) +"_num_epochs"+str(20)+"_pairs"+str(pairs)+'.h5')
net2g= load_model(wdir+'8_2_10_mrcp_ci_distinguisher.h5')
# net4g= load_model(wdir+'8_4_10_mrcp_ci_distinguisher.h5')
# net8g= load_model(wdir+'8_8_10_mrcp_ci_distinguisher.h5')

# for i in group_size:
#     # net6= load_model(wdir+'6_' + str(i) + '_5_mrcp_distinguisher.h5')
#     net7= load_model(wdir+'7_' + str(i) + '_10_rscp_distinguisher.h5')
#     net8= load_model(wdir+'8_' + str(i) + '_5_rscp_distinguisher.h5')

    # net8= load_model(wdir+"n=1000000model_8r_depth5_num_epochs20_pairs8_num_keys10000.h5")


    # X5,Y5 = sp.make_train_data(5*10**6,5,pairs)
    # X6,Y6 = sp.make_dataset_with_group_size(i*10**6,6, group_size=i)
    # X7,Y7 = sp.make_dataset_with_group_size(10**7,7, group_size=1)
X2g,Y2g = sp.make_dataset_with_group_size(2*10**6,8, group_size=2)
# X4g,Y4g = sp.make_dataset_with_group_size(4*10**6,8, group_size=4)
# X8g,Y8g = sp.make_dataset_with_group_size(8*10**6,8, group_size=8)

    # X5r, Y5r = sp.real_differences_data(5*10**6,5,pairs)
    # X6r, Y6r = sp.real_differences_data(10**6,6,pairs)
    # X7r, Y7r = sp.real_differences_data(10**6,7,pairs)


print('Testing neural distinguishers against 5 to 8 blocks in the ordinary real vs random setting')
    # print('5 rounds:')
    # evaluate(net5, X5, Y5)
    # print(f'6_{i} rounds:')
    # evaluate(net6, X6, Y6)
    # print(f'7_{i} rounds:')
    # evaluate(net7, X7, Y7)
print(f'8 rounds 2g:')
evaluate(net2g, X2g, Y2g)
# print(f'8 rounds 4g:')
# evaluate(net4g, X4g, Y4g)
# print(f'8 rounds 8g:')
# evaluate(net8g, X8g, Y8g)
    # print('\nTesting real differences setting now.')
    # print('5 rounds:')
    # evaluate(net5, X5r, Y5r)
    # print('6 rounds:')
    # evaluate(net6, X6r, Y6r)
    # print('7 rounds:')
    # evaluate(net7, X7r, Y7r)
    # print('8 rounds:')
    # evaluate(net8, X8r, Y8r)
    
    
# mrod:
# 1600/1600 [==============================] - 9s 4ms/step
# Accuracy:  0.887824375 TPR:  0.86918575 TNR:  0.906463 MSE: 0.07908730803240993
# Percentage of random pairs with score higher than median of real pairs: 0.490375
# 7_1 rounds:
# 1600/1600 [==============================] - 6s 4ms/step
# Accuracy:  0.65834775 TPR:  0.6601515 TNR:  0.656544 MSE: 0.21255347724977997
# Percentage of random pairs with score higher than median of real pairs: 20.588875
# Testing neural distinguishers against 5 to 8 blocks in the ordinary real vs random setting
# 6_2 rounds:
# 800/800 [==============================] - 8s 10ms/step
# Accuracy:  0.9515635 TPR:  0.943686 TNR:  0.959441 MSE: 0.03667938710568578
# Percentage of random pairs with score higher than median of real pairs: 0.02565
# 7_2 rounds:
# 800/800 [==============================] - 8s 10ms/step
# Accuracy:  0.71839725 TPR:  0.7128055 TNR:  0.723989 MSE: 0.18515528176667723
# Percentage of random pairs with score higher than median of real pairs: 12.1257
# Testing neural distinguishers against 5 to 8 blocks in the ordinary real vs random setting
# 6_4 rounds:
# 400/400 [==============================] - 10s 25ms/step
# Accuracy:  0.985065 TPR:  0.982855 TNR:  0.987275 MSE: 0.011482156492819499
# Percentage of random pairs with score higher than median of real pairs: 9.999999999999999e-05
# 7_4 rounds:
# 400/400 [==============================] - 10s 25ms/step
# Accuracy:  0.7904235 TPR:  0.791737 TNR:  0.78911 MSE: 0.1444345990864213
# Percentage of random pairs with score higher than median of real pairs: 5.1193
# Testing neural distinguishers against 5 to 8 blocks in the ordinary real vs random setting
# 6_8 rounds:
# 200/200 [==============================] - 11s 52ms/step
# Accuracy:  0.997646 TPR:  0.997196 TNR:  0.998096 MSE: 0.0018201457763420254
# Percentage of random pairs with score higher than median of real pairs: 0.0
# 7_8 rounds:
# 200/200 [==============================] - 11s 52ms/step
# Accuracy:  0.860948 TPR:  0.86972 TNR:  0.852176 MSE: 0.09942966035999547
# Percentage of random pairs with score higher than median of real pairs: 1.4727999999999999
# Testing neural distinguishers against 5 to 8 blocks in the ordinary real vs random setting
# 6_16 rounds:
# 100/100 [==============================] - 11s 105ms/step
# Accuracy:  0.999794 TPR:  0.99972 TNR:  0.999868 MSE: 0.00016490568667150372
# Percentage of random pairs with score higher than median of real pairs: 0.0
# 7_16 rounds:
# 100/100 [==============================] - 11s 105ms/step
# Accuracy:  0.913026 TPR:  0.904332 TNR:  0.92172 MSE: 0.06360609932118166
# Percentage of random pairs with score higher than median of real pairs: 0.2752

# mrcp:
# 8_8 rounds:
# 200/200 [==============================] - 15s 53ms/step
# Accuracy:  0.584447 TPR:  0.592788 TNR:  0.576106 MSE: 0.23887367002659782
# Percentage of random pairs with score higher than median of real pairs: 33.5528

# rscp:
# 7_1 rounds:
# 200/200 [==============================] - 4s 6ms/step
# Accuracy:  0.753871 TPR:  0.744604 TNR:  0.763138 MSE: 0.16679731794100863
# Percentage of random pairs with score higher than median of real pairs: 8.028