import cv2 
import glob #找寻文件列表
import numpy as np
import ipdb #调试
import time 
import logging #记录
from numpy.linalg import norm #二范数
from scipy.stats import multivariate_normal as gaussian #高斯分布
############### 生成记录日志 ##################
log_format = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('log.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
################### 生成图片列表 ####################
imgs = glob.glob('WavingTrees/b*.bmp')
imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('b')[-1])) #因为读取后是无序的需要排序
T = 200 #划分数据
train_list = imgs[:T] #划分数据
test_list = imgs[T:]
imgs_test_num = len(test_list)
################### init #################
img = cv2.imread(train_list[0]) #读取第一帧
H,W,C= img.shape
alpha = 1.0/T # from paper
K = 5 #高斯个数
D = 2.5 #比较阈值
cf = 0.75 #前景阈值

B = np.ones((H,W), dtype=np.int)
weight = np.array([[[1,0,0,0,0] for j in range(W)] for i in range(H)])
mu = np.array([[[np.zeros(3) for k in range(K)] for j in range(W)] for i in range(H)])
sd = np.array([[[225*np.eye(3) for k in range(K)] for j in range(W)] for i in range(H)])
for i in range(H):
    for j in range(W):
        for k in range(K):
            mu[i][j][k] = np.array(img[i][j]).reshape(1,3)
################### 训练 ###################
def match(pixel, mu, sigma):
    x = np.mat(np.reshape(pixel, (3, 1)))
    u = np.mat(mu).T
    sigma = np.mat(sigma)
    d = np.sqrt((x-u).T*sigma.I*(x-u))
    if d < 2.5:
        return True
    else:
        return False

start = time.time()
for ind, train in enumerate(train_list):
    img = cv2.imread(train)
    print("processing img.{0}".format(ind))
    for i in range(H):
        for j in range(W):
            match_flag = -1
            for k in range(K):
                # once match break
                if match(img[i][j], mu[i][j][k], sd[i][j][k]):
                    match_flag = k
                    break
            x = np.array(img[i][j]).reshape(1,3)
            if match_flag != -1:
                #update
                m = mu[i][j][match_flag]
                s = sd[i][j][match_flag]
                x = img[i][j].astype(np.float)
                delta = x - m
                p = gaussian.pdf(img[i][j],m,s)
                weight[i][j] = (1 - alpha)*weight[i][j]
                weight[i][j][match_flag] += alpha
                mu[i][j][match_flag] += delta*p
                sd[i][j][match_flag] += p*(np.matmul(delta,delta.T)-s)
            if match_flag == -1:
            # replace the least probable distribution
                w = [weight[i][j][k] for k in range(K)]
                min_id = w.index(min(w))
                mu[i][j][min_id] = x
                sd[i][j][min_id] = 225*np.eye(3)
    # sort
    for i in range(H):
        for j in range(W):
            rank = weight[i][j]*1.0/[norm(np.sqrt(sd[i][j][k])) for k in range(K)]
            rank_ind = [k for k in range(K)]
            rank_ind.sort(key=lambda x: -rank[x])
            weight[i][j] = weight[i][j][rank_ind]
            mu[i][j] =  mu[i][j][rank_ind]
            sd[i][j] = sd[i][j][rank_ind]
            cum = 0
            for ind, order in enumerate(rank_ind):
                cum += weight[i][j][ind]
                if cum > cf:
                    B[i][j] = ind + 1
                    break
end = time.time()
print("the cost of train is {0}min".format((end-start)/60))
################### test ####################
for ind, test in enumerate(test_list):
    img = cv2.imread(test)
    result = np.array(img)
    for i in range(H):
        for j in range(W):
            for k in range(B[i][j]):
                if match(img[i][j], mu[i][j][k], sd[i][j][k]):
                    result[i][j] = [255, 255, 255]    
                    break
    cv2.imwrite(r'./result/'+'%05d'%ind+'.bmp', img)

    









            



                

     


     


