import cv2
import numpy as np
import scipy
#from scipy.misc import imread
#import cPickle as pickle
import random
import os
import matplotlib.pyplot as plt
# Feature extractor 
# 特征提取器
def extract_features(image_path, vector_size=32):
    image = image_path
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        #此处为了简化安装步骤，使用KAZE，因为SIFT/ORB以及其他特征算子需要安
#装额外的模块
        alg = cv2.KAZE_create()
        # Finding image keypoints
        #寻找图像关键点
        kps, des = alg.detect(image)
        # Getting first 32 of them. 
        #计算前32个
        # Number of keypoints is varies depend on image size and color pallet
        #关键点的数量取决于图像大小以及彩色调色板
        # Sorting them based on keypoint response value(bigger is better)
        #根据关键点的返回值进行排序（越大越好）
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        #计算描述符向量
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        # 将其放在一个大的向量中，作为我们的特征向量
        dsc = dsc.flatten()
        # Making descriptor of same size
        # 使描述符的大小一致
        # Descriptor vector size is 64
        #描述符向量的大小为64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros 
            # at the end of our feature vector
#如果少于32个描述符，则在特征向量后面补零
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print('Error: ', e)
        return None


    return kps,des