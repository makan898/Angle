import cv2
import numpy as np
import matplotlib.pyplot as plt
from getpoint import *
from culc_rotate import *


#角度の計算
#上で検出した物体のリストの数だけ実行する
img6 = cv2.imread("2.jpg")
img1 = cv2.imread("20000101_100544.jpg")
img2 = cv2.imread("20000101_100551.jpg")
img3 = cv2.imread("20000101_100603.jpg")
img4 = cv2.imread("20000101_100609.jpg")
img5 = cv2.imread("20000101_100617.jpg")

#print("rotate : ",culc_rotate(img1,img6))
#print("rotate : ",culc_rotate(img1,img2))
#print("rotate : ",culc_rotate(img1,img3))
print("rotate : ",culc_rotate(img1,img4))
#print("rotate : ",culc_rotate(img1,img5))