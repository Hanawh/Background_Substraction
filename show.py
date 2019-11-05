import cv2
import glob
import numpy as np
imagepath = './WavingTrees'
resultpath = './result'
cv2.namedWindow('before')
cv2.namedWindow('after')
path1 = sorted(glob.glob(imagepath + "/*.bmp"))
path1 = path1[200:]

path2 = sorted(glob.glob(resultpath + "/*.bmp"))
for ind in range(len(path1)):
    img1 = cv2.imread(path1[ind]) #原图
    img2 = cv2.imread(path2[ind]) #结果图
    cv2.imshow("before", img1)
    cv2.imshow("after", img2)
    cv2.waitKey(100)