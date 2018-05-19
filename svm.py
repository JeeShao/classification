# -*- coding: utf-8 -*-
# @Time    : 18-5-18 下午3:10
# @Author  : JeeShao
# @File    : svm.py

import cv2
import doCsv
import sys
import numpy as np

class StatModel(object):
    '''parent class - starting point to add abstraction'''
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''
    def __init__(self):
        self.model = cv2.ml.SVM_create()

    def train(self, samples, responses):
        #setting algorithm parameters
        # params = dict( kernel_type=cv2.ml.SVM_LINEAR ,svm_type=cv2.ml.SVM_C_SVC,C=1 )
        self.model.setCoef0(0)
        self.model.setCoef0(0.0)
        self.model.setDegree(3)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
        self.model.setTermCriteria(criteria)
        self.model.setGamma(0)
        self.model.setKernel(cv2.ml.SVM_LINEAR)
        self.model.setNu(0.5)
        self.model.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
        self.model.setC(0.01)  # From paper, soft classifier
        self.model.setType(cv2.ml.SVM_EPS_SVR)  # C_SVC # EPSILON_SVR # may be also NU_SVR # do regression task
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples])


# wsize: 处理图片大小，通常64*128; 输入图片尺寸>= wsize
def computeHOGs(img_lst):
    gradient_lst = []
    winSize = (64,64)
    # winSize = (112, 88)
    # blockSize = (8, 8)
    blockSize = (16,16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    # compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (8, 8)
    padding = (8, 8)
    # locations = []  # (10, 10)# ((10,20),)

    for img in img_lst:
        gradient_lst.append(hog.compute(img, winStride, padding))
    return gradient_lst

# hog = cv2.HOGDescriptor(imageSize, Size(16, 16), cvSize(8, 8), cvSize(8, 8), 9)
#     # hog.winSize = wsize
#     for i in range(len(img_lst)):
#         if img_lst[i].shape[1] >= wsize[1] and img_lst[i].shape[0] >= wsize[0]:
#             roi = img_lst[i][(img_lst[i].shape[0] - wsize[0]) // 2: (img_lst[i].shape[0] - wsize[0]) // 2 + wsize[0], \
#                   (img_lst[i].shape[1] - wsize[1]) // 2: (img_lst[i].shape[1] - wsize[1]) // 2 + wsize[1]]
#             gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#             gradient_lst.append(hog.compute(gray))
#     # return gradient_lst

# docsv = doCsv.doCsv()
# vecImgs, vecLables = [], []
# img_list = docsv.csv_reader()
# if img_list:
#     for i in range(1, len(img_list)):
#         img_str = img_list[i].split(';')
#         filepath = img_str[0]
#         # try:
#         im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
#         c = img_str[1]
#         # except:
#             # traceback.print_exc()
#         vecImgs.append(np.asarray(im, dtype=np.uint8))
#         vecLables.append(c)
#
#     gradient_lst = computeHOGs(vecImgs)
#
#     vecLables = np.asarray(vecLables, dtype=np.int32)
#     vecImgs= np.array(gradient_lst)
#     svm = SVM()
#     # svm = cv2.ml.SVM_create()
#     # params = dict(kernel_type=cv2.ml.SVM_LINEAR ,svm_type=cv2.ml.SVM_C_SVC,C=1)
#     svm.train(vecImgs, vecLables)
#     svm.save("model.xml")

# if __name__=="__main__":
#     svm = SVM()
#     svm = StatModel.load("model.xml")



svm = cv2.ml.SVM_load("model.xml")
testimg = cv2.imread("./trainImgs/lighter/300.jpg", cv2.COLOR_BGR2GRAY)
# hog = cv2.HOGDescriptor()
hog = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
gradient_lst = hog.compute(testimg, (8,8), (8,8))
gradient_lst = gradient_lst.transpose()
# gradient_lst = computeHOGs(testimg)
out = svm.predict(np.asarray(gradient_lst,dtype=np.float32))
print(out)