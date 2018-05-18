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
        self.model.traiinn(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples])

docsv = doCsv.doCsv()
vecImgs, vecLables = [], []
img_list = docsv.csv_reader()
if img_list:
    for i in range(1, len(img_list)):
        img_str = img_list[i].split(';')
        filepath = img_str[0]
        try:
            im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            c = img_str[1]
        except:
            traceback.print_exc()
        vecImgs.append(np.asarray(im, dtype=np.uint8))
        vecLables.append(c)
    vecLables = np.asarray(vecLables, dtype=np.int32)
    vecImgs= np.array(vecImgs)
    # print vecImgs
    svm = SVM()
    # svm = cv2.ml.SVM_create()
    # params = dict(kernel_type=cv2.ml.SVM_LINEAR ,svm_type=cv2.ml.SVM_C_SVC,C=1)
    svm.train(vecImgs, vecLables)

