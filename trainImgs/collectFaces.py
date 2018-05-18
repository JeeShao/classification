# -*- coding: utf-8 -*-
# @Time    : 18-5-18 下午12:21
# @Author  : JeeShao
# @File    : collectFaces.py

import cv2
import os,shutil

fromPath = "./orl_faces/"
toPath = "./face/"
i=0



# create info.dat
dir_list = os.listdir(fromPath)
dir_list.sort(key=lambda x:int(x.split('s')[1]))
for dir in dir_list:
   imgs = os.listdir(os.path.join(fromPath,dir))
   imgs.sort(key=lambda x:int(x.split('.')[0]))
   for img_name in imgs:
      img_url = os.path.join(os.path.join(fromPath,dir), img_name)
      newName = toPath+"%d.pgm"%(i)
      i+=1
      shutil.copyfile(img_url,newName)
