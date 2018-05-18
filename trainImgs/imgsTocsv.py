# -*- coding: utf-8 -*-
# @Time    : 18-5-18 下午12:51
# @Author  : JeeShao
# @File    : imgsTocsv.py

import os
import csv
import doCsv



# def csv_writer(data):
#     try:
#         with open('../traindata.csv', 'w') as csvfile:
#             # csvfile.write(codecs.BOM_UTF8)
#             # now = time.strftime("%Y-%m-%d %H:%M:%S")
#             writer = csv.writer(csvfile, dialect='excel')
#             # writer.writerow([now])
#             writer.writerows(data)
#     except:
#         print("csv writer error")
#         raise

this_dir = os.path.abspath(os.path.dirname(__file__))
carPath =  os.path.join(this_dir,'car')
facePath =  os.path.join(this_dir,'face')
docsv = doCsv.doCsv('../traindata.csv')
data= []
carlable = 0
facelable = 1
carimgs = os.listdir(carPath)
carimgs.sort(key=lambda x:int((x.split('-')[1]).split('.')[0]))
for filename in carimgs:
    filepath = os.path.join(carPath,filename)
    data = data + [tuple(["%s;%d" % (filepath,carlable)])]
    docsv.csv_writer(data)

faceimgs = os.listdir(facePath)
faceimgs.sort(key=lambda x:int(x.split('.')[0]))
for filename in faceimgs:
    filepath = os.path.join(facePath,filename)
    data = data + [tuple(["%s;%d" % (filepath,facelable)])]
    docsv.csv_writer(data)