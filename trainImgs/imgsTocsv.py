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
lighterPath =  os.path.join(this_dir,'lighter')
membankPath =  os.path.join(this_dir,'membank')
docsv = doCsv.doCsv('../traindata.csv')
data= []
lighterlable = -1
membanklable = 1
lighterimgs = os.listdir(lighterPath)
lighterimgs.sort(key=lambda x:int(x.split('.')[0]))
for filename in lighterimgs:
    filepath = os.path.join(lighterPath,filename)
    data = data + [tuple(["%s;%d" % (filepath,lighterlable)])]
    docsv.csv_writer(data)

membankimgs = os.listdir(membankPath)
membankimgs.sort(key=lambda x:int(x.split('.')[0]))
for filename in membankimgs:
    filepath = os.path.join(membankPath,filename)
    data = data + [tuple(["%s;%d" % (filepath,membanklable)])]
    docsv.csv_writer(data)