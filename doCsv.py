# -*- coding: utf-8 -*-
# @Time    : 18-5-18 下午3:13
# @Author  : JeeShao
# @File    : doCsv.py
import csv
import time

class doCsv:
    def __init__(self,file= "traindata.csv"):
        self.file=file

    def csv_writer(self,data):
        try:
            with open(self.file, 'w',newline='') as csvfile:
                # csvfile.write(codecs.BOM_UTF8)
                now = time.strftime("%Y-%m-%d %H:%M:%S")
                writer = csv.writer(csvfile, dialect='excel')
                writer.writerow([now])
                writer.writerows(data)
        except:
            print("csv writer error")
            raise

    def csv_reader(self):
        resList = []
        try:
            with open(self.file, 'r',) as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    resList = resList + line
            if resList:
                return resList
            else:
                return False
        except:
            print("csv reader error")
            raise