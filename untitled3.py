# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:28:56 2019

@author: Robert
"""
import re
import os

def getDirAndFileList(dataFolder = r'C:\Users\Robert\Desktop\labdust-gan', pattern = '.*TIF', returnType = "file"):
    returnList  = []
    #for root, dirs, files in os.walk(dataFolder):       
    if returnType == "file":
        for file in next(os.walk(dataFolder))[2]:
            if re.match(pattern,file):
                #print(os.path.join(root, file))
                returnList.append(file)
    elif returnType == "dir":
        for folder in next(os.walk(dataFolder))[1]:
            if re.match(pattern,folder):
                #print(os.path.join(root, file))
                returnList.append(folder)
    return returnList

a = returnList