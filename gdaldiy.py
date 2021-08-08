# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 20:57:37 2019

@author: Neoooli
"""

import gdal
import numpy as np

def imgread(path):
    img=gdal.Open(path)
    w=img.RasterXSize
    h=img.RasterYSize
    c=img.RasterCount
    img_arr=img.ReadAsArray(0,0,w,h)
    if c>1:
        img_arr=img_arr.swapaxes(1,0)
        img_arr = img_arr.swapaxes(2,1)

    del img
    return img_arr

def imgwrite(path,narray):
    s=narray.shape
    if len(s)==2:
        w,h=s[0],s[1]
        c=1
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(path, w,h,1, gdal.GDT_Byte)
        dataset.GetRasterBand(1).WriteArray(narray)
        del dataset
    elif len(s)==3:
        w, h,c = s[0], s[1], s[2]
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(path, w, h, c, gdal.GDT_Byte)
        for i in range(c):
            dataset.GetRasterBand(i + 1).WriteArray(narray[:,:,i])
        del dataset