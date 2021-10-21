# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 15:10:17 2021

@author: ave41
"""

# print(comet_actual_ra,closest_ra_value_to_comet,moa_sources['xcentroid'][closest_ra_value_to_comet])
# print(comet_actual_dec,closest_dec_value_to_comet,moa_sources['ycentroid'][closest_dec_value_to_comet])

import numpy as np

# np.loadtxt("2021-02-18.txt",sep=" ")

# date_from_fits = []
# with open("2021-02-18.txt", "r") as datafile:
#     print(datafile.read().split())
#     yolo = datafile.read().split()
#     date_from_fits.append(yolo[0])
#     print(date_from_fits)
    
    
# columns = [[]] * 5
# with open('2021-02-18.txt','r') as token:
#     for line in token:
#         for field, value in enumerate(line.split()):
#              columns[field].append(value)
             
# col_num = 0 
# col_data = [] 
# delimiter = " " 
# with open('2021-02-18.txt') as f: 
#     col_data.append(f.readline().split(delimiter)[col_num]) 

d = np.genfromtxt('2021-02-18.txt',delimiter=",")
x_pixel_loc = d[0,:]
y_pixel_loc = d[1,:]

# 21-02-18T10:31:49,1151,2344
# 21-02-18T10:37:29,1163,2343
# 21-02-18T10:44:31,1176,2341
# 21-02-18T10:50:12,1186,2341
# 21-02-18T10:55:52,1197,2341