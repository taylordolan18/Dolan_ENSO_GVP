#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:36:59 2022
@author: taylordolan

Code to create histograms for the peak warm and cold intensity values
for members 1 - 10

Nino
Weak: 1.7 - 2.4
Mod: 2.41 - 2.81
Strong: 2.84 - 4.06

Nina
Weak: -1.75 to -2.32
Mod: -2.33 to -2.69
Strong: -2.7 to -4.46
"""
#Imports
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle

#Import the 2 csv files to work with
# file = open('NEW_Members_1to10_warmpeaks.csv')
# csvreader = csv.reader(file)
data = pd.read_csv('/Users/taylordolan/Documents/NEW_Members_1to10_warmpeaks.csv')  
data1 = pd.read_csv('/Users/taylordolan/Documents/NEW_Members_1to10_coldpeaks.csv')
df = pd.DataFrame(data)
df1 = pd.DataFrame(data1)
array = df.to_numpy()
array1 = df1.to_numpy()

#Make the histogram

cmap = plt.get_cmap('Accent')
weak = cmap(0.5)
moderate =cmap(0.9)
strong = cmap(0.7)

#El Nino
num_bins = 40
fig, ax = plt.subplots()
N, bins, patches = ax.hist(data, 40, edgecolor='white', linewidth=1)
#plt.axvline(2.4, color = 'purple', label = 'Weak Intensity') #end weak
#plt.axvline(2.81, color = 'black', label = 'Moderate Intensity') #moderate
for axis in [ax.yaxis]:
    axis.set_major_locator(ticker.MaxNLocator(integer=True))
for i in range(0, 11):
    patches[i].set_facecolor(weak)
for i in range(11,18):    
    patches[i].set_facecolor(moderate)
for i in range(18, len(patches)):
    patches[i].set_facecolor(strong)  
plt.xlabel('Intensities')
plt.ylabel('Count')
plt.title('Members 1-10 Warm Peak Intensity Frequency')
plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper left')
handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [weak,moderate, strong]]
labels= ["Weak","Moderate", "Strong"]
plt.legend(handles, labels)
plt.show()

#La Nina
# num_bins = 35
# fig, ax = plt.subplots()
# N, bins, patches = ax.hist(data1, 35, edgecolor='white', linewidth=1)
# # plt.axvline(-2.33, color = 'purple', label = 'Weak Intensity') #end weak
# # plt.axvline(-2.7, color = 'black', label = 'Moderate Intensity') #moderate
# for axis in [ax.yaxis]:
#     axis.set_major_locator(ticker.MaxNLocator(integer=True))
# for i in range(0, 18):
#     patches[i].set_facecolor(weak)
# for i in range(18,23):    
#     patches[i].set_facecolor(moderate)
# for i in range(23, len(patches)):
#     patches[i].set_facecolor(strong)  
# plt.xlabel('Intensities')
# plt.ylabel('Count')
# plt.title('Members 1-10 Cold Peak Intensity Frequency')
# plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper left')
# handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [weak,moderate, strong]]
# labels= ["Weak","Moderate", "Strong"]
# plt.legend(handles, labels)
# plt.show()


