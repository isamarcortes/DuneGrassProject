#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:40:44 2021

@author: isamarcortes
"""

###preprocessing included removing band 4 using gdal_translate

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.multioutput import MultiOutputClassifier


Tif_File = '/Users/isamarcortes/Downloads/DJI_output.tif'


def read_band(file, band_num):
    with rio.open(file, 'r') as f:
        band = f.read(band_num)
    return band

###reading bands/ training bands
band1 = read_band(Tif_File,1)
band2 = read_band(Tif_File,2)
band3 = read_band(Tif_File,3)

#### creating an empty array that I will fill based on a series of values that I determine to be each class
labeled_data = np.zeros(shape = band1.shape, dtype=np.str)
labeled_data[((band1 >= 161) & (band1 <= 255))]='Sand'
labeled_data[((band1 >= 0) & (band1 <= 160))]='Vegetation'
#labeled_data[((band1 >= 76) & (band1 <= 199))]='Other'

X_train, X_test, y_train, y_test = train_test_split(band1, labeled_data, test_size=0.33,random_state=66)
rfc = RandomForestClassifier(random_state=66)
rfc_multiclass = MultiOutputClassifier(rfc)

rfc_multiclass.fit(X_train,y_train)

rfc_predict = rfc_multiclass.predict(X_test) #y_pred_test
testpredict = rfc_multiclass.predict(band1)

####creating the classification map
rfc_predict_img = np.zeros(shape = testpredict.shape, dtype=np.integer)
rfc_predict_img[((testpredict == 'S'))] = 0
rfc_predict_img[((testpredict == 'V'))] = 1
plt.imshow(rfc_predict_img)
plt.colorbar()


#checking precision and accuracy
TP = ((rfc_predict == 'S') & (y_test == 'S')).sum()
FP = ((rfc_predict == 'S') & (y_test == 'V')).sum()
precision = TP / (TP+FP)
Y_test_flatten = y_test.flatten()
rfc_predict_flatten = rfc_predict.flatten()
print(accuracy_score(Y_test_flatten, rfc_predict_flatten))
print(confusion_matrix(Y_test_flatten,rfc_predict_flatten))
print(classification_report(Y_test_flatten,rfc_predict_flatten))
