import os
import shutil

import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from skimage.transform import resize
from skimage.io import imread, imsave
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
from PIL import Image

Categories = ['Nanotubes', 'Spherical', 'Vessicles']

model = pickle.load(open('C:/Users/saclashaj/PycharmProjects/NanoparticlesSVM/Data/img_model.p', 'rb'))

input_folder = 'C:/Users/sahaj/PycharmProjects/NanoparticlesSVM/Input_Images'

for filename in os.listdir(input_folder):
    if filename.endswith('.tif'):
        url = os.path.join(input_folder, filename)
        img = imread(url)
        img_resize = resize(img, (150, 150, 3))
        l = [img_resize.flatten()]
        probability = model.predict_proba(l)
        for ind, val in enumerate(Categories):
            print(f'{val} = {probability[0][ind]*100}%')
        result = Categories[model.predict(l)[0]]
        print("The predicted image is : ",filename,'is',result )


        if result == 'Nanotubes':
            imsave(os.path.join('C:/Users/sahaj/PycharmProjects/NanoparticlesSVM/Sorted_Data', 'Sorted_Nanotubes', filename), img)
            os.remove(url)
            print (filename, 'moved')
        if result == 'Spherical':
            imsave(os.path.join('C:/Users/sahaj/PycharmProjects/NanoparticlesSVM/Sorted_Data', 'Sorted_Spherical', filename), img)
            os.remove(url)
            print(filename, 'moved')
        if result == 'Vessicles':
            imsave(os.path.join('C:/Users/sahaj/PycharmProjects/NanoparticlesSVM/Sorted_Data', 'Sorted_Vessicles', filename), img)
            os.remove(url)
            print(filename, 'moved')
