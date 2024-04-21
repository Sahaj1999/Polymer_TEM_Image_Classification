import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix, precision_score, f1_score, recall_score
import pickle
from PIL import Image


Categories = ['Nanotubes','Spherical', 'Vessicles']
flat_data_arr = []  # input array
target_arr = []  # output array
datadir = 'C:/Users/sahaj/PycharmProjects/NanoparticlesSVM/Data/Polymer'
# path which contains all the categories of images
for i in Categories:
    print(f'loading... category : {i}')
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (150, 150, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data)  # dataframe
print(df)
df['Target'] = target
x = df.iloc[:, :-1]  # input data
y = df.iloc[:, -1]  # output data

param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
print('Splitted Successfully')
model.fit(x_train,y_train)
print('The Model is trained well with the given images')
# model.best_params_ contains the best parameters obtained from GridSearchCV
print(model.best_params_)

y_pred=model.predict(x_test)
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")
print(np.array(y_test))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100)

# Calculate Accuracy
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")

# Calculate Precision
print(f"The model is {precision_score(y_pred, y_test, average = 'weighted')*100}% precise")

# Calculate F1 Score
print(f"The F1 Score for the model is: {f1_score(y_pred, y_test, average = 'weighted')}")

# Calculate Recall Score
print(f"The model has {recall_score(y_pred, y_test, average = 'weighted')} recall sensitivity.")

pickle.dump(model, open('C:/Users/sahaj/PycharmProjects/NanoparticlesSVM/Data/img_model.p', 'wb'))
print("Pickle is dumped successfully")




# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100)

