#!/usr/bin/env python
# coding: utf-8

# # **Compiled by: Japesh Methuku**
# LinkedIn Profile: [Japesh Methuku](https://www.linkedin.com/in/japeshmethuku/)

# ## **Import required libraries**

# In[2]:


# %tensorflow_version 2.x

# import tensorflow and tensorflow.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

# import ImageDataGenerator and the related functions required for processing images
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

# import optimizers
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop

# import statements for building and loading the model
from tensorflow.keras import models
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.models import model_from_json

# import statements for callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

# import statements for initlializers and regularizers
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.regularizers import l2

# import statements for loading ResNet50 from keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# import statements for scikit-learn
import sklearn.metrics as metrics

# import os for file access
import os 

# import numpy, pandas
import numpy as np
import pandas as pd

# import opencv
import cv2

# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# import zipfile for unzipping the data
import zipfile

# import csv to access the csv files
import csv

# import seaborn
import seaborn as sns

# import time
from time import time
print("tensorflow version:",tf.__version__)


# ## **Loading images from the dataset**

# In[3]:


# Specifying the location of training images after extraction
training_dir = '../train/'
testing_dir = './test/'


# In[4]:


# Loading the csv file to access the details of the images
training_data = pd.read_csv('../Training_set_face_mask.csv', na_values='na')
testing_data = pd.read_csv('../Testing_set_face_mask.csv')


# In[5]:


# Displaying top 10 values of the csv file
training_data.head(10)


# In[9]:


datagen = ImageDataGenerator(validation_split=0.2)
train_data = datagen.flow_from_directory(training_dir, class_mode='categorical', target_size=(224,224), subset='training', batch_size=32)
valid_data = datagen.flow_from_directory(training_dir, class_mode='categorical', target_size=(224,224), subset='validation', batch_size=32,shuffle=False)


# ## **Model Creation**

# In[ ]:


# Loading the ResNet50 model
resnet_base = ResNet50(weights= 'imagenet', include_top=False, input_shape= (224,224,3))
resnet_base.summary()


# In[7]:


# Sequential building of the image classification model
# Using Keras Sequential API
model = models.Sequential()
model.add(resnet_base)
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(2, activation = 'softmax'))


# In[ ]:


# Visualizing the summary of the model
model.summary()


# ## **Model Compilation**

# In[9]:


# Using Stochastic Gradient Descent as optimization algorithm
OPTIMIZER = keras.optimizers.SGD(lr=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer = OPTIMIZER,
              metrics=['accuracy'])


# In[10]:


# Specifying the callbacks
callbacks_list= [keras.callbacks.ModelCheckpoint('Face_Mask_Model.hdf5', 
                                                 monitor='val_accuracy', 
                                                 verbose=1, 
                                                 save_best_only=True)]


# ## **Train Model**

# In[ ]:


# Fit the compiled model on the training data and validate with validation data
history= model.fit(train_data, steps_per_epoch= 11264//32, 
                    callbacks=callbacks_list, 
                    epochs = 50, verbose = 1, validation_data = valid_data)


# ## **Visualize the execution results**

# In[ ]:


# Visualize accuracy results
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training and Validation Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.minorticks_on()
plt.grid()
plt.figure()
# save image
plt.savefig('Classification Model Accuracy', dpi=250)
plt.show()

# Visualize loss results
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.minorticks_on()
plt.grid()
plt.figure()
# save image
plt.savefig('Classification Model Loss', dpi=250)
plt.show()


# ## **Load Model for Predictions**

# In[ ]:


final_model = load_model('Face_Mask_Model.hdf5')
print("Model is loaded")


# In[ ]:


# Display the validation accuracy and loss
results = final_model.evaluate(valid_data)
print("Loss: ", results[0])
print("Accuracy: ", results[1])


# In[ ]:


testing_data.head(5)


# In[ ]:


img_details = testing_data['filename']
print(len(img_details))
print(img_details[2])


# In[19]:


test_pred = []
for j in range(0,len(img_details)):
  img_name = testing_dir+"/"+img_details[j]
  # reading the images
  #print(img_name)
  img = cv2.imread(img_name)
  # Converting the color space from BGR to RGB
  img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
  # resizing the images to size required by ResNet50
  img = cv2.resize(img, (224,224))
  img = img.reshape(-1,224,224,3)
  prediction = final_model.predict(img)
  test_pred.append(np.argmax(prediction))


# In[ ]:


print(len(test_pred))


# In[ ]:


res = pd.DataFrame(test_pred)
res.columns=["prediction"]
res.to_csv("pred_results.csv")
from google.colab import files     
files.download('pred_results.csv')

