#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''
Lucien Derouet - Benoit Estrade
MODIFICATION POUR LA RUN NUMERO 3

Model 6 couches

'''


# In[2]:


import cv2                
import numpy as np         
import os                 
from random import shuffle 
from tqdm import tqdm      

TRAIN_DIR = '/home/estrade/Bureau/DL/DOG_VS_CAT/cat_dogs_kaggle/data_train/train'
TEST_DIR = '/home/estrade/Bureau/DL/DOG_VS_CAT/cat_dogs_kaggle/data_test'
LOAD_DIR = '/home/estrade/Bureau/DL/DOG_VS_CAT/'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME ='6conv-version'


# In[4]:


def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return [1,0]
    #                             [no cat, very doggo]
    elif word_label == 'dog': return [0,1]


# In[5]:


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


# In[6]:


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('data_test.npy', testing_data)
    return testing_data


# In[6]:


train_data = create_train_data()


# In[7]:


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
tf.reset_default_graph() #Permet de reset le noyau de tflearn. 

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


model = tflearn.DNN(convnet, tensorboard_dir='log')


# In[8]:


if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')


# In[9]:


path = os.path.join(LOAD_DIR, 'train_data.npy')
train_data = np.load(path)

train = train_data[:-1000]
test = train_data[-1000:]


# In[10]:


X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]


# In[12]:


model.fit({'input': X}, {'targets': Y}, n_epoch=6, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=1000, show_metric=True, run_id=MODEL_NAME)


# In[17]:


#tensorboard --logdir=/home/estrade/Bureau/DL/DOG_VS_CAT


# In[ ]:


process_test_data()


# In[1]:


import matplotlib.pyplot as plt

# if you need to create the data:
#test_data = process_test_data()
# if you already have some saved:

test_data = np.load('data_test.npy')

fig=plt.figure()

for num,data in enumerate(test_data[:12]):
    # cat: [1,0]
    # dog: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label='Dog'
    else: str_label='Cat'
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()


# In[ ]:




