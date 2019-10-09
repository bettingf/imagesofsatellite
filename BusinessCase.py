
# coding: utf-8

# # Images of satellites classification

# In[37]:


from google_images_download import google_images_download   #importing the library
import pandas
import cv2 #to load and displays images
import matplotlib.pyplot as plt
import imutils
import os
import pandas as pd
from shutil import copyfile


# ## Download images of satellites / non satellites

# In[2]:


nb_images = 2000 # The number of images to download (both from satellite and non satellite images)
download = False  # Are we doanloading the images ?
chromeDriverPath = "/usr/bin/chromedriver"


# In[3]:


def downloadImages(tags, nbImgs):
    response = google_images_download.googleimagesdownload()   #class instantiation

    arguments = {"keywords":','.join(tags),"limit":nbImgs,"print_urls":False, "extract_metadata":True, "thumbnail_only":True, "chromedriver":chromeDriverPath}   #creating list of arguments
    if download:
        response.download(arguments)   #passing the arguments to the function


# download list of satellites categories

# In[4]:


# categories taken from https://www.omicsonline.org/conferences-list/types-of-satellites-and-applications
categories = ["Communications Satellite","Remote Sensing Satellite","Navigation Satellite","LEO satellite", 
              "MEO satellite", "HEO satellite","GPS satellite","GEO satellite","Drone Satellite",
              "Polar Satellite","Nano Satellites","CubeSats","SmallSats"]


# In[5]:


downloadImages(categories, nb_images)


# In[6]:


catpd = [pandas.read_json("logs/" + str(cat) + ".json") for cat in categories]


# ### download of images

# In[7]:


downloadImages(["satellite","images"], nb_images)


# In[8]:


imgpd = pandas.read_json("logs/images.json")
satpd = pandas.read_json("logs/satellite.json")


# In[9]:


imgpd.image_filename.count(), satpd.image_filename.count()


# In[10]:


satpd.iloc[0]


# ### display of the first images

# In[11]:


def draw_imgs(imgs_list):
    l = int(len(imgs_list))
    n = int(len(imgs_list[0]))
    _, axs = plt.subplots(l, n, figsize=(17, 17))
    axs = axs.flatten()
    for img, ax in zip([item for sublist in imgs_list for item in sublist], axs):
        ax.imshow(img)
    plt.show()


# In[12]:


listnonsat = []
for i in range(0,7):
    img =cv2.imread("./downloads/images - thumbnail/"+imgpd.image_filename[i])
    img = cv2.resize(img, (244,244), cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    listnonsat.append(img)
    
listsat = []
for i in range(0,7):
    img =cv2.imread("./downloads/satellite - thumbnail/"+satpd.image_filename[i])
    img = cv2.resize(img, (244,244), cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    listsat.append(img)
    
draw_imgs([listnonsat, listsat])


# In[13]:


listcat = []
for current,cat in zip(catpd, categories):
    print(cat)
    listcurrent = []
    for i in range(0,7):
        img =cv2.imread("./downloads/"+str(cat)+" - thumbnail/"+current.image_filename[i])
        img = cv2.resize(img, (244,244), cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        listcurrent.append(img)
    listcat.append(listcurrent)
    
draw_imgs(listcat)


# ## Classify satellite images

# In[14]:


#saving images in the right directories for keras CNN

train_size = 200
test_size = 100

try:
    os.mkdir("./classification")
    os.mkdir("./classification/test")
    os.mkdir("./classification/test/images")
    os.mkdir("./classification/test/satellite")
    os.mkdir("./classification/train")
    os.mkdir("./classification/train/images")
    os.mkdir("./classification/train/satellite")
    os.mkdir("./classification/valid")
    os.mkdir("./classification/valid/images")
    os.mkdir("./classification/valid/satellite")
except:
    print("directories already in place")

def path_from_number(n):
    if i<=train_size:
        path="train"
    else:
        if i<=train_size+test_size:
            path = "test"
        else:
            path = "valid"
    return path

listnonsat = []
for i in range(0,imgpd.shape[0]):
    try:
        img =cv2.imread("./downloads/images - thumbnail/"+imgpd.image_filename[i])
        img = cv2.resize(img, (64,64), cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./classification/" + path_from_number(i)+"/images/"+str(i)+".jpg", img)
        listnonsat.append(img)
    except:
        print("image not resizable")
    
listsat = []
for i in range(0,satpd.shape[0]):
    try:
        img =cv2.imread("./downloads/satellite - thumbnail/"+satpd.image_filename[i])
        img = cv2.resize(img, (64,64), cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)         
        cv2.imwrite("./classification/" + path_from_number(i)+"/satellite/"+str(i)+".jpg", img)
        listsat.append(img)
    except:
        print("image not resizable")
    


# In[15]:


len(listnonsat), len(listsat)


# In[16]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import time
IMG_SIZE = 64 # Replace with the size of your images
NB_CHANNELS = 3 # 3 for RGB images or 1 for grayscale images
BATCH_SIZE = 32 # Typical values are 8, 16 or 32
NB_TRAIN_IMG = 200 # Replace with the total number training images
NB_VALID_IMG = 50 # Replace with the total number validation images


# In[17]:


cnn = Sequential()
cnn.add(Conv2D(filters=32, 
               kernel_size=(2,2), 
               strides=(1,1),
               padding='same',
               input_shape=(IMG_SIZE,IMG_SIZE,NB_CHANNELS),
               data_format='channels_last'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),
                     strides=2))
cnn.add(Conv2D(filters=64,
               kernel_size=(2,2),
               strides=(1,1),
               padding='valid'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),
                     strides=2))
cnn.add(Conv2D(filters=128,
               kernel_size=(2,2),
               strides=(1,1),
               padding='valid'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),
                     strides=2))
cnn.add(Flatten())        
cnn.add(Dense(32))
cnn.add(Activation('relu'))
cnn.add(Dropout(0.25))
cnn.add(Dense(1))
cnn.add(Activation('sigmoid'))
cnn.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[18]:


print(cnn.summary())


# In[19]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'classification/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'classification/valid',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# In[20]:


cnn.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=80)


# In[21]:


cnn.save_weights('cnn.h5')


# In[22]:


cnn.load_weights('cnn.h5')


# In[23]:


test_generator = test_datagen.flow_from_directory(
        'classification/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_imgs, test_labels = next(test_generator)


# In[24]:


test_imgs.shape


# In[25]:


predictions = cnn.predict_generator(test_generator, steps=1)


# In[26]:


predictions.shape


# In[27]:


# check satellites
for i in range(0,predictions.shape[0]):
    if predictions[i]>0.5:
        plt.figure()
        plt.imshow(test_imgs[i])


# In[28]:


# check non satellites
for i in range(0,predictions.shape[0]):
    if predictions[i]<0.5:
        plt.figure()
        plt.imshow(test_imgs[i])


# ## filtering specific satellites datasets

# In[29]:


try:
    os.mkdir("./specific")
    os.mkdir("./specific/Remote Sensing Satellite")
except:
    print("directory exists")
    
listspecific = []
for i in range(0,catpd[1].shape[0]):
    try:
        img =cv2.imread("./downloads/Remote Sensing Satellite - thumbnail/"+catpd[1].image_filename[i])
        img = cv2.resize(img, (64,64), cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./specific/Remote Sensing Satellite/"+str(i)+".jpg", img)
        listspecific.append(img)
    except:
        print("image not resizable")
        
draw_imgs([listspecific])


# In[30]:


specific_generator = test_datagen.flow_from_directory(
        'specific',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
specific_imgs, specific_labels = next(specific_generator)


# In[31]:


specific_predictions = cnn.predict_generator(specific_generator, steps=1)


# In[32]:


# check satellites
for i in range(0,specific_predictions.shape[0]):
    if specific_predictions[i]>0.5:
        plt.figure()
        plt.imshow(specific_imgs[i])


# In[33]:


# check non satellites
for i in range(0,specific_predictions.shape[0]):
    if specific_predictions[i]<0.5:
        plt.figure()
        plt.imshow(specific_imgs[i])


# In[34]:


pd.Series(specific_predictions.transpose()[0]).hist()


# #### It looks like filtering is not going to work properly so we're just leaving the specific data sets as downloaded

# ### labelling specific satellite datasets

# In[77]:


try:
    os.mkdir("./labelling")
    os.mkdir("./labelling/train")
    os.mkdir("./labelling/test")
    os.mkdir("./labelling/valid")
    for cat in categories:
        os.mkdir("./labelling/train/" + cat)
        os.mkdir("./labelling/test/" + cat)
        os.mkdir("./labelling/valid/" + cat)
except:
    print("directory exists")
    
listlabelling = []
for j in range(0, len(categories)):
    for i in range(0,catpd[j].shape[0]):
        
        try:
            image = "./downloads/"+categories[j]+" - thumbnail/"+catpd[j].image_filename[i]
            dstype = "train"
            if i%3 == 1:
                dstype = "test"
            if i%3 == 2:
                dstype = "valid"
            dest = "./labelling/"+dstype+"/"+categories[j]+"/"+str(i)+".jpg"  
            
            img =cv2.imread(image)
            img = cv2.resize(img, (244,244), cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(dest, img)
            #copyfile(image, dest)
        except:
            print("image not copied")
        


# #### We want to use th VGG16 architexture but with only 13 classes

# In[78]:


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.optimizers import SGD
import numpy as np

#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

#Create your own input format (here 3x64x64)
inputlayer = Input(shape=(244,244,3),name = 'image_input')

#Use the generated model 
output_vgg16_conv = model_vgg16_conv(inputlayer)

#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(2048, activation='relu', name='fc1')(x)
x = Dense(512, activation='relu', name='fc2')(x)
x = Dense(13, activation='softmax', name='predictions')(x)

#Create the model 
labelmodel = Model(input=inputlayer, output=x)

#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
labelmodel.summary()


# In[84]:


sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
labelmodel.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
#labelmodel.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[85]:


train_label_generator = train_datagen.flow_from_directory(
        'labelling/train',
        target_size=(244, 244),
        batch_size=32,
        class_mode='categorical')

validation_label_generator = test_datagen.flow_from_directory(
        'labelling/valid',
        target_size=(244, 244),
        batch_size=32,
        class_mode='categorical')


# In[88]:


labelmodel.fit_generator(
        train_label_generator,
        steps_per_epoch=10,
        epochs=5,
        validation_data=validation_label_generator,
        validation_steps=5)


# #### By looking at the evolution of the accuracy (and validation accuracy) the network is not learning properly
# #### some overfitting occurs.
# #### We need to improve the quality and quantity of images in the datasets of the different categories to be able 
# #### to get a better annotation model.
