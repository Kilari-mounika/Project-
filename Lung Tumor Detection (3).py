#!/usr/bin/env python
# coding: utf-8

# In[1]:


from builtins import range, input
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical


IMAGE_SIZE = [224, 224]

tumor_path = r'C:\Users\ADMIN\OneDrive\Desktop\4-1\Major Project\image\Tumor'
nontumor_path = r'C:\Users\ADMIN\OneDrive\Desktop\4-1\Major Project\image\Non-tumor'

 


tumor_files = glob(tumor_path + '/*')
nontumor_files = glob(nontumor_path + '/*')

print("First 5 Tumor Files: ",tumor_files[0:5])
print("Total Count: ",len(tumor_files))
print("First 5 NonTumor Files: ",nontumor_files[0:5])
print("Total Count: ",len(nontumor_files))





# In[2]:


tumor_labels = []
nontumor_labels = []
tumor_images=[]
nontumor_images=[]

for i in range(len(tumor_files)):
    image = cv2.imread(tumor_files[i]) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = cv2.resize(image,(224,224)) 
    tumor_images.append(image) 
    tumor_labels.append('CT_TUMOR')
    
for i in range(len(nontumor_files)):
    image = cv2.imread(nontumor_files[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(224,224))
    nontumor_images.append(image)
    nontumor_labels.append('CT_NonTUMOR')



# In[3]:


tumor_images[2].shape


# In[4]:


print(tumor_images[2])


# In[5]:


cv2.imwrite("img.jpg",tumor_images[2])


# In[6]:


tumor_images = np.array(tumor_images) / 255
nontumor_images = np.array(nontumor_images) / 255


# In[7]:


tumor_x_train, tumor_x_test, tumor_y_train, tumor_y_test = train_test_split(
tumor_images, tumor_labels, test_size=0.2)
nontumor_x_train, nontumor_x_test, nontumor_y_train, nontumor_y_test = train_test_split(
nontumor_images, nontumor_labels, test_size=0.2)


# In[8]:


X_train = np.concatenate((nontumor_x_train, tumor_x_train), axis=0)
X_test = np.concatenate((nontumor_x_test, tumor_x_test), axis=0)
y_train = np.concatenate((nontumor_y_train, tumor_y_train), axis=0)
y_test = np.concatenate((nontumor_y_test, tumor_y_test), axis=0)


# In[9]:


y_train = LabelBinarizer().fit_transform(y_train)
y_train = to_categorical(y_train)
y_test = LabelBinarizer().fit_transform(y_test)
y_test = to_categorical(y_test)
for i in range(0,5):
    image1=tumor_x_train[i]
    image2=tumor_x_test[i]
    plt.imshow(image1)
    plt.imshow(image2)
    plt.show()


# In[14]:


resnet = ResNet50(weights="imagenet", include_top=False,
input_tensor=Input(shape=(224, 224, 3)))
outputs = resnet.output
outputs = Flatten(name="flatten")(outputs)
outputs = Dropout(0.5)(outputs)
outputs = Dense(2, activation="softmax")(outputs)
model = Model(inputs=resnet.input, outputs=outputs)
for layer in resnet.layers:
    layer.trainable = False
model.compile(
loss='categorical_crossentropy',
optimizer='adam',
metrics=['accuracy']
)


# In[15]:


model.summary()
train_aug = ImageDataGenerator(
rotation_range=20,
width_shift_range=0.2,
height_shift_range=0.2,
horizontal_flip=True
)
history = model.fit(train_aug.flow(X_train, y_train, batch_size=batch_size),
validation_data=(X_test, y_test),
validation_steps=len(X_test) / batch_size,
steps_per_epoch=len(X_train) / batch_size,
epochs=epochs)






