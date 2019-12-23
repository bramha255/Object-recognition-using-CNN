import os
import numpy as np
import cv2
loc = (r'D:\Deep Learning\Datasets\natural_images\images15')
# LABEL EXTRACTION
labels=[]
for i in os.listdir(loc):  
    #print(i)
    #print(i.split('_'))
    if(i.split('_')[0]=='airplane'):
            labels.append(0)
    elif(i.split('_')[0]=='car'):
            labels.append(1)
    elif(i.split('_')[0]=='cat'):
            labels.append(2)
    elif(i.split('_')[0]=='dog'):
            labels.append(3)
    elif(i.split('_')[0]=='daisy'):
            labels.append(4)
    elif(i.split('_')[0]=='dandelion'):
            labels.append(5)
    elif(i.split('_')[0]=='flower'):
            labels.append(6)
    elif(i.split('_')[0]=='fruit'):
            labels.append(7)
    elif(i.split('_')[0]=='motorbike'):
            labels.append(8)
    elif(i.split('_')[0]=='person'):
            labels.append(9)
    elif(i.split('_')[0]=='rose'):
            labels.append(10)
    elif(i.split('_')[0]=='sunflower'):
            labels.append(11)
    elif(i.split('_')[0]=='tulips'):
            labels.append(12)
    #print(labels)
# FEATURES EXTRACTION
features=[]
for i in os.listdir(loc):
    path = os.path.join(loc,i)
    img = cv2.imread(path)
    img = cv2.resize(img,(100,100))
    features.append(img)
    
#%% TRAINING AND TESTING THE MODEL
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,MaxPooling2D,Conv2D,Dropout

#%% execute Final_code_for_label.py and execute label code

ft = np.array(features)
#ft = ft/255

lt = np.array(labels)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(ft,lt)

#""""""""""""""'''''''''''''''''"%% Model
model = Sequential()

model.add(Conv2D(146, (5, 5), activation='relu', input_shape=(100,100,3)))
model.add(MaxPooling2D(pool_size=5, strides=3, padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(142, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=3, padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(138, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=3, padding='same'))

model.add(Dense((512), activation='relu'))
model.add(Flatten())
model.add(Dense(13, activation='softmax'))

#%%
import tensorflow
adam = tensorflow.keras.optimizers.Adam(learning_rate=0.003, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

modelf = model.fit(xtrain, ytrain,
                   batch_size=130,epochs=10,
                   verbose=2,
                   validation_data=(xtest, ytest))
#%%PREDICTION
#%%loss and metrics
loss_and_metrics = model.evaluate(xtest, ytest, verbose=2)

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])

#%%GRAPH

import matplotlib.pyplot as plt
plt.subplot(2,1,1)
plt.plot(modelf.history['accuracy'])
plt.plot(modelf.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(modelf.history['loss'])
plt.plot(modelf.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

model.save('13class_75%acc.h5')

#%% PREDICTION FROM WEBCAM
#%%DEPLOYMENT
import cv2
from keras.models import load_model
model = load_model('13class_79%accuracy.model')

category = ['Airplane','Car','Cat','Dog','Daisy','Dandelion','Flower',
            'Fruit','Motorbike','Person','Rose','Sunflower','Tulips']

def prepare(filepath):
    #img = cv2.imread(filepath)
    img = cv2.resize(filepath,(100,100))
    return img.reshape(-1,100,100,3)   #put return
    
prediction = model.predict([prepare('Car_side_17.jpg')])
number = prediction.argmax()
d = category[number]
print(d)

#%%
video_capture = cv2.VideoCapture(0)
a=1
while True:
    check, frame = video_capture.read()
    canvas = prepare(frame)
    prediction = model.predict([prepare(frame)])
    number = prediction.argmax()
    d = category[number]
    print(d)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break
video_capture.release()
cv2.destroyAllWindows()
