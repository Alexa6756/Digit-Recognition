import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#Data Processing
(x_train,y_train),(x_test,y_test)=mnist.load_data()

#Normalize Data
x_train,x_test=x_train/255.0,x_test/255.0

#Reshape Only 4D batchsize height width channel
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

#vectors of 0 and 1
y_train=to_categorical(y_train,num_classes=10)
y_test=to_categorical(y_test,num_classes=10)

#Creating Model
model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(10,activation='softmax')
])

#Compiling Model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Training Model
model.fit(x_train,y_train,epochs=5,batch_size=32,validation_data=(x_test,y_test))

#Evaluate Model
test_loss,test_acc=model.evaluate(x_test,y_test)
print(f"test accuracy:{test_acc:.4f}")

#Predictions
index=np.random.randint(0,len(x_test))
reshape_img=x_test[index].reshape(1,28,28,1)

digit=np.argmax(model.predict(reshape_img))

plt.imshow(reshape_img.reshape(28,28),cmap='gray')
plt.title(f"digit:{digit}")
plt.show()