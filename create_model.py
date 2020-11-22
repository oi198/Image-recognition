from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Flatten, Activation
import matplotlib.pyplot as plt
# %matplotlib inline

plt.figure(figsize=(10,15))

batch_size = 128
num_classes = 10
epochs = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28,28,1)
x_test = x_test.reshape(10000, 28,28,1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.title(y_train[i].argmax())
    plt.axis('off')
    plt.imshow(x_train[i].reshape((28, 28)), cmap='gray')
"""    

model = Sequential()
model.add(Conv2D(5, (5, 5), padding='valid', input_shape=(28,28,1)))
model.add(Activation('relu')) # Conv2Dのところにまとめて書いても良い
model.add(MaxPool2D(strides=(2, 2)))
model.add(Conv2D(5, (3,3), padding='same', activation='relu'))
model.add(Conv2D(5, (3,3), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
#model.summary()

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test))

from google.colab import drive
drive.mount('/content/drive')

model.save('/content/drive/My Drive/model.h5')
