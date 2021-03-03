import keras

from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Activation

model = Sequential()

# 第一层 Convolution 和 Maxpooling
model.add(Convolution2D(25, 3, 3, input_shape=(1, 28, 28)))
model.add(MaxPooling2D((2, 2)))

# 第二层 Convolution 和 Maxpooling
model.add(Convolution2D(50, 3, 3))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

# Fully Connected Feedforward network
model.add(Dense(output_dim=100))
model.add(Activation('relu'))
model.add(Dense(output_dim=10))
model.add(Activation('softmax'))
