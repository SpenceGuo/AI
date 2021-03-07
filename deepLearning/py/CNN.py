import keras

from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Activation

model = Sequential()

"""
第一层 Convolution 和 Maxpooling
输入图像大小：28*28 的矩阵且深度为 1
卷积核 Filter： 25 个 3*3 大小的2D矩阵（此时每一个 Filter 的参数为 3*3=9 个）
池化层：在 2*2 大小内做Maxpooling
"""
model.add(Convolution2D(25, 3, 3, input_shape=(1, 28, 28)))
# 卷积后输入的矩阵变为 26*26 且深度为 25
model.add(MaxPooling2D((2, 2)))
# 池化后矩阵大小变为 13*13 且深度仍为 25

"""
第二层 Convolution 和 Maxpooling
输入图像大小：此时输入的矩阵为上一层的输出矩阵，即 13*13 的矩阵，且深度仍为 25
卷积核 Filter： 50 个 3*3 大小的2D矩阵（此时每一个 Filter 的参数为 3*3*25=225 个）
池化层：在 2*2 大小内做Maxpooling
"""
model.add(Convolution2D(50, 3, 3))
# 卷积后输入的矩阵变为 11*11 且深度为 50
model.add(MaxPooling2D((2, 2)))
# 池化后矩阵大小变为 5*5 且深度仍为 50

# 将上一层的结果拉直为以为向量，维度为 50*5*5=1250
model.add(Flatten())

# Fully Connected Feedforward network
model.add(Dense(output_dim=100))
model.add(Activation('relu'))
model.add(Dense(output_dim=10))
model.add(Activation('softmax'))
