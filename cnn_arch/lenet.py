import numba

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input

@numba.jit
def build_model(img_size, CLASSES):
    model = Sequential()
    # 第一層
    model.add(Convolution2D(32, 5, 5, input_shape=(img_size, img_size, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 第二層
    model.add(Convolution2D(64, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 全結合層
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(CLASSES))
    model.add(Activation('softmax'))

    #model.summary()
    return model
