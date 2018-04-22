import numba

from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Input, Flatten
from keras.applications.vgg19 import VGG19


@numba.jit
def build_model(img_size, CLASSES):
    input_tensor = Input(shape = (img_size, img_size, 3))
    top_model = VGG19(include_top=False, weights='imagenet', input_tensor=input_tensor)
    model = Sequential()
    model.add(Flatten(input_shape=top_model.output_shape[1:]))
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Dense(CLASSES))
    model.add(Activation('sigmoid'))

    model = Model(input=top_model.input, output=model(top_model.output))

    #model.summary()
    return model
