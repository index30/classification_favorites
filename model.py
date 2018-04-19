from datetime import datetime
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
from pathlib import Path, PurePath
import pickle
import tensorflow as tf

from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Input, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import keras.backend.tensorflow_backend as K
from keras.utils import np_utils
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception

# for filter
K_SIZE = 3
P_SIZE = (2, 2)
STRIDES = (2, 2)

# Directory
DATA_PATH = 'images/data'


def build_model(image_size, CLASSES, name='VGG19'):
    input_tensor = Input(shape = (image_size, image_size, 3))
    #top_model = VGG19(include_top=False, weights='imagenet', input_tensor=input_tensor)
    top_model = Xception(include_top=False, weights='imagenet', input_tensor=input_tensor)
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


def main(img_size, classes, batch_size, epoch):
    model = build_model(img_size, classes)
    train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                       #rotation_range=90,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       #shear_range=0.78,
                                       zoom_range=0.1,
                                       #channel_shift_range=100,
                                       horizontal_flip=True,
                                       vertical_flip=True)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    train_generator = train_datagen.flow_from_directory(
        str(Path(DATA_PATH, 'train')),
        #color_mode="grayscale",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical')
    validation_generator = val_datagen.flow_from_directory(
        str(Path(DATA_PATH, 'validation')),
        #color_mode="grayscale",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical')

    # modelの保存
    save_root_path = "./result"
    save_root = Path(save_root_path)
    if not save_root.exists():
        save_root.mkdir()

    now_time = datetime.now().strftime("%m%d%H%M%S")
    eval_path = Path(save_root, now_time)
    eval_path.mkdir()
    MODEL_NAME = str(Path(eval_path, "model.h5"))
    para_txt = str(Path(eval_path, "parameter.txt"))
    para_str = ["image_size", "batch_size", "epochs", "train_size", "val_size", "model_type"]
    parameter = [img_size, batch_size, epoch, 0, 0, 'vgg19']
    with open(para_txt, mode='w') as p:
        for (arg, txt) in zip(parameter, para_str):
            line = txt +": "+str(arg)+"\n"
            p.write(line)

    callbacks = [
        ReduceLROnPlateau(factor=0.1, patience=5, mode='auto', verbose=1, epsilon=0.0001, min_lr=1e-5),
        ModelCheckpoint(filepath=MODEL_NAME, verbose=1, save_best_only=True)
    ]

    history = model.fit_generator(
        train_generator,
        samples_per_epoch=60,#train_size // b_size,
        nb_epoch=epoch,
        validation_data=validation_generator,
        validation_steps=60,
        callbacks=callbacks)#val_size // b_size)

    pickle_file_name = str(Path(eval_path, "result.pickle"))
    with open(pickle_file_name, mode='wb') as h:
        pickle.dump(history.history, h)


def predict(img_path, model_name, img_size, classes):
    img = load_img(img_path, target_size=(img_size,img_size))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x /= 255
    model = build_model(img_size, classes)
    model.load_weights(model_name)
    #model = load_model(model_name)
    predict = model.predict(x)
    return predict


if __name__=="__main__":
    ### when train
    #main(256, 3, 10, 50)

    ### when predict
    #img_name = "images/data/validation/illust/illustFGOイラスト4.jpg"
    #img_name = "images/data/validation/scenery/scenery風景353.jpg"
    img_name = "images/data/validation/animal/animal猫374.jpg"
    model_name = "result/model.h5"
    predict = predict(img_name, model_name, 256, 3)
    print(predict)
