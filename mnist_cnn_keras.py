from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json

import matplotlib.pyplot as plt
import glob
import numpy as np
import scipy

batch_size = 32
num_classes = 4
epochs = 25

def get_label(pos, len_A, len_B, len_C = 0, len_D = 0, len_E = 0):
    a = len_A
    b = a + len_B
    c = b + len_C
    d = c + len_D
    e = d + len_E
    if (pos < a):
        label = 0
    elif (pos < b):
        label = 1
    elif (pos < c):
        label = 2
    elif (pos < d):
        label = 3
    elif (pos < e):
        label = 4
    else:
        label = 5
    return label


# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

train_list_A = glob.glob('A*_gr_28_train/*.png')
train_list_B = glob.glob('B*_gr_28_train/*.png')
train_list_C = glob.glob('C*_gr_28_train/*.png')
train_list_D = glob.glob('D*_gr_28_train/*.png')
train_list = train_list_A + train_list_B + train_list_C + train_list_D
eval_list_A = glob.glob('A*_gr_28_test/*.png')
eval_list_B = glob.glob('B*_gr_28_test/*.png')
eval_list_C = glob.glob('C*_gr_28_test/*.png')
eval_list_D = glob.glob('D*_gr_28_test/*.png')
eval_list = eval_list_A + eval_list_B + eval_list_C + eval_list_D

np.set_printoptions(precision=3, threshold=100000)
print(train_list)
print(eval_list)

x_train = np.array([np.array(scipy.ndimage.imread(fname)) for fname in train_list], dtype=np.float32)
x_test = np.array([np.array(scipy.ndimage.imread(fname)) for fname in eval_list], dtype=np.float32)

y_train = np.array([get_label(i, len(train_list_A), len(train_list_B), len(train_list_C), len(train_list_D)) for i in range(len(train_list))], dtype=np.int32)
y_test = np.array([get_label(i, len(eval_list_A), len(eval_list_B), len(eval_list_C), len(eval_list_D)) for i in range(len(eval_list))], dtype=np.int32)

print(x_train, x_test, y_train, y_test)

# permutation, will be used for data as well as labels
perm = np.random.permutation(len(train_list))
x_train = x_train[perm]
y_train = y_train[perm]

# perm = np.random.permutation(len(eval_list))
# x_test = x_test[perm]
# y_test = y_test[perm]

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

np.set_printoptions(precision=3, threshold=100000)
print(x_train)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
score = loaded_model.evaluate(x_test, y_test, verbose=2)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
print('Test loss:', score[0])
print('Test accuracy:', score[1])