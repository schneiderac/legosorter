import keras
from keras.models import model_from_json
import scipy
import glob
import numpy as np

batch_size = 16
num_classes = 4
epochs = 25

# load dataset
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

x_train = np.array([np.array(scipy.ndimage.imread(fname)) for fname in train_list], dtype=np.float32)
x_test = np.array([np.array(scipy.ndimage.imread(fname)) for fname in eval_list], dtype=np.float32)

y_train = np.array([get_label(i, len(train_list_A), len(train_list_B), len(train_list_C), len(train_list_D)) for i in range(len(train_list))], dtype=np.int32)
y_test = np.array([get_label(i, len(eval_list_A), len(eval_list_B), len(eval_list_C), len(eval_list_D)) for i in range(len(eval_list))], dtype=np.int32)

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

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

print(np.shape(y_test))

for i in range(499):
    value = max(loaded_model.predict(np.expand_dims(x_test[i], axis=0), verbose=0))
    index_max = np.argmax(value)
    if (index_max != np.argmax(y_test[i])):
        print(index_max, value[index_max], eval_list[i])