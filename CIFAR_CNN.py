import keras.backend as K
from keras.models import Model
from keras.layers import Input,Convolution2D,MaxPooling2D,Dense,Dropout,Flatten
from keras.utils import np_utils
import numpy as np
import os
import tarfile
import cPickle
import matplotlib.pyplot as plt
import keras

np.random.seed(123123)

K.set_image_dim_ordering('th')
os.environ['THEANO_FLAGS'] = "device=cuda,force_device=True,floatX=float32"
import theano

batch_size = 32
num_epochs = 30
kern_size = 3
pool_size = 2
conv_depth1 = 32
conv_depth2 = 64
drop_prob1 = 0.25
drop_prob2 = 0.5
hidden_size = 512
LR = 2.5e-3
decay_rate = LR/num_epochs
adam = keras.optimizers.adam(lr=LR,decay=decay_rate)

tar_file = tarfile.open('cifar-10-python.tar.gz','r:gz')
X_train = []

for batch in range(1,6):
    f = tar_file.extractfile('cifar-10-batches-py/data_batch_%d' % batch)
    array = cPickle.load(f)
    X_train.append(array)
    f.close()

x_train = np.concatenate([batch['data'].reshape(batch['data'].shape[0],3,32,32) for batch in X_train])
y_train = np.concatenate([np.array(batch['labels'],dtype=np.uint8) for batch in X_train])

f = tar_file.extractfile('cifar-10-batches-py/test_batch')
X_test = cPickle.load(f)

x_test = X_test['data'].reshape(X_test['data'].shape[0],3,32,32)
y_test = np.array(X_test['labels'],dtype=np.uint8)

n_train,depth,width,height = x_train.shape
n_test = x_test.shape[0]
n_class = np.unique(y_train).shape[0]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.

y_train = np_utils.to_categorical(y_train,n_class)
y_test = np_utils.to_categorical(y_test,n_class)

inp = Input(shape=(depth,height,width))
conv1 = Convolution2D(conv_depth1,(kern_size,kern_size),padding='same',activation='relu')(inp)
conv2 = Convolution2D(conv_depth1,(kern_size,kern_size),padding='same',activation='relu')(conv1)
pool1 = MaxPooling2D((pool_size,pool_size))(conv2)
drop1 = Dropout(drop_prob1)(pool1)
conv3 = Convolution2D(conv_depth2,(kern_size,kern_size),padding='same',activation='relu')(drop1)
conv4 = Convolution2D(conv_depth2,(kern_size,kern_size),padding='same',activation='relu')(conv3)
pool2 = MaxPooling2D((pool_size,pool_size))(conv4)
drop2 = Dropout(drop_prob2)(pool2)
flat = Flatten()(drop2)
hidden = Dense(hidden_size,activation='relu')(flat)
drop3 = Dropout(drop_prob2)(hidden)
out = Dense(n_class,activation='softmax')(drop3)

model = Model(inp,out)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
print K.eval(model.optimizer.lr)
hist = model.fit(x_train,y_train,batch_size,num_epochs,verbose=2,validation_split=0.1)
plt.figure(0)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train','test'],loc='upper left')
plt.figure(1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()
score = model.evaluate(x_test,y_test,verbose=2)
print score
