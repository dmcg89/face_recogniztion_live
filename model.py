import sys,os
# root_path = 'gdrive/My Drive/Colab Notebooks/tripletloss/'
# sys.path.append(root_path)

from preprocess import PreProcessing
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint

dataset = PreProcessing('ExtendedYaleB/', 'negative')

anchors = []
positives = []
negatives = []
n = 1000

for _ in range(n):
    triplet = dataset.generate_triplets()
    anchors.append(triplet[0])
    positives.append(triplet[1])
    negatives.append(triplet[2])

import numpy as np
def train_test_split(arr, p=.9):
    test = []
    train = []
    for i in range(len(arr)):
        if i < len(arr)*.9: train.append(arr[i])
        else: test.append(arr[i])
    return np.array(train), np.array(test)
#     return train, test

anchor_train, anchor_test = train_test_split(anchors)
print((anchor_train[1].shape))
print((anchor_train[3].shape))
positive_train, positive_test = train_test_split(positives)
negative_train, negative_test = train_test_split(negatives)

def triplet_loss(y_true, y_pred, alpha = 0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ',y_pred)
    
    total_lenght = y_pred.shape.as_list()[-1]
#     print('total_lenght=',  total_lenght)
#     total_lenght =12
    
    anchor = y_pred[:,0:int(total_lenght*1/3)]
    positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
    negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
 
    return loss

from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import os
from keras.callbacks import ModelCheckpoint
import keras
print(keras.__version__)
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

def create_base_network(in_dims):
    """
    Base network to be shared.
    """
    model = Sequential()
    model.add(Conv2D(128,(7,7),padding='same',input_shape=(in_dims[0],in_dims[1],in_dims[2],),activation='relu',name='conv1'))
    model.add(MaxPooling2D((2,2),(2,2),padding='same',name='pool1'))
    model.add(Conv2D(256,(5,5),padding='same',activation='relu',name='conv2'))
    model.add(MaxPooling2D((2,2),(2,2),padding='same',name='pool2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(4,name='embeddings'))
    # model.add(Dense(600))
    
    return model

adam_optim = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)

anchor_input = Input((480, 640, 1), name='anchor_input')
positive_input = Input((480, 640, 1), name='positive_input')
negative_input = Input((480, 640, 1), name='negative_input')

# Shared embedding layer for positive and negative items
Shared_DNN = create_base_network([480, 640, 1])


encoded_anchor = Shared_DNN(anchor_input)
encoded_positive = Shared_DNN(positive_input)
encoded_negative = Shared_DNN(negative_input)


merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')

model = Model(inputs=[anchor_input,positive_input, negative_input], outputs=merged_vector)
model.compile(loss=triplet_loss, optimizer=adam_optim)

model.summary()

import numpy as np
# RGB Images
# Anchor_train = anchor_train.reshape(-1,368,368,3)
# Positive_train = positive_train.reshape(-1,368,368,3)
# Negative_train = negative_train.reshape(-1,368,368,3)
# Anchor_test = anchor_test.reshape(-1,368,368,3)
# Positive_test = positive_test.reshape(-1,368,368,3)
# Negative_test = negative_test.reshape(-1,368,368,3)
# (480, 640, 1)
# Greyscale Images
Anchor_train = anchor_train.reshape(-1,480, 640, 1)
Positive_train = positive_train.reshape(-1,480, 640, 1)
Negative_train = negative_train.reshape(-1,480, 640, 1)
Anchor_test = anchor_test.reshape(-1,480, 640, 1)
Positive_test = positive_test.reshape(-1,480, 640, 1)
Negative_test = negative_test.reshape(-1,480, 640, 1)

trained_model = Model(inputs=anchor_input, outputs=encoded_anchor)

trained_model.load_weights("weights.hdf5")
print('here')

X_train_trm = trained_model.predict(dataset.images_train.reshape(-1,480, 640, 1))

len(X_train_trm)
print('here1')
X_train_trm_new = X_train_trm[:850]
X_test_trm = X_train_trm[850:922]
print('here2')

y_train_onehot = np.zeros((dataset.labels_train.size, dataset.labels_train.max()+1))
y_train_onehot[np.arange(dataset.labels_train.size),dataset.labels_train] = 1

print(len(y_train_onehot))
y_train_onehot_new = y_train_onehot[:850]
y_test_onehot = y_train_onehot[850:922]

dataset.labels_test
# y_test_onehot = np.zeros((dataset.labels_test.size, dataset.labels_test.max()+1))
# y_test_onehot[np.arange(dataset.labels_test.size),dataset.labels_test] = 1
# dataset.labels_test.max()
print(len(y_test_onehot))

Classifier_input = Input((4,))
Classifier_output = Dense(27, activation='softmax')(Classifier_input)
Classifier_model = Model(Classifier_input, Classifier_output)

Classifier_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# checkpointer = ModelCheckpoint(filepath="weights2.hdf5", verbose=1, save_best_only=True)
Classifier_model.fit(X_train_trm_new,y_train_onehot_new, validation_data=(X_test_trm,y_test_onehot),epochs=200)
Classifier_model.save('my_model.h5')