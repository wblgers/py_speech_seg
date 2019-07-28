from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential, Model
from keras.layers import Bidirectional, Input, Lambda, TimeDistributed, Dropout, Concatenate
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.backend import squeeze
from keras import regularizers
from keras.layers import LSTM
import numpy as np
import keras
import keras.backend as K
from smorms3 import SMORMS3

model = Sequential()

model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(32)))
model.add(TimeDistributed(Dense(32)))
# model.add(TimeDistributed(Dense(2)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))

model.build(input_shape=(None, 200, 35))

def weightedLoss(originalLossFunc, weightsList):

    def lossFunc(true, pred):

        axis = -1 #if channels last
        #axis=  1 #if channels first


        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index
        classSelectors = K.argmax(true, axis=axis)

        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index
        classSelectors = [K.equal(np.int64(i), classSelectors) for i in range(len(weightsList))]

        #casting boolean to float for calculations
        #each tensor in the list contains 1 where ground true class is equal to its index
        #if you sum all these, you will get a tensor full of ones.
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)]

        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]


        #make sure your originalLossFunc only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true,pred)
        loss = loss * weightMultiplier

        return loss
    return lossFunc


# model.compile(loss=weightedLoss(keras.losses.binary_crossentropy, [1.0, 2.0]), optimizer=Adam(), metrics=['accuracy'])
# model.compile(loss=keras.losses.binary_crossentropy, optimizer=Adam(), metrics=['accuracy'])
model.compile(loss=keras.losses.binary_crossentropy, optimizer=SMORMS3(), metrics=['accuracy'])
model.summary()

# all_x = np.load('x.npy')
# all_y = np.load('y.npy')
from prepare_dataset_new import load_dataset
all_x, all_y = load_dataset()
print(all_y.shape, np.sum(all_y))

subsample_all_x = []
subsample_all_y = []
for index in range(all_y.shape[0]):
    class_positive = sum(all_y[index])
    if class_positive > 5:
        subsample_all_x.append(all_x[index][np.newaxis, :, :])
        subsample_all_y.append(all_y[index])

all_x = np.vstack(subsample_all_x)
all_y = np.vstack(subsample_all_y)
print(all_y.shape, np.sum(all_y))

# all_y = keras.utils.to_categorical(all_y, num_classes=2)
all_y = all_y[:, :, np.newaxis]

indices = np.random.permutation(all_x.shape[0])
all_x_random = all_x[indices]
all_y_random = all_y[indices]

datasize = all_x_random.shape[0]
train_size = int(datasize*0.97)
train_x = all_x_random[0:train_size]
valid_x = all_x_random[train_size:]

train_y = all_y_random[0:train_size]
valid_y = all_y_random[train_size:]
print('train over')

model.fit(x=train_x, y=train_y, batch_size=256, epochs=200,
          validation_data=(valid_x, valid_y), shuffle=True)


def save_model(model, json_model_file, h5_model_file):
    # serialize model to JSON
    model_json = model.to_json()
    with open(json_model_file, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(h5_model_file)
    print("Saved model to disk")


model_name = 'speech_seg'
json_model_file = model_name+'.json'
h5_model_file = model_name+'.h5'
save_model(model, json_model_file, h5_model_file)
