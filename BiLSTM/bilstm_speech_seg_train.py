from keras.layers.core import Dense
from keras.models import Sequential
from keras.layers import Bidirectional, TimeDistributed, Dropout
from keras.layers import LSTM
import numpy as np
import keras
from BiLSTM.smorms3 import SMORMS3
from BiLSTM.prepare_dataset import load_dataset


def train_bilstm():

    model = Sequential()

    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(32)))
    model.add(TimeDistributed(Dense(32)))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    model.build(input_shape=(None, 200, 35))

    model.compile(loss=keras.losses.binary_crossentropy, optimizer=SMORMS3(), metrics=['accuracy'])
    model.summary()

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
