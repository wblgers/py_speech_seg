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

model = Sequential()

model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Bidirectional(LSTM(20, return_sequences=True)))
model.add(TimeDistributed(Dense(40)))
model.add(TimeDistributed(Dense(10)))
model.add(TimeDistributed(Dense(2)))

model.build(input_shape=(None, 200, 35))
# model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()

model_name = 'speech_seg'
json_model_file = model_name+'.json'
h5_model_file = model_name+'.h5'
model.load_weights(h5_model_file)

import librosa

def extract_feature():
    file = 'duihua_sample.wav'
    sr = 16000
    frame_size = 512
    frame_shift = 256
    y, sr = librosa.load(file, sr=sr)
    mfccs = librosa.feature.mfcc(y, sr, n_mfcc=12, hop_length=frame_shift, n_fft=frame_size)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    mfcc = mfccs[1:, ]
    norm_mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / np.std(mfcc, axis=1, keepdims=True)
    norm_mfcc_delta = (mfcc_delta - np.mean(mfcc_delta, axis=1, keepdims=True)) / np.std(mfcc_delta, axis=1, keepdims=True)
    norm_mfcc_delta2= (mfcc_delta2 - np.mean(mfcc_delta2, axis=1, keepdims=True)) / np.std(mfcc_delta2, axis=1, keepdims=True)

    ac_feature = np.vstack((norm_mfcc, norm_mfcc_delta, norm_mfcc_delta2))
    print(ac_feature.shape)

    sub_seq_len = int(2.56 * sr / frame_shift)
    sub_seq_step = int(0.64 * sr / frame_shift)

    feature_len = ac_feature.shape[1]

    sub_train_x = []
    sub_train_y = []
    for i in range(0, feature_len-sub_seq_len, sub_seq_step):
        sub_seq_x = np.transpose(ac_feature[:, i: i+sub_seq_len])
        sub_train_x.append(sub_seq_x[np.newaxis, :, :])
    return np.vstack(sub_train_x)


predict_x = extract_feature()
print(predict_x.shape)

predict_y = model.predict(predict_x)
print(predict_y.shape)
predict_y = np.argmax(predict_y, axis=-1)


print(predict_y.max(), predict_y.sum(axis=1))