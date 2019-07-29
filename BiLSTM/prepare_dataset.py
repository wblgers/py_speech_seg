import librosa
import numpy as np
file_list = ['ES2003a', 'ES2003b', 'ES2003c', 'ES2003d',
             'ES2011a', 'ES2011b', 'ES2011c', 'ES2011d',
             'TS3004a', 'TS3004b', 'TS3004c', 'TS3004d',
             'IS1008a', 'IS1008b', 'IS1008c', 'IS1008d',
             'TS3004a', 'TS3004b', 'TS3004c', 'TS3004d',
             'TS3006a', 'TS3006b', 'TS3006c', 'TS3006d'
             ]


def extract_feature(file_name):
    file = file_name+'.Mix-Headset.wav'
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

    change_point = []
    with open('dev.mdtm', 'r') as f:
        content = f.readlines()
        for line in content:
            sess_id = line.split()[0]
            if sess_id in file_name:
                start_p = float(line.split()[2])
                end_p = float(line.split()[2])+float(line.split()[3])
                # change_point.append(start_p)
                dur_1 = int((end_p-0.075)*sr)  # left 50ms
                dur_2 = int((end_p+0.075)*sr)  # right 50ms
                change_point.append((dur_1, dur_2))
    # change_point = np.array(change_point)

    sub_seq_len = int(3.2*sr/frame_shift)
    sub_seq_step= int(0.8*sr/frame_shift)

    feature_len = ac_feature.shape[1]

    def is_change_point(n):
        flag = False
        for x in change_point:
            if n > x[0] and n < x[1]:
                flag = True
                break

            if n+frame_size-1 > x[0] and n+frame_size-1 < x[1]:
                flag = True
                break
        return flag

    sub_train_x = []
    sub_train_y = []
    for i in range(0, feature_len-sub_seq_len, sub_seq_step):
        sub_seq_x = np.transpose(ac_feature[:, i: i+sub_seq_len])
        sub_train_x.append(sub_seq_x[np.newaxis, :, :])
        tmp = []
        for index in range(i, i+sub_seq_len):
            if is_change_point(index*frame_shift):
                tmp.append(1)
            else:
                tmp.append(0)
        lab_y = np.array(tmp)
        lab_y = np.reshape(lab_y, (1, sub_seq_len))
        sub_train_y.append(lab_y)
    return sub_train_x, sub_train_y


def load_dataset():
    all_x = []
    all_y = []
    for audio_file in file_list:
        new_train_x, new_train_y = extract_feature(audio_file)
        new_train_x = np.vstack(new_train_x)
        new_train_y = np.vstack(new_train_y)
        print(new_train_x.shape)
        print(new_train_y.shape)

        all_x.append(new_train_x)
        all_y.append(new_train_y)
    print(len(all_x))
    print(len(all_y))

    all_x_stack = np.vstack(all_x)
    all_y_stack = np.vstack(all_y)
    print(all_x_stack.shape, all_y_stack.shape)
    print('over')
    return all_x_stack, all_y_stack

