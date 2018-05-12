import numpy as np
import voice_activity_detect as vad
import vq_lbg as vqlbg
import librosa
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.cluster import KMeans


#Speech segmentation based on BIC
def compute_bic(mfcc_v,delta):
    m, n = mfcc_v.shape

    sigma0 = np.cov(mfcc_v).diagonal()
    eps = np.spacing(1)
    realmin = np.finfo(np.double).tiny
    det0 = max(np.prod(np.maximum(sigma0,eps)),realmin)

    flat_start = 5

    range_loop = range(flat_start,n,delta)
    x = np.zeros(len(range_loop))
    iter = 0
    for index in range_loop:
        part1 = mfcc_v[:, 0:index]
        part2 = mfcc_v[:, index:n]

        sigma1 = np.cov(part1).diagonal()
        sigma2 = np.cov(part2).diagonal()

        det1 = max(np.prod(np.maximum(sigma1, eps)), realmin)
        det2 = max(np.prod(np.maximum(sigma2, eps)), realmin)

        BIC = 0.5*(n*np.log(det0)-index*np.log(det1)-(n-index)*np.log(det2))-0.5*(m+0.5*m*(m+1))*np.log(n)
        x[iter] = BIC
        iter = iter + 1

    maxBIC = x.max()
    maxIndex = x.argmax()
    if maxBIC>0:
        return range_loop[maxIndex]-1
    else:
        return -1


def speech_segmentation(mfccs):
    wStart = 0
    wEnd = 200
    wGrow = 200
    delta = 25

    m, n = mfccs.shape

    store_cp = []
    index = 0
    while wEnd < n:
        featureSeg = mfccs[:, wStart:wEnd]
        detBIC = compute_bic(featureSeg, delta)
        index = index + 1
        if detBIC > 0:
            temp = wStart + detBIC
            store_cp.append(temp)
            wStart = wStart + detBIC + 200
            wEnd = wStart + wGrow
        else:
            wEnd = wEnd + wGrow

    return np.array(store_cp)

def multi_segmentation(file,sr,frame_size,frame_shift,plot_seg = False,save_seg = False,classify_seg = False):
    y, sr = librosa.load(file, sr=sr)

    mfccs = librosa.feature.mfcc(y, sr, n_mfcc=12, hop_length=frame_shift, n_fft=frame_size)
    seg_point = speech_segmentation(mfccs / mfccs.max())

    seg_point = seg_point * frame_shift
    seg_point = np.insert(seg_point, 0, 0)
    seg_point = np.append(seg_point, len(y))
    rangeLoop = range(len(seg_point) - 1)

    output_segpoint = []
    for i in rangeLoop:
        temp = y[seg_point[i]:seg_point[i + 1]]
        x1, x2 = vad.vad(temp, sr=sr, framelen=frame_size, frameshift=frame_shift)
        if len(x1) == 0 or len(x2) == 0:
            continue
        elif seg_point[i + 1] == len(y):
            continue
        else:
            output_segpoint.append(seg_point[i + 1])

    if plot_seg:
        plt.figure('speech segmentation plot')
        plt.plot(np.arange(0, len(y)) / (float)(sr), y, "b-")

        for i in range(len(output_segpoint)):
            plt.vlines(output_segpoint[i] / (float)(sr), -1, 1, colors="c", linestyles="dashed")
            plt.vlines(output_segpoint[i] / (float)(sr), -1, 1, colors="r", linestyles="dashed")
        plt.xlabel("Time/s")
        plt.ylabel("Speech Amp")
        plt.grid(True)
        plt.show()

    if save_seg:
        if not os.path.exists("save_audio"):
            os.makedirs("save_audio")
        else:
            shutil.rmtree("save_audio")
            os.makedirs("save_audio")
        save_segpoint = output_segpoint.copy()
        # Add the start and the end of the audio file
        save_segpoint.insert(0,0)
        save_segpoint.append(len(y))
        for i in range(len(save_segpoint)-1):
            tempAudio = y[save_segpoint[i]:save_segpoint[i+1]]
            librosa.output.write_wav("save_audio/%s.wav"%i,tempAudio,sr)

    if classify_seg:
        classify_segpoint = output_segpoint.copy()
        # Add the start and the end of the audio file
        classify_segpoint.insert(0, 0)
        classify_segpoint.append(len(y))

        # Length of codebook
        k = 16
        vq_features = np.zeros((len(classify_segpoint) - 1,k*12),dtype=np.float32)
        for i in range(len(classify_segpoint) - 1):
            tempAudio = y[classify_segpoint[i]:classify_segpoint[i + 1]]
            mfccs = librosa.feature.mfcc(tempAudio, sr, n_mfcc=12, hop_length=frame_shift, n_fft=frame_size)
            mfccs = mfccs / mfccs.max()
            vq_code = vqlbg.vqlbg(mfccs,k)
            vq_features[i,:] = vq_code.reshape(1,vq_code.shape[0]*vq_code.shape[1])

        K = range(1,len(classify_segpoint))
        square_error = []
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(vq_features)
            square_error.append(kmeans.inertia_)

        plt.figure('Kmeans Number of clusters evaluate')
        plt.plot(K, square_error, "bo-")
        plt.title('Please choose the best number of clusters under Elbow Criterion')
        plt.xlabel("Number of clusters")
        plt.ylabel("SSE For each step")
        plt.ylim(0,square_error[0]*1.5)
        plt.grid(True)
        plt.show()

        k_n = input("Please input the best K value: ")
        kmeans = KMeans(int(k_n), random_state=0).fit(vq_features)
        print("The lables for",len(kmeans.labels_),"speech segmentation belongs to the clusters below:")
        for i in range(len(kmeans.labels_)):
            print(kmeans.labels_[i],"")
    return (np.asarray(output_segpoint) / float(sr))