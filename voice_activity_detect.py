from __future__ import print_function
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def vad(x,framelen = None, sr = None, frameshift = None, plot = False):
    if sr is None:
        sr = 16000
    if framelen is None:
        framelen = 256
    if frameshift is None:
        frameshift = 128
    amp_th1 = 8
    amp_th2 = 20
    zcr_th  = 5

    maxsilence = 8
    minlen     = 15
    status     = 0
    count      = 0
    silence    = 0

    x = x/np.absolute(x).max()

    tmp1 = enframe(x[0:(len(x)-1)], framelen, frameshift)
    tmp2 = enframe(x[1:(len(x)-1)], framelen, frameshift)
    signs = (tmp1* tmp2) < 0
    diffs = (tmp1 - tmp2) > 0.05
    zcr = np.sum(signs* diffs, axis=1)

    filter_coeff = np.array([1, -0.9375])
    pre_emphasis = signal.convolve(x,filter_coeff)[0:len(x)]
    amp = np.sum(np.absolute(enframe(pre_emphasis, framelen, frameshift)), axis=1)

    amp_th1 = min(amp_th1, amp.max()/ 3)
    amp_th2 = min(amp_th2, amp.max() / 8)

    x1 = []
    x2 = []
    t = 0

    for n in range(len(zcr)):
        if status == 0 or status == 1:
            if amp[n] > amp_th1:
                x1.append(max(n - count - 1, 1))
                status = 2
                silence = 0
                count = count + 1
            elif amp[n] > amp_th2 or zcr[n]>zcr_th:
                status = 1
                count = count + 1
            else:
                status = 0
                count = 0
            continue
        if status == 2:
            if amp[n] > amp_th2 or zcr[n]>zcr_th:
                count = count + 1
            else:
                silence = silence + 1
                if silence < maxsilence:
                    count = count + 1
                elif count < minlen:
                    status = 0
                    silence = 0
                    count = 0
                else:
                    status = 0
                    count = count - silence / 2
                    x2.append(x1[t] + count - 1)
                    t = t + 1

    if plot:
        plt.figure('speech endpoint detect')
        plt.plot(np.arange(0,len(x))/(float)(sr),x, "b-")
        len_endpoint = min(len(x1), len(x2))
        for i in range(len_endpoint):
            plt.vlines(x1[i]*frameshift/(float)(sr), -1, 1, colors="c", linestyles="dashed")
            plt.vlines(x2[i] * frameshift/(float)(sr), -1, 1, colors="r", linestyles="dashed")
        plt.xlabel("Time/s")
        plt.ylabel("Normalized Amp")
        plt.grid(True)
        plt.show()
    return x1,x2


def enframe(x,framelen, frameshift):
    xlen = len(x)
    nf   = (int)((xlen-framelen+frameshift)/frameshift)
    f    = np.zeros((nf,framelen), dtype=np.float32)
    indf = frameshift * (np.arange(0,nf)).reshape(nf,1)
    inds = np.arange(0,framelen).reshape(1,framelen)
    indall = np.tile(indf, (1, framelen))+np.tile(inds, (nf, 1))
    f = x[indall]
    return f

