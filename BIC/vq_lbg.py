import numpy as np

def vqlbg(d,k):
    e = 0.01
    r = np.mean(d,axis=1)
    r = r.reshape((r.shape[0],1))
    dpr = 10000.0
    loop_cnt = int(np.log2(k))

    output_r = np.zeros((r.shape[0], k), dtype=np.float32)
    available_cols = 0
    for i in range(loop_cnt):
        r = np.concatenate((r * (1 + e), r * (1 - e)), axis=1)
        whileFlag = 0
        while 1:
            if whileFlag != 0:
                r = output_r[:, 0:available_cols]
            z = disteu(d, r)
            ind = np.argmin(z,axis = 1)
            t=0

            available_cols = np.power(2,i+1)
            for j in range(np.power(2,i+1)):
                equal_ind = np.where(ind == j)
                equal_ind = equal_ind[0]
                temp = d[:,equal_ind]
                temp_r = np.mean(temp,axis=1)
                output_r[:,j] = temp_r

                temp_r = temp_r.reshape((r.shape[0], 1))
                x = disteu(temp,temp_r)
                t = t+np.sum(x)
            if ((dpr-t)/t)<e:
                break
            else:
                dpr = t

            whileFlag = 1
    return output_r



def disteu(x, y):
    M = x.shape[0]
    N = x.shape[1]
    M2 = y.shape[0]
    P = y.shape[1]

    d = np.zeros((N,P), dtype=np.float32)

    if N<P:
        copies = np.zeros(P,dtype=np.int32)
        for n in range(N):
            d[n,:]= np.sum(np.power((x[:,n+copies]-y),2),axis=0)
    else:
        copies = np.zeros(N, dtype=np.int32)
        for p in range(P):
            temp = np.sum(np.power((x-y[:, p + copies]),2),axis=0)
            d[:,p] = temp.T
    d = np.power(d,0.5)
    return d