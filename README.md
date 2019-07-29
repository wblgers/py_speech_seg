# py_speech_seg
A toolkit to implement segmentation on speech based on BIC and nerual network, such as BiLSTM

## Dependency
- Python>=3.6
- tensorflow=1.13.1
- keras=2.2.4
- Librosa
- Numpy
- Scipy

You can use the installation of Anaconda to satisfy the required packages except [Librosa](https://github.com/librosa/librosa).

To install librosa, you can try the following command:

`conda install -c conda-forge librosa`
## Example Usage for BIC segmentation


1. Run script multi_detect.py to test the segmentation on a simple wav file:

    `python multi_detect_BIC.py`

   And you can get a speech segmentation result as showm below:

![Alt text](https://github.com/wblgers/py_speech_seg/raw/master/pictures/Demo1.png)

2. In the python script of multi_detect.py, there is a function call after some parameter settings:
    
    `seg_point = seg.multi_segmentation("dialog4.wav",sr,frame_size,frame_shift,plot_seg=False,save_seg=True)`
    
   To save the segmented audio into wav files, set the flag `save_seg=True`

   To plot out the wave figure in time domain with segmentation lines on, set the flag `plot_seg=True`

3. Add a new parameter interface to enable the "Clustering segmented audio fragment using Kmeans method", just set the flag:
    `classify_seg=True`
   
   To determine the number of cluster number, I plot out a figure with X axis the number of clusters, Y axis is the "Sum of squared distances of samples to their closest cluster center" for each Kmeans clustering. Choose the best K value under Elbow Criterion:

![Alt text](https://github.com/wblgers/py_speech_seg/raw/master/pictures/kmeans_number_of_clusters_evaluate.png)

   From the figure shown abvove, I choose K = 2 to be the best cluster numbers:
   
   Please input the best K value: 2

   The lables for 4 speech segmentation belongs to the clusters below:

   0 
   1 
   0 
   1 

   From the audio files stored in folder "save_audio", we can check that the clustering result is right.

4. Change the interface in 3 to be the definition of the clustering method you choose. Now the supported methods are "Kmeans" and "BIC distance". Also, the clustering method based on "BIC distance" is inspired by the Reference article.

   Meanwhile, I use a longer audio file to test the new clustering method, there are totally 7 segments in "duihua_sample.wav". The final clustering results is as below:
```
There are total 2 clusters and they are listed below: 
cluster 0 :  ['1', '3', '5']
cluster 1 :  ['0', '2', '4', '6']
```
## Example Usage for nerual network segmentation(To be continued)
1. Train the network

    `python train_bilstm_model.py`

2. Predict the segmentation points
    `python multi_detect_Nerual.py`

## My Blog for this project
[Python实现基于BIC的语音对话分割(一)](https://blog.csdn.net/wblgers1234/article/details/75896605)

[Python实现基于BIC的语音对话分割(二)](https://blog.csdn.net/wblgers1234/article/details/77103444)

## Reference
*Speaker, Environment and Channel Change Detection and Clustering via the Bayesian Information Criterion*, by IBM T.J. Watson Research Center
*Speaker Change Detection in Broadcast TV using Bidirectional Long Short-Term Memory Networks, by Ruiqing Yin, Herve Bredin, Claude Barras